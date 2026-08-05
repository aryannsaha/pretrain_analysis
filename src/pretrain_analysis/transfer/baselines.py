"""Competing transferability predictors, so LogME can be judged against them.

LogME on its own is not a result.  Every estimator here answers the same
question -- "will this frozen representation fine-tune well?" -- by a different
and cheaper route, and the point of the study is whether LogME beats them.

Implemented
-----------
:func:`ridge_probe`
    Fit a ridge regression on frozen features, report held-out R^2 and MAE.
    The "just try a linear probe" baseline that LogME claims to improve on.
:func:`h_score`
    Bao et al. (2019), adapted to regression by binning the target.  Both the
    plain and shrinkage-regularized variants.
:func:`gaussian_distance` / :func:`gmm_log_likelihood`
    Distributional distance between the pretraining and target embedding
    clouds.  Mahalanobis under a single Gaussian, and mean log-likelihood
    under a full-covariance GMM.  Motivated for MLIPs by Tan et al.,
    npj Comput. Mater. 2023 (arXiv:2305.01754).
:func:`participation_ratio`
    Effective dimensionality of the feature covariance spectrum.  Cheap proxy
    for representational richness; uses no target information at all.

Deliberately omitted
--------------------
**LEEP** (Nguyen et al. 2020) and **NCE** (Tran et al. 2019) both require the
source model to expose a classification head, so that source-label posteriors
can be pushed through to target labels.  A MACE energy/force regressor has no
such head -- its readout emits a scalar per atom -- and there is no
label-free substitute that preserves what those estimators measure.  They are
inapplicable here rather than merely inconvenient, and are reported as such
rather than silently dropped.

All functions take float64 and return plain floats or small dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GMMResult",
    "RidgeProbeResult",
    "effective_rank",
    "gaussian_distance",
    "gmm_log_likelihood",
    "h_score",
    "participation_ratio",
    "ridge_probe",
]

_DEFAULT_ALPHAS = np.logspace(-6, 6, 25)


# --------------------------------------------------------------------------
# ridge probe
# --------------------------------------------------------------------------


@dataclass
class RidgeProbeResult:
    r2: float
    mae: float
    rmse: float
    alpha: float
    n_train: int
    n_test: int
    d: int

    def as_dict(self) -> dict:
        return {
            "ridge_r2": self.r2,
            "ridge_mae": self.mae,
            "ridge_rmse": self.rmse,
            "ridge_alpha": self.alpha,
            "ridge_n_train": self.n_train,
            "ridge_n_test": self.n_test,
        }


def ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    test_fraction: float = 0.25,
    alphas: np.ndarray | None = None,
    seed: int = 0,
) -> RidgeProbeResult:
    """Ridge regression on frozen features, scored on a held-out split.

    The regularization strength is chosen on a *validation* split carved out of
    the training half, never on the test half, so the reported R^2 and MAE are
    honest held-out numbers.

    The whole ridge path is evaluated from a single SVD of the training
    features: with ``F = U S V^T`` the ridge solution for penalty ``a`` is
    ``w(a) = V diag(s / (s^2 + a)) U^T y``, so sweeping ``a`` costs one
    length-``r`` vector operation per value rather than a fresh solve.

    Returns held-out R^2 (1 - SSE/SST, so it can go negative), MAE and RMSE in
    the units of ``targets``.
    """
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).ravel()
    if f.shape[0] != y.shape[0]:
        raise ValueError("features and targets disagree on the number of rows")
    if alphas is None:
        alphas = _DEFAULT_ALPHAS

    rng = np.random.default_rng(seed)
    n = f.shape[0]
    perm = rng.permutation(n)
    n_test = max(1, int(round(test_fraction * n)))
    n_val = max(1, int(round(test_fraction * (n - n_test))))
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    if train_idx.size < 2:
        raise ValueError(f"Not enough rows to split: n={n}")

    # Standardize on training statistics only.  Unlike LogME (where the
    # isotropic prior makes column scaling meaningful), ridge with a single
    # penalty genuinely needs comparable column scales, so standardizing here
    # is the correct choice rather than an arbitrary one.
    mu = f[train_idx].mean(axis=0)
    sd = f[train_idx].std(axis=0)
    sd[sd < 1e-12] = 1.0
    fs = (f - mu) / sd
    y_mu = y[train_idx].mean()

    u, s, vt = np.linalg.svd(fs[train_idx], full_matrices=False)
    z = u.T @ (y[train_idx] - y_mu)

    def _predict(idx: np.ndarray, a: float) -> np.ndarray:
        w = vt.T @ (s * z / (s**2 + a))
        return fs[idx] @ w + y_mu

    val_err = [np.abs(_predict(val_idx, a) - y[val_idx]).mean() for a in alphas]
    best_alpha = float(alphas[int(np.argmin(val_err))])

    pred = _predict(test_idx, best_alpha)
    resid = pred - y[test_idx]
    sst = float(np.sum((y[test_idx] - y[test_idx].mean()) ** 2))

    return RidgeProbeResult(
        r2=float(1.0 - np.sum(resid**2) / sst) if sst > 0 else float("nan"),
        mae=float(np.abs(resid).mean()),
        rmse=float(np.sqrt(np.mean(resid**2))),
        alpha=best_alpha,
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
        d=int(f.shape[1]),
    )


# --------------------------------------------------------------------------
# H-score
# --------------------------------------------------------------------------


def h_score(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 20,
    shrinkage: float | None = None,
) -> float:
    """H-score (Bao et al., 2019), adapted to a continuous target.

    The original is defined for classification as

        H = tr( cov(f)^-1 cov( E[f | y] ) )

    i.e. how much of the feature variance is explained by variation of the
    class-conditional mean.  A regression target has no classes, so ``y`` is
    binned into ``n_bins`` equal-frequency (quantile) bins and each bin is
    treated as a class.  This is the standard adaptation, but it *is* an
    adaptation: the score depends on ``n_bins``, and that dependence is a
    caveat rather than a nuisance parameter to tune away.

    Parameters
    ----------
    shrinkage:
        Ledoit-Wolf-style shrinkage of the covariance toward a scaled identity,
        ``(1 - g) * S + g * (tr(S)/d) * I``.  MACE feature covariances are
        badly conditioned, and the plain H-score inverts ``cov(f)`` directly,
        so the unregularized value is dominated by near-null directions.
        ``None`` (default) picks a small automatic value; pass ``0.0`` to force
        the unregularized estimator.
    """
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).ravel()
    n, d = f.shape

    fc = f - f.mean(axis=0, keepdims=True)
    cov_f = (fc.T @ fc) / max(n - 1, 1)

    # Equal-frequency binning: robust to the heavy-tailed energy and
    # force-magnitude distributions, where equal-width bins would leave most
    # bins empty.
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return float("nan")  # target is effectively constant
    bin_id = np.clip(np.digitize(y, edges[1:-1]), 0, edges.size - 2)

    # cov of the bin-conditional means, weighted by bin occupancy.
    means = np.zeros((edges.size - 1, d))
    weights = np.zeros(edges.size - 1)
    for b in range(edges.size - 1):
        m = bin_id == b
        if not m.any():
            continue
        means[b] = fc[m].mean(axis=0)
        weights[b] = m.sum() / n
    mw = means * weights[:, None]
    cov_cond = means.T @ mw

    if shrinkage is None:
        # Enough to make the inverse well-posed without washing out structure.
        shrinkage = 1e-3 if n > d else 1e-1
    if shrinkage > 0:
        mu_trace = np.trace(cov_f) / d
        cov_f = (1 - shrinkage) * cov_f + shrinkage * mu_trace * np.eye(d)

    try:
        solved = np.linalg.solve(cov_f, cov_cond)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(cov_f) @ cov_cond
    return float(np.trace(solved))


# --------------------------------------------------------------------------
# distributional distance between pretraining and target embeddings
# --------------------------------------------------------------------------


def gaussian_distance(
    source: np.ndarray,
    target: np.ndarray,
    *,
    shrinkage: float = 1e-3,
) -> dict:
    """Mahalanobis distance of target embeddings under a source-fitted Gaussian.

    Fits a single full-covariance Gaussian to ``source`` (a sample of OMat24
    embeddings from the frozen model) and reports how far the ``target``
    embeddings sit from it.  Larger distance means the downstream set is
    further outside the pretraining distribution.

    Returns mean and median Mahalanobis distance, and the mean log-likelihood,
    which is the same quantity plus the log-determinant normalization.
    """
    s = np.asarray(source, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    d = s.shape[1]

    mu = s.mean(axis=0)
    sc = s - mu
    cov = (sc.T @ sc) / max(s.shape[0] - 1, 1)
    cov = (1 - shrinkage) * cov + shrinkage * (np.trace(cov) / d) * np.eye(d)

    # Cholesky then triangular solve: numerically better than forming cov^-1,
    # and gives the log-determinant for free.
    chol = np.linalg.cholesky(cov)
    delta = np.linalg.solve(chol, (t - mu).T)
    m2 = np.einsum("ij,ij->j", delta, delta)
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    log_lik = -0.5 * (m2 + log_det + d * np.log(2 * np.pi))

    return {
        "mahalanobis_mean": float(np.sqrt(m2).mean()),
        "mahalanobis_median": float(np.median(np.sqrt(m2))),
        "gaussian_loglik_mean": float(log_lik.mean()),
    }


@dataclass
class GMMResult:
    n_components: int
    converged: bool
    n_iter: int
    source_loglik: float
    target_loglik: float

    def as_dict(self) -> dict:
        return {
            "gmm_target_loglik": self.target_loglik,
            "gmm_source_loglik": self.source_loglik,
            "gmm_loglik_gap": self.source_loglik - self.target_loglik,
            "gmm_converged": self.converged,
        }


def _gmm_log_prob(x: np.ndarray, means, chols, log_weights) -> np.ndarray:
    """Per-component log N(x | mu_k, Sigma_k) + log pi_k. Shape (n, k)."""
    n, d = x.shape
    out = np.empty((n, len(means)))
    for k, (mu, chol) in enumerate(zip(means, chols, strict=True)):
        delta = np.linalg.solve(chol, (x - mu).T)
        m2 = np.einsum("ij,ij->j", delta, delta)
        log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
        out[:, k] = -0.5 * (m2 + log_det + d * np.log(2 * np.pi)) + log_weights[k]
    return out


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    return (amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))).squeeze(axis)


def gmm_log_likelihood(
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_components: int = 8,
    max_iter: int = 200,
    tol: float = 1e-5,
    reg_covar: float = 1e-4,
    seed: int = 0,
) -> GMMResult:
    """Mean log-likelihood of target embeddings under a GMM fit to the source.

    Full-covariance EM, written out rather than pulled from scikit-learn so the
    package runs unchanged in the MACE conda environment (which has no sklearn)
    and so the regularization is visible.

    ``reg_covar`` is added to the covariance diagonal at every M step.  It is
    load-bearing, not cosmetic: MACE invariant features have near-null
    directions, and without a floor a component collapses onto a handful of
    points and the likelihood diverges to ``+inf``.

    Initialization is k-means++ style seeding followed by hard assignment,
    which is deterministic given ``seed``.
    """
    s = np.asarray(source, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    n, d = s.shape
    k = min(n_components, n)
    rng = np.random.default_rng(seed)

    # k-means++ seeding
    centers = [s[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(
            np.stack([np.sum((s - c) ** 2, axis=1) for c in centers], axis=1), axis=1
        )
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(s[rng.choice(n, p=probs)])
    means = np.stack(centers)

    # A few Lloyd iterations to get sane starting responsibilities.
    for _ in range(10):
        assign = np.argmin(
            np.stack([np.sum((s - m) ** 2, axis=1) for m in means], axis=1), axis=1
        )
        for j in range(k):
            m = assign == j
            if m.any():
                means[j] = s[m].mean(axis=0)

    weights = np.full(k, 1.0 / k)
    base_cov = np.cov(s, rowvar=False) + reg_covar * np.eye(d)
    covs = np.stack([base_cov.copy() for _ in range(k)])

    prev_ll = -np.inf
    converged = False
    it = 0
    chols = [np.linalg.cholesky(c) for c in covs]

    for it in range(1, max_iter + 1):
        # E step
        log_prob = _gmm_log_prob(s, means, chols, np.log(weights))
        ll_per_point = _logsumexp(log_prob, axis=1)
        ll = float(ll_per_point.mean())
        resp = np.exp(log_prob - ll_per_point[:, None])

        if abs(ll - prev_ll) <= tol * abs(prev_ll if prev_ll != -np.inf else 1.0):
            converged = True
            break
        prev_ll = ll

        # M step
        nk = resp.sum(axis=0) + 1e-12
        weights = nk / n
        means = (resp.T @ s) / nk[:, None]
        chols = []
        for j in range(k):
            dev = s - means[j]
            cov_j = (dev * resp[:, j : j + 1]).T @ dev / nk[j]
            cov_j.flat[:: d + 1] += reg_covar
            covs[j] = cov_j
            chols.append(np.linalg.cholesky(cov_j))

    src_ll = float(_logsumexp(_gmm_log_prob(s, means, chols, np.log(weights)), axis=1).mean())
    tgt_ll = float(_logsumexp(_gmm_log_prob(t, means, chols, np.log(weights)), axis=1).mean())

    return GMMResult(
        n_components=k,
        converged=converged,
        n_iter=it,
        source_loglik=src_ll,
        target_loglik=tgt_ll,
    )


# --------------------------------------------------------------------------
# effective dimensionality
# --------------------------------------------------------------------------


def participation_ratio(features: np.ndarray) -> float:
    """Participation ratio of the feature covariance spectrum.

    ``PR = (sum_i lam_i)^2 / sum_i lam_i^2``, between 1 (all variance in one
    direction) and ``d`` (isotropic).  A target-free measure of how many
    directions the representation actually uses.

    Computed from the singular values of the centered feature matrix rather
    than by forming the covariance, which would square the condition number.
    """
    f = np.asarray(features, dtype=np.float64)
    fc = f - f.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(fc, compute_uv=False)
    lam = sv**2
    denom = float(np.sum(lam**2))
    return float(np.sum(lam) ** 2 / denom) if denom > 0 else float("nan")


def effective_rank(features: np.ndarray) -> float:
    """Effective rank: ``exp(H)`` of the normalized eigenvalue distribution.

    Reported next to :func:`participation_ratio` because the two weight the
    spectrum tail differently -- participation ratio is dominated by the top
    eigenvalues, entropy-based effective rank is more sensitive to the tail.
    Where they disagree, the spectrum is heavy-tailed.
    """
    f = np.asarray(features, dtype=np.float64)
    fc = f - f.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(fc, compute_uv=False)
    lam = sv**2
    total = lam.sum()
    if total <= 0:
        return float("nan")
    p = lam / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))
