"""LogME: evidence-maximization transferability estimation.

Implements the estimator of You et al., "LogME: Practical Assessment of
Pre-trained Models for Transfer Learning", ICML 2021 (arXiv:2102.11005),
from scratch so that every step is auditable.

Model
-----
Given a frozen feature matrix ``F`` of shape ``(n, d)`` and a target ``y`` of
shape ``(n,)``, fit the Bayesian linear model

    p(y | F, w, beta) = N(y | F w, beta^-1 I)
    p(w | alpha)      = N(w | 0, alpha^-1 I)

and maximize the marginal likelihood (evidence) over ``(alpha, beta)``.  The
maximized evidence, normalized by ``n``, is the transferability score.

Marginalizing ``w`` analytically gives

    log p(y | F, alpha, beta) = (d/2) log alpha + (n/2) log beta
                                - (beta/2) ||F m - y||^2
                                - (alpha/2) ||m||^2
                                - (1/2) log|A|
                                - (n/2) log(2 pi)

with posterior precision ``A = alpha I_d + beta F^T F`` and posterior mean
``m = beta A^-1 F^T y``.

Everything is evaluated through a single thin SVD ``F = U S V^T``.  Writing
``sigma_i`` for the singular values and ``z = U^T y``:

    gamma        = sum_i (beta sigma_i^2) / (alpha + beta sigma_i^2)
    ||m||^2      = sum_i (beta sigma_i z_i)^2 / (alpha + beta sigma_i^2)^2
    ||F m - y||^2 = sum_i (alpha z_i / (alpha + beta sigma_i^2))^2
                    + (||y||^2 - sum_i z_i^2)
    log|A|       = sum_i log(alpha + beta sigma_i^2) + (d - r) log alpha

The fixed-point updates (MacKay evidence maximization) are

    alpha <- gamma / ||m||^2
    beta  <- (n - gamma) / ||F m - y||^2

Two details that are easy to get wrong, and that this implementation is
explicit about:

1. The ``||y||^2 - sum_i z_i^2`` term is the component of ``y`` outside the
   column space of ``F``.  It is nonzero whenever ``r < n`` (in particular
   whenever ``d < n``, the usual case here) and dropping it makes ``beta``
   diverge.
2. ``A`` is ``d x d`` and always has ``d`` eigenvalues, but the thin SVD only
   produces ``r = numerical_rank(F) <= min(n, d)`` of them.  The remaining
   ``d - r`` eigenvalues equal ``alpha`` exactly, contributing
   ``(d - r) log alpha`` to ``log|A|``.  The reference implementation
   (``thuml/LogME``) omits this term; it vanishes when ``r == d`` (the common
   ``n > d`` full-rank case, where the two agree to machine precision) but is
   required for correctness when ``n < d`` or when ``F`` is rank deficient.
   See ``tests/transfer/test_logme.py::test_matches_reference_formula_when_full_rank``.

Invariances (verified in the tests, and worth keeping in mind when comparing
scores):

* Invariant to right-multiplying ``F`` by an orthogonal matrix, and to a
  global rescaling ``F -> c F``, which is absorbed by ``alpha -> c^2 alpha``
  (the weights that reproduce the same function shrink to ``w / c``, so the
  prior precision that keeps the same prior over functions grows by ``c^2``).
* **Not** invariant to per-column rescaling: the prior ``N(0, alpha^-1 I)`` is
  isotropic, so it is not exchangeable under anisotropic reparameterization of
  the feature space.  Standardizing columns therefore changes the score, and a
  single convention must be fixed and applied to every model being compared.
* Under ``y -> c y`` the score shifts by ``-log|c|`` (a change-of-variables
  Jacobian).  Scores are therefore comparable across *models* for a fixed
  target, but LogME values for different targets (e.g. energy vs force
  magnitude) live on different scales and must never be compared to each other.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "LogMEResult",
    "logme",
    "logme_multi",
]

# Below this ratio to the largest singular value, a direction is treated as
# numerically absent from the column space of F.  Standard LAPACK convention.
_RANK_RTOL_FACTOR = 1.0


class LogMEConvergenceWarning(UserWarning):
    """Raised when the evidence fixed-point iteration hits ``max_iter``."""


@dataclass
class LogMEResult:
    """Outcome of a single-target LogME fit.

    Attributes
    ----------
    score:
        The transferability score, ``log p(y | F) / n``.  Higher is better.
        Units are nats per observation.
    alpha, beta:
        Maximized prior and noise precisions.  ``1 / sqrt(beta)`` is the fitted
        residual standard deviation in the units of ``y``, which is a useful
        sanity check against the known scale of the target.
    evidence:
        Unnormalized ``log p(y | F)``.
    n, d:
        Shape of the feature matrix actually used.
    rank:
        Numerical rank of ``F``.  If this is much smaller than ``min(n, d)``,
        the features are degenerate and the score should be read with care.
    gamma:
        Effective number of well-determined parameters at the optimum,
        ``sum_i beta sigma_i^2 / (alpha + beta sigma_i^2)`` in ``[0, rank]``.
        This is the quantity that gives LogME its resistance to overfitting:
        uninformative directions contribute nothing to ``gamma``.
    n_iter:
        Fixed-point iterations used.
    converged:
        Whether the iteration met ``tol`` before ``max_iter``.
    residual_out_of_span:
        ``||y||^2 - sum_i z_i^2``, the part of ``y`` unreachable by any linear
        map of ``F``.  Reported because silently dropping it is the single most
        common LogME implementation bug.
    """

    score: float
    alpha: float
    beta: float
    evidence: float
    n: int
    d: int
    rank: int
    gamma: float
    n_iter: int
    converged: bool
    residual_out_of_span: float

    def as_dict(self) -> dict:
        return {
            "logme": self.score,
            "alpha": self.alpha,
            "beta": self.beta,
            "evidence": self.evidence,
            "n": self.n,
            "d": self.d,
            "rank": self.rank,
            "gamma": self.gamma,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "residual_out_of_span": self.residual_out_of_span,
        }


@dataclass
class _Decomposition:
    """Thin SVD of ``F``, truncated to numerical rank. Computed once, reused."""

    u: np.ndarray  # (n, r)
    sigma_sq: np.ndarray  # (r,) squared singular values
    n: int
    d: int
    rank: int
    sq_sum_all: np.ndarray = field(default_factory=lambda: np.empty(0))


def _decompose(features: np.ndarray, rank_rtol: float | None) -> _Decomposition:
    """Thin SVD of ``features``, truncated at the numerical rank.

    Truncation matters for stability: directions with ``sigma`` at the level of
    floating-point noise carry no information, and keeping them injects
    ``z_i / sigma_i``-scale garbage into ``||m||^2``.  Their evidence
    contribution ``log(alpha + beta sigma_i^2) -> log alpha`` is exactly what
    the ``(d - r) log alpha`` term supplies, so truncating is not an
    approximation -- it is the same quantity, computed stably.
    """
    n, d = features.shape
    # full_matrices=False gives the thin SVD: u is (n, k), sv is (k,) with
    # k = min(n, d).  We never need V, which is the expensive part when d is
    # large, so this stays cheap even for wide feature matrices.
    u, sv, _ = np.linalg.svd(features, full_matrices=False)

    if rank_rtol is None:
        rank_rtol = _RANK_RTOL_FACTOR * max(n, d) * np.finfo(np.float64).eps
    cutoff = rank_rtol * (sv[0] if sv.size else 0.0)
    rank = int(np.count_nonzero(sv > cutoff))

    return _Decomposition(
        u=u[:, :rank],
        sigma_sq=(sv[:rank] ** 2),
        n=n,
        d=d,
        rank=rank,
    )


def _evidence(
    alpha: float,
    beta: float,
    m_sq: float,
    res_sq: float,
    sigma_sq: np.ndarray,
    n: int,
    d: int,
    rank: int,
) -> float:
    """Unnormalized ``log p(y | F, alpha, beta)``.

    ``log|A|`` is split into the ``rank`` directions the SVD resolved plus the
    ``d - rank`` directions where ``A`` reduces to ``alpha I``.
    """
    log_det_a = float(np.sum(np.log(alpha + beta * sigma_sq)))
    log_det_a += (d - rank) * np.log(alpha)
    return (
        0.5 * d * np.log(alpha)
        + 0.5 * n * np.log(beta)
        - 0.5 * beta * res_sq
        - 0.5 * alpha * m_sq
        - 0.5 * log_det_a
        - 0.5 * n * np.log(2.0 * np.pi)
    )


def _fit_one(
    dec: _Decomposition,
    y: np.ndarray,
    *,
    max_iter: int,
    tol: float,
    warn: bool,
) -> LogMEResult:
    """Evidence maximization for a single target vector, given a cached SVD."""
    n, d, rank = dec.n, dec.d, dec.rank

    z = dec.u.T @ y  # (rank,) projection of y onto the column space of F
    z_sq = z**2
    sigma_sq = dec.sigma_sq

    # Component of y orthogonal to span(F).  Clipped at 0 because catastrophic
    # cancellation can make it very slightly negative when F spans y almost
    # exactly; a negative value would drive beta negative and crash the log.
    res_out = float(np.dot(y, y) - z_sq.sum())
    res_out = max(res_out, 0.0)

    if not np.isfinite(res_out) or (res_out == 0.0 and rank == 0):
        raise ValueError("Degenerate target: y has no finite variation.")

    alpha, beta = 1.0, 1.0
    m_sq = res_sq = np.nan
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        denom = alpha + beta * sigma_sq  # (rank,)

        gamma = float(np.sum(beta * sigma_sq / denom))
        m_sq = float(np.sum((beta**2) * sigma_sq * z_sq / denom**2))
        res_sq = float(np.sum((alpha**2) * z_sq / denom**2)) + res_out

        # Guard the denominators.  m_sq -> 0 means the posterior mean collapsed
        # to the origin (no signal); res_sq -> 0 means the fit is exact.  Both
        # are legitimate limits in which the corresponding precision diverges,
        # so we floor rather than error, and let the convergence check report.
        new_alpha = gamma / max(m_sq, np.finfo(np.float64).tiny)
        new_beta = (n - gamma) / max(res_sq, np.finfo(np.float64).tiny)

        # Converge on the ratio alpha/beta: it is the only quantity the fixed
        # point actually determines (the evidence is a function of it plus the
        # data), and it is scale-free, so one relative tolerance works for
        # targets of any magnitude.
        ratio, new_ratio = alpha / beta, new_alpha / new_beta
        alpha, beta = new_alpha, new_beta
        if abs(new_ratio - ratio) <= tol * abs(ratio):
            converged = True
            break

    if not converged and warn:
        warnings.warn(
            f"LogME fixed-point iteration did not converge in {max_iter} "
            f"iterations (n={n}, d={d}, rank={rank}, alpha={alpha:.6g}, "
            f"beta={beta:.6g}). The score may be unreliable.",
            LogMEConvergenceWarning,
            stacklevel=3,
        )

    # Recompute the sufficient statistics at the final (alpha, beta) so that
    # the reported evidence is consistent with the reported hyperparameters
    # rather than lagging one iteration behind.
    denom = alpha + beta * sigma_sq
    gamma = float(np.sum(beta * sigma_sq / denom))
    m_sq = float(np.sum((beta**2) * sigma_sq * z_sq / denom**2))
    res_sq = float(np.sum((alpha**2) * z_sq / denom**2)) + res_out

    evidence = _evidence(alpha, beta, m_sq, res_sq, sigma_sq, n, d, rank)

    return LogMEResult(
        score=evidence / n,
        alpha=alpha,
        beta=beta,
        evidence=evidence,
        n=n,
        d=d,
        rank=rank,
        gamma=gamma,
        n_iter=n_iter,
        converged=converged,
        residual_out_of_span=res_out,
    )


def _prepare(
    features: np.ndarray,
    targets: np.ndarray,
    center_features: bool,
    center_targets: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate, cast to float64, and optionally center.

    float64 is not negotiable.  A float32 SVD of MACE invariant features -- which
    are routinely ill-conditioned, with condition numbers past 1e6 -- loses the
    small singular values entirely, and those are exactly the directions that
    distinguish an informative representation from a saturated one.
    """
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)

    if f.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {f.shape}")
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError(f"targets must be 1-D or 2-D, got shape {y.shape}")
    if f.shape[0] != y.shape[0]:
        raise ValueError(
            f"features has {f.shape[0]} rows but targets has {y.shape[0]}"
        )
    if f.shape[0] < 2:
        raise ValueError("Need at least 2 observations.")
    if not np.all(np.isfinite(f)):
        raise ValueError("features contains non-finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("targets contains non-finite values.")

    # Centering stands in for an intercept.  The Bayesian model has no bias
    # term, and its isotropic prior would shrink one toward zero anyway, which
    # is wrong for a target like energy-per-atom whose mean is large and
    # carries no transferability information.  Centering both sides removes the
    # offset from the problem instead of asking the features to explain it.
    if center_features:
        f = f - f.mean(axis=0, keepdims=True)
    if center_targets:
        y = y - y.mean(axis=0, keepdims=True)

    return f, y


def logme(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-9,
    center_features: bool = True,
    center_targets: bool = True,
    rank_rtol: float | None = None,
    warn: bool = True,
) -> LogMEResult:
    """LogME score for a single target.

    Parameters
    ----------
    features:
        ``(n, d)`` frozen feature matrix.  Cast to float64 internally.
    targets:
        ``(n,)`` target vector.  For a ``(n, k)`` array use :func:`logme_multi`.
    max_iter, tol:
        Fixed-point iteration budget and relative tolerance on ``alpha / beta``.
    center_features, center_targets:
        Subtract column means before fitting, standing in for an intercept.
        Both default to True; see :func:`_prepare` for why.  The choice must be
        held fixed across every model in a comparison.
    rank_rtol:
        Relative cutoff for the numerical rank.  ``None`` uses the LAPACK
        default ``max(n, d) * eps``.
    warn:
        Emit :class:`LogMEConvergenceWarning` if the iteration hits ``max_iter``.

    Returns
    -------
    LogMEResult
        ``.score`` is the transferability estimate; the other fields are
        diagnostics that should be checked before trusting it.
    """
    f, y = _prepare(features, targets, center_features, center_targets)
    if y.shape[1] != 1:
        raise ValueError(
            f"logme() takes a single target; got {y.shape[1]} columns. "
            "Use logme_multi() for multi-dimensional targets."
        )
    dec = _decompose(f, rank_rtol)
    return _fit_one(dec, y[:, 0], max_iter=max_iter, tol=tol, warn=warn)


def logme_multi(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    max_iter: int = 1000,
    tol: float = 1e-9,
    center_features: bool = True,
    center_targets: bool = True,
    rank_rtol: float | None = None,
    warn: bool = True,
) -> tuple[float, list[LogMEResult]]:
    """LogME for a multi-dimensional target, averaged over dimensions.

    The SVD is computed once and shared across every target dimension, which is
    the whole reason LogME is cheap enough to run over a grid of checkpoints
    and layers.

    Returns
    -------
    (mean_score, per_dimension_results)

    Notes
    -----
    Averaging per-dimension scores is the convention from the paper, and it is
    the right thing for genuinely multi-output targets.  It is *not* a licence
    to feed in per-component force vectors against invariant features: those
    three columns are related by rotations that the invariant features cannot
    see, so each per-component score measures an arbitrary choice of frame.
    """
    f, y = _prepare(features, targets, center_features, center_targets)
    dec = _decompose(f, rank_rtol)
    results = [
        _fit_one(dec, y[:, j], max_iter=max_iter, tol=tol, warn=warn)
        for j in range(y.shape[1])
    ]
    return float(np.mean([r.score for r in results])), results
