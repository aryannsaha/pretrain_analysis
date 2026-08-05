"""Rank-correlation machinery for judging transferability predictors.

The question is whether a predictor computed on frozen features ranks
checkpoints the same way the post-fine-tune error does.  That is a rank
problem, so the metrics are rank correlations: Spearman's rho, Kendall's tau,
and the weighted Kendall's tau_w that the LogME paper uses as its headline
(tau_w upweights agreement among the top-ranked models, which is what matters
when the estimator is used to *pick* a checkpoint).

Sign convention
---------------
The ground truth here is an error (MAE), where lower is better, while every
predictor is oriented so that higher is better.  A raw correlation between a
predictor and an MAE column is therefore *negative* when the predictor works,
which is an easy way to publish a sign error.  This module removes the trap by
converting the ground truth to a performance score (``-MAE``) exactly once, in
:func:`rank_agreement`, so that **positive always means the predictor works**.

Statistical power
-----------------
With six pretraining scales, only a near-perfect ranking can clear
significance.  The exact one-sided permutation distribution over all 720
orderings (computed, not estimated) says:

======================  ===========  ==================================
Spearman rho at n = 6   exact p      interpretation
======================  ===========  ==================================
1.000 (perfect)         0.00139      the floor; nothing can beat it
0.943 (one swap)        0.00833      clears p < 0.01
0.886 (two swaps)       0.01667      clears p < 0.05
0.829                   0.02917      the *last* value that clears 0.05
0.771 or below          > 0.05       indistinguishable from chance
======================  ===========  ==================================

So the usable threshold is **rho >= 0.83**: at n = 6 a predictor must get the
ordering right to within about two adjacent transpositions or it has shown
nothing.  A respectable-looking rho = 0.77 is *not* significant here, and the
gap between 0.83 and 0.77 is a single pair of adjacent models swapping.  For
contrast, at n = 8 the 0.05 threshold falls to rho = 0.64.

The bootstrap CI on a 6-point rank correlation typically spans most of
``[-1, 1]``.  :func:`rank_agreement` computes it anyway, because seeing the
width is the point.

Every result carries ``n_models`` and an exact permutation p-value so a bare
``rho = 0.9`` cannot be read as solid.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = [
    "RankAgreement",
    "compare_predictors",
    "rank_agreement",
]

# Above this many models an exact permutation test gets expensive (n! terms);
# fall back to sampling.  8! = 40320 is still instant, 12! is not.
_EXACT_PERM_MAX = 8


@dataclass
class RankAgreement:
    """Agreement between one predictor and one ground-truth error column."""

    predictor: str
    target: str
    n_models: int

    spearman: float
    spearman_p: float
    kendall: float
    kendall_p: float
    weighted_kendall: float

    perm_p_spearman: float
    perm_p_weighted_kendall: float
    perm_exact: bool

    spearman_ci: tuple[float, float]
    weighted_kendall_ci: tuple[float, float]

    n_bootstrap: int
    notes: list[str] = field(default_factory=list)

    @property
    def underpowered(self) -> bool:
        return self.n_models < 10

    def as_dict(self) -> dict:
        return {
            "predictor": self.predictor,
            "target": self.target,
            "n_models": self.n_models,
            "spearman": self.spearman,
            "spearman_p": self.spearman_p,
            "kendall": self.kendall,
            "kendall_p": self.kendall_p,
            "weighted_kendall": self.weighted_kendall,
            "perm_p_spearman": self.perm_p_spearman,
            "perm_p_weighted_kendall": self.perm_p_weighted_kendall,
            "perm_exact": self.perm_exact,
            "spearman_ci_lo": self.spearman_ci[0],
            "spearman_ci_hi": self.spearman_ci[1],
            "weighted_kendall_ci_lo": self.weighted_kendall_ci[0],
            "weighted_kendall_ci_hi": self.weighted_kendall_ci[1],
            "underpowered": self.underpowered,
            "notes": "; ".join(self.notes),
        }


def _weighted_tau(x: np.ndarray, y: np.ndarray) -> float:
    """scipy's weighted Kendall tau, with the LogME paper's default settings.

    ``rank=True`` (scipy's default) averages the hyperbolic weighting over both
    orderings.  The LogME reference code calls ``weightedtau`` bare, so this
    matches it; the choice is recorded here because tau_w is the headline
    metric and its weighting scheme is not unique.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.weightedtau(x, y).statistic)


def _permutation_p(
    pred: np.ndarray,
    perf: np.ndarray,
    observed: float,
    statistic,
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[float, bool]:
    """One-sided permutation p-value for "predictor ranks better than chance".

    Exact (all ``n!`` permutations) when ``n <= 8``, which covers the six-model
    design with room to spare; sampled otherwise.  One-sided because the
    hypothesis under test is directional -- we are asking whether the predictor
    ranks checkpoints correctly, not whether it correlates at all.
    """
    n = len(pred)
    if n <= _EXACT_PERM_MAX:
        perms = itertools.permutations(range(n))
        total = math.factorial(n)
        count = sum(
            1 for p in perms if statistic(pred, perf[list(p)]) >= observed - 1e-12
        )
        return count / total, True

    count = sum(
        1
        for _ in range(n_samples)
        if statistic(pred, rng.permutation(perf)) >= observed - 1e-12
    )
    return (count + 1) / (n_samples + 1), False


def _bootstrap_ci(
    pred: np.ndarray,
    perf: np.ndarray,
    statistic,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI over resampled (predictor, performance) pairs.

    Resamples are drawn with replacement, so at small ``n`` many contain
    duplicates and some are degenerate (all-identical, giving an undefined
    correlation).  Those are dropped, and the fraction dropped is what makes
    the interval so wide.  That width is the honest answer, not a defect.
    """
    n = len(pred)
    stats_out = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(pred[idx])) < 2 or len(np.unique(perf[idx])) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = statistic(pred[idx], perf[idx])
        if np.isfinite(s):
            stats_out.append(s)
    if len(stats_out) < 20:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(stats_out, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def rank_agreement(
    predictor_values: np.ndarray,
    true_error: np.ndarray,
    *,
    predictor: str = "predictor",
    target: str = "target",
    higher_predictor_is_better: bool = True,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> RankAgreement:
    """Correlate one predictor against one post-fine-tune error column.

    Parameters
    ----------
    predictor_values:
        One value per model.  By convention higher means "should transfer
        better"; set ``higher_predictor_is_better=False`` for predictors that
        run the other way (a Mahalanobis distance, say, where larger means
        further out of distribution).
    true_error:
        The post-fine-tune MAE per model.  Lower is better; the conversion to a
        performance score happens here and only here.

    Returns
    -------
    RankAgreement
        Positive correlations mean the predictor ranks checkpoints correctly.
    """
    pred = np.asarray(predictor_values, dtype=np.float64)
    err = np.asarray(true_error, dtype=np.float64)
    if pred.shape != err.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {err.shape}")

    notes: list[str] = []
    finite = np.isfinite(pred) & np.isfinite(err)
    if not finite.all():
        notes.append(f"dropped {int((~finite).sum())} model(s) with missing values")
        pred, err = pred[finite], err[finite]
    if pred.size < 3:
        raise ValueError(f"need at least 3 models to correlate, got {pred.size}")

    if not higher_predictor_is_better:
        pred = -pred
        notes.append("predictor sign flipped (lower was better)")

    # The single place the ground truth changes orientation.
    perf = -err

    n = pred.size
    if n < 10:
        notes.append(
            f"n_models={n}: rank correlations are severely underpowered; "
            "treat the point estimate as a direction, not a measurement"
        )
    if len(np.unique(pred)) < n:
        notes.append("ties in the predictor values")

    rng = np.random.default_rng(seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = stats.spearmanr(pred, perf)
        kt = stats.kendalltau(pred, perf)
    wt = _weighted_tau(pred, perf)

    perm_p_sp, exact = _permutation_p(
        pred, perf, float(sp.statistic), lambda a, b: stats.spearmanr(a, b).statistic, rng, 20_000
    )
    perm_p_wt, _ = _permutation_p(pred, perf, wt, _weighted_tau, rng, 20_000)

    return RankAgreement(
        predictor=predictor,
        target=target,
        n_models=n,
        spearman=float(sp.statistic),
        spearman_p=float(sp.pvalue),
        kendall=float(kt.statistic),
        kendall_p=float(kt.pvalue),
        weighted_kendall=wt,
        perm_p_spearman=perm_p_sp,
        perm_p_weighted_kendall=perm_p_wt,
        perm_exact=exact,
        spearman_ci=_bootstrap_ci(
            pred, perf, lambda a, b: stats.spearmanr(a, b).statistic, rng, n_bootstrap
        ),
        weighted_kendall_ci=_bootstrap_ci(pred, perf, _weighted_tau, rng, n_bootstrap),
        n_bootstrap=n_bootstrap,
        notes=notes,
    )


def compare_predictors(
    table,
    predictor_columns,
    error_columns,
    *,
    lower_is_better_predictors=(),
    n_bootstrap: int = 10_000,
    seed: int = 0,
):
    """Run :func:`rank_agreement` over a grid of predictors and error columns.

    ``table`` is anything with ``__getitem__`` returning a 1-D sequence per
    column name -- a pandas DataFrame or a plain dict of arrays.  Returns a
    list of :class:`RankAgreement`, one per (predictor, error) pair.
    """
    out = []
    for err_col in error_columns:
        for pred_col in predictor_columns:
            try:
                out.append(
                    rank_agreement(
                        np.asarray(table[pred_col], dtype=np.float64),
                        np.asarray(table[err_col], dtype=np.float64),
                        predictor=pred_col,
                        target=err_col,
                        higher_predictor_is_better=pred_col
                        not in lower_is_better_predictors,
                        n_bootstrap=n_bootstrap,
                        seed=seed,
                    )
                )
            except ValueError as exc:
                warnings.warn(
                    f"skipping {pred_col} vs {err_col}: {exc}", RuntimeWarning, stacklevel=2
                )
    return out


def minimum_achievable_p(n: int) -> float:
    """Smallest one-sided exact permutation p-value attainable with ``n`` models.

    ``1 / n!``.  Provided so a report can state up front what significance is
    even reachable: at ``n = 6`` a *flawless* ranking gives ``p = 0.0014``, and
    anything short of flawless degrades fast.
    """
    return 1.0 / math.factorial(n)
