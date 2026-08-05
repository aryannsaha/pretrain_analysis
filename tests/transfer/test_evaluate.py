"""Tests for the rank-agreement machinery.

The sign convention is tested hardest, because a sign error here would turn a
useless predictor into a headline result and would not look wrong on a plot.
"""

from __future__ import annotations

import numpy as np
import pytest

from pretrain_analysis.transfer.evaluate import (
    compare_predictors,
    minimum_achievable_p,
    rank_agreement,
)


def test_a_perfect_predictor_scores_plus_one():
    """Predictor increases exactly as error falls -> +1 on every metric."""
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    err = np.array([60.0, 50.0, 40.0, 30.0, 20.0, 10.0])

    res = rank_agreement(pred, err)
    assert res.spearman == pytest.approx(1.0)
    assert res.kendall == pytest.approx(1.0)
    assert res.weighted_kendall == pytest.approx(1.0)


def test_an_anti_predictor_scores_minus_one():
    """Predictor increases as error increases -> the predictor is backwards."""
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    err = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

    res = rank_agreement(pred, err)
    assert res.spearman == pytest.approx(-1.0)
    assert res.weighted_kendall == pytest.approx(-1.0)


def test_lower_is_better_predictors_are_flipped():
    """A distance-style predictor (larger = more OOD = worse) reads positive."""
    distance = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    err = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])  # distance tracks error

    flipped = rank_agreement(distance, err, higher_predictor_is_better=False)
    assert flipped.spearman == pytest.approx(1.0)
    assert "sign flipped" in "; ".join(flipped.notes)


def test_sign_convention_is_applied_exactly_once():
    """Guard against a double negation silently cancelling out."""
    pred = np.array([5.0, 3.0, 1.0, 2.0, 4.0, 6.0])
    err = np.array([1.0, 3.0, 5.0, 4.0, 2.0, 0.5])  # error falls as pred rises
    forward = rank_agreement(pred, err)
    # Correlating against -err with the flag flipped must give the same answer.
    backward = rank_agreement(-pred, err, higher_predictor_is_better=False)
    assert forward.spearman == pytest.approx(backward.spearman)


# --------------------------------------------------------------------------
# power: the thing this study is short on
# --------------------------------------------------------------------------


def test_perfect_ranking_of_six_reaches_only_p_0p0014():
    """The best attainable exact p-value at n = 6, stated in the docs."""
    pred = np.arange(6.0)
    err = -np.arange(6.0)
    res = rank_agreement(pred, err)
    assert res.perm_exact
    assert res.perm_p_spearman == pytest.approx(1 / 720, abs=1e-9)
    assert minimum_achievable_p(6) == pytest.approx(1 / 720)


def test_exact_significance_thresholds_at_n_six():
    """Pin the n=6 power table quoted in the module docstring and the report.

    These are exact over all 720 permutations, not asymptotic approximations,
    and they are the basis for the claim that rho >= 0.83 is the bar.
    """
    pred = np.arange(6.0)

    # One adjacent swap: rho = 0.943, clears p < 0.01.
    one_swap = rank_agreement(pred, -np.array([0.0, 1.0, 2.0, 4.0, 3.0, 5.0]))
    assert one_swap.spearman == pytest.approx(0.9429, abs=1e-3)
    assert one_swap.perm_p_spearman == pytest.approx(0.00833, abs=1e-4)

    # Two adjacent swaps: rho = 0.886, still clears p < 0.05.
    two_swaps = rank_agreement(pred, -np.array([1.0, 0.0, 2.0, 3.0, 5.0, 4.0]))
    assert two_swaps.spearman == pytest.approx(0.8857, abs=1e-3)
    assert two_swaps.perm_p_spearman < 0.05

    # rho = 0.771 is the first value that fails at 0.05 -- a respectable-looking
    # correlation that has, at n = 6, demonstrated nothing.
    below = rank_agreement(pred, -np.array([0.0, 1.0, 4.0, 3.0, 2.0, 5.0]))
    assert below.spearman == pytest.approx(0.7714, abs=1e-3)
    assert below.perm_p_spearman > 0.05


def test_six_models_are_flagged_underpowered():
    res = rank_agreement(np.arange(6.0), -np.arange(6.0))
    assert res.underpowered
    assert any("underpowered" in n for n in res.notes)


def test_bootstrap_interval_at_n_six_is_very_wide():
    """The CI is computed precisely so its width is visible."""
    rng = np.random.default_rng(0)
    pred = rng.standard_normal(6)
    err = rng.standard_normal(6)
    res = rank_agreement(pred, err, n_bootstrap=5_000)
    lo, hi = res.spearman_ci
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi - lo > 1.0, f"expected a wide interval at n=6, got [{lo:.2f}, {hi:.2f}]"


def test_random_predictors_are_not_significant():
    """Across many random draws at n=6, few should clear p < 0.05."""
    rng = np.random.default_rng(1)
    hits = 0
    trials = 60
    for _ in range(trials):
        res = rank_agreement(
            rng.standard_normal(6), rng.standard_normal(6), n_bootstrap=200
        )
        hits += res.perm_p_spearman < 0.05
    assert hits < 0.2 * trials, f"{hits}/{trials} false positives"


def test_larger_n_uses_sampled_permutations():
    rng = np.random.default_rng(2)
    n = 12
    pred = rng.standard_normal(n)
    res = rank_agreement(pred, -pred, n_bootstrap=200)
    assert not res.perm_exact
    assert res.spearman == pytest.approx(1.0)


# --------------------------------------------------------------------------
# housekeeping
# --------------------------------------------------------------------------


def test_missing_values_are_dropped_and_reported():
    pred = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    err = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    res = rank_agreement(pred, err)
    assert res.n_models == 5
    assert any("dropped 1" in n for n in res.notes)


def test_ties_are_reported():
    pred = np.array([1.0, 1.0, 3.0, 4.0, 5.0, 6.0])
    err = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    assert any("ties" in n for n in rank_agreement(pred, err).notes)


def test_too_few_models_raises():
    with pytest.raises(ValueError, match="at least 3"):
        rank_agreement(np.array([1.0, 2.0]), np.array([2.0, 1.0]))


def test_compare_predictors_covers_the_grid():
    table = {
        "logme": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        "log_n_pretrain": np.log(np.array([1e5, 5e5, 1e6, 2e6, 5e6, 1e7])),
        "mahalanobis": np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
        "mae_energy": np.array([60.0, 50.0, 40.0, 30.0, 20.0, 10.0]),
        "mae_force": np.array([600.0, 500.0, 400.0, 300.0, 200.0, 100.0]),
    }
    out = compare_predictors(
        table,
        ["logme", "log_n_pretrain", "mahalanobis"],
        ["mae_energy", "mae_force"],
        lower_is_better_predictors=("mahalanobis",),
        n_bootstrap=500,
    )
    assert len(out) == 6
    assert {r.predictor for r in out} == {"logme", "log_n_pretrain", "mahalanobis"}
    assert {r.target for r in out} == {"mae_energy", "mae_force"}
    # All three predictors are perfectly informative in this toy table.
    assert all(r.spearman == pytest.approx(1.0) for r in out)
    assert all("predictor" in r.as_dict() for r in out)
