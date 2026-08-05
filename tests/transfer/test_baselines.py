"""Tests for the competing transferability predictors.

These are sanity tests rather than correctness proofs: each baseline is checked
to move in the right direction on data where the right direction is known, and
to survive the degenerate inputs (near-null feature directions, n < d) that
MACE features will actually produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from pretrain_analysis.transfer.baselines import (
    effective_rank,
    gaussian_distance,
    gmm_log_likelihood,
    h_score,
    participation_ratio,
    ridge_probe,
)


def _linear_problem(n=2_000, d=40, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, d))
    y = f @ rng.standard_normal(d) + noise * rng.standard_normal(n)
    return f, y


# --------------------------------------------------------------------------
# ridge probe
# --------------------------------------------------------------------------


def test_ridge_probe_recovers_a_linear_signal():
    f, y = _linear_problem(noise=0.2)
    res = ridge_probe(f, y)
    assert res.r2 > 0.95
    assert res.n_train + res.n_test < f.shape[0]  # a validation split was held out


def test_ridge_probe_reports_near_zero_r2_on_random_features():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((1_500, 40))
    y = rng.standard_normal(1_500)  # independent of f
    res = ridge_probe(f, y)
    assert res.r2 < 0.1


def test_ridge_probe_r2_degrades_with_noise():
    scores = [ridge_probe(*_linear_problem(noise=s)).r2 for s in (0.1, 1.0, 5.0)]
    assert scores == sorted(scores, reverse=True), scores


def test_ridge_probe_regularizes_more_on_noisier_targets():
    """The penalty grid is really being searched, and in the sensible direction.

    Averaged over seeds, because a single held-out selection is noisy -- which
    is itself worth knowing: at the ``n`` available in the per-structure energy
    setting, the selected ``alpha`` is not a stable quantity and should not be
    read as a property of the representation.  (An earlier version of this test
    asserted that small ``n`` selects a larger penalty; it does not, because the
    validation split shrinks along with everything else.)
    """
    def mean_log_alpha(noise):
        return np.mean(
            [
                np.log10(ridge_probe(*_linear_problem(n=2_000, d=60, noise=noise, seed=s)).alpha)
                for s in range(5)
            ]
        )

    assert mean_log_alpha(10.0) > mean_log_alpha(0.1)


def test_ridge_probe_survives_n_less_than_d():
    rng = np.random.default_rng(4)
    f = rng.standard_normal((60, 200))
    y = f @ rng.standard_normal(200) / np.sqrt(200) + 0.2 * rng.standard_normal(60)
    res = ridge_probe(f, y)
    assert np.isfinite(res.r2) and np.isfinite(res.mae)


def test_ridge_probe_handles_dead_columns():
    f, y = _linear_problem(d=20)
    f = np.hstack([f, np.zeros((f.shape[0], 5))])
    assert np.isfinite(ridge_probe(f, y).r2)


# --------------------------------------------------------------------------
# H-score
# --------------------------------------------------------------------------


def test_h_score_is_higher_for_informative_features():
    rng = np.random.default_rng(5)
    n, d = 3_000, 30
    f = rng.standard_normal((n, d))
    y = f @ rng.standard_normal(d) + 0.2 * rng.standard_normal(n)
    f_rand = rng.standard_normal((n, d))

    assert h_score(f, y) > h_score(f_rand, y)


def test_h_score_decreases_with_noise():
    rng = np.random.default_rng(6)
    n, d = 3_000, 30
    f = rng.standard_normal((n, d))
    signal = f @ rng.standard_normal(d)
    scores = [
        h_score(f, signal + s * rng.standard_normal(n)) for s in (0.1, 1.0, 10.0)
    ]
    assert scores == sorted(scores, reverse=True), scores


def test_h_score_is_finite_with_ill_conditioned_features():
    """Shrinkage must keep the inverse well-posed on a rank-deficient input."""
    rng = np.random.default_rng(7)
    f = rng.standard_normal((500, 20))
    f = np.hstack([f, f])  # exactly rank deficient
    y = f[:, :20] @ rng.standard_normal(20) + 0.3 * rng.standard_normal(500)
    assert np.isfinite(h_score(f, y))


def test_h_score_depends_on_bin_count():
    """Documents the adaptation's free parameter rather than hiding it."""
    f, y = _linear_problem()
    assert h_score(f, y, n_bins=5) != h_score(f, y, n_bins=50)


def test_h_score_on_constant_target_is_nan():
    f, _ = _linear_problem()
    assert np.isnan(h_score(f, np.ones(f.shape[0])))


# --------------------------------------------------------------------------
# distributional distance
# --------------------------------------------------------------------------


def test_mahalanobis_grows_as_the_target_shifts_away():
    rng = np.random.default_rng(8)
    source = rng.standard_normal((2_000, 20))
    distances = []
    for shift in (0.0, 1.0, 4.0):
        target = rng.standard_normal((500, 20)) + shift
        distances.append(gaussian_distance(source, target)["mahalanobis_mean"])
    assert distances == sorted(distances), distances


def test_gaussian_loglik_falls_as_the_target_shifts_away():
    rng = np.random.default_rng(9)
    source = rng.standard_normal((2_000, 20))
    near = gaussian_distance(source, rng.standard_normal((500, 20)))
    far = gaussian_distance(source, rng.standard_normal((500, 20)) + 3.0)
    assert near["gaussian_loglik_mean"] > far["gaussian_loglik_mean"]


def test_gmm_scores_in_distribution_targets_higher():
    rng = np.random.default_rng(10)
    # A genuinely multi-modal source, which is the case a single Gaussian misses.
    source = np.vstack(
        [
            rng.standard_normal((1_000, 10)) - 4.0,
            rng.standard_normal((1_000, 10)) + 4.0,
        ]
    )
    in_dist = rng.standard_normal((400, 10)) + 4.0
    out_dist = rng.standard_normal((400, 10)) + 20.0

    res_in = gmm_log_likelihood(source, in_dist, n_components=4)
    res_out = gmm_log_likelihood(source, out_dist, n_components=4)
    assert res_in.target_loglik > res_out.target_loglik
    assert res_in.converged


def test_gmm_beats_a_single_gaussian_on_bimodal_data():
    """If the GMM were not doing anything, this baseline would be redundant."""
    rng = np.random.default_rng(11)
    source = np.vstack(
        [
            rng.standard_normal((1_500, 8)) - 5.0,
            rng.standard_normal((1_500, 8)) + 5.0,
        ]
    )
    held_out = rng.standard_normal((500, 8)) + 5.0

    gmm = gmm_log_likelihood(source, held_out, n_components=4).target_loglik
    gauss = gaussian_distance(source, held_out)["gaussian_loglik_mean"]
    assert gmm > gauss


def test_gmm_does_not_diverge_on_degenerate_directions():
    """reg_covar has to stop a component collapsing onto a null direction."""
    rng = np.random.default_rng(12)
    source = rng.standard_normal((800, 10))
    source[:, -3:] = 0.0  # exactly degenerate
    target = rng.standard_normal((200, 10))
    target[:, -3:] = 0.0
    res = gmm_log_likelihood(source, target, n_components=3)
    assert np.isfinite(res.source_loglik) and np.isfinite(res.target_loglik)


def test_gmm_is_deterministic_given_a_seed():
    rng = np.random.default_rng(13)
    source = rng.standard_normal((600, 6))
    target = rng.standard_normal((200, 6))
    a = gmm_log_likelihood(source, target, seed=3).target_loglik
    b = gmm_log_likelihood(source, target, seed=3).target_loglik
    assert a == b


# --------------------------------------------------------------------------
# effective dimensionality
# --------------------------------------------------------------------------


def test_participation_ratio_spans_one_to_d_across_spectra():
    rng = np.random.default_rng(14)
    n, d = 4_000, 50

    isotropic = rng.standard_normal((n, d))
    assert participation_ratio(isotropic) == pytest.approx(d, rel=0.15)

    # Rank-1: all variance in a single direction.
    rank_one = np.outer(rng.standard_normal(n), rng.standard_normal(d))
    assert participation_ratio(rank_one) == pytest.approx(1.0, abs=1e-6)


def test_participation_ratio_falls_as_the_spectrum_concentrates():
    rng = np.random.default_rng(15)
    n, d = 3_000, 40
    base = rng.standard_normal((n, d))
    prs = []
    for decay in (0.0, 0.5, 2.0):
        scaled = base * np.exp(-decay * np.arange(d))
        prs.append(participation_ratio(scaled))
    assert prs == sorted(prs, reverse=True), prs


def test_effective_rank_agrees_with_participation_ratio_on_extremes():
    rng = np.random.default_rng(16)
    n, d = 3_000, 30
    isotropic = rng.standard_normal((n, d))
    assert effective_rank(isotropic) == pytest.approx(d, rel=0.15)

    rank_one = np.outer(rng.standard_normal(n), rng.standard_normal(d))
    assert effective_rank(rank_one) == pytest.approx(1.0, abs=1e-4)


def test_effective_dimensionality_ignores_the_mean():
    """Both measures are on the covariance, so a constant offset is irrelevant."""
    rng = np.random.default_rng(17)
    f = rng.standard_normal((1_000, 20))
    assert participation_ratio(f + 100.0) == pytest.approx(participation_ratio(f))
    assert effective_rank(f + 100.0) == pytest.approx(effective_rank(f))
