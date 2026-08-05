"""Correctness tests for the LogME estimator.

The four tests the design brief asked for are, in order:

* ``test_recovers_generative_hyperparameters`` -- synthetic data from the exact
  generative model, known ``alpha`` and ``beta``, checked for recovery.
* ``test_noise_columns_do_not_increase_logme`` -- overfitting immunity, with a
  naive linear probe alongside as the contrast.
* ``test_random_features_score_far_below_informative`` -- signal vs noise.
* ``test_column_standardization_changes_score`` and its neighbours -- the
  feature-scaling question, answered empirically and documented.
"""

from __future__ import annotations

import numpy as np
import pytest

from pretrain_analysis.transfer.logme import (
    LogMEConvergenceWarning,
    logme,
    logme_multi,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _draw_from_generative_model(n, d, alpha, beta, rng):
    """Sample exactly from p(w|alpha) p(y|F, w, beta)."""
    f = rng.standard_normal((n, d))
    w = rng.standard_normal(d) / np.sqrt(alpha)
    noise = rng.standard_normal(n) / np.sqrt(beta)
    return f, f @ w + noise, w


def _reference_logme(f, y, max_iter=500, tol=1e-9, eps=0.0):
    """Literal transcription of ``thuml/LogME``'s ``_fit_fixed_point`` loop.

    Kept deliberately close to the published source so that any divergence from
    our implementation is attributable.  The two known departures, both
    deliberate on our side, are:

    * the reference omits the ``(D - rank) log alpha`` piece of ``log|A|``,
      so this function is only a valid cross-check when ``F`` is full column
      rank with ``D <= N``;
    * the reference hardcodes 11 iterations, a 1e-3 tolerance and a 1e-5 floor
      on both denominators.  Those are loosened here via arguments so the
      comparison isolates the formulas rather than the stopping rule.
    """
    n, d = f.shape
    u, s, _ = np.linalg.svd(f, full_matrices=False)
    sigma = s**2
    x = u.T @ y
    x2 = x**2
    res_x2 = (y**2).sum() - x2.sum()

    alpha, beta = 1.0, 1.0
    evidence = np.nan
    for _ in range(max_iter):
        t = alpha / beta
        gamma = (sigma / (sigma + t)).sum()
        m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
        res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
        alpha = gamma / (m2 + eps)
        beta = (n - gamma) / (res2 + eps)
        t_ = alpha / beta
        evidence = (
            d / 2.0 * np.log(alpha)
            + n / 2.0 * np.log(beta)
            - 0.5 * np.sum(np.log(alpha + beta * sigma))
            - beta / 2.0 * res2
            - alpha / 2.0 * m2
            - n / 2.0 * np.log(2 * np.pi)
        ) / n
        if abs(t_ - t) / t <= tol:
            break
    return evidence, alpha, beta


def _probe_train_r2(f, y):
    """Ordinary least squares R^2 on the *training* data.

    This is the quantity that LogME is supposed to beat: it rises whenever you
    add columns, informative or not.
    """
    coef, *_ = np.linalg.lstsq(f, y, rcond=None)
    resid = y - f @ coef
    return 1.0 - resid.var() / y.var()


# --------------------------------------------------------------------------
# 1. hyperparameter recovery on data from the exact generative model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_true,beta_true", [(4.0, 25.0), (1.0, 100.0), (20.0, 4.0)])
def test_recovers_generative_hyperparameters(alpha_true, beta_true):
    """Evidence maximization should recover the alpha, beta that generated y.

    Tolerances are set by how much information the data actually carries about
    each hyperparameter, not by wishful thinking:

    * ``beta`` is pinned down by ``n - gamma ~ 9900`` residual degrees of
      freedom, so it is recovered tightly (a few percent).
    * ``alpha`` is inferred from a *single* draw of ``w``.  ``||w||^2`` is
      ``chi^2_d / alpha``, whose relative spread is ``sqrt(2/d)`` = 14% at
      ``d = 100``.  Recovering alpha to better than that is not possible even
      in principle, so we average over seeds and allow 25%.
    """
    n, d = 10_000, 100
    recovered_alpha, recovered_beta = [], []

    for seed in range(8):
        rng = np.random.default_rng(seed)
        f, y, _ = _draw_from_generative_model(n, d, alpha_true, beta_true, rng)
        # No centering: this test probes the raw generative model, which has
        # no intercept and zero-mean data by construction.
        res = logme(f, y, center_features=False, center_targets=False)

        assert res.converged, f"did not converge for seed {seed}"
        assert res.rank == d
        recovered_alpha.append(res.alpha)
        recovered_beta.append(res.beta)

    mean_alpha = float(np.mean(recovered_alpha))
    mean_beta = float(np.mean(recovered_beta))

    assert mean_alpha == pytest.approx(alpha_true, rel=0.25), (
        f"alpha: recovered {mean_alpha:.4g} vs true {alpha_true:.4g}"
    )
    assert mean_beta == pytest.approx(beta_true, rel=0.05), (
        f"beta: recovered {mean_beta:.4g} vs true {beta_true:.4g}"
    )


def test_beta_matches_residual_noise_scale():
    """1/sqrt(beta) should equal the noise std used to generate the data.

    This is the diagnostic used on real runs to tell "the features explain the
    target" from "the fit collapsed": if 1/sqrt(beta) comes back at the scale
    of the target's own standard deviation, the features explained nothing.
    """
    rng = np.random.default_rng(0)
    noise_std = 0.15
    f, y, _ = _draw_from_generative_model(20_000, 60, 4.0, noise_std**-2, rng)
    res = logme(f, y, center_features=False, center_targets=False)
    assert 1.0 / np.sqrt(res.beta) == pytest.approx(noise_std, rel=0.05)


# --------------------------------------------------------------------------
# 2. overfitting immunity: pure-noise columns must not help
# --------------------------------------------------------------------------


def test_noise_columns_do_not_increase_logme():
    """Appending pure-noise features must not raise the score.

    There is an exact argument for this, which the test confirms numerically:
    an added column contributes ``+0.5 log alpha`` through the ``(d/2) log
    alpha`` term and ``-0.5 log(alpha + beta sigma_new^2)`` through ``log|A|``,
    a strictly non-positive net change.

    The training-R^2 probe is measured alongside as the contrast: it rises
    monotonically with the junk columns, which is precisely the failure mode
    LogME exists to avoid.
    """
    rng = np.random.default_rng(11)
    n, d_sig = 2_000, 30
    f_sig = rng.standard_normal((n, d_sig))
    w = rng.standard_normal(d_sig)
    y = f_sig @ w + 0.5 * rng.standard_normal(n)

    base = logme(f_sig, y).score
    base_r2 = _probe_train_r2(f_sig, y)

    scores, r2s = [], []
    for n_noise in (10, 50, 200, 800):
        f_aug = np.hstack([f_sig, rng.standard_normal((n, n_noise))])
        scores.append(logme(f_aug, y).score)
        r2s.append(_probe_train_r2(f_aug, y))

    # LogME never improves.  A tiny tolerance absorbs the re-optimization of
    # (alpha, beta), which is not exactly held fixed by the argument above.
    for n_noise, s in zip((10, 50, 200, 800), scores, strict=True):
        assert s <= base + 1e-6, (
            f"{n_noise} noise columns raised LogME: {s:.6f} > {base:.6f}"
        )

    # ... and it degrades monotonically as more junk is added.
    assert scores == sorted(scores, reverse=True), f"non-monotone: {scores}"

    # The naive probe does the opposite, which is the whole point.
    assert all(r2 > base_r2 for r2 in r2s)
    assert r2s == sorted(r2s), f"probe R^2 should rise monotonically: {r2s}"


def test_gamma_counts_data_constrained_directions_not_informative_ones():
    """Pin down what gamma actually measures, because it is easy to over-read.

    ``gamma = sum_i beta sigma_i^2 / (alpha + beta sigma_i^2)`` is the effective
    number of parameters the *data* determines rather than the prior.  It is a
    property of the feature spectrum and the noise level -- **not** a count of
    informative features.  With ``n >> d`` and a well-conditioned ``F`` every
    direction is data-constrained, so ``gamma -> d`` even when most columns are
    pure noise.  (An earlier version of this test asserted the opposite and was
    wrong; the resistance to junk columns lives in the evidence trade-off
    exercised by ``test_noise_columns_do_not_increase_logme``, not in gamma.)

    Both halves are asserted here so the distinction stays documented.
    """
    rng = np.random.default_rng(12)
    n, d_sig, d_junk = 4_000, 30, 800
    f_sig = rng.standard_normal((n, d_sig))
    y = f_sig @ rng.standard_normal(d_sig) + 0.3 * rng.standard_normal(n)

    # (a) Well-conditioned junk: gamma climbs to the full width.
    f_flat = np.hstack([f_sig, rng.standard_normal((n, d_junk))])
    gamma_flat = logme(f_flat, y).gamma
    assert gamma_flat > 0.95 * (d_sig + d_junk), (
        f"gamma={gamma_flat:.1f}: with n >> d every direction should be "
        "data-constrained"
    )

    # (b) Junk in near-null directions: those fall below the prior and drop out.
    f_spiked = np.hstack([f_sig, 1e-4 * rng.standard_normal((n, d_junk))])
    gamma_spiked = logme(f_spiked, y).gamma
    assert gamma_spiked < 2 * d_sig, (
        f"gamma={gamma_spiked:.1f}: near-null directions should not count"
    )


# --------------------------------------------------------------------------
# 3. informative vs random features
# --------------------------------------------------------------------------


def test_random_features_score_far_below_informative():
    rng = np.random.default_rng(7)
    n, d = 3_000, 64
    f_good = rng.standard_normal((n, d))
    y = f_good @ rng.standard_normal(d) + 0.2 * rng.standard_normal(n)
    f_rand = rng.standard_normal((n, d))  # independent of y

    good = logme(f_good, y).score
    rand = logme(f_rand, y).score
    assert good > rand + 1.0, f"informative {good:.4f} vs random {rand:.4f}"


def test_partially_informative_features_rank_between():
    """Scores should be ordered by how much signal the features carry."""
    rng = np.random.default_rng(8)
    n, d = 4_000, 40
    f = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    signal = f @ w

    scores = []
    for noise_std in (0.1, 0.5, 2.0, 10.0):
        y = signal + noise_std * rng.standard_normal(n)
        scores.append(logme(f, y).score)

    assert scores == sorted(scores, reverse=True), f"not ordered by SNR: {scores}"


# --------------------------------------------------------------------------
# 4. feature scaling: what changes the score and what does not
# --------------------------------------------------------------------------


def test_global_feature_rescaling_leaves_score_invariant():
    """F -> cF is absorbed by alpha -> c^2 alpha, so the maximized evidence is fixed.

    Direction of the alpha shift, since it is easy to get backwards: the weight
    vector reproducing the same function under ``cF`` is ``w / c``, so the
    prior precision that leaves the prior over *functions* unchanged must grow
    as ``c^2``.  The fixed point agrees -- ``alpha = gamma / ||m||^2`` with
    ``gamma`` invariant and ``||m||^2`` shrinking by ``c^2``.
    """
    rng = np.random.default_rng(3)
    f = rng.standard_normal((1_500, 40))
    y = f @ rng.standard_normal(40) + 0.4 * rng.standard_normal(1_500)

    base = logme(f, y)
    for c in (1e-3, 0.5, 2.0, 1e3):
        scaled = logme(c * f, y)
        assert scaled.score == pytest.approx(base.score, abs=1e-8)
        assert scaled.alpha == pytest.approx(base.alpha * c**2, rel=1e-5)
        assert scaled.beta == pytest.approx(base.beta, rel=1e-5)


def test_orthogonal_feature_rotation_leaves_score_invariant():
    """The isotropic prior is rotation invariant, so F -> FQ must not matter."""
    rng = np.random.default_rng(4)
    f = rng.standard_normal((1_200, 30))
    y = f @ rng.standard_normal(30) + 0.4 * rng.standard_normal(1_200)
    q, _ = np.linalg.qr(rng.standard_normal((30, 30)))

    assert logme(f @ q, y).score == pytest.approx(logme(f, y).score, abs=1e-8)


def test_column_standardization_changes_score():
    """ANSWER TO THE FEATURE-SCALING QUESTION: yes, per-column scaling matters.

    The prior ``N(0, alpha^-1 I)`` is isotropic, so it is exchangeable only
    under transformations that preserve the Euclidean geometry of weight space
    -- rotations and global scalings (both tested above).  Rescaling column
    ``j`` by ``c_j`` is equivalent to rescaling ``w_j`` by ``1/c_j``, which the
    single shared ``alpha`` cannot absorb when the ``c_j`` differ.  The score
    therefore changes, sometimes substantially.

    Consequence for this project: a convention has to be chosen and applied
    identically to every checkpoint being compared.  The package standardizes
    on *centering only, no per-column scaling*, because MACE's own readout is a
    linear map on the raw (uncentered, unstandardized) invariant features, so
    the raw column geometry is the one the fine-tune actually inherits.
    """
    rng = np.random.default_rng(5)
    n, d = 2_000, 40
    # Columns with deliberately heterogeneous scales, as MACE features have.
    scales = np.exp(rng.uniform(-3, 3, size=d))
    f = rng.standard_normal((n, d)) * scales
    y = f @ (rng.standard_normal(d) / scales) + 0.3 * rng.standard_normal(n)

    raw = logme(f, y).score
    standardized = logme(f / f.std(axis=0, keepdims=True), y).score

    assert not np.isclose(raw, standardized, atol=1e-3), (
        "column standardization unexpectedly left the score unchanged; the "
        "documented convention rests on it mattering"
    )


def test_target_rescaling_shifts_score_by_log_c():
    """y -> c y shifts the score by -log|c| exactly (a Jacobian, not a signal).

    This is why ``logme_energy`` and ``logme_forcemag`` are never comparable to
    each other in absolute terms -- only within a column, across models.
    """
    rng = np.random.default_rng(6)
    f = rng.standard_normal((1_500, 25))
    y = f @ rng.standard_normal(25) + 0.4 * rng.standard_normal(1_500)

    base = logme(f, y).score
    for c in (0.01, 3.0, 250.0):
        assert logme(f, c * y).score == pytest.approx(base - np.log(c), abs=1e-7)


# --------------------------------------------------------------------------
# cross-check against the published reference implementation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_reference_formula_when_full_rank(seed):
    """Agree with thuml/LogME where the reference is valid (n > d, full rank)."""
    rng = np.random.default_rng(seed)
    n, d = 2_000, 50
    f = rng.standard_normal((n, d))
    y = f @ rng.standard_normal(d) + 0.5 * rng.standard_normal(n)

    ref_score, ref_alpha, ref_beta = _reference_logme(f, y)
    ours = logme(f, y, center_features=False, center_targets=False)

    assert ours.score == pytest.approx(ref_score, rel=1e-8)
    assert ours.alpha == pytest.approx(ref_alpha, rel=1e-6)
    assert ours.beta == pytest.approx(ref_beta, rel=1e-6)


def test_diverges_from_reference_when_d_exceeds_n():
    """Where the reference drops ``(d - rank) log alpha``, we must not.

    With ``n < d`` the omitted term is large, so this asserts a *disagreement* --
    documenting that the difference is intentional rather than a porting slip.
    """
    rng = np.random.default_rng(21)
    n, d = 80, 400
    f = rng.standard_normal((n, d))
    y = f @ rng.standard_normal(d) / np.sqrt(d) + 0.3 * rng.standard_normal(n)

    ref_score, _, _ = _reference_logme(f, y)
    ours = logme(f, y, center_features=False, center_targets=False)

    assert ours.rank == min(n, d)
    assert abs(ours.score - ref_score) > 1e-3

    # And the gap is accounted for by the missing log-determinant term.  The
    # residual ~1e-6 mismatch is a second, benign difference: the reference
    # evaluates the evidence *inside* the loop using sufficient statistics
    # computed at the previous (alpha, beta), so its reported score lags the
    # hyperparameters it reports by one iteration.  We recompute after the loop
    # so the two are self-consistent.
    missing = 0.5 * (d - ours.rank) * np.log(ours.alpha) / n
    assert ours.score == pytest.approx(ref_score - missing, rel=1e-4)


# --------------------------------------------------------------------------
# the out-of-span residual term
# --------------------------------------------------------------------------


def test_out_of_span_residual_is_reported_and_nonzero():
    """``||y||^2 - sum z_i^2`` must be carried; it is the bulk of the residual."""
    rng = np.random.default_rng(9)
    n, d = 1_000, 20
    f = rng.standard_normal((n, d))
    y = f @ rng.standard_normal(d) + 1.0 * rng.standard_normal(n)

    res = logme(f, y)
    assert res.residual_out_of_span > 0
    # With d << n the noise lives almost entirely outside span(F): roughly
    # (n - d)/n of the unit-variance noise power, i.e. ~980 of ~1000.
    assert res.residual_out_of_span > 0.5 * n
    # And beta must reflect the true noise scale (std 1.0), not a collapsed fit.
    assert 1.0 / np.sqrt(res.beta) == pytest.approx(1.0, rel=0.1)


# --------------------------------------------------------------------------
# n < d, rank deficiency, numerics, and API
# --------------------------------------------------------------------------


def test_handles_n_less_than_d():
    """Aggressive subsampling can make n < d; the score must stay finite."""
    rng = np.random.default_rng(13)
    n, d = 40, 256
    f = rng.standard_normal((n, d))
    y = f @ rng.standard_normal(d) / np.sqrt(d) + 0.2 * rng.standard_normal(n)

    res = logme(f, y)
    assert np.isfinite(res.score)
    assert res.converged
    assert res.rank <= n
    assert res.d == d


def test_handles_rank_deficient_features():
    """Duplicated columns must not blow up the score or the iteration."""
    rng = np.random.default_rng(14)
    n, d = 800, 30
    f = rng.standard_normal((n, d))
    y = f @ rng.standard_normal(d) + 0.3 * rng.standard_normal(n)
    f_dup = np.hstack([f, f, f])  # rank 30, width 90

    res = logme(f_dup, y)
    assert res.rank == d
    assert res.d == 3 * d
    assert np.isfinite(res.score)


def test_constant_column_is_tolerated():
    """A dead channel (MACE features have them) must not break the SVD path."""
    rng = np.random.default_rng(15)
    f = rng.standard_normal((500, 10))
    f = np.hstack([f, np.full((500, 1), 3.7)])
    y = f[:, :10] @ rng.standard_normal(10) + 0.2 * rng.standard_normal(500)

    res = logme(f, y)  # centering kills the constant column -> rank 10
    assert res.rank == 10
    assert np.isfinite(res.score)


def test_float32_input_is_promoted_and_matches_float64():
    """float32 in must not mean float32 arithmetic."""
    rng = np.random.default_rng(16)
    f = rng.standard_normal((900, 32))
    y = f @ rng.standard_normal(32) + 0.3 * rng.standard_normal(900)

    from_f32 = logme(f.astype(np.float32), y.astype(np.float32))
    from_f64 = logme(f.astype(np.float32).astype(np.float64), y.astype(np.float32).astype(np.float64))
    assert from_f32.score == pytest.approx(from_f64.score, rel=1e-12)


def test_multi_target_averages_and_shares_the_svd():
    rng = np.random.default_rng(17)
    n, d = 1_500, 24
    f = rng.standard_normal((n, d))
    y = np.stack(
        [
            f @ rng.standard_normal(d) + 0.2 * rng.standard_normal(n),
            f @ rng.standard_normal(d) + 2.0 * rng.standard_normal(n),
        ],
        axis=1,
    )

    mean_score, per_dim = logme_multi(f, y)
    assert len(per_dim) == 2
    assert mean_score == pytest.approx(np.mean([r.score for r in per_dim]))
    # Each dimension agrees with an independent single-target call.
    for j, r in enumerate(per_dim):
        assert r.score == pytest.approx(logme(f, y[:, j]).score, rel=1e-12)
    # The cleaner target scores higher.
    assert per_dim[0].score > per_dim[1].score


def test_is_deterministic():
    rng = np.random.default_rng(18)
    f = rng.standard_normal((700, 20))
    y = f @ rng.standard_normal(20) + 0.3 * rng.standard_normal(700)
    assert logme(f, y).score == logme(f, y).score


def test_rejects_malformed_input():
    rng = np.random.default_rng(19)
    f = rng.standard_normal((100, 10))
    y = rng.standard_normal(100)

    with pytest.raises(ValueError, match="must be 2-D"):
        logme(rng.standard_normal(100), y)
    with pytest.raises(ValueError, match="rows"):
        logme(f, rng.standard_normal(99))
    with pytest.raises(ValueError, match="non-finite"):
        bad = f.copy()
        bad[0, 0] = np.nan
        logme(bad, y)
    with pytest.raises(ValueError, match="single target"):
        logme(f, np.stack([y, y], axis=1))


def test_warns_when_iteration_budget_is_exhausted():
    rng = np.random.default_rng(20)
    f = rng.standard_normal((500, 20))
    y = f @ rng.standard_normal(20) + 0.3 * rng.standard_normal(500)

    with pytest.warns(LogMEConvergenceWarning):
        res = logme(f, y, max_iter=1, tol=1e-15)
    assert not res.converged
