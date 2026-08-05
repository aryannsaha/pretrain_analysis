# LogME for MACE: results and honest assessment

**Bottom line.** The question as posed cannot be answered yet — the fine-tune
MAEs that are supposed to be the ground truth do not exist (the runs are at
epoch 0–5 of 40). Against the *zero-shot* ground truth that is available,
LogME ranks the six checkpoints perfectly on both downstream targets, and it
beats the trivial `log N_pretrain` baseline on one of them — the only arm where
that baseline is beatable at all. With n = 6 that result is one adjacent
transposition wide, so it is suggestive, not established.

The deeper finding is about the experimental design: **on MOF-OFF, downstream
performance is strictly monotonic in pretraining scale, so sorting checkpoints
by dataset size already gives a perfect ranking and no feature-based estimator
can beat it.** Fixing this does not need more atoms or more targets — it needs
points where the data budget is the *wrong* answer. Section 5 says how to get
them cheaply.

Read [`DESIGN_AND_CAVEATS.md`](DESIGN_AND_CAVEATS.md) alongside this.

---

## 1. What was actually available

The brief assumed final fine-tuned energy and force MAEs for all twelve
(pretrain scale x target) combinations. They do not exist yet. As of
2026-08-04 the fine-tune runs were queued or a few epochs in:

| family | target | status |
|---|---|---|
| `mof_off_ft_rs` | MOF-OFF | 3 of 6 runs started, 1-4 epochs of 40 |
| `am_ft_rs` | AM (MPtrj+sAlex) | 6 runs launched today, no results files yet |
| `matpes_ft_rs` | **also MOF-OFF** (misconfigured) | 2 of 6 started |

Two further departures from the brief, both confirmed on disk:

- **The scale ladder is 100k / 500k / 1M / 2M / 5M / 10M — there is no 10k
  model** in the `stratified_random_nested` family.
- **The `matpes_ft_rs` / `oam` family trains on MOF-OFF, not MatPES.** Every one
  of its six configs points at `mof_off/R2SCAN/R2SCAN_train.lmdb`; the launcher
  was cloned from the MOF one and the data paths were never changed. Those runs
  are a learning-rate ablation (1e-5 vs 1e-4) on one target, not a second
  target.

So the post-fine-tune correlation is **blocked**, not computed. Everything that
does not depend on it is final.

---

## 2. What *is* final: LogME and baselines on the frozen checkpoints

Six checkpoints x two downstream targets x five feature sets x two poolings.
All scores converged; all feature matrices were full rank (`rank = d = 128` for
`layer0/inv` at `n = 3000` structures).

### 2.1 LogME rises monotonically with pretraining scale, and it is not noise

MOF-OFF, energy target, mean-pool (the self-consistent pairing):

| checkpoint | `layer0/inv` | `layer0/normed` | `layer1/inv` | `readout_hidden` |
|---|---|---|---|---|
| omat_100k | 0.599 | 0.732 | 0.165 | −0.749 |
| omat_500k | 0.899 | 1.093 | 0.573 | −0.733 |
| omat_1m | 1.039 | 1.262 | 0.471 | −0.643 |
| omat_2m | 1.230 | 1.424 | 0.558 | −0.711 |
| omat_5m | 1.239 | 1.546 | 0.622 | −0.744 |
| omat_10m | 1.242 | 1.599 | 0.826 | −0.624 |

The across-subsample standard deviation is **0.008–0.021** everywhere, against a
between-model range of 0.87 for `layer0/normed`. Signal-to-noise is roughly
**58:1**, so the ranking is not a sampling artefact. This was the first thing
worth checking and it passes comfortably.

Three observations:

- **`normed` beats `inv` at every scale**, and its advantage widens with scale
  (+0.13 at 100k, +0.36 at 10M). The `l > 0` norms carry real information that
  the scalar channels alone discard — so the extra feature variant earned its
  place rather than being decoration.
- **`layer0` beats `layer1`** everywhere, and `layer1/inv` is non-monotonic (it
  dips at 1M). The hypothesis that middle layers are most predictive cannot be
  tested properly here: with only 2 interaction blocks there is no middle layer.
- **`readout_hidden` is useless** — flat, negative, no trend. Its 16 dimensions
  are too few to carry the target, which is worth knowing given the intuition
  that the readout is what a fine-tune head most directly reuses.

### 2.2 The MACE invariant features are startlingly low-dimensional

The participation ratio of the `layer0/inv` covariance spectrum is **~2.5 out of
128 dimensions**, at every scale. Effective rank tells the same story. Over 98%
of the feature variance lives in about two or three directions.

This is a side observation, not something the brief asked for, but it is
striking and it bears on the whole enterprise: if the representation is
effectively 2-3 dimensional on this dataset, then most transferability
estimators are measuring a very low-dimensional object and their apparent
agreement may be trivial.

---

## 3. Correlation against the zero-shot ground truth

Since the fine-tune MAEs do not exist yet, the ground truth used here is
**zero-shot error of the frozen model on the downstream set**. This is a
different and weaker question than the one the brief asks — a model can start
poorly and adapt well — but it is a real downstream performance measure and it
is available now. Treat what follows as a sanity check, not the answer.

All numbers below are at a **pre-specified** feature set (`layer0/normed`,
mean-pool), chosen on principle before looking at results: `layer0` is the only
block with `l > 0` content, `normed` is a strict superset of `inv`, and
mean-pool is the pairing that matches a per-atom energy target exactly.
`build_results_table.py` also reports the best feature set per predictor, but
that maximizes over 10 combinations and is badly biased upward at n = 6 — it is
for exploration, not for the headline.

### 3.1 MOF-OFF (out of distribution) — the ground truth is monotonic, so LogME ties

Zero-shot force MAE falls **strictly** with pretraining scale:
0.7296 → 0.4351 → 0.3782 → 0.3294 → 0.3229 → 0.3199 eV/Å.

| predictor | tau_w | rho | exact p | significant? |
|---|---|---|---|---|
| **LogME (energy)** | **+1.000** | **+1.000** | 0.0014 | yes |
| **`log N_pretrain`** (trivial) | **+1.000** | **+1.000** | 0.0014 | yes |
| Gaussian log-likelihood | +0.864 | +0.943 | 0.0083 | yes |
| GMM log-likelihood | +0.864 | +0.943 | 0.0083 | yes |
| Ridge probe R² (energy) | +0.755 | +0.943 | 0.0083 | yes |
| LogME (force magnitude) | +0.755 | +0.943 | 0.0083 | yes |
| H-score (force magnitude) | +0.592 | +0.829 | 0.0292 | yes |
| Mahalanobis distance | +0.218 | +0.371 | 0.249 | no |
| Participation ratio | −0.491 | −0.543 | 0.879 | no |
| Effective rank | −0.780 | −0.829 | 0.983 | no |
| **H-score (energy)** | **−1.000** | **−1.000** | 1.000 | no (perfectly backwards) |

Because the ground truth is strictly monotonic in scale, **the trivial baseline
is unbeatable here** — it already achieves a perfect ranking with no model, no
features and no computation. LogME matches it; nothing can do better. A tie at
the ceiling is not evidence for LogME.

LogME does, however, beat the ridge probe on this arm (tau_w 1.000 vs 0.755):
ridge R² is compressed into 0.977–0.996 and inverts between 5M and 10M, while
LogME stays strictly monotonic. That is a resolution advantage, not a
demonstration of predictive power.

### 3.2 AM (nearer in distribution) — the ground truth is NOT monotonic, and LogME wins

This arm is the more informative one. Zero-shot force MAE is **not** monotonic:
0.1156 → 0.0863 → 0.0819 → 0.0784 → 0.0757 → **0.0769**. The 10M model is
slightly *worse* than the 5M model, so `log N_pretrain` is wrong about one pair.

| predictor | tau_w | rho | exact p | significant? |
|---|---|---|---|---|
| **LogME (energy)** | **+1.000** | **+1.000** | 0.0014 | yes |
| **LogME (force magnitude)** | **+1.000** | **+1.000** | 0.0014 | yes |
| **Ridge probe R² (energy)** | **+1.000** | **+1.000** | 0.0014 | yes |
| Ridge probe R² (force mag) | +0.864 | +0.943 | 0.0083 | yes |
| H-score (energy) | +0.791 | +0.886 | 0.0167 | yes |
| **`log N_pretrain`** (trivial) | **+0.755** | **+0.943** | 0.0083 | yes |
| Mahalanobis distance | +0.744 | +0.771 | 0.051 | no |
| GMM log-likelihood | +0.592 | +0.829 | 0.0292 | yes |
| Gaussian log-likelihood | +0.592 | +0.829 | 0.0292 | yes |
| Participation ratio | −0.155 | −0.257 | 0.718 | no |
| Effective rank | −0.614 | −0.714 | 0.949 | no |

**Here LogME beats the trivial baseline** (tau_w 1.000 vs 0.755, rho 1.000 vs
0.943): it correctly captures the 5M/10M inversion that pretraining-set-size
cannot see, because it reads the weights rather than the data budget. LogME's
energy score is 0.732 → ... → −0.446 (5M) → −0.500 (10M), reproducing exactly
the dip in the ground truth.

**This is the study's one genuine positive result, and it is fragile.** It rests
on a single pair of adjacent models swapping. Ridge R² captures the same
inversion and ties LogME here, so this is not evidence that LogME beats a linear
probe — only that both beat the data-budget heuristic when the data budget is
wrong.

### 3.3 Verdict

| comparison | MOF-OFF | AM |
|---|---|---|
| LogME vs `log N_pretrain` | tie (both perfect) | **LogME wins** (1.000 vs 0.755) |
| LogME vs ridge probe | **LogME wins** (1.000 vs 0.755) | tie (both perfect) |

LogME is never worse than either baseline on these two arms and is better than
each of them once. That is the most that can honestly be claimed, and it is
weak: with n = 6 every one of these gaps is one adjacent transposition wide.

### 3.4 Where LogME and the baselines fail

- **H-score (energy) is perfectly anti-correlated on MOF-OFF** (rho = −1.000).
  It ranks the *worst* checkpoint best. The binned-regression adaptation of
  H-score is evidently not measuring what it does in the classification setting;
  it should not be used for MLIP regression without further work.
- **`readout_hidden` is useless**: flat and negative at every scale (−0.749 to
  −0.624, no trend). Its 16 dimensions cannot carry the energy target. Worth
  knowing, given the intuition that the readout is what a fine-tune head most
  directly reuses.
- **The force-magnitude LogME is a weak signal on MOF-OFF**: the range across
  all six checkpoints is −1.600 to −1.553, i.e. 0.047 nats against a subsample
  SD of ~0.01 (S/N ≈ 5:1, an order of magnitude worse than energy). Ridge R² on
  force magnitude is only 0.21–0.25 — the invariant features barely explain it.
  On AM it works much better (rho = 1.000), so the weakness is dataset-specific.
- **Effective rank and participation ratio are negatively correlated** with
  performance on both arms. Representational "richness" as measured this way is
  not a proxy for transferability here — if anything, the better models have
  *more* concentrated spectra.
- **Zero-shot energy MAE is unpredictable on MOF-OFF**: it barely varies with
  scale (0.363–0.375 eV/atom, non-monotonic) because it is dominated by the
  E0/reference mismatch between OMat24 and r2SCAN-D4, which pretraining does not
  fix. No predictor reached p < 0.05 against it. The offset-corrected energy MAE
  is simply not a usable target on this arm.

---

### 3.5 Which layer is most predictive?

The hypothesis was middle layers. **This cannot be tested on these
checkpoints**: with `num_interactions = 2` there is no middle layer. What the
data does say (see `figures/per_layer_mean.png`):

- `layer0` (first interaction block) beats `layer1` on both targets, by a wide
  margin on MOF-OFF (LogME 1.60 vs 0.83 at 10M).
- Within `layer0`, `normed` beats `inv` at every scale, and the gap **widens**
  with scale (+0.13 at 100k, +0.36 at 10M) — the `l > 0` norms carry
  information the scalar channels discard, and pretraining puts more into them.
- `readout_hidden` is worst everywhere.

Testing the middle-layer hypothesis properly needs a deeper MACE (4+
interactions).

## 4. Statistical power: the binding constraint

From the exact permutation distribution over all 720 orderings of 6 items:

| Spearman rho | exact one-sided p |
|---|---|
| 1.000 | 0.00139 |
| 0.943 (one adjacent swap) | 0.00833 |
| 0.886 | 0.01667 |
| 0.829 | 0.02917 (last value clearing 0.05) |
| ≤ 0.771 | > 0.05 |

**A predictor must reach rho ≥ 0.83 to have shown anything at all.** At n = 6
there are effectively two outcomes: a near-perfect ranking, or nothing. There is
no middle ground in which one predictor is "somewhat better" than another in a
way that survives.

Bootstrap CIs on a 6-point rank correlation span most of [−1, 1]; they are
computed and reported so the width is visible.

---

## 5. What to do about it

Ranked by value per unit of effort.

1. **Use intermediate pretraining checkpoints.** This is the single highest-value
   change and it costs no new pretraining. Each run wrote per-epoch checkpoints;
   treating (scale, epoch) pairs as separate pretrained models takes n from 6 to
   several dozen. Crucially it also **breaks the monotonicity**: a 10M model at
   epoch 3 and a 500k model at epoch 90 are not ordered by `log N_pretrain`, so
   the trivial baseline finally has a chance to be wrong and LogME finally has a
   chance to beat it. It simultaneously fixes the training-completeness confound
   in §1.2 of the caveats, which is otherwise unaddressable.

2. **Fix the `matpes_ft_rs` configs** so the second target is real, or relabel
   the family as the lr ablation it currently is.

3. **Add downstream targets.** MatPES, mad, and mpaloe are already processed in
   this repo. Each is an independent test of the same estimator.

4. **Re-run the AM arm against `train_small.lmdb`** rather than `val.lmdb`.
   `AM/val.lmdb` is sAlex-only while the fine-tune trains on a 13% MPtrj / 87%
   sAlex mixture, so the two splits are not interchangeable for this dataset
   (they are for MOF-OFF).

Until at least (1) is done, my assessment is that **on the MOF-OFF arm this
experiment cannot distinguish LogME from sorting the checkpoints by pretraining
set size**, and reporting rho = 1.0 there as support for LogME would be
misleading. The AM arm is the exception that proves the point: it is informative
precisely *because* its ground truth is non-monotonic, and it is informative on
the strength of exactly one model pair.

The claim the data currently supports is: *LogME ranked six frozen MACE
checkpoints in exact agreement with their zero-shot downstream error on two
targets, including one case where pretraining-set-size did not.* That is worth
following up. It is not yet "a training-free model-selection signal for MLIP
foundation models."

---

## 6. Figures

All in `outputs/logme/figures/` (light-mode PNGs, validated categorical
palette, series direct-labeled as well as legended).

| figure | what it shows |
|---|---|
| `logme_vs_scale_mean.png` | LogME vs pretraining scale, per feature set, both targets. Subsample bands are drawn but invisible — they are ~0.015 against a range of 0.87. |
| `per_layer_mean.png` | Grouped bars: which layer scores highest at each scale. |
| `stability_mean.png` | Between-model spread vs within-model subsample noise. **Every point sits well above the diagonal**, so the ranking is signal, not sampling noise. |
| `predictor_grid_mean.png` | Every predictor vs scale, z-scored within panel. Predictors that trace a straight line are indistinguishable from `log N_pretrain`. |

The `_sum.png` variants repeat each figure with sum-pooling (paired with total
energy) as the robustness check described in the caveats.

## 7. Reproducing

```bash
sbatch scripts/transfer/submit_logme.sh      # LogME + baselines, ~25 min on 1x H200
sbatch scripts/transfer/submit_zeroshot.sh   # zero-shot ground truth, ~10 min
python scripts/transfer/build_results_table.py   # correlations + verdict
python scripts/transfer/plot_logme.py            # figures
```

Outputs land in `outputs/logme/`. Re-running `build_results_table.py` after the
fine-tunes finish picks up the post-fine-tune MAEs automatically and switches
the ground truth from zero-shot to fine-tuned.
