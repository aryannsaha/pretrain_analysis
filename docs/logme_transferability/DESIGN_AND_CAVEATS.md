# LogME for MACE: design decisions and caveats

Every place where a choice could have changed the answer. Read this before the
results.

---

## 1. What the study can and cannot currently measure

### 1.1 The scale ladder is 100k / 500k / 1M / 2M / 5M / 10M — there is no 10k

The brief specified 10k, 100k, 1M, 2M, 5M, 10M. What exists under
`runs/mace/omat24/stratified_random_nested/` is **100k, 500k, 1M, 2M, 5M, 10M**.
A `sr_10k` *dataset* exists but no model was trained on it in this family. An
older, differently-sampled family (`runs/mace/mace_omat_full_10k/`) does contain
a 10k model, but it was trained on a different subset construction
(`mace_omat_full_*` rather than the nested stratified sample), so mixing it in
would confound the scaling axis with a change of sampling scheme.

**Consequence:** n = 6 either way, but the low end of the ladder is 100k, not
10k, so the dynamic range in `log N_pretrain` is 2 orders of magnitude rather
than 3.

### 1.2 The six foundations are not matched on training completeness

Per `runs/mace/matpes_ft_rs/FOUNDATION_MODEL_EPOCHS.md`, the checkpoints sit at
different fractions of their epoch budgets:

| scale | saved epoch | budget | % of budget |
|---|---|---|---|
| 100k | 115 | 150 | 77% |
| 500k | 94 | 100 | 94% |
| 1M | 57 | 60 | 95% |
| 2M | 40 | 45 | 89% |
| 5M | 20 | 25 | 80% |
| 10M | 10 | 15 | 67% |

**This is the most serious confound in the study.** The independent variable is
supposed to be pretraining *data* scale, but pretraining *duration* co-varies
non-monotonically with it (77, 94, 95, 89, 80, 67). The two least-converged
models are at opposite ends of the ladder. Any correlation between a
transferability score and downstream MAE is therefore a correlation with a
mixture of data scale and training completeness, and the two cannot be
separated with six points.

If LogME turns out to track downstream MAE *better* than `log N_pretrain` does,
one plausible reading is precisely that LogME sees training completeness (which
is in the weights) while `log N_pretrain` does not. That would be a genuinely
interesting result, but it is a different claim from "LogME predicts the
benefit of pretraining data scale", and the write-up must not conflate them.

### 1.3 Both fine-tune families originally targeted MOF-OFF

At the time of the first survey, the two families of six fine-tunes were:

- `runs/mace/mof_off_ft_rs/mace_mof_off_rs_small_ft_*` — lr 1e-4
- `runs/mace/matpes_ft_rs/mace_oam_rs_small_ft_*` — lr 1e-5

Despite the directory name `matpes_ft_rs` and the run name `oam`, **every one of
the latter six configs points at MOF-OFF**:

```yaml
# runs/mace/matpes_ft_rs/mace_oam_rs_small_ft_1m/omat_1m_mof_off_r2scan_20260804.yaml
train_file: .../data/processed/mof_off/R2SCAN/R2SCAN_train.lmdb
valid_file: .../data/processed/mof_off/R2SCAN/R2SCAN_val.lmdb
```

confirmed at 100k / 1M / 10M and in the 500k runtime log
(`Loading single LMDB file: .../mof_off/R2SCAN/R2SCAN_train.lmdb`,
`training dataset size: 80643`). The launcher was evidently cloned from the MOF
one and the data paths were never changed. So those twelve runs are **6 scales
x 2 learning rates on one target**, not 6 x 2 targets.

A third family, `runs/mace/am_ft_rs/mace_am_rs_small_ft_*`, was launched on
2026-08-04 and *does* target AM (MPtrj + sAlex), via
`data/processed/AM/train_small.lmdb`. Those runs supply the in-distribution arm.

**Consequence:** the MOF-vs-MPtrj contrast rests on the `mof_off_ft_rs` and
`am_ft_rs` families. The `matpes_ft_rs` family should be reported as a
learning-rate ablation on MOF-OFF, or excluded — it is not a second target.

### 1.4 The AM target is not exactly MPtrj

`data/processed/AM/train_small.lmdb` is a 100k-structure sample of a combined
corpus that is 13% MPtrj and 87% sAlex (by the metadata: MPTrj 13,181 + sAlex
86,819). `AM/val.lmdb` is **sAlex only — no MPtrj at all**. Describing this arm
as "MPtrj" overstates it; it is an Alexandria-dominated mixture, and the
validation split contains none of the MPtrj distribution.

Two further provenance notes recorded in the dataset metadata: MPtrj energies
come from `uncorrected_total_energy` (raw VASP, **not** MP2020-corrected), and
MPtrj stresses were converted from VASP kbar **with a sign reversal**.

### 1.5 The MAEs were not available when this was built

The `mof_off_ft_rs` and `am_ft_rs` runs were queued or running as this package
was written. LogME and every baseline are computed on the **frozen** checkpoints
and are final; the rank correlations are the only part that waits. Re-running
`build_results_table.py` after the fine-tunes complete fills them in.

---

## 2. Estimator design decisions

### 2.1 The `(d - rank) log alpha` term

`A = alpha I_d + beta F^T F` has `d` eigenvalues, but a thin SVD only produces
`rank <= min(n, d)` of them; the remaining `d - rank` equal `alpha` exactly.
The reference implementation (`thuml/LogME`) omits their contribution to
`log|A|`. It vanishes when `rank == d` — the ordinary `n > d` full-rank case,
where our implementation and the reference agree to 1e-8 — but it is required
when `n < d` or `F` is rank deficient.

*Could have changed the answer:* only for aggressive atom subsampling or highly
degenerate features. All production runs here have `n >> d` (n = 3000 structures
or 50,000 atoms against d <= 256), so this term is inactive on the real data.
It is implemented and tested because the brief asked for the `n < d` case to be
handled, not because it bites here.

### 2.2 Centering, and the absent intercept

The Bayesian model `y = Fw` has no bias term, and its isotropic prior would
shrink one toward zero anyway — wrong for a target like energy-per-atom whose
mean is large and carries no transferability information. Both `F` and `y` are
therefore **centered** by default, which removes the offset from the problem
rather than asking the features to explain it.

*Could have changed the answer:* yes, substantially. Without centering, a large
part of the evidence is spent explaining the target's mean, and models whose
features happen to contain a strong constant direction score higher for a reason
unrelated to transferability.

### 2.3 Column standardization: not applied, deliberately

LogME is invariant to rotations of feature space and to a global rescaling
`F -> cF` (absorbed by `alpha -> c^2 alpha`), but **not** to per-column
rescaling — the isotropic prior is not exchangeable under an anisotropic
reparameterization. Both facts are verified in
`tests/transfer/test_logme.py`.

The convention adopted is **center, do not standardize**, on the grounds that
MACE's own readout is a linear map on the raw invariant features, so the raw
column geometry is the one the fine-tune actually inherits.

*Could have changed the answer:* yes. Standardizing would upweight low-variance
channels, which in a MACE layer are often near-dead. Because the choice is
applied identically to all six checkpoints it should largely cancel in the
*ranking*, but that is an assumption, not a guarantee.

### 2.4 Target scaling makes LogME columns non-comparable

Under `y -> c y` the score shifts by `-log|c|` — a change-of-variables
Jacobian, not a change in fit quality. So `logme_energy` and `logme_forcemag`
live on different scales and **must never be compared to each other**. Only
comparisons within a column, across models, are meaningful.

### 2.5 Pooling must match the energy target

MACE computes `E_inter = sum_i readout(h_i)`. Therefore:

- total interaction energy is exactly linear in **sum**-pooled features;
- per-atom energy is exactly linear in **mean**-pooled features.

The brief asked for sum-pooling against per-atom energy. That pairing is
**inconsistent**: `E/N = w . F_sum` is only linear if `N` is constant, and MOF
structures vary in size. Both pairings are computed and reported; the
self-consistent one (`mean` pool with per-atom energy) is the default.

*Could have changed the answer:* yes, and this is the most likely place for a
silent error in a naive implementation. Compare the `pooling=mean` and
`pooling=sum` rows before drawing conclusions.

### 2.6 E0 is not subtracted

MACE's total energy is `E0 + E_inter`, where `E0` is a fixed per-element sum the
readout never sees. A linear model on node features can only explain `E_inter`.
The features do encode element identity, so a linear map can approximate the E0
term, but not exactly, and not with the downstream dataset's own E0s.

*Could have changed the answer:* plausibly. The residual E0 mismatch is a
composition-dependent offset that inflates the apparent noise floor equally for
all six models. Since it is identical across models it should not affect the
*ranking*, but it does mean the absolute LogME values are pessimistic.

---

## 3. Feature-extraction decisions

### 3.1 Only invariant reductions are used

`inv` takes the `l = 0` channels; `normed` adds per-`l`, per-channel L2 norms of
the `l > 0` blocks. Raw `l > 0` components are never used — under a rotation of
the structure they mix among themselves, so a linear model on them fits an
arbitrary choice of frame.

This was **verified against a real checkpoint**, not just asserted: rotating a
periodic MOF structure moves `layer0`'s raw irreps by 0.23 in absolute value
while `layer0/normed` moves by 4e-8 relative
(`scripts/transfer/smoke_test_extraction.py`). That test also confirms the
e3nn `(mul, 2l+1)` multiplicity-outermost layout assumption, which is where a
silent channel-mixing bug would otherwise hide.

### 3.2 `layer1/normed` is identical to `layer1/inv`

These checkpoints have `hidden_irreps = 128x0e + 128x1o` with 2 interactions,
and MACE's **final** product block emits scalars only (`128x0e`). So layer 1 has
no `l > 0` block and `normed == inv` bit-for-bit. It is emitted under both names
to keep the grid rectangular, and collapsed in the figures. Do not read the two
as independent measurements.

Feature widths: `layer0/inv` 128, `layer0/normed` 256, `layer1/inv` 128,
`readout_hidden` 16.

### 3.3 An equivariant force-vector estimator is NOT included

Per-component force vectors against invariant features would not respect
equivariance and the score would be meaningless. The force target used here is
the **magnitude** `||F_i||`, which is rotation invariant, and it is a proxy for
force accuracy, not force accuracy itself.

A principled equivariant variant exists — an equivariant linear readout from the
`l = 1` features, i.e. `F_i = sum_c w_c h_i^{(1,c)}` with a scalar weight per
channel, whose evidence could be maximized the same way. That is a *different
estimator* and is proposed as an extension, not substituted into the main path.

### 3.4 LEEP and NCE are inapplicable, not omitted

Both require the source model to expose a classification head so that source
label posteriors can be pushed through to target labels. A MACE energy/force
regressor has no such head — its readout emits a scalar per atom — and there is
no label-free substitute that preserves what those estimators measure.

---

## 4. Sampling decisions

- **3000 structures** per downstream target, a fixed random subset shared across
  all six checkpoints (same seed), so differences are between models rather than
  between samples.
- **50,000 atoms** subsampled for the per-atom (force-magnitude) target, with the
  *same rows* used for every feature set so per-atom scores stay comparable.
- **1500 OMat24 structures** from `sr_100k/train.lmdb` for the distributional
  baseline. `sr_100k` is a subset of every larger nested subset, so it is
  in-distribution for all six models, and holding it fixed means the distance
  varies only through the model's representation.
- LogME is recomputed on **8 random half-subsamples** per row and the standard
  deviation reported. If that spread is comparable to the between-model spread,
  the ranking is noise — see the `stability_*.png` figure.

### 4.1 LogME is computed on the *validation* split, not the training split

Both arms use the same LMDB the corresponding fine-tune passes as `valid_file`:
`mof_off/R2SCAN/R2SCAN_val.lmdb` and `AM/val.lmdb`.

The argument for this is that the quantity being predicted is the fine-tune's
*validation* MAE, so the LogME target should be drawn from the distribution that
metric is measured on. The argument against is that transferability is a
property of the task the model will be trained on, which is the training split.

For MOF-OFF the two splits are drawn from one pool, so the choice is immaterial.
**For AM it is not**: `AM/train_small.lmdb` is 13% MPtrj / 87% sAlex, while
`AM/val.lmdb` is sAlex only. Computing LogME on the training split would change
the AM numbers. This is a defensible choice, not a free one, and re-running the
AM arm against `train_small.lmdb` is a one-flag change worth doing as a check.

*Note on the nested design:* because the pretraining subsets are nested random
samples of one pool, the six models saw the **same data distribution**, differing
only in sample size. So the distributional-distance baseline cannot distinguish
them through the data; it can only distinguish them through how each model's
learned representation places the target relative to pretraining. That is still
a meaningful measurement, but it is weaker than the Tan et al. setting, where the
pretraining distributions genuinely differ.

---

## 5. Statistical power — the binding constraint

With six models, the exact one-sided permutation distribution over all 720
orderings gives:

| Spearman rho | exact p | verdict |
|---|---|---|
| 1.000 (perfect) | 0.00139 | the floor — nothing beats it |
| 0.943 (one adjacent swap) | 0.00833 | clears p < 0.01 |
| 0.886 (two swaps) | 0.01667 | clears p < 0.05 |
| 0.829 | 0.02917 | the **last** value that clears 0.05 |
| 0.771 or below | > 0.05 | indistinguishable from chance |

**A predictor must reach rho >= 0.83 to have shown anything at all**, i.e. get
the ordering right to within about two adjacent transpositions. A
respectable-looking rho = 0.77 is not significant here, and the difference
between 0.83 and 0.77 is a single pair of adjacent models swapping.

Bootstrap CIs on a 6-point rank correlation span most of [-1, 1]. They are
computed and reported so that width is visible rather than inferred.

**This means the study as specified cannot produce a strong result even in
principle.** The honest outcomes are "LogME ranks these six near-perfectly" or
"nothing was demonstrated". Reporting a bare rho = 0.9 as evidence would be
overclaiming; at n = 6 that is one swap away from p > 0.05.

### What would fix it

More points on the x-axis, not more atoms per point. Options, roughly in order
of cost:

1. **Intermediate checkpoints.** Each pretraining run wrote per-epoch
   checkpoints. Treating (scale, epoch) pairs as separate "pretrained models"
   could take n from 6 to several dozen at zero extra pretraining cost, and
   would additionally *decouple* data scale from training duration — which is
   the confound in §1.2. This is the single highest-value change.
2. **Both learning rates as separate points.** The `matpes_ft_rs` family is a
   real lr ablation on MOF-OFF; if fine-tune outcomes differ, that is 12 points
   on one target rather than 6.
3. **More downstream targets.** Each additional target is another independent
   test of the same estimator, and MatPES/mad/mpaloe are already processed in
   this repo.
