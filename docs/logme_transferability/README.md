# LogME transferability estimation for pretrained MACE potentials

Can LogME, computed on a **frozen** pretrained model's features, predict
downstream fine-tuned performance without running the fine-tune?

This directory documents the implementation and the study design. The honest
assessment of what has and has not been shown is in
[`RESULTS.md`](RESULTS.md); every decision that could have changed the answer is
in [`DESIGN_AND_CAVEATS.md`](DESIGN_AND_CAVEATS.md).

---

## Package layout

```
src/pretrain_analysis/transfer/
  logme.py      LogME (You et al., ICML 2021) via evidence maximization
  irreps.py     O(3)-invariant reductions of MACE node features
  extract.py    Forward-hook activation extraction from frozen MACE models
  baselines.py  Ridge probe, H-score, distributional distance, effective dim
  evaluate.py   Spearman / Kendall / weighted-Kendall + bootstrap + exact perms

tests/transfer/     90 unit tests
scripts/transfer/
  smoke_test_extraction.py   Validates hooks and rotation invariance on a real checkpoint
  run_logme_analysis.py      Driver: scores every checkpoint x target x layer x variant
  build_results_table.py     Joins fine-tune MAEs and reports rank agreement
  plot_logme.py              Figures
  submit_logme.sh            SLURM (ailab, 1x H200)
```

Nothing outside numpy/scipy is required by the package itself, so it runs
unchanged in `pretrain_analysis_env_mace` (which has MACE but no scikit-learn)
as well as `pretrain_analysis_env`. The GMM and ridge path are written out
rather than imported for exactly this reason, and because the brief asked for an
auditable implementation.

---

## How to run it

```bash
# 1. Unit tests (no GPU, no checkpoints needed)
conda activate pretrain_analysis_env
python -m pytest tests/transfer/ -q

# 2. Validate extraction against a real checkpoint (needs MACE)
conda activate pretrain_analysis_env_mace
python scripts/transfer/smoke_test_extraction.py

# 3. Score all six checkpoints on both targets (SLURM, ~25 min on one H200)
sbatch scripts/transfer/submit_logme.sh

# 4. Once the fine-tunes finish, join the MAEs and correlate
python scripts/transfer/build_results_table.py
python scripts/transfer/plot_logme.py
```

---

## What is being scored

**Six frozen MACE checkpoints**, pretrained on nested stratified random subsets
of OMat24: **100k, 500k, 1M, 2M, 5M, 10M**. All share one architecture
(`hidden_irreps = 128x0e + 128x1o`, 2 interactions, `r_max` 6.0, 89 elements,
`ScaleShiftMACE` with an identity scale/shift), so feature widths are identical
across the ladder and only the weights differ.

Snapshotted with SHA-256 manifests to
`models/logme_frozen_foundations/` — the originals under
`runs/mace/omat24/stratified_random_nested/*/models/` are transient and will be
overwritten by the queued pretraining-resume jobs.

**Feature sets**, all O(3)-invariant:

| name | width | what it is |
|---|---|---|
| `layer0/inv` | 128 | `l=0` channels of product block 0 |
| `layer0/normed` | 256 | plus per-channel norms of the `128x1o` block |
| `layer1/inv` | 128 | `l=0` channels of product block 1 |
| `layer1/normed` | 128 | **identical to `layer1/inv`** — the last block is scalars only |
| `readout_hidden` | 16 | post-activation hidden layer of the final readout |

**Targets:**

- *energy* — per-structure pooled invariant features against DFT energy.
  Mean-pool pairs with per-atom energy, sum-pool with total energy; both are
  computed, and the pairing matters (see caveats §2.5).
- *force magnitude* — per-atom invariant features against `||F_i||`, which is
  rotation invariant. **A proxy for force accuracy, not force accuracy.**

Per-component force vectors against invariant features are deliberately not
attempted: they are not equivariance-respecting and the score would be
meaningless. An equivariant `l=1` readout variant is proposed as an extension in
the caveats, not substituted in.

---

## Correctness

The estimator is implemented from scratch and cross-checked line-by-line against
`thuml/LogME`, with a literal transcription of the reference fixed point kept in
the test file so any divergence is attributable. Two deliberate departures:

1. **`(d - rank) log alpha` in `log|A|`.** `A = alpha I_d + beta F^T F` always
   has `d` eigenvalues; a thin SVD yields only `rank` of them and the rest equal
   `alpha` exactly. The reference drops this. It vanishes at full rank with
   `n > d` (where the two agree to 1e-8, asserted in the tests) and is required
   when `n < d`.
2. **Self-consistent evidence.** The reference evaluates the evidence inside its
   loop using sufficient statistics from the *previous* iterate, so its reported
   score lags its reported hyperparameters by one step. We recompute after
   convergence.

Verified properties (all in `tests/transfer/`):

- Recovers the generating `alpha` and `beta` on data drawn from the exact model,
  to within the precision the data can carry (`beta` to 5%, `alpha` to 25% —
  `||w||^2` is `chi^2_d/alpha`, so `alpha` cannot be pinned tighter from one
  draw of `w`).
- Pure-noise feature columns never raise the score and degrade it monotonically,
  while a training-R^2 probe rises monotonically on the same data — the
  overfitting immunity that motivates LogME, shown as a contrast rather than
  asserted.
- Invariant to feature-space rotation and to global rescaling
  (`alpha -> c^2 alpha`); **not** invariant to per-column standardization; shifts
  by exactly `-log c` under `y -> cy`.
- Handles `n < d`, rank-deficient features, dead channels, and float32 input.

Extraction correctness rests on one physical check rather than on trusting the
e3nn layout convention: rotating a real periodic MOF structure moves `layer0`'s
raw irreps by 0.23 absolute while `layer0/normed` moves by 4e-8 relative.

---

## Status

The LogME and baseline scores are computed on the frozen checkpoints and are
final. **The rank correlations are blocked**: the fine-tune runs that supply the
ground-truth MAEs were launched on 2026-08-04 and are at epoch 0-3 of 40. See
[`RESULTS.md`](RESULTS.md).
