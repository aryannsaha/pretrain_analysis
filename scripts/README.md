# Scripts

All scripts assume they are launched from the repository root after:

```bash
conda activate pretrain_analysis_env
```

- `run_uma.sh`: launches `fairchem -c <config>`.
- `1_UMA_prepare_uma_finetune_from_traj.py`: converts train/validation `.traj` files into UMA ASE-LMDB folders and runnable YAML configs.
- `run_upet.sh`: launches `mtt train <options.yaml>`.
- `run_mace.sh`: launches `mace_run_train --config=<config>`.
- `setup_submitit_proxy.sh`: installs the proxy-aware `submitit` fork into the active environment.
- `make_uma_data_paths_absolute.py`: updates UMA data-task YAML `src` fields to absolute paths.
- `evaluate_uma_parity_batched.py`: runs the UMA MatPES parity evaluations and writes the figures.
- `uma/evaluate_uma_checkpoint_on_traj.py`: evaluates one UMA checkpoint on one `.traj`/ASE-LMDB.

## UMA Parity Evaluation

The UMA counterpart of `evaluate_mace_parity_batched.py`. Both write the same
long-form `predictions.csv`, apply the same 25 eV/A reference-force cut, and use
the same panel layout, so UMA and MACE MAEs are directly comparable.

`scripts/internal/uma_parity_inference.py` streams frames through a fine-tuned
UMA `inference_ckpt.pt` in batches and caches predictions; re-running with the
same checkpoint, data, and options reuses the CSV instead of recomputing it.

Evaluate a single checkpoint against a single trajectory:

```bash
python scripts/uma/evaluate_uma_checkpoint_on_traj.py \
  --checkpoint runs/uma/.../checkpoints/final/inference_ckpt.pt \
  --data data/processed/matpes/r2scan_flatiron/r2scan_3/lte_val.traj \
  --output-dir runs/matpes_parity/uma_adhoc/lte3_val
```

Run the full MatPES r2SCAN 75k sweep (lte3/lte4 x val/gt) on Slurm, then build
the summary once the array finishes:

```bash
sbatch scripts/uma/submit_uma_matpes_parity.slurm
python scripts/evaluate_uma_parity_batched.py --summary-only
```

`--base-model` selects which fine-tune family to evaluate: `1p2p1` (default),
`1p1`, or `all` to put both in one grouped figure. The Slurm array index maps
onto that family's evaluations: `0` = lte3 val, `1` = lte3 gt, `2` = lte4 val,
`3` = lte4 gt; override the family with `BASE_MODEL=1p1 sbatch ...`. Summary
files are named per scope (`uma_matpes_efs_mae_summary_<scope>.png`), so the
families never overwrite each other.

Parity panels are hexbin density maps over **every** point with a log colour
scale, not a subsample; `--plot-style scatter` restores plain markers and
`--gridsize` sets the hexbin resolution. Reported MAEs are accumulated in
float64 while the plotted pairs are stored as float32, so the rendering choice
never moves a number.

Predictions are cached per evaluation, so re-running a finished index only
redraws its figure — a few seconds each, up to ~20 s for the 8.4M-point `gt`
splits. Redraws need no GPU, so run them directly rather than queueing them.
Set `PRETRAIN_ANALYSIS_ROOT` when the code runs from a git worktree so `data/`
and `runs/` still resolve to the main checkout.

## UMA From `.traj`

FAIR-Chem fine-tuning trains from ASE database folders, not directly from a
single `.traj`. Keep the raw split files in `data/processed/...`, then generate
the training-ready folders and YAMLs:

Edit `TRAIN_INPUT`, `VAL_INPUT`, `OUTPUT_DIR`, `LOCAL_CHECKPOINT`, and the job
settings in `scripts/1_UMA_prepare_uma_finetune_from_traj.py`, then run:

```bash
python scripts/1_UMA_prepare_uma_finetune_from_traj.py
```

This creates:

```text
data/processed/mof-off/r2scan-d4/uma_efs/
  train/
  val/
  data/<generated data-task yaml>
  uma_sm_finetune_template.yaml
```

Run the self-contained generated config:

```bash
fairchem -c configs/uma/uma_sm_finetune_template.yaml
```

The script is intentionally just a small wrapper around FAIR-Chem's
`create_uma_finetune_dataset.py`. Each input can be a single `.traj` file or a
directory containing many `.traj` files. After FAIR-Chem generates the YAMLs in
`OUTPUT_DIR`, the script patches the `job` block and copies the training YAML
plus matching data YAML into `configs/uma/`. It also replaces FAIR-Chem's
checkpoint downloader with the local path in `LOCAL_CHECKPOINT`.
