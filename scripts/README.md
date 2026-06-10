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
