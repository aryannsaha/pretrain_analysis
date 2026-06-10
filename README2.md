# File Trace for Fine-Tuning Runs

Run commands from the repository root:

```bash
cd /scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
```

The shared starting data in the current scripts is:

- Train split: `data/processed/mof-off/r2scan-d4/train_10k.traj`
- Validation split: `data/processed/mof-off/r2scan-d4/val_1k.traj`
- Pretrained checkpoints: `models/pretrained/`

The usual workflow is:

1. Edit the relevant `scripts/1_*_prepare_*_from_traj.py`.
2. Run the prepare script to create model-specific data files, a config, and a ready-to-submit SLURM script.
3. Submit the generated SLURM script printed by the prepare script.

Important: for UPET and MACE, prefer the generated scripts in `runs/upet/<run_name>/submit_upet.sh` and `runs/mace/<run_name>/submit_mace.sh`. The older static `scripts/run_upet.sh` and `scripts/run_mace.sh` can still have stale hard-coded paths.

## UMA

### 1. Edit `scripts/1_UMA_prepare_uma_finetune_from_traj.py`

Specify only the run-defining values:

- `TRAIN_INPUT` and `VAL_INPUT`: source `.traj` files or directories of `.traj` files.
- `OUTPUT_DIR`: where the generated UMA ASE-LMDB data will be written.
- `LOCAL_CHECKPOINT`: local pre-trained UMA checkpoint, currently `models/pretrained/**.pt`.
- `RUN_DIR` and `RUN_NAME`: where FAIR-Chem stores the run and how W&B labels it.
- `REGRESSION_TASKS`: `e`, `ef`, or `efs`.
- `MAIL_USER`, `WANDB_ENTITY`, and `WANDB_PROJECT`.
- Optional training/job settings such as `NUM_WORKERS`, SLURM time, CPUs, memory, GPU constraint, `epochs`, `batch_size`, and `lr`.

Current script example:

```text
OUTPUT_DIR = data/processed/{DATASET}/{SPECIFIED_DIR}
RUN_DIR    = runs/uma/{SPECIFIED_DIR}
RUN_NAME   = {model}_{dataset}_{additional terms}
```

### 2. Prepare UMA data and configs

Command:

```bash
python scripts/1_UMA_prepare_uma_finetune_from_traj.py
```

This runs FAIR-Chem's `create_uma_finetune_dataset.py`.

Produced under `OUTPUT_DIR`:

```text
data/processed/mof-off/r2scan-d4/uma_moff_off_test2/
  train/
    data.0000.aselmdb
    data.0001.aselmdb
    ...
    metadata.npz
  val/
    data.0000.aselmdb
    data.0001.aselmdb
    ...
    metadata.npz
  data/
    uma_conserving_data_task_energy_force_stress.yaml
  uma_sm_finetune_template.yaml
```

Also copied into `configs/uma/`:

```text
configs/uma/uma-s-1p1_mof_off_r2scan_d4_efs_YYYYMMDD.yaml
configs/uma/data/uma-s-1p1_mof_off_r2scan_d4_efs_YYYYMMDD_data.yaml
```

The copied training YAML points to the copied data YAML through Hydra defaults. The data YAML contains absolute `train_dataset.splits.train.src` and `val_dataset.splits.val.src` paths that point back to `OUTPUT_DIR/train` and `OUTPUT_DIR/val`.

### 3. Files to check before running

- `configs/uma/<generated>.yaml`: check `job.run_dir`, `job.run_name`, `timestamp_id`, `runner.train_eval_unit.model.checkpoint_location`, `epochs`, `batch_size`, and `lr`.
- `configs/uma/data/<generated>_data.yaml`: check the `src` paths and energy/force/stress loss `coefficient` values.
- `scripts/run_uma.sh`: pass the generated config path as the first argument, or edit the default `CONFIG`.
- If running on `della-gpu` with W&B, make sure the proxy-aware `submitit` setup is installed. The helper is `scripts/setup_submitit_proxy.sh`.

### 4. Launch UMA

Either:

```bash
bash scripts/run_uma.sh configs/uma/uma-s-1p1_mof_off_r2scan_d4_efs_YYYYMMDD.yaml
```

or directly:

```bash
fairchem -c configs/uma/uma-s-1p1_mof_off_r2scan_d4_efs_YYYYMMDD.yaml
```

With `job.scheduler.mode: SLURM`, `fairchem` submits a SLURM job through `submitit`.

### 5. UMA run outputs

Outputs go under:

```text
runs/uma/<run_dir_name>/<timestamp_id>/
  canonical_config.yaml
  checkpoints/
    step_0/
      inference_ckpt.pt
      resume.yaml
      __0_0.distcp
    step_1000/
      inference_ckpt.pt
      resume.yaml
      __0_0.distcp
  logs/
    <slurm_id>_0_log.out
    <slurm_id>_0_log.err
    <slurm_id>_submission.sh
```

Current example:

```text
runs/uma/moff_off_test2/uma-s-1p1_mof_off_r2scan_d4_efs_20260610/
```

## UPET

### 1. Edit `scripts/1_UPET_prepare_upet_finetune_from_traj.py`

Specify only the run-defining values:

- `TRAIN_INPUT` and `VAL_INPUT`: source `.traj`, `.xyz`, `.extxyz`, or directories containing those formats.
- `OUTPUT_DIR`: where UPET-ready `.extxyz` files are written.
- `PRETRAINED`: starting PET checkpoint, currently `models/pretrained/pet-omat-s-v1.0.0.ckpt`.
- `DATASET_NAME`: used in the generated config filename.
- `RUN_DIR`, `RUN_NAME`, and W&B settings.
- `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `NUM_WORKERS`, and `DEVICE`.
- `TARGET` and the ASE target keys if the source files use nonstandard energy, force, or stress names.

Current script example:

```text
OUTPUT_DIR = data/processed/mof-off/r2scan-d4/upet_moff_off_test
CONFIG_OUT = configs/upet/pet-omat-s-v1.0.0_mof_off_r2scan_d4_YYYYMMDD.yaml
RUN_DIR    = runs/upet/upet_moff_off_test
RUN_NAME   = upet_moff_off_test10
```

### 2. Prepare UPET data and config

Command:

```bash
python scripts/1_UPET_prepare_upet_finetune_from_traj.py
```

Produced under `OUTPUT_DIR`:

```text
data/processed/mof-off/r2scan-d4/upet_moff_off_test/
  train.extxyz
  val.extxyz
```

Those files contain UPET target keys:

```text
moff_energy
moff_forces
moff_stress
```

Also produced:

```text
configs/upet/pet-omat-s-v1.0.0_mof_off_r2scan_d4_YYYYMMDD.yaml
runs/upet/upet_moff_off_test/submit_upet.sh
```

The generated YAML contains:

- `architecture.training.finetune.read_from`: the pretrained checkpoint, or the newest `.ckpt` found in `RUN_DIR` when resuming.
- `training_set` and `validation_set`: absolute paths to the generated `train.extxyz` and `val.extxyz`.
- `wandb`: entity, project, run name, resume setting, and optional run id from environment variables.
- `submit_upet.sh`: SLURM output/error paths, conda environment, W&B dirs, `RUN_DIR`, generated config path, and output model name.

### 3. Files to check before running

- `scripts/1_UPET_prepare_upet_finetune_from_traj.py`:
  - Check the `SLURM_*` constants if you want different time, memory, CPUs, GPU constraint, or job name.
  - Check `CONDA_ENV` if you changed environments.
  - Check `OUTPUT_MODEL` if you want a different final `.pt` name.
- `runs/upet/<run_name>/submit_upet.sh`: this should already be ready. Skim the generated `#SBATCH` lines and `CONFIG_FILE` if you want to verify before submitting.
- For resuming the same W&B run, set `WANDB_RUN_ID`, `WANDB_NAME`, and `WANDB_RESUME=allow`, and use the metatrain resume source-code changes described in `Fine-tuning Procedures.md`.

### 4. Launch UPET

Submit to SLURM:

```bash
sbatch runs/upet/upet_moff_off_test/submit_upet.sh
```

The generated submit script does:

```bash
cd "$RUN_DIR"
mtt train "$CONFIG_FILE" --restart auto -o upet_moff_off_test.pt
```

### 5. UPET run outputs

SLURM logs go to:

```text
runs/upet/<run_name>/logs/
  upet_moff_<job_id>.out
  upet_moff_<job_id>.err
```

Metatrain outputs are written relative to `RUN_DIR` because the run script changes into that directory:

```text
runs/upet/<run_name>/
  outputs/YYYY-MM-DD/HH-MM-SS/
    train.log
    train.csv
    options_restart.yaml
    model_0.ckpt
    model_1.ckpt
    ...
    <output_name>.ckpt
    <output_name>.pt
  <output_name>.ckpt
  <output_name>.pt
  wandb/
```

Current existing example:

```text
runs/upet/upet_moff_off/outputs/2026-06-10/08-46-58/
```

## MACE

### 1. Edit `scripts/1_MACE_prepare_mace_finetune_from_traj.py`

Specify only the run-defining values:

- `TRAIN_INPUT` and `VAL_INPUT`: source `.traj`, `.xyz`, `.extxyz`, or directories containing those formats.
- `OUTPUT_DIR`: where MACE-ready `.extxyz` files are written.
- `PRETRAINED`: starting MACE model, currently `models/pretrained/mace-omat-0-medium.model`.
- `DATASET_NAME`: used in the generated config filename.
- `RUN_NAME` and `RUN_DIR`: controls both the MACE config name field and output directory.
- `MAX_NUM_EPOCHS`, `BATCH_SIZE`, `LR`, `DEVICE`, and W&B settings.
- `DEFAULT_E0S`: must contain E0 values for every atomic number present in train or validation data.
- Target keys if the source files use nonstandard energy, force, or stress names.

Current script example:

```text
OUTPUT_DIR = data/processed/mof-off/r2scan-d4/mace_moff_off_test
RUN_NAME   = mace_moff_off_TEST
RUN_DIR    = runs/mace/mace_moff_off_TEST
CONFIG_OUT = runs/mace/mace_moff_off_TEST/mace-omat-0-medium_mof_off_r2scan_d4_YYYYMMDD.yaml
```

### 2. Prepare MACE data and config

Command:

```bash
python scripts/1_MACE_prepare_mace_finetune_from_traj.py
```

Produced under `OUTPUT_DIR`:

```text
data/processed/mof-off/r2scan-d4/mace_moff_off_test/
  train.extxyz
  val.extxyz
```

Those files contain MACE target keys:

```text
mace_energy
mace_forces
mace_stress
```

Also produced inside `RUN_DIR`:

```text
runs/mace/mace_moff_off_TEST/mace-omat-0-medium_mof_off_r2scan_d4_YYYYMMDD.yaml
runs/mace/mace_moff_off_TEST/submit_mace.sh
```

The generated YAML contains:

- `foundation_model`: the pretrained MACE model, or the newest `.model` already in `RUN_DIR` when continuing.
- `restart_latest`: `true` if continuing from an existing model in `RUN_DIR`.
- `train_file` and `valid_file`: absolute paths to generated `.extxyz` files.
- `atomic_numbers` and matching `E0s`.
- stress settings: `compute_stress: true`, `loss: stress`, and `error_table: PerAtomMAEstressvirials`.
- `submit_mace.sh`: SLURM output/error paths, conda environment, W&B dirs, `MPLCONFIGDIR`, `RUN_DIR`, and generated config filename.

### 3. Files to check before running

- `scripts/1_MACE_prepare_mace_finetune_from_traj.py`:
  - Check the `SLURM_*` constants if you want different time, memory, CPUs, GPU constraint, or job name.
  - Check `CONDA_ENV` if you changed environments.
  - Check `RUN_NAME`, because it controls the MACE output names.
- `runs/mace/<run_name>/submit_mace.sh`: this should already be ready. Skim the generated `#SBATCH` lines and `CONFIG` if you want to verify before submitting.
- If the data contains a new element, add its E0 to `DEFAULT_E0S` in the prepare script before running.

### 4. Launch MACE

Submit to SLURM:

```bash
sbatch runs/mace/mace_moff_off_TEST/submit_mace.sh
```

The generated submit script does:

```bash
cd "$RUN_DIR"
mace_run_train --config="$CONFIG"
```

### 5. MACE run outputs

SLURM logs and MACE outputs go under `RUN_DIR`:

```text
runs/mace/<run_name>/
  logs/
    mace_moff_<job_id>.out
    mace_moff_<job_id>.err
    <run_name>_run-123.log
    <run_name>_run-123_debug.log
  checkpoints/
    <run_name>_run-123_epoch-0.pt
    ...
  results/
    <run_name>_run-123_train.txt
    <run_name>_run-123_train_Default_stage_one.png
  <generated_config>.yaml
  <run_name>.model
  <run_name>_compiled.model
  wandb/
```

Current existing examples:

```text
runs/mace/mace_moff_off/
runs/mace/mace_moff_off_TEST/
```
