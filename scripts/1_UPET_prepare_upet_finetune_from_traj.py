#!/usr/bin/env python
"""Edit the values below, then run this file from the repo root."""

from datetime import date
import os
from pathlib import Path
import shlex

import numpy as np
import yaml
from ase.io import read, write


def slurm_gpus_per_node(gres):
    for item in str(gres).split(","):
        fields = item.strip().split(":")
        if fields and fields[0] == "gpu":
            if len(fields) == 1:
                return 1
            try:
                return int(fields[-1])
            except ValueError:
                return 1
    return 0


ROOT = Path(__file__).resolve().parents[1]
TRAIN_INPUT = ROOT / "data/processed/mof-off/r2scan-d4/train_10k.traj"
VAL_INPUT = ROOT / "data/processed/mof-off/r2scan-d4/val_1k.traj"
OUTPUT_DIR = ROOT / "data/processed/mof-off/r2scan-d4/upet_moff_off_test"
PRETRAINED = ROOT / "models/pretrained/pet-omat-s-v1.0.0.ckpt"
DATASET_NAME = "mof_off_r2scan_d4"
MODEL_NAME = PRETRAINED.stem
DATE_TAG = date.today().strftime("%Y%m%d")
CONFIG_OUT = ROOT / "configs/upet" / f"{MODEL_NAME}_{DATASET_NAME}_{DATE_TAG}.yaml"
RUN_DIR = ROOT / "runs/upet/upet_moff_off_test"
WANDB_ENTITY = "rosengroup-general"
WANDB_PROJECT = "finetuning"
RUN_NAME = "upet_moff_off_test10"
WANDB_ID = os.environ.get("WANDB_RUN_ID")
WANDB_NAME = os.environ.get("WANDB_NAME", RUN_NAME)
WANDB_RESUME = os.environ.get("WANDB_RESUME", "never")
DEVICE = "cuda"
NUM_EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1.0e-5
NUM_WORKERS = 0
TARGET = "energy/r2scan_d4"
ENERGY_KEY, FORCES_KEY, STRESS_KEY = "moff_energy", "moff_forces", "moff_stress"
ENERGY_IN, FORCES_IN, STRESS_IN = ("energy", ENERGY_KEY), ("forces", FORCES_KEY), ("stress", STRESS_KEY)
SLURM_JOB_NAME = "upet_moff"
SLURM_TIME = "6:00:00"
SLURM_NODES = 1
SLURM_CPUS_PER_TASK = 8
SLURM_MEM = "32G"
SLURM_GRES = "gpu:1"
SLURM_GPUS_PER_NODE = slurm_gpus_per_node(SLURM_GRES)
SLURM_NTASKS_PER_NODE = SLURM_GPUS_PER_NODE if DEVICE == "cuda" and SLURM_GPUS_PER_NODE else 1
SLURM_NTASKS = SLURM_NODES * SLURM_NTASKS_PER_NODE
SLURM_CONSTRAINT = "intel&gpu80"
CONDA_ENV = "pretrain_analysis_env"
CONDA_SH = Path("/scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh")
OUTPUT_MODEL = f"{RUN_DIR.name}.pt"
SUBMIT_SCRIPT = RUN_DIR / "submit_upet.sh"


def q(value):
    return shlex.quote(str(value))


def frames(path):
    path = path.expanduser().resolve()
    files = sorted(path.rglob("*.traj")) + sorted(path.rglob("*.xyz")) + sorted(path.rglob("*.extxyz")) if path.is_dir() else [path]
    return [atoms for file in files for atoms in read(file, ":")]


def get(atoms, keys, result):
    for key in keys:
        if key in atoms.info:
            return atoms.info[key]
        if key in atoms.arrays:
            return atoms.arrays[key]
    if atoms.calc and result in atoms.calc.results:
        return atoms.calc.results[result]
    return atoms.get_potential_energy() if result == "energy" else atoms.get_forces() if result == "forces" else atoms.get_stress(voigt=False)


def stress9(value):
    value = np.asarray(value, dtype=float)
    if value.shape == (6,):
        xx, yy, zz, yz, xz, xy = value
        value = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return value.reshape(9)


def write_split(src, dst):
    out = []
    for source in frames(src):
        atoms = source.copy()
        atoms.info[ENERGY_KEY] = float(np.asarray(get(source, ENERGY_IN, "energy")))
        atoms.arrays[FORCES_KEY] = np.asarray(get(source, FORCES_IN, "forces"), dtype=float).reshape(len(atoms), 3)
        atoms.info[STRESS_KEY] = stress9(get(source, STRESS_IN, "stress")).tolist()
        atoms.calc = None
        out.append(atoms)
    if not out:
        raise SystemExit(f"No ASE-readable structures found in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    write(dst, out)
    first = read(dst, "0")
    if ENERGY_KEY not in first.info or FORCES_KEY not in first.arrays or STRESS_KEY not in first.info:
        raise SystemExit(f"{dst} is missing metatrain target keys")
    if first.arrays[FORCES_KEY].shape != (len(first), 3) or np.asarray(first.info[STRESS_KEY]).size != 9:
        raise SystemExit(f"{dst} has bad force/stress shapes")
    return len(out)


def latest_checkpoint():
    ckpts = []
    for root in (RUN_DIR, ROOT / "outputs"):
        if not root.exists():
            continue
        for ckpt in root.rglob("*.ckpt"):
            opts = ckpt.parent / "options_restart.yaml"
            if root.name == "outputs" and (WANDB_ID is None or not opts.exists() or WANDB_ID not in opts.read_text()):
                continue
            ckpts.append(ckpt)
    return max(ckpts, key=lambda path: path.stat().st_mtime) if ckpts else PRETRAINED


def split(path):
    path = str(path.resolve())
    return {
        "systems": {"read_from": path, "reader": "ase", "length_unit": "angstrom"},
        "targets": {TARGET: {"quantity": "energy", "read_from": path, "reader": "ase", "key": ENERGY_KEY, "unit": "eV", "description": "MOF-OFF R2SCAN-D4 total energy", "forces": {"read_from": path, "reader": "ase", "key": FORCES_KEY}, "stress": {"read_from": path, "reader": "ase", "key": STRESS_KEY}}},
    }


def write_slurm_script(config_path: Path) -> Path:
    logs_dir = RUN_DIR / "logs"
    wandb_dir = RUN_DIR / "wandb"
    wandb_cache_dir = RUN_DIR / ".wandb_cache"
    wandb_config_dir = RUN_DIR / ".wandb_config"
    for directory in (logs_dir, wandb_dir, wandb_cache_dir, wandb_config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    content = f"""#!/bin/bash
#SBATCH --job-name={SLURM_JOB_NAME}
#SBATCH --output={logs_dir.resolve()}/%x_%j.out
#SBATCH --error={logs_dir.resolve()}/%x_%j.err
#SBATCH --time={SLURM_TIME}
#SBATCH --nodes={SLURM_NODES}
#SBATCH --ntasks={SLURM_NTASKS}
#SBATCH --ntasks-per-node={SLURM_NTASKS_PER_NODE}
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}
#SBATCH --mem={SLURM_MEM}
#SBATCH --gres={SLURM_GRES}
#SBATCH --constraint={SLURM_CONSTRAINT}

set -euo pipefail

CONFIG_FILE={q(config_path.resolve())}
ROOT={q(ROOT)}
RUN_DIR={q(RUN_DIR)}
OUTPUT_MODEL={q(OUTPUT_MODEL)}
CONDA_ENV="${{CONDA_ENV:-{CONDA_ENV}}}"

module load proxy/default
source {q(CONDA_SH)}
conda activate "$CONDA_ENV"

mkdir -p "$RUN_DIR/logs"
export WANDB_MODE="${{WANDB_MODE:-online}}"
export WANDB_RESUME="${{WANDB_RESUME:-{WANDB_RESUME}}}"
export WANDB_DIR="${{WANDB_DIR:-$RUN_DIR/wandb}}"
export WANDB_CACHE_DIR="${{WANDB_CACHE_DIR:-$RUN_DIR/.wandb_cache}}"
export WANDB_CONFIG_DIR="${{WANDB_CONFIG_DIR:-$RUN_DIR/.wandb_config}}"
export WANDB_CONSOLE="${{WANDB_CONSOLE:-off}}"
export WANDB_DISABLE_CODE="${{WANDB_DISABLE_CODE:-true}}"
export WANDB_DISABLE_GIT="${{WANDB_DISABLE_GIT:-true}}"
export WANDB_SAVE_CODE="${{WANDB_SAVE_CODE:-false}}"
export WANDB_X_SAVE_REQUIREMENTS="${{WANDB_X_SAVE_REQUIREMENTS:-false}}"
export WANDB_IGNORE_GLOBS="${{WANDB_IGNORE_GLOBS:-config.yaml,requirements.txt,wandb-metadata.json,wandb-summary.json,output.log,*.ckpt,*.pt,*.model,*.extxyz}}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

cd "$RUN_DIR"
srun --ntasks={SLURM_NTASKS} mtt train "$CONFIG_FILE" --restart auto -o "$OUTPUT_MODEL"
"""
    SUBMIT_SCRIPT.write_text(content)
    SUBMIT_SCRIPT.chmod(0o755)
    return SUBMIT_SCRIPT


train_xyz, val_xyz = OUTPUT_DIR / "train.extxyz", OUTPUT_DIR / "val.extxyz"
RUN_DIR.mkdir(parents=True, exist_ok=True)
n_train, n_val = write_split(TRAIN_INPUT, train_xyz), write_split(VAL_INPUT, val_xyz)
read_from = latest_checkpoint()
if not read_from.is_file():
    raise SystemExit(f"Checkpoint not found: {read_from}")

wandb_cfg = {"entity": WANDB_ENTITY, "project": WANDB_PROJECT, "name": WANDB_NAME, "resume": WANDB_RESUME}
if WANDB_ID is not None:
    wandb_cfg["id"] = WANDB_ID

cfg = {
    "seed": 42,
    "device": DEVICE,
    "wandb": wandb_cfg,
    "architecture": {"name": "pet", "training": {"batch_size": BATCH_SIZE, "num_epochs": NUM_EPOCHS, "learning_rate": LEARNING_RATE, "log_interval": 1, "checkpoint_interval": 1, "num_workers": NUM_WORKERS, "distributed": SLURM_NTASKS > 1, "finetune": {"method": "full", "read_from": str(read_from.resolve())}, "loss": {TARGET: {"type": "mse", "weight": 20.0, "reduction": "mean", "forces": {"type": "mse", "weight": 2.0, "reduction": "mean"}, "stress": {"type": "mse", "weight": 1.0, "reduction": "mean"}}}}},
    "training_set": split(train_xyz),
    "validation_set": split(val_xyz),
}
CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
CONFIG_OUT.write_text(yaml.safe_dump(cfg, sort_keys=False))
submit_script = write_slurm_script(CONFIG_OUT)
print(
    f"Wrote {n_train} train / {n_val} val structures and {CONFIG_OUT}\n"
    f"Wrote SLURM script: {submit_script}\n"
    f"Submit with: sbatch {submit_script}"
)
