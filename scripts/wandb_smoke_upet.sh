#!/bin/bash
#SBATCH --job-name=upet_wandb_smoke
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/wandb_smoke/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/wandb_smoke/logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
CONDA_ENV="${CONDA_ENV:-pretrain_analysis_env}"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/$CONDA_ENV/bin/python

RUN_ID="upet_wandb_smoke_${SLURM_JOB_ID:-manual}"
RUN_NAME="$RUN_ID"
SMOKE_ROOT="$ROOT/runs/wandb_smoke/$RUN_ID"
CONFIG="$SMOKE_ROOT/upet_wandb_smoke.yaml"

module load proxy/default
source /scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$SMOKE_ROOT" "$SMOKE_ROOT/wandb" "$SMOKE_ROOT/.wandb_cache" "$SMOKE_ROOT/.wandb_config"

export ROOT
export RUN_ID
export RUN_NAME
export SMOKE_ROOT
export WANDB_MODE=online
export WANDB_RUN_ID="$RUN_ID"
export WANDB_NAME="$RUN_NAME"
export WANDB_RESUME=never
export WANDB_DIR="$SMOKE_ROOT/wandb"
export WANDB_CACHE_DIR="$SMOKE_ROOT/.wandb_cache"
export WANDB_CONFIG_DIR="$SMOKE_ROOT/.wandb_config"
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
export WANDB_DISABLE_GIT=true
export WANDB_SAVE_CODE=false
export WANDB_X_SAVE_REQUIREMENTS=false
export WANDB_IGNORE_GLOBS="config.yaml,requirements.txt,wandb-metadata.json,wandb-summary.json,output.log,*.ckpt,*.pt,*.model,*.extxyz"
export PYTHONUNBUFFERED=1

cd "$ROOT"

"$ENV_PY" - <<'PY'
import os
from pathlib import Path

import numpy as np
import yaml
from ase.io import read, write


root = Path(os.environ["ROOT"])
smoke_root = Path(os.environ["SMOKE_ROOT"])
run_id = os.environ["RUN_ID"]
run_name = os.environ["RUN_NAME"]

train_input = root / "data/processed/mof-off/r2scan-d4/train_10k.traj"
val_input = root / "data/processed/mof-off/r2scan-d4/val_1k.traj"
pretrained = root / "models/pretrained/pet-omat-s-v1.0.0.ckpt"
data_dir = smoke_root / "data"
train_xyz = data_dir / "train.extxyz"
val_xyz = data_dir / "val.extxyz"
config_out = smoke_root / "upet_wandb_smoke.yaml"

target = "energy/r2scan_d4"
energy_key = "moff_energy"
forces_key = "moff_forces"
stress_key = "moff_stress"


def get(atoms, keys, result):
    for key in keys:
        if key in atoms.info:
            return atoms.info[key]
        if key in atoms.arrays:
            return atoms.arrays[key]
    if atoms.calc and result in atoms.calc.results:
        return atoms.calc.results[result]
    if result == "energy":
        return atoms.get_potential_energy()
    if result == "forces":
        return atoms.get_forces()
    return atoms.get_stress(voigt=False)


def stress9(value):
    value = np.asarray(value, dtype=float)
    if value.shape == (6,):
        xx, yy, zz, yz, xz, xy = value
        value = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return value.reshape(9)


def write_split(src, dst, selection):
    out = []
    for source in read(src, selection):
        atoms = source.copy()
        atoms.info[energy_key] = float(np.asarray(get(source, ("energy", energy_key), "energy")))
        atoms.arrays[forces_key] = np.asarray(
            get(source, ("forces", forces_key), "forces"), dtype=float
        ).reshape(len(atoms), 3)
        atoms.info[stress_key] = stress9(get(source, ("stress", stress_key), "stress")).tolist()
        atoms.calc = None
        out.append(atoms)

    dst.parent.mkdir(parents=True, exist_ok=True)
    write(dst, out)
    return len(out)


def split(path):
    path = str(path.resolve())
    return {
        "systems": {"read_from": path, "reader": "ase", "length_unit": "angstrom"},
        "targets": {
            target: {
                "quantity": "energy",
                "read_from": path,
                "reader": "ase",
                "key": energy_key,
                "unit": "eV",
                "description": "MOF-OFF R2SCAN-D4 total energy",
                "forces": {"read_from": path, "reader": "ase", "key": forces_key},
                "stress": {"read_from": path, "reader": "ase", "key": stress_key},
            }
        },
    }


n_train = write_split(train_input, train_xyz, ":2")
n_val = write_split(val_input, val_xyz, ":1")

cfg = {
    "seed": 42,
    "device": "cuda",
    "wandb": {
        "entity": "rosengroup-general",
        "project": "finetuning",
        "name": run_name,
        "resume": "never",
        "id": run_id,
    },
    "architecture": {
        "name": "pet",
        "training": {
            "batch_size": 1,
            "num_epochs": 1,
            "learning_rate": 1.0e-5,
            "log_interval": 1,
            "checkpoint_interval": 1,
            "num_workers": 0,
            "finetune": {"method": "full", "read_from": str(pretrained.resolve())},
            "loss": {
                target: {
                    "type": "mse",
                    "weight": 20.0,
                    "reduction": "mean",
                    "forces": {"type": "mse", "weight": 2.0, "reduction": "mean"},
                    "stress": {"type": "mse", "weight": 1.0, "reduction": "mean"},
                }
            },
        },
    },
    "training_set": split(train_xyz),
    "validation_set": split(val_xyz),
}

config_out.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"WANDB_SMOKE_RUN_ID={run_id}")
print(f"Wrote {n_train} train / {n_val} val structures to {smoke_root}")
PY

cd "$SMOKE_ROOT"
"$ENV_PY" -m metatrain train "$CONFIG" -o upet_wandb_smoke.pt

"$ENV_PY" - <<'PY'
import os
import sys
import time

import wandb


entity = "rosengroup-general"
project = "finetuning"
run_id = os.environ["RUN_ID"]
expected = {"training/loss", "validation/loss"}
api = wandb.Api(timeout=30)

last_found = set()
last_rows = 0
url = ""

for attempt in range(18):
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        url = run.url
        rows = list(run.scan_history(keys=sorted(expected), page_size=100))
        found = {key for row in rows for key in expected if row.get(key) is not None}
        last_found = found
        last_rows = len(rows)
        print(f"WANDB_HISTORY_ATTEMPT={attempt} rows={last_rows} found={sorted(found)}")
        if expected <= found:
            print(f"WANDB_SMOKE_OK url={url}")
            sys.exit(0)
    except Exception as err:
        print(f"WANDB_HISTORY_WAIT attempt={attempt} error={err!r}")

    time.sleep(10)

missing = sorted(expected - last_found)
print(
    f"WANDB_SMOKE_FAIL missing={missing} rows={last_rows} url={url}",
    file=sys.stderr,
)
sys.exit(2)
PY
