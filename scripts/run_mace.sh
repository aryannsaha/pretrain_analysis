#!/bin/bash
#SBATCH --job-name=mace_moff
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/mace/mace_moff_off/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/mace/mace_moff_off/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=intel&gpu80

set -euo pipefail


CONFIG_FILE=runs/mace/mace_moff_off_TEST/mace-omat-0-medium_mof_off_r2scan_d4_20260610_test10.yaml
CONFIG=mace-omat-0-medium_mof_off_r2scan_d4_20260610_test10.yaml
ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
RUN_DIR="$ROOT/runs/mace/mace_moff_off_TEST"
CONDA_ENV="${CONDA_ENV:-pretrain_analysis_env_mace}"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/$CONDA_ENV/bin/python

module load proxy/default
source /scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$RUN_DIR/logs"
export MPLCONFIGDIR="$RUN_DIR/mplconfig"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RESUME="${WANDB_RESUME:-never}"
export WANDB_DIR="${WANDB_DIR:-$RUN_DIR/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$RUN_DIR/.wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$RUN_DIR/.wandb_config}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export WANDB_DISABLE_CODE="${WANDB_DISABLE_CODE:-true}"
export WANDB_DISABLE_GIT="${WANDB_DISABLE_GIT:-true}"
export WANDB_SAVE_CODE="${WANDB_SAVE_CODE:-false}"
export WANDB_X_SAVE_REQUIREMENTS="${WANDB_X_SAVE_REQUIREMENTS:-false}"
export WANDB_IGNORE_GLOBS="${WANDB_IGNORE_GLOBS:-config.yaml,requirements.txt,wandb-metadata.json,wandb-summary.json,output.log,*.ckpt,*.pt,*.model,*.extxyz}"
mkdir -p "$MPLCONFIGDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

cd "$ROOT"
"$ENV_PY" scripts/1_MACE_prepare_mace_finetune_from_traj.py

cd "$RUN_DIR"
mace_run_train --config="$CONFIG"