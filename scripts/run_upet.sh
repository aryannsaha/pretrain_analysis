#!/bin/bash
#SBATCH --job-name=upet_moff
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/upet/upet_moff_off_test/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/upet/upet_moff_off_test/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=intel&gpu80

set -euo pipefail

CONFIG_FILE=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/configs/upet/pet-omat-s-v1.0.0_mof_off_r2scan_d4_20260610.yaml

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
RUN_DIR="$ROOT/runs/upet/upet_moff_off_test"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python

module load proxy/default

source /scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
conda activate pretrain_analysis_env

mkdir -p "$RUN_DIR/logs"
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
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

cd "$ROOT"
"$ENV_PY" scripts/1_UPET_prepare_upet_finetune_from_traj.py

cd "$RUN_DIR"
mtt train "$CONFIG_FILE" --restart auto -o upet_moff_off_test.pt
