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

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
RUN_DIR="$ROOT/runs/mace/mace_moff_off"
CONDA_ENV="${CONDA_ENV:-pretrain_analysis_env}"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/$CONDA_ENV/bin/python

module load proxy/default
source /scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

mkdir -p "$RUN_DIR/logs"
export MPLCONFIGDIR="$RUN_DIR/mplconfig"
export WANDB_RUN_ID=mace_moff_off
export WANDB_RESUME=allow
mkdir -p "$MPLCONFIGDIR"

cd "$ROOT"
"$ENV_PY" scripts/1_MACE_prepare_mace_finetune_from_traj.py

cd "$RUN_DIR"
CONFIG=$(ls -t mace-omat-0-medium_mof_off_r2scan_d4_*.yaml | head -n 1)
command -v mace_run_train >/dev/null 2>&1 || { echo "mace_run_train is not available in $CONDA_ENV" >&2; exit 127; }
mace_run_train --config="$CONFIG"
