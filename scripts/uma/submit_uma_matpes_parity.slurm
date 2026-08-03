#!/bin/bash
#SBATCH --job-name=uma_matpes_parity
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# UMA parity evaluation for the MatPES r2SCAN 75k fine-tunes.
# Array index maps onto EVALUATIONS in scripts/evaluate_uma_parity_batched.py:
#   uma-s-1p1   base: 0 = lte3 val  1 = lte3 gt  2 = lte4 val  3 = lte4 gt
#   uma-s-1p2p1 base: 4 = lte3 val  5 = lte3 gt  6 = lte4 val  7 = lte4 gt
# Completed evaluations reuse their cached predictions.csv, so re-running an
# already-finished index only redraws its figure.

set -euo pipefail

ROOT="${ROOT:-/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis}"
SCRIPT_ROOT="${SCRIPT_ROOT:-$ROOT}"
CONDA_ENV="${CONDA_ENV:-pretrain_analysis_env}"
CONDA_SH=/scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
ENV_PY="${ENV_PY:-/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/$CONDA_ENV/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-32}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "UMA MatPES parity evaluation"
echo "  task_id: $TASK_ID"
echo "  root: $ROOT"
echo "  script_root: $SCRIPT_ROOT"
echo "  python: $ENV_PY"
echo "  batch_size: $BATCH_SIZE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$ROOT/logs/slurm"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/outputs/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"
# Data and run trees live in the main checkout even when the code runs from a worktree.
export PRETRAIN_ANALYSIS_ROOT="$ROOT"

cd "$SCRIPT_ROOT"
"$ENV_PY" scripts/evaluate_uma_parity_batched.py \
  --task-index "$TASK_ID" \
  --device cuda \
  --batch-size "$BATCH_SIZE"
