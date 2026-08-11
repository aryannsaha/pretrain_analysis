#!/bin/bash
#SBATCH --job-name=uma_parity_hse25_1to3
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/uma_parity/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/uma_parity/logs/%x_%j.err

# UMA parity plot for the hse25 MatPES finetunes.
# Smoke-tests on a few frames first so a broken path fails in seconds, not hours.

set -euo pipefail

REPO=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
PYTHON=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python3.11
SCRIPT="${PARITY_SCRIPT:-$REPO/scripts/uma/evaluate_uma_parity_batched.py}"
SMOKE_DIR="${SMOKE_DIR:-$REPO/runs/uma_parity/smoke}"

mkdir -p "$REPO/runs/uma_parity/logs" "$SMOKE_DIR"
cd "$REPO"

echo "=== smoke: 20 frames ==="
"$PYTHON" -B "$SCRIPT" \
  --max-frames 20 --plot-points 1000 --force \
  --output-root "$SMOKE_DIR" --plot-dir "$SMOKE_DIR/plots"

echo
echo "=== full run ==="
"$PYTHON" -B "$SCRIPT" --force

echo "Done."
