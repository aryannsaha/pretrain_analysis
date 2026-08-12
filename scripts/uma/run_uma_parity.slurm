#!/bin/bash
#SBATCH --job-name=uma_parity_hse25
# No --partition: della rejects an explicit "gpu" partition and routes any
# --gres=gpu job to the gpu partition on its own. Naming ailab would override
# that, so leave it unset.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/uma_parity/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/uma_parity/logs/%x_%j.err

# UMA parity plots for the hse25 MatPES finetunes.
# Smoke-tests on a few frames first so a broken path fails in seconds, not hours.
#
# ONLY=<substring>  restrict to matching evaluations (see --only)
# FORCE=1           recompute even when a prediction CSV is already cached

set -euo pipefail

REPO=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
PYTHON=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python3.11
SCRIPT="${PARITY_SCRIPT:-$REPO/scripts/uma/evaluate_uma_parity_batched.py}"
SMOKE_DIR="${SMOKE_DIR:-$REPO/runs/uma_parity/smoke}"

only_args=()
if [[ -n "${ONLY:-}" ]]; then
  only_args=(--only "$ONLY")
fi

# Cached CSVs are reused by default so re-plotting one model does not
# recompute the others.
force_args=()
if [[ -n "${FORCE:-}" ]]; then
  force_args=(--force)
fi

mkdir -p "$REPO/runs/uma_parity/logs" "$SMOKE_DIR"
cd "$REPO"

echo "=== smoke: 20 frames ==="
"$PYTHON" -B "$SCRIPT" \
  --max-frames 20 --plot-points 1000 --force \
  --output-root "$SMOKE_DIR" --plot-dir "$SMOKE_DIR/plots" "${only_args[@]}"

echo
echo "=== full run ==="
"$PYTHON" -B "$SCRIPT" "${only_args[@]}" "${force_args[@]}"

echo "Done."
