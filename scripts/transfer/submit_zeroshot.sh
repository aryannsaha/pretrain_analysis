#!/bin/bash
#SBATCH --job-name=logme_zeroshot
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/outputs/logme/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/outputs/logme/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=fail
#SBATCH --mail-user=as7959@princeton.edu

# Zero-shot error of the six frozen checkpoints on both downstream targets.
# Gives the rank-correlation machinery a real ground-truth column while the
# fine-tunes are still running. Not a substitute for the fine-tune result.

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
WORKTREE="${WORKTREE:-$ROOT/.claude/worktrees/logme-transferability}"
OUT="${OUT:-$ROOT/outputs/logme}"

module load proxy/default 2>/dev/null || true
source /scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
conda activate pretrain_analysis_env_mace

mkdir -p "$OUT/logs"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUT/mplconfig}"
mkdir -p "$MPLCONFIGDIR"
cd "$WORKTREE"

N_STRUCT="${N_STRUCT:-3000}"

echo "=== MOF-OFF ==="
python scripts/transfer/zeroshot_eval.py \
    --target-lmdb "$ROOT/data/processed/mof_off/R2SCAN/R2SCAN_val.lmdb" \
    --target-name mof_off \
    --n-structures "$N_STRUCT" \
    --out "$OUT"

echo "=== AM ==="
python scripts/transfer/zeroshot_eval.py \
    --target-lmdb "$ROOT/data/processed/AM/val.lmdb" \
    --target-name am \
    --n-structures "$N_STRUCT" \
    --out "$OUT"

echo "=== done ==="
