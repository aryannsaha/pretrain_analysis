#!/bin/bash
#SBATCH --job-name=logme_transfer
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/outputs/logme/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/outputs/logme/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=ailab
#SBATCH --mail-type=fail
#SBATCH --mail-user=as7959@princeton.edu

# LogME transferability analysis over the six frozen OMat24 MACE checkpoints.
#
# Runs both downstream targets in one job: MOF-OFF (out of distribution) and
# AM/MPtrj (closer to in distribution). The OMat24 source sample for the
# distributional-distance baseline is fixed across checkpoints so that
# differences are differences between models, not between samples.

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
N_SOURCE="${N_SOURCE:-1500}"
MAX_ATOMS="${MAX_ATOMS:-50000}"
BATCH="${BATCH:-16}"

SOURCE_LMDB=$ROOT/data/processed/omat24/stratified_random_nested/sr_100k/train.lmdb

echo "=== MOF-OFF (out of distribution) ==="
python scripts/transfer/run_logme_analysis.py \
    --target-lmdb "$ROOT/data/processed/mof_off/R2SCAN/R2SCAN_val.lmdb" \
    --target-name mof_off \
    --source-lmdb "$SOURCE_LMDB" \
    --n-structures "$N_STRUCT" \
    --n-source-structures "$N_SOURCE" \
    --max-atoms "$MAX_ATOMS" \
    --batch-size "$BATCH" \
    --out "$OUT"

echo "=== AM / MPtrj (nearer in distribution) ==="
python scripts/transfer/run_logme_analysis.py \
    --target-lmdb "$ROOT/data/processed/AM/val.lmdb" \
    --target-name am \
    --source-lmdb "$SOURCE_LMDB" \
    --n-structures "$N_STRUCT" \
    --n-source-structures "$N_SOURCE" \
    --max-atoms "$MAX_ATOMS" \
    --batch-size "$BATCH" \
    --out "$OUT"

echo "=== done ==="
