#!/bin/bash
#SBATCH --job-name=matpes_all_uma_latents
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.err
#SBATCH --array=0-3
# <=1 h routes to gputest (3-concurrent QOS), which schedules fast; the four
# array tasks therefore run 3-then-1. The previous MatPES extraction did
# 278,632 structures / 992,746 atoms in 157 s per shard; this set is 387,897
# structures / 3,059,679 atoms (3.1x the atom work, mean 7.9 vs 3.6 atoms,
# max 240 vs 4), so ~10-15 min per shard is expected -- well inside the window.
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
CONDA_ENV="${CONDA_ENV:-pretrain_analysis_env}"
CONDA_SH=/scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
ENV_PY="${ENV_PY:-/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/$CONDA_ENV/bin/python}"

# The full MatPES-R2SCAN-2025.1 set: 387,897 frames, 100% r2SCAN, atoms 1-240,
# stresses corrected to ASE tension-positive by build_matpes_all_lmdb.py.
# NOT mace_matpes_test_full, which is mixed-functional and capped at 4 atoms.
INPUT="${INPUT:-$ROOT/data/processed/matpes/all.lmdb}"
OUT_DIR="${OUT_DIR:-$ROOT/data/processed/matpes/uma_latents_all/parts}"
OUTPUT_STEM="${OUTPUT_STEM:-matpes_all_r2scan}"
NUM_JOBS="${NUM_JOBS:-4}"
# The OMAT24 reference matrix was built with these; they must not drift or the
# resulting latents are not comparable and every kNN against OMAT24 is invalid.
MODEL="${MODEL:-uma-s-1p2}"
TASK_NAME="${TASK_NAME:-omat}"
# 240-atom cells are ~60x the previous maximum; the extractor splits a batch on
# OOM and records oom_batch_splits, so this stays at the proven value.
BATCH_SIZE="${BATCH_SIZE:-128}"

ARGS=(
  --lmdb "$INPUT"
  --out-dir "$OUT_DIR"
  --output-stem "$OUTPUT_STEM"
  --model "$MODEL"
  --task-name "$TASK_NAME"
  --job-index "${SLURM_ARRAY_TASK_ID:-0}"
  --num-jobs "$NUM_JOBS"
  --batch-size "$BATCH_SIZE"
)

echo "MatPES (full r2SCAN) batched UMA latent extraction"
echo "  task: ${SLURM_ARRAY_TASK_ID:-0}/$NUM_JOBS"
echo "  input: $INPUT"
echo "  output: $OUT_DIR ($OUTPUT_STEM)"
echo "  model/task: $MODEL / $TASK_NAME"
echo "  batch size: $BATCH_SIZE"

module load proxy/default
source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$OUT_DIR" "$ROOT/logs/slurm"
cd "$ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$ENV_PY" -u scripts/uma/extract_uma_lmdb_latents_batched.py "${ARGS[@]}"
