#!/bin/bash
#SBATCH --job-name=matpes_uma_latents
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=01:55:00
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

INPUT="${INPUT:-$ROOT/data/processed/matpes/r2scan/mace_matpes_test_full/matpes_r2scan.lmdb}"
OUT_DIR="${OUT_DIR:-$ROOT/data/processed/matpes/uma_latents/parts}"
OUTPUT_STEM="${OUTPUT_STEM:-matpes_r2scan}"
NUM_JOBS="${NUM_JOBS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_STRUCTURES="${MAX_STRUCTURES:-}"

ARGS=(
  --lmdb "$INPUT"
  --out-dir "$OUT_DIR"
  --output-stem "$OUTPUT_STEM"
  --job-index "${SLURM_ARRAY_TASK_ID:-0}"
  --num-jobs "$NUM_JOBS"
  --batch-size "$BATCH_SIZE"
)
if [[ -n "$MAX_STRUCTURES" ]]; then
  ARGS+=(--max-structures "$MAX_STRUCTURES")
fi

echo "MatPES batched UMA latent extraction"
echo "  task: ${SLURM_ARRAY_TASK_ID:-0}/$NUM_JOBS"
echo "  input: $INPUT"
echo "  output directory: $OUT_DIR"
echo "  output stem: $OUTPUT_STEM"
echo "  batch size: $BATCH_SIZE"
echo "  max structures: ${MAX_STRUCTURES:-all assigned}"
echo "  python: $ENV_PY"

module load proxy/default
source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$OUT_DIR" "$ROOT/logs/slurm"
cd "$ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$ENV_PY" -u scripts/uma/extract_uma_lmdb_latents_batched.py "${ARGS[@]}"
