#!/bin/bash
#SBATCH --job-name=mad_uma_latents
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.err
#SBATCH --array=0-23%4
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT="${PRETRAIN_ROOT:-/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis}"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
CONDA_SH=/scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
OUT_ROOT="$ROOT/data/processed/mad/uma_latents"
PARTS_PER_DATASET=8

# MAD ships three native splits (95,595 structures: 76,483 train / 9,566 val / 9,546 test).
# Each split is one "subdataset" for the extractor, mirroring MOF_OFF R2SCAN/PBE.
# Measured throughput on the gpu partition is ~14 structures/s, so the largest
# part (76,483 / 8 = 9,561 frames) takes ~12 min plus model load.
DATASETS=(train val test)
TRAJECTORIES=(
  "$ROOT/data/processed/mad/train/mad_train.traj"
  "$ROOT/data/processed/mad/val/mad_val.traj"
  "$ROOT/data/processed/mad/test/mad_test.traj"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET_INDEX=$((TASK_ID / PARTS_PER_DATASET))
PART_INDEX=$((TASK_ID % PARTS_PER_DATASET))
DATASET="${DATASETS[$DATASET_INDEX]}"
TRAJECTORY="${TRAJECTORIES[$DATASET_INDEX]}"

module load proxy/default
source "$CONDA_SH"
conda activate pretrain_analysis_env
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

cd "$ROOT"
"$ENV_PY" -u scripts/uma/extract_uma_latent_vectors.py \
  --trajectory "$TRAJECTORY" \
  --subdataset "$DATASET" \
  --out-root "$OUT_ROOT" \
  --job-index "$PART_INDEX" \
  --num-jobs "$PARTS_PER_DATASET"
