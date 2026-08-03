#!/bin/bash
#SBATCH --job-name=mpaloe_uma_latents
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%A_%a.err
#SBATCH --array=0-23%8
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail

# Partition note: della rejects an explicit "--partition=gpu" and routes GPU
# jobs by shape. Walltime is the lever -- a <=1 h single-GPU job lands on
# gputest (gpu-test QOS: only 3 concurrent), while >1 h lands on the gpu
# partition (gpu-short QOS: 44 concurrent). The 02:00:00 request below both
# covers the ~50 min of work per part and keeps this array off gputest.
# ailab is never used here.

ROOT="${PRETRAIN_ROOT:-/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis}"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
CONDA_SH=/scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
OUT_ROOT="$ROOT/data/processed/mpaloe/uma_latents"
PARTS_PER_DATASET=24

# MP-ALOE has no native splits: one dataset of 909,792 structures, which divides
# exactly into 24 parts of 37,908 frames each. At the ~13.9 structures/s measured
# on MAD (whose structures average 27 atoms vs MP-ALOE's 6.5) a part needs
# ~46 min, so 02:00:00 leaves comfortable headroom.
DATASETS=(all)
TRAJECTORIES=(
  "$ROOT/data/processed/mpaloe/mpaloe_all.traj"
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
