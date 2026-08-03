#!/bin/bash
#SBATCH --job-name=mpaloe_uma_pca
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

ROOT="${PRETRAIN_ROOT:-/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis}"
PYTHON=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
LATENT_ROOT="$ROOT/data/processed/mpaloe/uma_latents"
BACKGROUND="$ROOT/data/processed/omat24/uma_latents_copy/all_uma_embeddings.npy"
PCA_STATE="$ROOT/runs/uma_latents_copy_full_pca/uma_100824585_rapids_pca_state_11212254.npz"
DENSITY="$ROOT/runs/direct_sampling_rapids_pass_pca/full_100m_pca_density.npz"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# MP-ALOE has no native splits: a single 909,792 x 128 matrix built up front by
# scripts/uma/combine_uma_latent_parts.py, so this job only projects it onto the
# OMAT24 100M PCA background.
NAME=mpaloe_all
LATENTS="$LATENT_ROOT/all/mpaloe_all.npy"
LABEL="MP-ALOE (r2SCAN)"

if [[ ! -f "$LATENTS" ]]; then
  echo "Missing combined latent matrix: $LATENTS" >&2
  exit 1
fi

RUN_DIR="$ROOT/runs/direct_sampling_rapids_pass_pca/$NAME"
mkdir -p "$RUN_DIR/mplconfig"
export MPLCONFIGDIR="$RUN_DIR/mplconfig"

cd "$ROOT"
"$PYTHON" scripts/plot_uma_direct_pca.py \
  --matrix "$BACKGROUND" \
  --pca-state "$PCA_STATE" \
  --density-cache "$DENSITY" \
  --overlay-matrix "$LATENTS" \
  --overlay-label "$LABEL" \
  --output "$RUN_DIR/${NAME}_on_OMAT_100m_PCA.png" \
  --max-selected-points 0 \
  --selected-marker-size 0.7 \
  --selected-alpha 0.5 \
  --plot-xlim -3.034101891597502 14.780749281886083 \
  --plot-ylim -4.606675380009836 5.826277607171946
