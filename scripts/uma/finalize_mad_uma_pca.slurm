#!/bin/bash
#SBATCH --job-name=mad_uma_pca
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
LATENT_ROOT="$ROOT/data/processed/mad/uma_latents"
BACKGROUND="$ROOT/data/processed/omat24/uma_latents_copy/all_uma_embeddings.npy"
PCA_STATE="$ROOT/runs/uma_latents_copy_full_pca/uma_100824585_rapids_pca_state_11212254.npz"
DENSITY="$ROOT/runs/direct_sampling_rapids_pass_pca/full_100m_pca_density.npz"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# The per-split matrices and the combined mad_all matrix are built up front by
# scripts/uma/combine_uma_latent_parts.py (95,595 = 76,483 + 9,566 + 9,546), so
# this job only projects them onto the OMAT24 100M PCA background.
NAMES=(mad_all mad_train mad_val mad_test)
MATRICES=(
  "$LATENT_ROOT/mad_all.npy"
  "$LATENT_ROOT/train/mad_train.npy"
  "$LATENT_ROOT/val/mad_val.npy"
  "$LATENT_ROOT/test/mad_test.npy"
)
LABELS=(
  "MAD (all splits)"
  "MAD train"
  "MAD val"
  "MAD test"
)

cd "$ROOT"
for INDEX in "${!NAMES[@]}"; do
  NAME="${NAMES[$INDEX]}"
  LATENTS="${MATRICES[$INDEX]}"
  LABEL="${LABELS[$INDEX]}"

  if [[ ! -f "$LATENTS" ]]; then
    echo "Missing combined latent matrix: $LATENTS" >&2
    exit 1
  fi

  RUN_DIR="$ROOT/runs/direct_sampling_rapids_pass_pca/$NAME"
  mkdir -p "$RUN_DIR/mplconfig"
  export MPLCONFIGDIR="$RUN_DIR/mplconfig"

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
done
