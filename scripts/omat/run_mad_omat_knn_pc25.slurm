#!/bin/bash
#SBATCH --job-name=mad_knn_pc25
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
# No --partition directive: della rejects an explicit "gpu" partition and routes
# GPU jobs itself; a <=1 h walltime routes to gputest, whose queue is short and
# whose 3-concurrent QOS is fine for a single job. Expected runtime ~5-15 min.
# ailab is never used here.
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

ROOT="${PRETRAIN_ROOT:-/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis}"
PYTHON=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python

PC25_ROOT="$ROOT/runs/omat_knn_probe/omat_knn_pc25"
PROJECTED_REFERENCE="$PC25_ROOT/omat_pc25_unweighted.npy"
PCA_STATE="$ROOT/runs/uma_latents_copy_full_pca/uma_100824585_rapids_pca_state_11212254.npz"
CALIBRATION="$ROOT/runs/omat_knn_probe/omat_knn_pc25_final/OMAT_calibration"
MANIFEST="$ROOT/data/processed/omat24/uma_latents_copy/all_uma_embeddings_manifest.csv"
RUN_ROOT="$ROOT/runs/omat_knn_probe/omat_knn_pc25_mad"
QUERY="$ROOT/data/processed/mad/uma_latents/mad_all.npy"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$RUN_ROOT/mplconfig"

mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR"
cd "$ROOT"

# Same geometry and index parameters as the MOF-off pc25 run
# (runs/omat_knn_probe/omat_knn_pc25_mof_off): unweighted standardized 25-PC
# scores, IVF-Flat nlist=32768 nprobe=1024, k=10, against the same projected
# 100,824,585 x 25 reference -- so distances are directly comparable.
# The 128-d query matrix is projected with the same PCA state inside the script.
"$PYTHON" -u scripts/omat/compute_omat_knn_novelty.py query \
  --projected-reference "$PROJECTED_REFERENCE" \
  --pca-state "$PCA_STATE" \
  --output-root "$RUN_ROOT" \
  --query "MAD_all=$QUERY" \
  --dimensions 25 \
  --neighbors 10 \
  --nlist 32768 \
  --nprobe 1024 \
  --batch-rows 50000 \
  --shard-index 0 \
  --num-shards 1 \
  --validation-query "$QUERY" \
  --validation-rows 128

"$PYTHON" -u scripts/omat/compute_omat_knn_novelty.py merge \
  --output-root "$RUN_ROOT" \
  --dataset MAD_all \
  --calibration-dir "$CALIBRATION" \
  --manifest "$MANIFEST"
