#!/bin/bash
#SBATCH --job-name=matpes_all_knn_pc25
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
# <=1 h routes to gputest. The 161,262-row subset run (job 12114977) took 2m53s
# end to end, most of it the 76 s index build; 387,897 rows adds ~1 min of query.
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
RUN_ROOT="$ROOT/runs/omat_knn_probe/omat_knn_pc25_matpes_all"

# The FULL MatPES-R2SCAN-2025.1 set: 387,897 frames, 100% r2SCAN, atoms 1-240,
# built from data/raw/matpes/all.traj via data/processed/matpes/all.lmdb.
# Supersedes the omat_knn_pc25_matpes run, which used the mixed-functional
# mace_matpes_test_full r2SCAN subset (161,262 frames, all <=4 atoms).
QUERY="$ROOT/data/processed/matpes/uma_latents_all/matpes_all_r2scan.npy"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$RUN_ROOT/mplconfig"

mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$ROOT/logs/slurm"
cd "$ROOT"

test -f "$QUERY" || { echo "missing query matrix: $QUERY" >&2; exit 1; }

# Same geometry and index parameters as every other pc25 run so the result is
# directly comparable and drops into scripts/omat/pc25_subdataset_table.py.
"$PYTHON" -u scripts/omat/compute_omat_knn_novelty.py query \
  --projected-reference "$PROJECTED_REFERENCE" \
  --pca-state "$PCA_STATE" \
  --output-root "$RUN_ROOT" \
  --query "MatPES_r2SCAN_all=$QUERY" \
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
  --dataset MatPES_r2SCAN_all \
  --calibration-dir "$CALIBRATION" \
  --manifest "$MANIFEST"
