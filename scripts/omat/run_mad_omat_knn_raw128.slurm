#!/bin/bash
#SBATCH --job-name=mad_knn_raw128
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
# No --partition directive: della rejects an explicit "gpu" partition
# ("You specified a partition of gpu. This is not allowed") and routes GPU jobs
# itself. Asking for a full 80 GB A100 for 2 h keeps this off ailab and off the
# MIG/gputest-sized queues.
#SBATCH --constraint=gpu80
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

ROOT="${PRETRAIN_ROOT:-/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis}"
PYTHON=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

cd "$ROOT"

# Exact raw-128-d Euclidean top-10: GPU scan (+ fp64 audit), merge checks, CSV join.
# Query sets are the three native MAD splits (95,595 rows total) against the full
# 100,824,585 x 128 OMAT24 reference, which stays resident on the GPU (51.6 GB).
#
# NOTE ON SHARDING: --num-shards/--shard-index split the QUERY rows, not the
# reference (shard_interval is called on query.shape[0], and load_reference_gpu
# always allocates the full matrix). Sharding therefore cannot lower the resident
# reference footprint, so the reference must fit on one card: 51.62 GB rules out
# the 40 GB A100s and requires --constraint=gpu80 (80 GB, 59 nodes in this
# partition). MIG slices are excluded by asking for gpu:a100:1.
#
# Peak GPU memory on an 80 GB A100 with --block-rows 32768:
#   reference 51.62 + reference_sq 0.40 + scan buffers ~4.3 (gram/squared
#   temporaries at 8192 x 32768 fp32) + re-rank ~1.6 + context ~1.0 = ~59 GB,
#   leaving ~20 GB headroom. The default 262144 block would need ~26 GB of scan
#   buffers alone (~79 GB peak) and is not safe here.
"$PYTHON" -u scripts/omat/mad_omat_knn_raw128.py query --num-shards 1 --shard-index 0 --block-rows 32768
"$PYTHON" -u scripts/omat/mad_omat_knn_raw128.py merge --num-shards 1
"$PYTHON" -u scripts/omat/mad_omat_knn_raw128.py join
