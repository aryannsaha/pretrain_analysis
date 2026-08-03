#!/bin/bash
#SBATCH --job-name=mpaloe_knn_raw128
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
# No --partition directive: della rejects an explicit "gpu" partition and routes
# GPU jobs itself; a >1 h walltime keeps this on the gpu partition rather than
# gputest. ailab is never used here.
#SBATCH --constraint=gpu80
#SBATCH --time=04:00:00
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
# Query set is the single MP-ALOE matrix (909,792 rows) against the full
# 100,824,585 x 128 OMAT24 reference, which stays resident on the GPU (48.08 GiB).
#
# --num-shards/--shard-index split the QUERY rows, not the reference, so the
# reference must fit on one card: --constraint=gpu80 (80 GB A100) is required and
# the 40 GB A100s cannot run this. Peak with --block-rows 32768 is ~55 GiB
# (48.08 reference + 0.38 norms + ~5.5 scan buffers + ~1 context).
#
# Runtime: MAD's 95,595 queries scanned in ~9 min wall; at ~9.5x the query count
# expect ~1.5-2 h here, so 04:00:00 carries >2x headroom (and keeps it off gputest).
"$PYTHON" -u scripts/omat/mpaloe_omat_knn_raw128.py query --num-shards 1 --shard-index 0 --block-rows 32768
"$PYTHON" -u scripts/omat/mpaloe_omat_knn_raw128.py merge --num-shards 1
"$PYTHON" -u scripts/omat/mpaloe_omat_knn_raw128.py join
