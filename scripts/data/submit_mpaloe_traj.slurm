#!/bin/bash
#SBATCH --job-name=mpaloe_traj
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

ROOT="${PRETRAIN_ROOT:-/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis}"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python

export PYTHONUNBUFFERED=1

cd "$ROOT"
# Streams 909,792 pymatgen Structure records into one ASE .traj (~20 min).
"$ENV_PY" -u scripts/data/build_mpaloe_traj.py --overwrite
