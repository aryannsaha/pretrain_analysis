#!/bin/bash
#SBATCH --job-name=upet_moff
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/upet/upet_moff_off/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/upet/upet_moff_off/logs/%x_%j.err
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=intel&gpu80

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
RUN_DIR="$ROOT/runs/upet/upet_moff_off"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python

module load proxy/default

source /scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
conda activate pretrain_analysis_env

mkdir -p "$RUN_DIR/logs"

cd "$ROOT"
"$ENV_PY" scripts/1_UPET_prepare_upet_finetune_from_traj.py

cd "$RUN_DIR"
CONFIG=pet-omat-s-v1.0.0_mof_off_r2scan_d4_20260610.yaml
if [[ ! -f "$CONFIG" ]]; then
  echo "UPET config not found: $RUN_DIR/$CONFIG" >&2
  exit 1
fi
mtt train "$CONFIG" --restart auto -o upet_moff_off.pt
