#!/bin/bash
#SBATCH --job-name=matpes_lmdb
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
CONDA_ENV="${CONDA_ENV:-pretrain_analysis_env}"
CONDA_SH=/scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
ENV_PY="${ENV_PY:-/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/$CONDA_ENV/bin/python}"

# Checkout that supplies the builder; override when running from a worktree.
SCRIPT_ROOT="${SCRIPT_ROOT:-$ROOT}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT/data/processed/matpes/r2scan/mace_matpes_test_full}"
OUTPUT="${OUTPUT:-$SOURCE_DIR/matpes_r2scan.lmdb}"

echo "MatPES mace_matpes_test_full -> single LMDB"
echo "  source directory: $SOURCE_DIR"
echo "  output: $OUTPUT"
echo "  script root: $SCRIPT_ROOT"
echo "  python: $ENV_PY"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$ROOT/logs/slurm"
cd "$ROOT"

export PYTHONUNBUFFERED=1

"$ENV_PY" -u "$SCRIPT_ROOT/scripts/matpes/build_matpes_test_full_lmdb.py" \
  --source-dir "$SOURCE_DIR" \
  --output "$OUTPUT"
