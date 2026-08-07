#!/bin/bash
#SBATCH --job-name=matpes_uma_combine
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
CONDA_ENV="${CONDA_ENV:-pretrain_analysis_env}"
CONDA_SH=/scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
ENV_PY="${ENV_PY:-/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/$CONDA_ENV/bin/python}"

LATENT_ROOT="${LATENT_ROOT:-$ROOT/data/processed/matpes/uma_latents}"
OUTPUT_STEM="${OUTPUT_STEM:-matpes_r2scan}"
NUM_JOBS="${NUM_JOBS:-4}"
LMDB="${LMDB:-$ROOT/data/processed/matpes/r2scan/mace_matpes_test_full/matpes_r2scan.lmdb}"
MANIFEST="${MANIFEST:-$ROOT/data/processed/matpes/r2scan/mace_matpes_test_full/matpes_r2scan_manifest.csv}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$ROOT/logs/slurm"
cd "$ROOT"

export PYTHONUNBUFFERED=1

# The latent matrix must have exactly one row per LMDB structure, in ID order.
EXPECTED_ROWS="$("$ENV_PY" - "$LMDB" <<'PY'
import sys, zlib, orjson, lmdb
env = lmdb.open(sys.argv[1], subdir=False, readonly=True, lock=False)
with env.begin() as txn:
    print(int(orjson.loads(zlib.decompress(txn.get(b"nextid")))) - 1)
PY
)"

echo "Combining MatPES UMA latent parts"
echo "  parts directory: $LATENT_ROOT/parts"
echo "  stem: $OUTPUT_STEM"
echo "  parts: $NUM_JOBS"
echo "  expected rows: $EXPECTED_ROWS"

"$ENV_PY" -u scripts/uma/combine_uma_latent_parts.py \
  --parts-dir "$LATENT_ROOT/parts" \
  --stem "$OUTPUT_STEM" \
  --num-parts "$NUM_JOBS" \
  --expected-rows "$EXPECTED_ROWS" \
  --output "$LATENT_ROOT/$OUTPUT_STEM.npy"

cp "$MANIFEST" "$LATENT_ROOT/$(basename "$MANIFEST")"
echo "Row i of $LATENT_ROOT/$OUTPUT_STEM.npy corresponds to lmdb_id i+1 in $(basename "$MANIFEST")"
