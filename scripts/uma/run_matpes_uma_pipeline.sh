#!/bin/bash
# Chain the MatPES UMA embedding pipeline: build the LMDB, extract latents, combine parts.
#
# The stress-sign correction (scripts/matpes/fix_extxyz_stress_sign.py) must already have
# been applied to the extxyz splits; the LMDB builder refuses to run otherwise.

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
SCRIPT_ROOT="${SCRIPT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
NUM_JOBS="${NUM_JOBS:-4}"

export SCRIPT_ROOT NUM_JOBS

BUILD_ID=$(sbatch --parsable "$SCRIPT_ROOT/scripts/matpes/build_matpes_test_full_lmdb.slurm")
echo "build LMDB:      $BUILD_ID"

EXTRACT_ID=$(sbatch --parsable \
  --dependency="afterok:$BUILD_ID" \
  --array="0-$((NUM_JOBS - 1))" \
  "$SCRIPT_ROOT/scripts/uma/submit_matpes_uma_latents.slurm")
echo "extract latents: $EXTRACT_ID (array 0-$((NUM_JOBS - 1)))"

COMBINE_ID=$(sbatch --parsable \
  --dependency="afterok:$EXTRACT_ID" \
  "$SCRIPT_ROOT/scripts/uma/combine_matpes_uma_latents.slurm")
echo "combine parts:   $COMBINE_ID"

echo
echo "watch with: squeue -j $BUILD_ID,$EXTRACT_ID,$COMBINE_ID"
