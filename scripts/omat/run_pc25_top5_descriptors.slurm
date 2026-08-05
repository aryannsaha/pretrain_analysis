#!/bin/bash
#SBATCH --job-name=pc25_top5_desc
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Coordination / bonding / connectivity descriptors for both query sets and
# their pc25 top-5 OMAT24 neighbours, plus the two OMAT24 baselines.  Extraction
# is CPU-only and I/O bound on the .aselmdb shards.  8 workers is deliberate: the
# whole sweep is ~95 core-minutes, so a small short-walltime ask backfills on the
# busy shared cpu partition within minutes instead of queueing for hours.

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
SCRIPTS="$ROOT/.claude/worktrees/mof-omat-neighbor-chem/scripts/omat"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
WORKERS=${SLURM_CPUS_PER_TASK:-8}

cd "$ROOT"

# MOF-off carries the two OMAT24 baselines; MAD reuses them, so it only needs
# its own queries and neighbours.
"$ENV_PY" "$SCRIPTS/pc25_top5_structure_descriptors.py" --root "$ROOT" extract \
    --query mof_off --workers "$WORKERS" --chunk 1000 --background 50000

"$ENV_PY" "$SCRIPTS/pc25_top5_structure_descriptors.py" --root "$ROOT" extract \
    --query mad --workers "$WORKERS" --chunk 1000

echo "descriptor extraction complete"
