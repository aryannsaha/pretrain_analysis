#!/bin/bash
#SBATCH --job-name=mof_omat_coord
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# CrystalNN + ChemEnv + bond-valence analysis of one frame per distinct MOF and
# every pc25 top-5 OMAT24 neighbour of those frames, at ~1.2 s per structure.
# MOF_LIMIT caps the sample: 1,500 of the 3,269 distinct MOFs is +-1.3% on a 50%
# proportion, which is far finer than any effect here, and keeps the job inside a
# one-hour backfill window on the saturated shared cpu partition.

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
SCRIPTS="$ROOT/.claude/worktrees/mof-omat-neighbor-chem/scripts/omat"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
WORKERS=${SLURM_CPUS_PER_TASK:-8}

cd "$ROOT"

"$ENV_PY" "$SCRIPTS/mof_omat_coordination_chemistry.py" --root "$ROOT" analyse \
    --workers "$WORKERS" --chunk 40 --limit-mofs "${MOF_LIMIT:-1500}"

echo "coordination analysis complete"
