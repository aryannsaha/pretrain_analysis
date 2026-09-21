#!/bin/bash
#SBATCH --job-name=omat24_efs_stats
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/omat24_subdataset_efs_stats/logs/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs/omat24_subdataset_efs_stats/logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=cpu

# Full-population E/F/S statistics for all 11 OMAT24 train subdatasets.
# mkdir -p the logs directory above before submitting. Override SCRIPT to run a
# copy that lives outside the main checkout (e.g. a worktree).

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
PYTHON=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
SCRIPT=${SCRIPT:-$ROOT/scripts/omat/omat24_subdataset_efs_stats.py}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runs/omat24_subdataset_efs_stats}
NUM_TASKS=${NUM_TASKS:-128}

export PYTHON SCRIPT OUTPUT_DIR NUM_TASKS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=$OUTPUT_DIR/.mplconfig
mkdir -p "$MPLCONFIGDIR"

cd "$ROOT"
seq 0 $((NUM_TASKS - 1)) | xargs -P "$SLURM_CPUS_PER_TASK" -n 1 bash -c '
  set -euo pipefail
  "$PYTHON" -u "$SCRIPT" map --task-index "$1" --num-tasks "$NUM_TASKS" --output-dir "$OUTPUT_DIR"
' _

"$PYTHON" -u "$SCRIPT" reduce --num-tasks "$NUM_TASKS" --output-dir "$OUTPUT_DIR"
