#!/bin/bash
# pc25 top-5 attribution over the union pool (OMAT24 train + every external corpus).
#
#   STAGE=pilot  sbatch scripts/omat/run_union_pool_pc25_knn.slurm   # project + 1 set + validate
#   STAGE=full   sbatch scripts/omat/run_union_pool_pc25_knn.slurm   # every query set + table
#
#SBATCH --job-name=union_pool_knn
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
WORKTREE="${WORKTREE:-$ROOT/.claude/worktrees/union-pool-pc25-knn}"
PYTHON=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
SCRIPT="$WORKTREE/scripts/omat/union_pool_pc25_knn.py"
STAGE="${STAGE:-full}"

export PRETRAIN_ANALYSIS_ROOT="$ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "stage=$STAGE  script=$SCRIPT"

"$PYTHON" "$SCRIPT" project

if [[ "$STAGE" == "pilot" ]]; then
  "$PYTHON" "$SCRIPT" search --dataset "MOF-off R2SCAN-D4"
  "$PYTHON" "$SCRIPT" search --dataset "MAD"
  "$PYTHON" "$SCRIPT" validate --rows 2000
else
  "$PYTHON" "$SCRIPT" search --dataset all
  "$PYTHON" "$SCRIPT" validate --rows 2000
  "$PYTHON" "$SCRIPT" table
fi

echo "done"
