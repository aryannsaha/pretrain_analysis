#!/bin/bash
#SBATCH --job-name=pc25_top5_report
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

# Everything downstream of the descriptor extraction: the comparison tables and
# figures, the composition/calibration/PC controls, and the exported CIF examples.
# Memory is sized for the composition baseline, which holds a ~118k x 119 pool
# matrix and blocks an 80,643-row query against it.

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
SCRIPTS="$ROOT/.claude/worktrees/mof-omat-neighbor-chem/scripts/omat"
ENV_PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
OUT="$ROOT/runs/omat_knn_probe/pc25_top5_chemistry"

export MPLCONFIGDIR="$OUT/mplconfig"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"

"$ENV_PY" "$SCRIPTS/pc25_top5_chemistry_report.py" --root "$ROOT"

"$ENV_PY" "$SCRIPTS/mof_omat_composition_baseline.py" --root "$ROOT" --report

"$ENV_PY" "$SCRIPTS/plot_pc25_top5_chemistry.py" --analysis "$OUT/analysis"

# Coordination-dependent panels are skipped cleanly if the coordination job has
# not landed yet; re-run this script once it has.
"$ENV_PY" "$SCRIPTS/plot_mof_omat_chemical_relationship.py" \
    --analysis "$OUT/analysis" --root "$ROOT"

for QUERY in mof_off mad; do
    "$ENV_PY" "$SCRIPTS/export_pc25_top5_examples.py" --root "$ROOT" --query "$QUERY"
done

echo "pc25 top-5 report stage complete"
