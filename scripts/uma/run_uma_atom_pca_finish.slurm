#!/bin/bash
# After the per-atom extraction array: wait until all NUM_PARTS parts report
# complete, assemble the atom bank + per-atom label sets, fit the streaming
# standardized PCA (pca_state.npz, pc12/pc25 scores, per-group densities) and
# draw the figures.  CPU only.
#
#     sbatch --dependency=afterany:<array job id> scripts/uma/run_uma_atom_pca_finish.slurm
#SBATCH --job-name=uma_atom_pca_finish
#SBATCH --output=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/logs/slurm/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

ROOT=/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
BANK_DIR="${BANK_DIR:-$ROOT/data/processed/omat24/uma_atom_latents_sr_1m}"
PARTS_DIR="${PARTS_DIR:-$BANK_DIR/parts}"
NUM_PARTS="${NUM_PARTS:-16}"
EXPECTED_FRAMES="${EXPECTED_FRAMES:-1000000}"
FRAME_LABELS="${FRAME_LABELS:-$ROOT/runs/mace/official_omat_mace_pca/background}"
PCA_DIR="${PCA_DIR:-$ROOT/runs/uma_atom_pca_sr_1m}"
WAIT_MINUTES="${WAIT_MINUTES:-90}"
# When FRAME_LABELS has no row_labels.npy yet, derive it from the selection CSV +
# LMDB with scripts/omat/label_train_lmdb_rows.py (needs the MACE env).
LABEL_CSV="${LABEL_CSV:-}"
LABEL_LMDB="${LABEL_LMDB:-}"
SAMPLE_LABEL="${SAMPLE_LABEL:-sr_1m (1,000,000 stratified random OMAT24 training frames)}"

source /scratch/gpfs/ROSENGROUP/aryan/software/miniconda/etc/profile.d/conda.sh
conda activate pretrain_analysis_env
cd "$ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="$ROOT/.tmp/matplotlib"
mkdir -p "$PCA_DIR" "$MPLCONFIGDIR"

if [[ ! -f "$FRAME_LABELS/row_labels.npy" ]]; then
  if [[ -z "$LABEL_CSV" || -z "$LABEL_LMDB" ]]; then
    echo "no frame labels in $FRAME_LABELS and LABEL_CSV/LABEL_LMDB unset" >&2
    exit 1
  fi
  conda activate pretrain_analysis_env_mace
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -B scripts/omat/label_train_lmdb_rows.py \
    --csv "$LABEL_CSV" --lmdb "$LABEL_LMDB" --out-dir "$FRAME_LABELS" --verify 64 --seed 0
  conda activate pretrain_analysis_env
fi

# Poll for completeness (an extraction task may still be finishing on another lane).
count_complete() {
  python - "$PARTS_DIR" "$NUM_PARTS" <<'PY'
import json, sys
from pathlib import Path
parts, n = Path(sys.argv[1]), int(sys.argv[2])
done = 0
for i in range(n):
    m = parts / f"train.part-{i:05d}-of-{n:05d}.metadata.json"
    if m.is_file() and json.loads(m.read_text()).get("complete"):
        done += 1
print(done)
PY
}
for ((minute = 0; minute <= WAIT_MINUTES; minute++)); do
  done_parts="$(count_complete)"
  echo "$(date '+%F %T') complete parts: $done_parts/$NUM_PARTS"
  if (( done_parts == NUM_PARTS )); then break; fi
  if (( minute == WAIT_MINUTES )); then
    echo "gave up waiting for parts" >&2
    exit 1
  fi
  sleep 60
done

if [[ ! -f "$BANK_DIR/atom_latents.npy" ]]; then
  python -B scripts/uma/assemble_uma_atom_latents.py \
    --parts-dir "$PARTS_DIR" --stem train --num-parts "$NUM_PARTS" \
    --expected-frames "$EXPECTED_FRAMES" --frame-labels-dir "$FRAME_LABELS" \
    --out-dir "$BANK_DIR"
else
  echo "atom bank exists, skipping assembly: $BANK_DIR/atom_latents.npy"
fi

python -B scripts/mace/fit_descriptor_bank_pca.py \
  --bank "$BANK_DIR/atom_latents.npy" \
  --labels-dir "$BANK_DIR/labels_subdataset" \
  --labels-dir "$BANK_DIR/labels_element" \
  --labels-dir "$BANK_DIR/labels_element_group" \
  --out-dir "$PCA_DIR" --chunk-rows 2000000 --save-pcs 25

python -B scripts/uma/plot_uma_atom_pca.py --pca-dir "$PCA_DIR" --bank-dir "$BANK_DIR" \
  --sample-label "$SAMPLE_LABEL"
