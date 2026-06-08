#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/uma/uma_sm_finetune_template.yaml}"

if ! command -v fairchem >/dev/null 2>&1; then
  echo "fairchem is not available in the active environment." >&2
  echo "Install FAIR-Chem first, then rerun this script." >&2
  exit 127
fi

fairchem -c "${CONFIG}"
