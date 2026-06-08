#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/mace/configs.yaml}"

if ! command -v mace_run_train >/dev/null 2>&1; then
  echo "mace_run_train is not available in the active environment." >&2
  echo "Install MACE first, then rerun this script." >&2
  exit 127
fi

mace_run_train --config="${CONFIG}"
