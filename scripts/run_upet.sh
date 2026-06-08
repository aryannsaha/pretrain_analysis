#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/upet/options.yaml}"

if ! command -v mtt >/dev/null 2>&1; then
  echo "mtt is not available in the active environment." >&2
  echo "Install metatrain first, then rerun this script." >&2
  exit 127
fi

mtt train "${CONFIG}"
