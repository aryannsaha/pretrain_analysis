#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-models/external_repos/submitit}"
REPO_URL="${SUBMITIT_PROXY_REPO:-https://github.com/sihoonchoi/submitit.git}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Activate pretrain_analysis_env before running this script." >&2
  exit 2
fi

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

python -m pip uninstall -y submitit
python -m pip install -e "${TARGET_DIR}"
