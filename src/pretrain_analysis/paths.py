"""Common project paths used by scripts and notebooks."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
LOGS_DIR = PROJECT_ROOT / "logs"

WORKFLOW_DIRS = {
    "uma": {
        "configs": CONFIGS_DIR / "uma",
        "runs": RUNS_DIR / "uma",
    },
    "upet": {
        "configs": CONFIGS_DIR / "upet",
        "runs": RUNS_DIR / "upet",
    },
    "mace": {
        "configs": CONFIGS_DIR / "mace",
        "runs": RUNS_DIR / "mace",
    },
}
