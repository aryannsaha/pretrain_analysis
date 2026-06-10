#!/usr/bin/env python
"""Edit the values below, then run this file from the repo root.

Creates the ASE LMBD used for fine-tuning for FAIRChem work
also creates 

file to run fine-tuning with:
fairchem -c ...finetune_template.yaml
"""

from pathlib import Path
from datetime import date
import shutil, subprocess, sys, tempfile

import yaml

"""
specify: new directory name
run_id automatically specified


"""


ROOT = Path(__file__).resolve().parents[1]
FAIRCHEM = ROOT / "fairchem"
TRAIN_INPUT = ROOT / "data/processed/mof-off/r2scan-d4/train_10k.traj"  # file or dir
VAL_INPUT = ROOT / "data/processed/mof-off/r2scan-d4/val_1k.traj"  # file or dir
OUTPUT_DIR = ROOT / "data/processed/mof-off/r2scan-d4/uma_moff_off_test2"
CONFIG_DIR = ROOT / "configs/uma"
UMA_TASK = "omat"
REGRESSION_TASKS = "efs"  # e, ef, or efs
FINETUNE_DATASET = "mof_off_r2scan_d4"
NUM_WORKERS = 8
BASE_MODEL = "uma-s-1p1"
LOCAL_CHECKPOINT = ROOT / "models/pretrained/uma-s-1p1.pt"
RUN_DIR = ROOT / "runs/uma/moff_off_test2"
RUN_NAME = "uma_moff_off_test2_coefficient_test"
MAIL_USER = "as7959@princeton.edu"
WANDB_ENTITY = "rosengroup-general"
WANDB_PROJECT = "finetuning"

DATE_TAG = date.today().strftime("%Y%m%d")
TEMPLATE = f"{BASE_MODEL}_{FINETUNE_DATASET}_{REGRESSION_TASKS}_{DATE_TAG}"

TEMPLATE_NAME = f"{TEMPLATE}.yaml"
RUN_ID = Path(TEMPLATE_NAME).stem
DATA_CONFIG_STEM = f"{TEMPLATE}_data"
DATA_CONFIG_NAME = f"{DATA_CONFIG_STEM}.yaml"


def as_dir(path: Path, tmp: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    out = tmp / name
    out.mkdir()
    files = sorted(path.rglob("*.traj")) if path.is_dir() else [path]
    if not files:
        raise SystemExit(f"No .traj files found in {path}")
    for i, file in enumerate(files):
        (out / f"{i:06d}_{file.name}").symlink_to(file.resolve())
    return out


def set_data_default(cfg: dict, data_config_stem: str) -> None:
    defaults = cfg.get("defaults")
    if not isinstance(defaults, list):
        raise SystemExit("Generated UMA template is missing a Hydra defaults list")
    for item in defaults:
        if isinstance(item, dict) and "data" in item:
            item["data"] = data_config_stem
            return
    defaults.insert(0, {"data": data_config_stem})


with tempfile.TemporaryDirectory(prefix="uma-traj-input-") as tmp:
    tmp = Path(tmp)
    subprocess.run([
        sys.executable,
        str(FAIRCHEM / "src/fairchem/core/scripts/create_uma_finetune_dataset.py"),
        "--train-dir",
        str(as_dir(TRAIN_INPUT, tmp, "train")),
        "--val-dir",
        str(as_dir(VAL_INPUT, tmp, "val")),
        "--output-dir",
        str(OUTPUT_DIR.resolve()),
        "--uma-task",
        UMA_TASK,
        "--regression-tasks",
        REGRESSION_TASKS,
        "--num-workers",
        str(NUM_WORKERS),
        "--base-model",
        BASE_MODEL,
    ], cwd=FAIRCHEM, check=True)

template = OUTPUT_DIR / "uma_sm_finetune_template.yaml"
data_yaml = next((OUTPUT_DIR / "data").glob("uma_conserving_data_task_*.yaml"))
if not LOCAL_CHECKPOINT.is_file():
    raise SystemExit(f"Local UMA checkpoint not found: {LOCAL_CHECKPOINT}")
cfg = yaml.safe_load(template.read_text())
cfg["job"] = {
    "device_type": "CUDA",
    "scheduler": {
        "mode": "SLURM",
        "ranks_per_node": 1,
        "num_nodes": 1,
        "slurm": {
            "timeout_hr": 12,
            "cpus_per_task": 6,
            "mem_gb": 32,
            "additional_parameters": {
                "gres": "gpu:1",
                "constraint": "intel&gpu80",
                "mail_user": MAIL_USER,
                "mail_type": "begin,end,fail",
            },
        },
    },
    "debug": False,
    "run_dir": str(RUN_DIR),
    "run_name": RUN_NAME,
    # Keep this stable so rerunning the same config auto-loads the latest checkpoint
    # from run_dir/timestamp_id/checkpoints when runner_state_path is null.
    "timestamp_id": RUN_ID,
    "runner_state_path": None,
    "logger": {
        "_target_": "fairchem.core.common.logger.WandBSingletonLogger.init_wandb",
        "_partial_": True,
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
    },
}
cfg.pop("base_model_name", None)
set_data_default(cfg, DATA_CONFIG_STEM)
cfg["runner"]["train_eval_unit"]["model"]["checkpoint_location"] = str(
    LOCAL_CHECKPOINT.resolve()
)

(CONFIG_DIR / "data").mkdir(parents=True, exist_ok=True)
template_out = CONFIG_DIR / TEMPLATE_NAME
data_out = CONFIG_DIR / "data" / DATA_CONFIG_NAME
template_out.write_text(yaml.safe_dump(cfg, sort_keys=False))
shutil.copy2(data_yaml, data_out)

print(f"\nRun fine-tuning with:\nfairchem -c {template_out}")
