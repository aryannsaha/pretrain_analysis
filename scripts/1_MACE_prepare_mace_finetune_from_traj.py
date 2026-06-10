#!/usr/bin/env python
"""Edit the values below, then run this file from the repo root."""

from datetime import date
from pathlib import Path

import numpy as np
import yaml
from ase.io import read, write


ROOT = Path(__file__).resolve().parents[1]
TRAIN_INPUT = ROOT / "data/processed/mof-off/r2scan-d4/train_10k.traj"
VAL_INPUT = ROOT / "data/processed/mof-off/r2scan-d4/val_1k.traj"
OUTPUT_DIR = ROOT / "data/processed/mof-off/r2scan-d4/mace_moff_off"
PRETRAINED = ROOT / "models/pretrained/mace-omat-0-medium.model"
DATASET_NAME = "mof_off_r2scan_d4"
MODEL_NAME = PRETRAINED.stem
RUN_NAME = WANDB_ID = "mace_moff_off"
RUN_DIR = ROOT / "runs/mace" / RUN_NAME
CONFIG_OUT = RUN_DIR / f"{MODEL_NAME}_{DATASET_NAME}_{date.today():%Y%m%d}.yaml"
WANDB_ENTITY = "rosengroup-general"
WANDB_PROJECT = "finetuning"
MAX_NUM_EPOCHS = 10
BATCH_SIZE = 8
LR = 1.0e-3
DEVICE = "cuda"
ENERGY_KEY, FORCES_KEY, STRESS_KEY = "mace_energy", "mace_forces", "mace_stress"
ENERGY_IN, FORCES_IN, STRESS_IN = ("energy", ENERGY_KEY), ("forces", FORCES_KEY), ("stress", STRESS_KEY)
DEFAULT_E0S = {1: -27.65549603, 5: -89.23771986, 6: -20.04349189, 7: -73.2266828, 8: -11.96387168, 9: -12.5632481, 11: -7.11092969, 12: -8.3717352, 13: -6.85897879, 14: -17.78753248, 15: -766.57107979, 16: -260.19445295, 17: -24.89623942, 19: -12.00265242, 21: -22.96878401, 22: -38.89004647, 23: -14.17689136, 24: -29.79709436, 25: -421.58229264, 26: -28.95919981, 27: -13.38079667, 28: -12.10812657, 29: -11.0266493, 30: -18.06215903, 31: -46.38896315, 33: -59.40125422, 34: -923.00144496, 35: -53.40723705, 39: -40.76883293, 40: -45.62578474, 41: -24.8398042, 42: -26.30801155, 43: -52.34648696, 44: -51.21775841, 45: -24.44165061, 46: -23.1264227, 47: -86.19860042, 48: -40.48605081, 49: -22.70338887, 50: -49.70920903, 51: -51.72793151, 52: -76.71399767, 53: -98.70226559, 57: -119.30846814, 58: -30.95223674, 59: -188.51867757, 60: -132.63495916, 62: -147.21732286, 63: -79.53287722, 64: -82.37644592, 65: -117.18450916, 66: -151.71573468, 67: -74.66460718, 68: -72.18001754, 69: -70.93001583, 70: -35.62976638, 71: -114.34368382, 74: -51.71014625, 75: -210.69676258, 77: -52.65615428, 78: -51.74737156, 79: -101.61334916, 80: -148.50975734, 82: -56.44531344, 83: -117.31067972, 90: -73.6452058, 92: -160.38560063, 93: -667.12743208}


def frames(path):
    path = path.expanduser().resolve()
    files = sorted(path.rglob("*.traj")) + sorted(path.rglob("*.xyz")) + sorted(path.rglob("*.extxyz")) if path.is_dir() else [path]
    return [atoms for file in files for atoms in read(file, ":")]


def get(atoms, keys, result):
    for key in keys:
        if key in atoms.info:
            return atoms.info[key]
        if key in atoms.arrays:
            return atoms.arrays[key]
    if atoms.calc and result in atoms.calc.results:
        return atoms.calc.results[result]
    return atoms.get_potential_energy() if result == "energy" else atoms.get_forces() if result == "forces" else atoms.get_stress(voigt=False)


def stress33(value):
    value = np.asarray(value, dtype=float)
    if value.shape == (6,):
        xx, yy, zz, yz, xz, xy = value
        value = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return value.reshape(3, 3)


def write_split(src, dst):
    out, zs = [], set()
    for source in frames(src):
        atoms = source.copy()
        atoms.info[ENERGY_KEY] = float(np.asarray(get(source, ENERGY_IN, "energy")))
        atoms.arrays[FORCES_KEY] = np.asarray(get(source, FORCES_IN, "forces"), dtype=float).reshape(len(atoms), 3)
        atoms.info[STRESS_KEY] = stress33(get(source, STRESS_IN, "stress")).tolist()
        atoms.calc = None
        zs.update(map(int, atoms.numbers))
        out.append(atoms)
    if not out:
        raise SystemExit(f"No ASE-readable structures found in {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    write(dst, out)
    first = read(dst, "0")
    if ENERGY_KEY not in first.info or FORCES_KEY not in first.arrays or STRESS_KEY not in first.info:
        raise SystemExit(f"{dst} is missing MACE target keys")
    if first.arrays[FORCES_KEY].shape != (len(first), 3) or np.asarray(first.info[STRESS_KEY]).shape != (3, 3):
        raise SystemExit(f"{dst} has bad force/stress shapes")
    return len(out), zs


def latest_model():
    models = [p for p in RUN_DIR.rglob("*.model") if p.resolve() != PRETRAINED.resolve()]
    return max(models, key=lambda path: path.stat().st_mtime) if models else PRETRAINED


train_xyz, val_xyz = OUTPUT_DIR / "train.extxyz", OUTPUT_DIR / "val.extxyz"
RUN_DIR.mkdir(parents=True, exist_ok=True)
n_train, train_zs = write_split(TRAIN_INPUT, train_xyz)
n_val, val_zs = write_split(VAL_INPUT, val_xyz)
atomic_numbers = sorted(train_zs | val_zs)
missing = [z for z in atomic_numbers if z not in DEFAULT_E0S]
if missing:
    raise SystemExit(f"Missing E0s for atomic numbers: {missing}")
foundation_model = latest_model()
if not foundation_model.is_file():
    raise SystemExit(f"MACE model not found: {foundation_model}")

cfg = {
    "name": RUN_NAME,
    "foundation_model": str(foundation_model.resolve()),
    "restart_latest": foundation_model.resolve() != PRETRAINED.resolve(),
    "train_file": str(train_xyz.resolve()),
    "valid_file": str(val_xyz.resolve()),
    "atomic_numbers": str(atomic_numbers),
    "E0s": {z: DEFAULT_E0S[z] for z in atomic_numbers},
    "energy_key": ENERGY_KEY,
    "forces_key": FORCES_KEY,
    "stress_key": STRESS_KEY,
    "error_table": "PerAtomMAEstressvirials",
    "compute_stress": True,
    "loss": "stress",
    "energy_weight": 1,
    "forces_weight": 20,
    "stress_weight": 1,
    "lr": LR,
    "scaling": "rms_forces_scaling",
    "batch_size": BATCH_SIZE,
    "max_num_epochs": MAX_NUM_EPOCHS,
    "ema": True,
    "ema_decay": 0.99,
    "amsgrad": True,
    "default_dtype": "float32",
    "device": DEVICE,
    "wandb": True,
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "wandb_name": RUN_NAME,
}
CONFIG_OUT.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"Wrote {n_train} train / {n_val} val structures and {CONFIG_OUT}\nRun from {RUN_DIR} with: mace_run_train --config={CONFIG_OUT.name}")
