#!/usr/bin/env python
"""Plot validation E/F/S MAEs from UPET, MACE, and UMA runs.

Edit RUNS below, or pass paths on the command line:
  python scripts/plot_efs_results.py
  python scripts/plot_efs_results.py MACE=runs/mace/.../logs/mace_moff_123.out UMA=runs/uma/.../output.log
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

MPLCONFIGDIR = Path("outputs/.mplconfig")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# RUNS = [
#     ("UPET", "path/to/train.csv"),
#     ("MACE", "path/to/mace_log.out"),
#     ("UMA", "path/to/output.log"),
# ]

RUNS = [
    ("UPET", "runs/upet/upet_moff_off_test/outputs/2026-06-10/12-12-49/train.csv"),
    ("MACE_original", "runs/mace/mace_moff_off/logs/mace_moff_9508496.out"),
    # ("MACE_test1", "runs/mace/mof_off_ft_test/test_1/logs/mace_moff_9514712.out"),
    # ("MACE_test2", "runs/mace/mof_off_ft_test/test_2_lr4/logs/mace_moff_9515217.out"),
    # ("MACE_foundation_E", "runs/mace/mof_off_ft_test/test_4_foundation_E/logs/mace_moff_9548429.out"),
    ("MACE_estimated_E_d", "runs/mace/mof_off_ft_test/test_5_average_E/logs/mace_moff_d_9585056.out"),
    (
        "UMA",
        "runs/uma/moff_off_test2/uma-s-1p1_mof_off_r2scan_d4_efs_20260610/"
        "logs/wandb/run-20260610_121510-uma-s-1p1_mof_off_r2scan_d4_efs_20260610/files/output.log",
    ),
]

OUT_PNG = Path("outputs/ft_test_4.png")
OUT_CSV = Path("outputs/efs_errors.csv")


def n(x: str) -> float:
    return float(x.strip())


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", s).replace("\r", "")


def col(cols: list[str], *needles: str) -> str:
    matches = [c for c in cols if all(t in c for t in needles)]
    if not matches:
        raise ValueError(f"missing CSV column containing {needles}")
    return matches[0]


def parse_upet(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as f:
        r = csv.reader(f)
        header = next(r)
        next(r, None)
        e, fcol = col(header, "validation energy/", "MAE (per atom)"), col(header, "validation forces[", "MAE")
        s = col(header, "validation ", "MAE (per atom)", "virial[")
        for raw in r:
            if raw and raw[0].strip():
                rec = dict(zip(header, raw))
                rows.append({"epoch": int(rec["Epoch"]), "E": n(rec[e]), "F": n(rec[fcol]), "S": n(rec[s])})
    return rows


def parse_mace_log(path: Path) -> list[dict]:
    pat = re.compile(
        r"Epoch\s+(\d+):.*?MAE_E_per_atom=\s*([0-9.eE+-]+).*?"
        r"MAE_F=\s*([0-9.eE+-]+).*?MAE_stress=\s*([0-9.eE+-]+)"
    )
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            rows.append({"epoch": int(m[1]), "E": n(m[2]), "F": n(m[3]), "S": n(m[4])})
    return rows


def parse_mace_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        if d.get("mode") == "eval" and d.get("epoch") is not None and "mae_e_per_atom" in d:
            rows.append({"epoch": int(d["epoch"]), "E": 1000 * d["mae_e_per_atom"], "F": 1000 * d["mae_f"], "S": 1000 * d["mae_stress"]})
    return rows


def parse_uma(path: Path) -> list[dict]:
    rows, cur, vals = [], None, {}
    pats = {
        "E": r"val/.*energy,per_atom_mae:\s*([0-9.eE+-]+)",
        "F": r"val/.*forces,mae:\s*([0-9.eE+-]+)",
        "S": r"val/.*stress,mae:\s*([0-9.eE+-]+)",
    }
    for raw in path.read_text(errors="replace").splitlines():
        line = strip_ansi(raw)
        m = re.search(r"Eval Epoch\s+(\d+)", line)
        if m:
            cur, vals = int(m[1]), {}
        for k, pat in pats.items():
            mm = re.search(pat, line)
            if cur is not None and mm:
                vals[k] = 1000 * n(mm[1])
                if len(vals) == 3:
                    rows.append({"epoch": cur, **vals})
                    vals = {}
    return rows


def candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    names = ["train.csv", "output.log", "*.out", "*_train.txt"]
    found = [p for pat in names for p in path.rglob(pat)]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def load(label: str, path: str) -> tuple[str, list[dict], Path]:
    parsers = [parse_upet, parse_mace_log, parse_mace_jsonl, parse_uma]
    for p in candidates(Path(path)):
        for parser in parsers:
            try:
                rows = parser(p)
            except Exception:
                rows = []
            if rows:
                return label, sorted(rows, key=lambda r: r["epoch"]), p
    raise SystemExit(f"No E/F/S metrics found under {path}")


def run_arg(s: str) -> tuple[str, str]:
    if "=" in s:
        label, path = s.split("=", 1)
        return label, path
    return Path(s).stem, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="Optional LABEL=path entries. If omitted, edit/use RUNS in this file.")
    ap.add_argument("--out", default=str(OUT_PNG))
    args = ap.parse_args()

    runs = [run_arg(x) for x in args.runs] if args.runs else RUNS
    data = [load(label, path) for label, path in runs]

    out_png, out_csv = Path(args.out), OUT_CSV if args.out == str(OUT_PNG) else Path(args.out).with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "epoch", "E_meV_atom", "F_meV_A", "S_meV_scale", "source"])
        for label, rows, src in data:
            for r in rows:
                w.writerow([label, r["epoch"], r["E"], r["F"], r["S"], src])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, key, ylabel in zip(axes, ["E", "F", "S"], ["Energy MAE (meV/atom)", "Force MAE (meV/A)", "Stress/Virial MAE (meV scale)"]):
        for label, rows, _ in data:
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o", linewidth=2, label=label)
        ax.set_title(key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Validation E/F/S Errors")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Wrote {out_png}\nWrote {out_csv}")


if __name__ == "__main__":
    main()
