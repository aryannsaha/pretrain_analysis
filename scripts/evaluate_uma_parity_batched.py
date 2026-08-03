#!/usr/bin/env python
"""Edit EVALUATIONS below, then make batched UMA parity plots.

UMA counterpart of ``evaluate_mace_parity_batched.py``. Figures deliberately use
the same panel layout and MAE definitions so UMA and MACE results are directly
comparable.

Run every evaluation:      python scripts/evaluate_uma_parity_batched.py
Run one (Slurm array):     python scripts/evaluate_uma_parity_batched.py --task-index 0
Re-plot without inference: python scripts/evaluate_uma_parity_batched.py --plot-only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Data and run trees live outside git, so allow pointing at the main checkout
# when this script is executed from a worktree.
DATA_ROOT = Path(os.environ.get("PRETRAIN_ANALYSIS_ROOT", ROOT))

from scripts.internal.uma_parity_inference import (  # noqa: E402
    CSV_HEADER,
    STRESS_COMPONENTS,
    file_fingerprint,
    prepare_predictions,
    stress_to_voigt,
    validate_cache,
    write_json,
)

__all__ = [
    "STRESS_COMPONENTS",
    "file_fingerprint",
    "stress_to_voigt",
    "validate_cache",
]


# Each fine-tune differs only in the pretrained checkpoint it started from, so
# the two families are directly comparable on identical data and hyperparameters.
BASE_MODELS = [
    {
        "tag": "1p1",
        "run_dir_suffix": "",
        "run_tag": "uma-s-1p1_matpes_r2scan_lte{lte}_s75k_efs_20260802",
    },
    {
        "tag": "1p2p1",
        "run_dir_suffix": "_1p2p1",
        "run_tag": "uma-s-1p2p1_matpes_r2scan_lte{lte}_s75k_efs_20260802",
    },
]
EVALUATIONS = [
    {
        "model": DATA_ROOT
        / f"runs/uma/matpes/r2scan_flatiron/lte{lte}_s75k{base['run_dir_suffix']}"
        / base["run_tag"].format(lte=lte)
        / "checkpoints/final/inference_ckpt.pt",
        "data": DATA_ROOT / f"data/processed/matpes/r2scan_flatiron/r2scan_{lte}/{filename}",
        "output_dir": DATA_ROOT
        / "runs/matpes_parity/uma_initial"
        / f"lte{lte}__{base['run_tag'].format(lte=lte)}__{dataset}",
        "task_name": "omat",
        "base_model": base["tag"],
        "split": f"lte{lte} {dataset}",
        "label": f"{base['tag']} lte{lte} {dataset}",
    }
    for base in BASE_MODELS
    for lte in (3, 4)
    for dataset, filename in (("lte_val", "lte_val.traj"), ("gt", "gt_.traj"))
]
PLOT_DIR = DATA_ROOT / "runs/matpes_parity/uma_initial_pub"
DEVICE, BATCH_SIZE, MAX_FRAMES = "cuda", 16, None
MAX_ATOMS_PER_BATCH = 4096
PROGRESS_EVERY, PLOT_POINTS = 10_000, 200_000
MAX_DFT_FORCE_EV_PER_A = 25
MODEL_LABEL = "UMA"


@dataclass
class Summary:
    sample_size: int
    seed: int
    count: int = 0
    absolute_error: float = 0.0
    low: float = math.inf
    high: float = -math.inf
    sample: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.random = random.Random(self.seed)

    def add(self, reference: float, prediction: float) -> None:
        if not (math.isfinite(reference) and math.isfinite(prediction)):
            raise ValueError("CSV contains a non-finite reference or prediction")
        self.count += 1
        self.absolute_error += abs(reference - prediction)
        self.low = min(self.low, reference, prediction)
        self.high = max(self.high, reference, prediction)
        if len(self.sample) < self.sample_size:
            self.sample.append((reference, prediction))
        else:
            index = self.random.randrange(self.count)
            if index < self.sample_size:
                self.sample[index] = (reference, prediction)

    @property
    def mae(self) -> float:
        return self.absolute_error / self.count

    @property
    def limits(self) -> tuple[float, float]:
        low, high = np.asarray(self.sample).min(), np.asarray(self.sample).max()
        if low == high:
            padding = max(abs(low) * 0.05, 1.0e-6)
        else:
            padding = 0.02 * (high - low)
        return low - padding, high + padding


def summarize_csv(csv_path: Path, sample_size: int):
    summaries = {
        "energy_eV_per_atom": Summary(sample_size, 20260729),
        "force_eV_per_A": Summary(sample_size, 20260730),
        "stress_eV_per_A3": Summary(sample_size, 20260731),
    }
    raw_counts = dict.fromkeys(summaries, 0)
    excluded_frames = 0
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, []) != CSV_HEADER:
            raise ValueError(f"Unexpected CSV header in {csv_path}")
        for _, frame in groupby(reader, lambda row: row[1]):
            rows = list(frame)
            for row in rows:
                if row[0] in summaries:
                    raw_counts[row[0]] += 1
            forces = np.asarray(
                [float(row[4]) for row in rows if row[0] == "force_eV_per_A"]
            ).reshape(-1, 3)
            if np.any(np.linalg.norm(forces, axis=1) > MAX_DFT_FORCE_EV_PER_A):
                excluded_frames += 1
                continue
            for row in rows:
                if row[0] in summaries:
                    summaries[row[0]].add(float(row[4]), float(row[5]))
    if (
        not summaries["energy_eV_per_atom"].count
        or not summaries["force_eV_per_A"].count
    ):
        raise ValueError("CSV must contain energy and force predictions")
    return summaries, raw_counts, excluded_frames


def make_parity_plot(
    csv_path: Path,
    output_path: Path,
    sample_size: int,
    title: str,
    expected_counts: dict | None = None,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries, raw_counts, excluded_frames = summarize_csv(csv_path, sample_size)
    energy = summaries["energy_eV_per_atom"]
    force = summaries["force_eV_per_A"]
    stress = summaries["stress_eV_per_A3"]
    if expected_counts is not None:
        actual = tuple(raw_counts[name] for name in summaries)
        expected = (
            expected_counts["frames"],
            expected_counts["force_rows"],
            expected_counts["stress_rows"],
        )
        if actual != expected:
            raise ValueError(f"CSV counts {actual} do not match metadata {expected}")

    panels = [
        (energy, "energy (eV/atom)"),
        (force, "force component (eV/A)"),
    ]
    if stress.count:
        panels.append((stress, "stress component (eV/A^3)"))

    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    for axis, (summary, label) in zip(np.atleast_1d(axes), panels, strict=False):
        points = np.asarray(summary.sample)
        limits = summary.limits
        axis.scatter(points[:, 0], points[:, 1], s=4, alpha=0.25, rasterized=True)
        axis.plot(limits, limits, "k--", linewidth=1)
        axis.set(
            xlim=limits,
            ylim=limits,
            aspect="equal",
            xlabel=f"DFT {label}",
            ylabel=f"{MODEL_LABEL} {label}",
            title=f"{label}\nMAE = {summary.mae:.3g} ({energy.count:,} structures)",
        )
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, format="png", dpi=200)
    plt.close(figure)

    return {
        "energy_mae_eV_per_atom": energy.mae,
        "force_mae_eV_per_A": force.mae,
        "stress_mae_eV_per_A3": stress.mae if stress.count else None,
        "energy_points": energy.count,
        "force_points": force.count,
        "stress_points": stress.count,
        "sample_size": sample_size,
        "included_frames": energy.count,
        "excluded_frames": excluded_frames,
        "max_dft_force_eV_per_A": MAX_DFT_FORCE_EV_PER_A,
    }


BASE_MODEL_COLORS = {"1p1": "#4C72B0", "1p2p1": "#DD8452"}


def write_summary(rows: list[dict], plot_dir: Path, scope: str) -> None:
    """One CSV plus a grouped bar chart comparing base models on each split."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = plot_dir / f"uma_matpes_efs_mae_summary_{scope}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["label", "base_model", "split", "E_meV_per_atom", "F_meV_per_A",
             "S_meV_per_A3", "frames", "excluded_frames"]
        )
        for row in rows:
            writer.writerow([
                row["label"],
                row.get("base_model", ""),
                row.get("split", ""),
                1000 * row["energy_mae_eV_per_atom"],
                1000 * row["force_mae_eV_per_A"],
                None if row["stress_mae_eV_per_A3"] is None else 1000 * row["stress_mae_eV_per_A3"],
                row["included_frames"],
                row["excluded_frames"],
            ])

    # Group bars by split so the two base models sit side by side.
    splits, bases = [], []
    for row in rows:
        if row.get("split") not in splits:
            splits.append(row.get("split"))
        if row.get("base_model") not in bases:
            bases.append(row.get("base_model"))
    by_key = {(row.get("base_model"), row.get("split")): row for row in rows}

    positions = np.arange(len(splits))
    width = 0.8 / max(len(bases), 1)
    panels = [
        ("energy_mae_eV_per_atom", "Energy MAE (meV/atom)"),
        ("force_mae_eV_per_A", "Force MAE (meV/A)"),
        ("stress_mae_eV_per_A3", "Stress MAE (meV/A^3)"),
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5))
    for axis, (key, title) in zip(np.atleast_1d(axes), panels, strict=False):
        for index, base in enumerate(bases):
            values = []
            for split in splits:
                row = by_key.get((base, split))
                value = None if row is None else row[key]
                values.append(math.nan if value is None else 1000 * value)
            offset = (index - (len(bases) - 1) / 2) * width
            bars = axis.bar(
                positions + offset, values, width=width,
                label=f"uma-s-{base}", color=BASE_MODEL_COLORS.get(base),
            )
            axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)
        axis.set_xticks(positions, splits, rotation=15, ha="right")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    if len(bases) > 1:
        np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    based = " vs ".join(f"uma-s-{base}" for base in bases)
    figure.suptitle(
        f"{based} fine-tuned on MatPES r2SCAN 75k: val vs held-out larger cells"
    )
    figure.tight_layout()
    summary_png = plot_dir / f"uma_matpes_efs_mae_summary_{scope}.png"
    figure.savefig(summary_png, format="png", dpi=200)
    plt.close(figure)
    print(f"Saved {summary_png}")
    print(f"Saved {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default="1p2p1",
        choices=[base["tag"] for base in BASE_MODELS] + ["all"],
        help="Which pretrained checkpoint's fine-tunes to evaluate and summarise.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        help="Run only this entry of the selected base model's evaluations (for Slurm arrays).",
    )
    parser.add_argument("--device", default=DEVICE, choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-atoms-per-batch", type=int, default=MAX_ATOMS_PER_BATCH)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild figures from cached prediction CSVs; never run inference.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Rebuild only the cross-evaluation summary from existing metadata.",
    )
    return parser.parse_args()


def active(args: argparse.Namespace) -> list[dict]:
    """Every evaluation belonging to the requested base model."""
    if args.base_model == "all":
        return EVALUATIONS
    return [e for e in EVALUATIONS if e["base_model"] == args.base_model]


def selected(args: argparse.Namespace) -> list[dict]:
    evaluations = active(args)
    if args.task_index is None:
        return evaluations
    if not 0 <= args.task_index < len(evaluations):
        raise SystemExit(
            f"--task-index must be in [0, {len(evaluations) - 1}] for "
            f"--base-model {args.base_model}, got {args.task_index}"
        )
    return [evaluations[args.task_index]]


def main() -> None:
    args = parse_args()
    plot_dir = args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        rows, missing = [], []
        for evaluation in active(args):
            path = Path(evaluation["output_dir"]) / "metadata.json"
            try:
                metadata = json.loads(path.read_text())
                plot = metadata["plot"]
            except (OSError, KeyError, json.JSONDecodeError):
                missing.append(evaluation["label"])
                continue
            rows.append({
                "label": evaluation["label"],
                "base_model": evaluation["base_model"],
                "split": evaluation["split"],
                **plot,
            })
        if not rows:
            raise SystemExit("No evaluations have completed yet")
        if missing:
            print(f"Skipping {len(missing)} evaluation(s) without results: {', '.join(missing)}")
        write_summary(rows, plot_dir, args.base_model)
        return

    rows = []
    for evaluation in selected(args):
        label = evaluation["label"]
        output_dir = Path(evaluation["output_dir"])
        csv_path = output_dir / "predictions.csv"
        metadata_path = output_dir / "metadata.json"

        if args.plot_only:
            if not csv_path.is_file():
                raise SystemExit(f"--plot-only needs cached predictions at {csv_path}")
            metadata = json.loads(metadata_path.read_text())
            model_path, data_path = Path(evaluation["model"]), Path(evaluation["data"])
        else:
            inference_args = SimpleNamespace(
                model=evaluation["model"],
                data=evaluation["data"],
                output_dir=evaluation["output_dir"],
                task_name=evaluation["task_name"],
                manifest=None,
                task_index=None,
                device=args.device,
                batch_size=args.batch_size,
                max_atoms_per_batch=args.max_atoms_per_batch,
                max_frames=args.max_frames,
                progress_every=args.progress_every,
            )
            (
                csv_path,
                metadata_path,
                metadata,
                model_path,
                data_path,
            ) = prepare_predictions(inference_args)

        plot_path = plot_dir / f"{output_dir.name}.png"
        plot = make_parity_plot(
            csv_path,
            plot_path,
            PLOT_POINTS,
            f"uma-s-{evaluation['base_model']} fine-tune, "
            f"{evaluation['split']} ({data_path.name})",
            metadata["counts"],
        )
        metadata["plot"] = {
            "complete": True,
            "path": str(plot_path),
            "size_bytes": plot_path.stat().st_size,
            **plot,
        }
        write_json(metadata_path, metadata)
        rows.append({
            "label": label,
            "base_model": evaluation["base_model"],
            "split": evaluation["split"],
            **plot,
        })

        print(f"[{label}] Energy MAE: {plot['energy_mae_eV_per_atom']:.6g} eV/atom")
        print(f"[{label}] Force MAE: {plot['force_mae_eV_per_A']:.6g} eV/A")
        stress = plot["stress_mae_eV_per_A3"]
        if stress is None:
            print(f"[{label}] Stress MAE: unavailable")
        else:
            print(f"[{label}] Stress MAE: {stress:.6g} eV/A^3")
        print(
            f"[{label}] Excluded frames above {MAX_DFT_FORCE_EV_PER_A} eV/A: "
            f"{plot['excluded_frames']:,}"
        )
        print(f"[{label}] Saved {plot_path}")

    if args.task_index is None and len(rows) == len(active(args)):
        write_summary(rows, plot_dir, args.base_model)


if __name__ == "__main__":
    main()
