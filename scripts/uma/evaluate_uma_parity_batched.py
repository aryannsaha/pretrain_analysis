#!/usr/bin/env python
"""Edit EVALUATIONS below, then make UMA parity plots.

UMA counterpart of scripts/evaluate_mace_parity_batched.py. The MACE script
cannot be pointed at a UMA run: its loader torch.loads a pickled MACE module and
reads model.atomic_numbers / model.r_max / model.heads. UMA ships a fairchem
inference checkpoint instead, so inference is done here through
pretrained_mlip.load_predict_unit + FAIRChemCalculator.

Everything that affects the numbers is shared with the MACE pipeline: the same
long-form CSV schema (CSV_HEADER) and the same summarize_csv, so MAEs, reservoir
sampling and the MAX_DFT_FORCE_EV_PER_A frame filter are identical across model
families. Only the figure is drawn locally, so the y-axis can say UMA rather
than the MACE plotter's hardcoded label.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Paths resolve against the canonical checkout, not this file's tree, so the
# script still finds runs/ and data/ when executed from a git worktree.
REPO = Path("/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
# Import the shared CSV schema and plotter from the canonical checkout: the
# parity pipeline is untracked there, so a worktree copy would not have it.
sys.path.insert(0, str(REPO))

from scripts.evaluate_mace_parity_batched import (  # noqa: E402
    MAX_DFT_FORCE_EV_PER_A,
    summarize_csv,
)
from scripts.internal.mace_parity_inference import (  # noqa: E402
    CSV_HEADER,
    STRESS_COMPONENTS,
    file_fingerprint,
    references,
    stress_to_voigt,
    write_json,
)

RUN = "uma-s-1p1_hse25_matpes_1to3_uc_efs_20260811"
EVALUATIONS = [
    {
        "model": REPO / f"runs/uma/hse25_matpes_1to3_uc/{RUN}/checkpoints/final/inference_ckpt.pt",
        "data": REPO / "data/processed/hse_matpes/HSE25/hse_matpes_1to3_uc_val.lmdb",
        "output_dir": REPO / f"runs/uma_parity/initial/hse25_1to3_uc__{RUN}__val",
        # dataset_name from configs/uma/data/<run>_data.yaml; also the wandb
        # metric prefix (val/omat.val,*).
        "task": "omat",
    }
]
PLOT_DIR = REPO / "runs/uma_parity/initial_pub"
DEVICE, MAX_FRAMES = "cuda", None
PROGRESS_EVERY, PLOT_POINTS = 1_000, 200_000

# fairchem's ASE reader needs the aselmdb connector spelled out for a .lmdb path.
CONNECT_ARGS = {"type": "aselmdb", "readonly": True, "use_lock_file": False}


def iter_frames(path: Path):
    """Yield ASE Atoms with their reference E/F/S attached."""
    if path.suffix.lower() in {".lmdb", ".aselmdb", ".db"}:
        from fairchem.core.datasets import AseDBDataset

        dataset = AseDBDataset(config={"src": str(path), "connect_args": CONNECT_ARGS})
        for index in range(len(dataset)):
            yield dataset.get_atoms(index)
    else:
        import ase.io

        yield from ase.io.iread(str(path), index=":")


def load_calculator(model_path: Path, task: str, device: str):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.load_predict_unit(str(model_path), device=device)
    return FAIRChemCalculator(predictor, task_name=task)


def run_inference(
    model_path: Path,
    data_path: Path,
    csv_path: Path,
    task: str,
    device_name: str,
    max_frames: int | None,
    progress_every: int,
) -> dict:
    """Stream frames through UMA and write one long-form CSV."""
    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    calculator = load_calculator(model_path, task, device_name)

    counts = {"frames": 0, "atoms": 0, "stress_frames": 0}
    temporary = csv_path.with_suffix(".csv.tmp")
    started = time.time()
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)

        for index, atoms in enumerate(iter_frames(data_path)):
            if max_frames is not None and index >= max_frames:
                break
            frame = index + 1
            reference_energy, reference_forces, reference_stress = references(atoms, frame)

            # Attach the calculator to a clean copy so the SinglePointCalculator
            # holding the DFT reference is never overwritten before it is read.
            predicted = atoms.copy()
            predicted.calc = calculator
            model_energy = float(predicted.get_potential_energy())
            model_forces = np.asarray(predicted.get_forces(), dtype=float)

            n_atoms = len(atoms)
            counts["frames"] += 1
            counts["atoms"] += n_atoms
            writer.writerows([
                ["energy_eV", frame, "", "", reference_energy, model_energy],
                ["energy_eV_per_atom", frame, "", "",
                 reference_energy / n_atoms, model_energy / n_atoms],
            ])
            writer.writerows(
                ["force_eV_per_A", frame, atom, "xyz"[axis],
                 reference_forces[atom, axis], model_forces[atom, axis]]
                for atom in range(n_atoms) for axis in range(3)
            )
            if reference_stress is not None:
                model_stress = stress_to_voigt(predicted.get_stress(), frame, "predicted")
                counts["stress_frames"] += 1
                writer.writerows(
                    ["stress_eV_per_A3", frame, "", component, ref, pred]
                    for component, ref, pred in zip(
                        STRESS_COMPONENTS, reference_stress, model_stress, strict=False
                    )
                )

            if progress_every and counts["frames"] % progress_every == 0:
                rate = counts["frames"] / max(time.time() - started, 1e-9)
                print(f"  {counts['frames']:,} frames ({rate:.1f} frames/s)", flush=True)

    if not counts["frames"]:
        raise SystemExit(f"No frames read from {data_path}")
    temporary.replace(csv_path)
    return {
        "frames": counts["frames"],
        "force_rows": counts["atoms"] * 3,
        "stress_rows": counts["stress_frames"] * len(STRESS_COMPONENTS),
        "seconds": round(time.time() - started, 3),
    }


def make_parity_plot(
    csv_path: Path,
    output_path: Path,
    sample_size: int,
    title: str,
    model_name: str = "UMA",
) -> dict:
    """Same figure as the MACE pipeline, with the y-axis named for this model.

    Statistics come from the shared summarize_csv, so MAEs, reservoir sampling
    and the MAX_DFT_FORCE_EV_PER_A frame filter are identical to the MACE plots
    by construction; only the axis label differs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries, _raw_counts, excluded_frames = summarize_csv(csv_path, sample_size)
    energy = summaries["energy_eV_per_atom"]
    force = summaries["force_eV_per_A"]
    stress = summaries["stress_eV_per_A3"]

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
            ylabel=f"{model_name} {label}",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--plot-points", type=int, default=PLOT_POINTS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Redirect prediction CSVs here, keeping each evaluation's directory "
        "name. Use for smoke tests so they cannot be mistaken for a full run.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore any cached CSV")
    args = parser.parse_args()

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    for evaluation in EVALUATIONS:
        model_path = Path(evaluation["model"])
        data_path = Path(evaluation["data"])
        output_dir = Path(evaluation["output_dir"])
        if args.output_root is not None:
            output_dir = args.output_root / output_dir.name
        for path in (model_path, data_path):
            if not path.exists():
                raise SystemExit(f"Missing input: {path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "predictions.csv"

        name = output_dir.name
        print(f"=== {name} ===", flush=True)
        if csv_path.is_file() and not args.force:
            print(f"Reusing cached predictions: {csv_path}", flush=True)
            counts = None
        else:
            counts = run_inference(
                model_path, data_path, csv_path, evaluation["task"],
                args.device, args.max_frames, PROGRESS_EVERY,
            )
            print(f"Wrote {counts['frames']:,} frames in {counts['seconds']:.1f}s", flush=True)

        plot_path = args.plot_dir / f"{name}.png"
        plot = make_parity_plot(csv_path, plot_path, args.plot_points, name)

        metadata = {
            "model": file_fingerprint(model_path),
            "data": file_fingerprint(data_path),
            "task": evaluation["task"],
            "device": args.device,
            "max_frames": args.max_frames,
            "counts": counts,
            "plot": {"path": str(plot_path), **plot},
        }
        write_json(output_dir / "metadata.json", metadata)

        print(f"Energy MAE: {plot['energy_mae_eV_per_atom']:.6g} eV/atom")
        print(f"Force MAE: {plot['force_mae_eV_per_A']:.6g} eV/A")
        stress = plot["stress_mae_eV_per_A3"]
        if stress is not None:
            print(f"Stress MAE: {stress:.6g} eV/A^3")
        print(f"Excluded frames: {plot['excluded_frames']:,}")
        print(f"Saved {plot_path}", flush=True)


if __name__ == "__main__":
    main()
