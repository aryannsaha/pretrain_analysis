#!/usr/bin/env python
"""Create or reuse a UMA prediction CSV; this module does no plotting.

This is the fairchem/UMA counterpart of ``mace_parity_inference.py`` and writes
the identical long-form CSV so the same summarising and plotting code works for
both model families.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

CACHE_VERSION = 1
CSV_HEADER = ["property", "frame", "atom_index", "component", "reference", "prediction"]
STRESS_COMPONENTS = ("xx", "yy", "zz", "yz", "xz", "xy")


def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument(
        "--task-name",
        help="UMA dataset/task name (e.g. omat). Required only for multi-task checkpoints.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-atoms-per-batch",
        type=int,
        default=4096,
        help="Cap on atoms per batch; keeps memory bounded when frames are large.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--progress-every", "--progress", type=int, default=10000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    return parser.parse_args()


def apply_manifest_task(args: argparse.Namespace) -> None:
    """Replace direct arguments with one row from a Slurm task manifest."""
    if args.manifest is None:
        if not all((args.model, args.data, args.output_dir)):
            raise SystemExit("Pass --model, --data, and --output-dir")
        if args.task_index is not None:
            raise SystemExit("--task-index requires --manifest")
        return

    if any((args.model, args.data, args.output_dir, args.task_name)):
        raise SystemExit("Do not mix --manifest with direct model/data arguments")
    if args.task_index is None:
        raise SystemExit("--manifest requires --task-index")

    with args.manifest.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        task = next(
            (row for row in rows if row["task_id"] == str(args.task_index)), None
        )
    if task is None:
        raise SystemExit(f"Task {args.task_index} not found in {args.manifest}")
    args.model = Path(task["model_path"])
    args.data = Path(task["data_path"])
    args.output_dir = Path(task["output_dir"])
    args.task_name = task.get("task_name", "").strip() or None


def file_fingerprint(path: Path) -> dict:
    path = path.resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def validate_cache(
    csv_path: Path,
    metadata_path: Path,
    model_fingerprint: dict,
    data_fingerprint: dict,
    options: dict,
) -> tuple[bool, str, dict | None]:
    """Return an existing cache when it belongs to these exact inputs."""
    if not csv_path.is_file() or not metadata_path.is_file():
        return False, "cache files are missing", None
    try:
        metadata = json.loads(metadata_path.read_text())
        counts = metadata["counts"]
        with csv_path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        valid = (
            metadata["cache_version"] == CACHE_VERSION
            and metadata["complete"] is True
            and metadata["model"] == model_fingerprint
            and metadata["data"] == data_fingerprint
            and metadata["options"] == options
            and (
                options["requested_task"] is None
                or metadata["resolved_task"] == options["requested_task"]
            )
            and metadata["csv"]["size_bytes"] == csv_path.stat().st_size
            and header == CSV_HEADER
            and counts["energy_rows"] == 2 * counts["frames"]
            and counts["force_rows"] == 3 * counts["atoms"]
            and counts["stress_rows"]
            == len(STRESS_COMPONENTS) * counts["stress_frames"]
            and counts["rows"]
            == counts["energy_rows"] + counts["force_rows"] + counts["stress_rows"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, "cache metadata is invalid", None
    return (True, "cache is valid", metadata) if valid else (
        False,
        "cache does not match this run",
        metadata,
    )


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def iter_frames(path: Path):
    if path.suffix.lower() in {".lmdb", ".aselmdb", ".db"}:
        from fairchem.core.datasets.ase_datasets import AseDBDataset

        dataset = AseDBDataset(config={"src": str(path)})
        for index in range(len(dataset)):
            yield dataset.get_atoms(index)
    else:
        import ase.io

        yield from ase.io.iread(str(path), index=":")


def stress_to_voigt(value, frame: int, source: str) -> np.ndarray:
    """Normalize 3x3, flat-9, or Voigt-6 stress to ASE Voigt order."""
    from ase.stress import full_3x3_to_voigt_6_stress

    stress = np.asarray(value, dtype=float)
    if stress.shape == (9,):
        stress = stress.reshape(3, 3)
    if stress.shape == (3, 3):
        stress = full_3x3_to_voigt_6_stress(stress)
    if stress.shape != (6,) or not np.isfinite(stress).all():
        raise ValueError(f"frame {frame} has invalid {source} stress")
    return stress


def reference_stress(atoms, frame: int) -> np.ndarray | None:
    values = [atoms.info.get("stress"), atoms.info.get("REF_stress")]
    if atoms.calc is not None:
        values.insert(0, getattr(atoms.calc, "results", {}).get("stress"))
    for value in values:
        if value is not None:
            return stress_to_voigt(value, frame, "reference")
    try:
        return stress_to_voigt(atoms.get_stress(), frame, "reference")
    except (AttributeError, AssertionError, KeyError, NotImplementedError, RuntimeError):
        return None


def references(atoms, frame: int):
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return energy, forces, reference_stress(atoms, frame)


def batches(items, size: int, max_atoms: int | None = None):
    """Group frames into batches of `size`, splitting early past `max_atoms`."""
    batch: list = []
    atoms_in_batch = 0
    for atoms in items:
        n_atoms = len(atoms)
        if batch and (
            len(batch) >= size
            or (max_atoms is not None and atoms_in_batch + n_atoms > max_atoms)
        ):
            yield batch
            batch, atoms_in_batch = [], 0
        batch.append(atoms)
        atoms_in_batch += n_atoms
    if batch:
        yield batch


def load_predictor(path: Path, device: str, requested_task: str | None):
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    predictor = MLIPPredictUnit(str(path), device=device)
    tasks = list(predictor.dataset_to_tasks.keys())
    if not tasks:
        raise SystemExit(f"Checkpoint exposes no tasks: {path}")
    if requested_task is None:
        if len(tasks) != 1:
            raise SystemExit(f"Multi-task checkpoints require --task-name; choose from {tasks}")
        task = tasks[0]
    elif requested_task not in tasks:
        raise SystemExit(f"Unknown task {requested_task}; choose from {tasks}")
    else:
        task = requested_task
    dtype = str(predictor.inference_settings.base_precision_dtype).removeprefix("torch.")
    return predictor, tasks, task, dtype


def run_inference(
    model_path: Path,
    data_path: Path,
    csv_path: Path,
    requested_task: str | None,
    device_name: str,
    batch_size: int,
    max_atoms_per_batch: int | None,
    max_frames: int | None,
    progress_every: int,
) -> dict:
    """Stream frames through UMA in batches and write one long-form CSV."""
    import torch

    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    if device_name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    predictor, tasks, task, model_dtype = load_predictor(
        model_path, device_name, requested_task
    )
    target_dtype = predictor.inference_settings.base_precision_dtype

    frames = iter_frames(data_path)
    if max_frames is not None:
        frames = islice(frames, max_frames)

    temporary = csv_path.with_suffix(".csv.tmp")
    counts = {"frames": 0, "atoms": 0, "stress_frames": 0}
    started = time.time()
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)

        for atoms_batch in batches(frames, batch_size, max_atoms_per_batch):
            first_frame = counts["frames"] + 1
            refs = [
                references(atoms, first_frame + i)
                for i, atoms in enumerate(atoms_batch)
            ]
            graphs = [
                AtomicData.from_ase(
                    atoms,
                    task_name=task,
                    r_edges=False,
                    r_data_keys=["spin", "charge"],
                    target_dtype=target_dtype,
                )
                for atoms in atoms_batch
            ]
            batch = atomicdata_list_to_batch(graphs)
            output = predictor.predict(batch, undo_element_references=True)

            pred_e = output["energy"].detach().cpu().double().reshape(-1).numpy()
            pred_f = output["forces"].detach().cpu().double().reshape(-1, 3).numpy()
            pred_s = (
                output["stress"].detach().cpu().double().numpy()
                if "stress" in output
                else None
            )
            offsets = np.concatenate(
                ([0], np.cumsum([len(atoms) for atoms in atoms_batch]))
            )

            for index, (atoms, (ref_e, ref_f, ref_s)) in enumerate(
                zip(atoms_batch, refs, strict=False)
            ):
                frame = first_frame + index
                model_e = float(pred_e[index])
                model_f = pred_f[offsets[index] : offsets[index + 1]]
                n_atoms = len(atoms)
                writer.writerows([
                    ["energy_eV", frame, "", "", ref_e, model_e],
                    ["energy_eV_per_atom", frame, "", "", ref_e / n_atoms, model_e / n_atoms],
                ])
                writer.writerows(
                    ["force_eV_per_A", frame, atom, "xyz"[axis],
                     ref_f[atom, axis], model_f[atom, axis]]
                    for atom in range(n_atoms) for axis in range(3)
                )
                if ref_s is not None:
                    if pred_s is None:
                        raise SystemExit(
                            "Reference stress is present but the checkpoint predicts no stress"
                        )
                    model_s = stress_to_voigt(pred_s[index], frame, "predicted")
                    writer.writerows(
                        ["stress_eV_per_A3", frame, "", component, ref, pred]
                        for component, ref, pred in zip(
                            STRESS_COMPONENTS, ref_s, model_s, strict=False
                        )
                    )

            counts["frames"] += len(atoms_batch)
            counts["atoms"] += sum(len(atoms) for atoms in atoms_batch)
            counts["stress_frames"] += sum(stress is not None for _, _, stress in refs)
            if progress_every and counts["frames"] % progress_every < len(atoms_batch):
                elapsed = time.time() - started
                print(f"Processed {counts['frames']:,} frames "
                      f"({counts['frames'] / elapsed:.1f} frames/s)", flush=True)

    if counts["frames"] == 0:
        raise SystemExit("Input dataset is empty")
    temporary.replace(csv_path)

    counts.update(
        energy_rows=2 * counts["frames"],
        force_rows=3 * counts["atoms"],
        stress_rows=len(STRESS_COMPONENTS) * counts["stress_frames"],
    )
    counts["rows"] = sum(counts[key] for key in ("energy_rows", "force_rows", "stress_rows"))
    return {
        "resolved_task": task,
        "available_tasks": tasks,
        "model_dtype": model_dtype,
        "elapsed_s": time.time() - started,
        "counts": counts,
    }


def prepare_predictions(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict, Path, Path]:
    """Create or reuse predictions and return their paths and metadata."""
    apply_manifest_task(args)
    if args.batch_size < 1:
        raise SystemExit("Batch size must be positive")
    if args.max_atoms_per_batch is not None and args.max_atoms_per_batch < 1:
        raise SystemExit("Max atoms per batch must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise SystemExit("Max frames must be positive")
    if args.progress_every < 0:
        raise SystemExit("Progress interval cannot be negative")

    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    if not model_path.is_file() or not data_path.is_file():
        raise SystemExit("Model or data file does not exist")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "predictions.csv"
    metadata_path = output_dir / "metadata.json"

    model_info = file_fingerprint(model_path)
    data_info = file_fingerprint(data_path)
    options = {
        "requested_task": args.task_name,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_atoms_per_batch": args.max_atoms_per_batch,
        "max_frames": args.max_frames,
    }
    cached, reason, metadata = validate_cache(
        csv_path, metadata_path, model_info, data_info, options
    )
    if cached:
        print(f"Reusing {csv_path}")
    else:
        print(f"{reason}; running inference", flush=True)
        result = run_inference(
            model_path, data_path, csv_path, args.task_name, args.device,
            args.batch_size, args.max_atoms_per_batch, args.max_frames,
            args.progress_every,
        )
        metadata = {
            "cache_version": CACHE_VERSION,
            "complete": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": model_info,
            "data": data_info,
            "options": options,
            **result,
            "csv": {
                "path": str(csv_path),
                "size_bytes": csv_path.stat().st_size,
                "header": CSV_HEADER,
            },
        }
        write_json(metadata_path, metadata)
        print(f"Wrote {metadata['counts']['frames']:,} frames to {csv_path}", flush=True)

    return csv_path, metadata_path, metadata, model_path, data_path


def main() -> None:
    prepare_predictions(parse_args())


if __name__ == "__main__":
    main()
