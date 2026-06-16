#!/usr/bin/env python
"""Build ASE trajectory splits from selected MatPES source indices."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.units import GPa
from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDICES = ROOT / "subsampled_pbe_nsites_4.npy"
DEFAULT_DATASET = Path(
    "/scratch/gpfs/ROSENGROUP/ng8249/mof_off_jsons/"
    "matpes_mpaloe/MatPES-R2SCAN-2025.1.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/matpes/r2scan"

SEED = 20260612
TRAIN_FRACTION = 0.8
STRESS_INPUT_UNIT = "kbar"
STRESS_OUTPUT_UNIT = "eV/Angstrom^3"
KBAR_TO_EV_PER_A3 = 0.1 * GPa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select MatPES records by source index, convert them to ASE .traj, "
            "and write full plus half-size 80/20 train/validation splits."
        )
    )
    parser.add_argument("--indices", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument(
        "--record-source",
        choices=("auto", "dataset", "index-file"),
        default="auto",
        help=(
            "Where to read selected records from. 'auto' uses dataset_path when "
            "all indices fit and falls back to dict values stored in the .npy."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit applied after sorting source indices.",
    )
    return parser.parse_args()


def looks_like_record(value: object) -> bool:
    return (
        isinstance(value, dict)
        and "structure" in value
        and "energy" in value
        and "forces" in value
        and "stress" in value
    )


def load_selected_records(
    path: Path, limit: int | None = None
) -> tuple[np.ndarray, dict[int, dict] | None]:
    selected = np.load(path, allow_pickle=True).item()
    if isinstance(selected, dict):
        selected_by_index = {int(index): value for index, value in selected.items()}
        indices = np.array(sorted(selected_by_index), dtype=int)
        if limit is not None:
            indices = indices[:limit]
        records = {
            int(index): selected_by_index[int(index)]
            for index in indices
            if looks_like_record(selected_by_index[int(index)])
        }
        records_by_index = records if len(records) == len(indices) else None
    else:
        indices = np.array(sorted(int(index) for index in selected), dtype=int)
        if limit is not None:
            indices = indices[:limit]
        records_by_index = None
    if indices.size == 0:
        raise SystemExit(f"No source indices found in {path}")
    return indices, records_by_index


def structure_from_record(record: dict) -> Structure:
    structure = record["structure"]
    if isinstance(structure, Structure):
        return structure
    if isinstance(structure, dict):
        return Structure.from_dict(structure)
    raise TypeError(f"Unsupported structure type: {type(structure)!r}")


def stress_to_matrix(stress: Iterable[float]) -> np.ndarray:
    stress = np.asarray(stress, dtype=float)
    if stress.shape == (3, 3):
        matrix = stress
    elif stress.shape == (6,):
        xx, yy, zz, yz, xz, xy = stress
        matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)
    else:
        raise ValueError(f"Expected stress shape (6,) or (3, 3), got {stress.shape}")
    return matrix * KBAR_TO_EV_PER_A3


def record_to_atoms(record: dict, source_index: int) -> Atoms:
    atoms = AseAtomsAdaptor.get_atoms(structure_from_record(record))
    energy = float(record["energy"])
    forces = np.asarray(record["forces"], dtype=float).reshape(len(atoms), 3)
    stress = stress_to_matrix(record["stress"])

    atoms.info["source_index"] = int(source_index)
    for key in ("matpes_id", "formula_pretty", "functional"):
        if key in record and record[key] is not None:
            atoms.info[key] = record[key]
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=energy,
        forces=forces,
        stress=stress,
    )
    return atoms


def write_indices(path: Path, indices: np.ndarray) -> None:
    path.write_text("\n".join(str(int(index)) for index in indices) + "\n")


def write_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, sort_keys=False) + "\n")


def metadata_for(
    name: str,
    output: Path,
    sample_indices_file: Path,
    sample_indices: np.ndarray,
    args: argparse.Namespace,
    source_frame_count: int | None,
    desired_frame_count: int,
    split: str,
    record_source: str,
) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "ASE .traj with SinglePointCalculator results",
        "output": str(output.resolve()),
        "sample_indices_file": str(sample_indices_file.resolve()),
        "sample_size": int(len(sample_indices)),
        "split": split,
        "split_name": name,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "source": str(args.dataset_path.resolve()),
        "source_frame_count": (
            int(source_frame_count) if source_frame_count is not None else None
        ),
        "record_source": record_source,
        "desired_frame_count": int(desired_frame_count),
        "indices_source": str(args.indices.resolve()),
        "output_order": "source-index order for all files; seeded shuffled order for split files",
        "energy_unit": "eV",
        "forces_unit": "eV/Angstrom",
        "stress_input_unit": STRESS_INPUT_UNIT,
        "stress_output_unit": STRESS_OUTPUT_UNIT,
        "stress_conversion_factor": KBAR_TO_EV_PER_A3,
    }


def write_dataset(
    name: str,
    frames: list,
    source_indices: np.ndarray,
    frame_positions: np.ndarray,
    args: argparse.Namespace,
    source_frame_count: int | None,
    desired_frame_count: int,
    split: str,
    record_source: str,
) -> None:
    output = args.output_dir / f"{name}.traj"
    sample_indices_file = args.output_dir / f"{name}_sample_indices.txt"
    metadata_file = args.output_dir / f"{name}_metadata.json"
    selected_indices = source_indices[frame_positions]
    selected_frames = [frames[int(position)] for position in frame_positions]

    write(output, selected_frames)
    write_indices(sample_indices_file, selected_indices)
    write_metadata(
        metadata_file,
        metadata_for(
            name,
            output,
            sample_indices_file,
            selected_indices,
            args,
            source_frame_count,
            desired_frame_count,
            split,
            record_source,
        ),
    )

    first = read(output, "0")
    if first.calc is None:
        raise SystemExit(f"{output} is missing calculator results")
    required = {"energy", "forces", "stress"}
    missing = required - set(first.calc.results)
    if missing:
        raise SystemExit(f"{output} is missing calculator results: {sorted(missing)}")
    print(f"Wrote {len(selected_frames):>7} frames -> {output}")


def split_positions(
    n_frames: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise SystemExit("--train-fraction must be between 0 and 1")
    order = rng.permutation(n_frames)
    n_train = int(train_fraction * n_frames)
    if n_train == 0 or n_train == n_frames:
        raise SystemExit("Need at least one frame in both train and validation splits")
    return order[:n_train], order[n_train:]


def records_from_dataset(
    args: argparse.Namespace,
    source_indices: np.ndarray,
) -> tuple[list[dict], int]:
    print(f"Loading {args.dataset_path}")
    structures = loadfn(args.dataset_path)
    source_frame_count = len(structures)
    max_index = int(source_indices.max())
    if max_index >= source_frame_count:
        raise IndexError(
            f"Index {max_index} is outside source dataset with "
            f"{source_frame_count} records"
        )
    return [structures[int(source_index)] for source_index in source_indices], source_frame_count


def records_from_index_file(
    source_indices: np.ndarray, records_by_index: dict[int, dict] | None
) -> list[dict]:
    if records_by_index is None:
        raise SystemExit(
            "The index file does not contain usable MatPES records; provide a "
            "matching --dataset-path or use a .npy dict whose values contain "
            "structure, energy, forces, and stress."
        )
    return [records_by_index[int(source_index)] for source_index in source_indices]


def selected_records(
    args: argparse.Namespace,
    source_indices: np.ndarray,
    records_by_index: dict[int, dict] | None,
) -> tuple[list[dict], int | None, str]:
    if args.record_source == "index-file":
        print(f"Using MatPES records stored in {args.indices}")
        return (
            records_from_index_file(source_indices, records_by_index),
            None,
            "index-file values",
        )

    try:
        records, source_frame_count = records_from_dataset(args, source_indices)
        return records, source_frame_count, "dataset_path"
    except IndexError as exc:
        if args.record_source == "dataset":
            raise SystemExit(str(exc)) from exc
        print(f"{exc}; falling back to records stored in {args.indices}")
        return records_from_index_file(source_indices, records_by_index), None, "index-file values"


def main() -> None:
    args = parse_args()
    args.indices = args.indices.expanduser().resolve()
    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_indices, records_by_index = load_selected_records(args.indices, args.limit)
    records, source_frame_count, record_source = selected_records(
        args, source_indices, records_by_index
    )

    print(f"Converting {len(source_indices)} selected MatPES records")
    frames = [
        record_to_atoms(record, int(source_index))
        for record, source_index in zip(records, source_indices, strict=True)
    ]
    positions = np.arange(len(frames), dtype=int)
    rng = np.random.default_rng(args.seed)

    train_positions, val_positions = split_positions(
        len(frames), args.train_fraction, rng
    )
    half_positions = np.sort(rng.permutation(len(frames))[: len(frames) // 2])
    half_train_rel, half_val_rel = split_positions(
        len(half_positions), args.train_fraction, rng
    )
    half_train_positions = half_positions[half_train_rel]
    half_val_positions = half_positions[half_val_rel]

    write_dataset(
        "all",
        frames,
        source_indices,
        positions,
        args,
        source_frame_count,
        len(frames),
        "all selected data",
        record_source,
    )
    write_dataset(
        "train",
        frames,
        source_indices,
        train_positions,
        args,
        source_frame_count,
        len(frames),
        "80% train split of all selected data",
        record_source,
    )
    write_dataset(
        "val",
        frames,
        source_indices,
        val_positions,
        args,
        source_frame_count,
        len(frames),
        "20% validation split of all selected data",
        record_source,
    )
    write_dataset(
        "half",
        frames,
        source_indices,
        half_positions,
        args,
        source_frame_count,
        len(frames),
        "seeded half-size subset of all selected data",
        record_source,
    )
    write_dataset(
        "half_train",
        frames,
        source_indices,
        half_train_positions,
        args,
        source_frame_count,
        len(frames),
        "80% train split of half-size subset",
        record_source,
    )
    write_dataset(
        "half_val",
        frames,
        source_indices,
        half_val_positions,
        args,
        source_frame_count,
        len(frames),
        "20% validation split of half-size subset",
        record_source,
    )


if __name__ == "__main__":
    main()
