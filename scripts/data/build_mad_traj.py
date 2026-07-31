#!/usr/bin/env python
"""Convert the MAD dataset extxyz splits into ASE .traj files.

The UMA latent extractor (scripts/uma/extract_uma_latent_vectors.py) reads its
``--trajectory`` input with ``ase.io.trajectory.Trajectory``, which only accepts
ASE .traj files.  MAD ships as extended XYZ, so each native split
(train/val/test) is streamed into one .traj that becomes a "subdataset" for the
extractor, mirroring the MOF_OFF R2SCAN/PBE layout.

Per-frame provenance (subset label, atom count, pbc, source row) is written to a
sidecar TSV so that downstream PCA/kNN work can colour rows of the extracted
(N, 128) .npy by MAD subset without re-reading the structures.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    from ase.io import iread, read
    from ase.io.trajectory import Trajectory
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script needs ASE and NumPy. Run it with the pretrain_analysis_env "
        "python used for the rest of the UMA/MACE data preparation."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "data/raw/mad"
OUTPUT_ROOT = ROOT / "data/processed/mad"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream MAD extxyz splits into ASE .traj files."
    )
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional per-split smoke-test limit.")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser.parse_args()


def output_paths(output_root: Path, split: str) -> tuple[Path, Path, Path]:
    out_dir = output_root / split
    stem = f"mad_{split}"
    return (
        out_dir / f"{stem}.traj",
        out_dir / f"{stem}_metadata.json",
        out_dir / f"{stem}_frames.tsv",
    )


def prepare_outputs(paths: tuple[Path, ...], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Output files already exist. Use --overwrite to replace them:\n"
            + "\n".join(f"  {path}" for path in existing)
        )
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.unlink()


def validate(path: Path) -> None:
    atoms = read(path, "0")
    if atoms.calc is None or "energy" not in atoms.calc.results:
        raise SystemExit(f"{path} is missing calculator energy on the first frame")
    if np.asarray(atoms.calc.results["forces"]).shape != (len(atoms), 3):
        raise SystemExit(f"{path} has malformed forces on the first frame")


def convert_split(split: str, args: argparse.Namespace) -> dict:
    source = args.source_root / f"mad-{split}.xyz"
    traj_path, metadata_path, frames_path = output_paths(args.output_root, split)
    if not source.is_file():
        raise SystemExit(f"Missing MAD source file: {source}")

    prepare_outputs((traj_path, metadata_path, frames_path), args.overwrite)
    print(f"Writing MAD {split}: {source} -> {traj_path}", flush=True)

    count = 0
    subsets: dict[str, int] = {}
    pbc_counts: dict[str, int] = {}
    total_atoms = 0
    elements: set[str] = set()

    with Trajectory(str(traj_path), mode="w") as traj, frames_path.open("w") as frames:
        frames.write("frame\tsubset\tsplit\tnatoms\tpbc\n")
        for row, atoms in enumerate(iread(str(source), index=":", format="extxyz")):
            if args.limit is not None and count >= args.limit:
                break
            subset = atoms.info.get("subset", "<none>")
            pbc = "".join("T" if flag else "F" for flag in atoms.pbc)
            atoms.info.setdefault("mad_split", split)
            atoms.info["mad_split_row"] = row
            traj.write(atoms)
            frames.write(f"{row}\t{subset}\t{split}\t{len(atoms)}\t{pbc}\n")
            subsets[subset] = subsets.get(subset, 0) + 1
            pbc_counts[pbc] = pbc_counts.get(pbc, 0) + 1
            total_atoms += len(atoms)
            elements.update(atoms.get_chemical_symbols())
            count += 1
            if args.progress_every and count % args.progress_every == 0:
                print(f"  MAD {split}: {count:,} frames", flush=True)

    if count == 0:
        raise SystemExit(f"No structures were written from {source}")
    validate(traj_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "ASE .traj with SinglePointCalculator results",
        "dataset": "MAD",
        "split": split,
        "source_file": str(source.resolve()),
        "output": str(traj_path.resolve()),
        "frames_file": str(frames_path.resolve()),
        "sample_size": count,
        "total_atoms": total_atoms,
        "subset_counts": dict(sorted(subsets.items(), key=lambda kv: -kv[1])),
        "pbc_counts": dict(sorted(pbc_counts.items())),
        "n_elements": len(elements),
        "elements": sorted(elements),
        "energy_unit": "eV",
        "forces_unit": "eV/Angstrom",
        "stress_unit": "eV/Angstrom^3",
        "limit": args.limit,
        "output_order": "source extxyz order",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished MAD {split}: {count:,} frames, {total_atoms:,} atoms", flush=True)
    return metadata


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    args.output_root = args.output_root.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()

    splits = SPLITS if args.split == "all" else (args.split,)
    total = 0
    for split in splits:
        total += convert_split(split, args)["sample_size"]
    print(f"Done. Wrote {total:,} total frames.")


if __name__ == "__main__":
    main()
