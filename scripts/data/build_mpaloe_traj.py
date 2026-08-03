#!/usr/bin/env python
"""Convert the MP-ALOE JSONL dataset into a single ASE .traj file.

The UMA latent extractor (scripts/uma/extract_uma_latent_vectors.py) reads its
``--trajectory`` input with ``ase.io.trajectory.Trajectory``, which only accepts
ASE .traj files.  MP-ALOE ships as one JSON-lines file of pymatgen ``Structure``
records, so it is streamed into one .traj that becomes a single "subdataset"
("all") for the extractor -- MP-ALOE has no native train/val/test split, unlike
MAD.

Per-frame provenance (mp_aloe_id, formula, atom count, pbc) is written to a
sidecar TSV aligned to frame order so that downstream PCA/kNN work can annotate
rows of the extracted (N, 128) .npy without re-reading the structures.

Stress is deliberately NOT stored: MP-ALOE reports a 6-vector ``stress`` in the
VASP kbar / compression-positive convention, and converting it to the ASE
eV/Angstrom^3 tension-positive convention requires assumptions this script does
not need to make (UMA embeddings use only positions, cell, pbc and species).
Energy (eV) and forces (eV/Angstrom) are unambiguous and are preserved.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read
    from ase.io.trajectory import Trajectory
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script needs ASE and NumPy. Run it with the pretrain_analysis_env "
        "python used for the rest of the UMA/MACE data preparation."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/mpaloe/MP_ALOE_data.jsonl"
OUTPUT_ROOT = ROOT / "data/processed/mpaloe"
DATASET = "mpaloe_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream the MP-ALOE JSONL into one ASE .traj file."
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit.")
    parser.add_argument("--progress-every", type=int, default=50_000)
    return parser.parse_args()


def species_symbol(site: dict) -> str:
    """Highest-occupancy element for a pymatgen site (MOF_OFF convention)."""
    species = site.get("species") or []
    if species:
        return max(species, key=lambda item: float(item.get("occu", 1.0)))["element"]
    if "species_string" in site:
        return site["species_string"]
    if "label" in site:
        return "".join(char for char in site["label"] if char.isalpha())
    raise ValueError(f"Could not determine species for site: {site}")


def record_to_atoms(record: dict, row: int) -> Atoms:
    structure = record["structure"]
    sites = structure["sites"]
    lattice = structure["lattice"]
    atoms = Atoms(
        symbols=[species_symbol(site) for site in sites],
        positions=np.asarray([site["xyz"] for site in sites], dtype=float),
        cell=np.asarray(lattice["matrix"], dtype=float),
        pbc=lattice.get("pbc", [True, True, True]),
    )
    atoms.info["mpaloe_row"] = row
    identifier = record.get("mp_aloe_id")
    if identifier is not None:
        atoms.info["mp_aloe_id"] = identifier

    energy = record.get("energy")
    forces = record.get("forces")
    if energy is not None and forces is not None:
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(energy),
            forces=np.asarray(forces, dtype=float).reshape(len(atoms), 3),
        )
    return atoms


def output_paths(output_root: Path) -> tuple[Path, Path, Path]:
    return (
        output_root / f"{DATASET}.traj",
        output_root / f"{DATASET}_metadata.json",
        output_root / f"{DATASET}_frames.tsv",
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


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    args.output_root = args.output_root.expanduser().resolve()
    args.source = args.source.expanduser().resolve()
    if not args.source.is_file():
        raise SystemExit(f"Missing MP-ALOE source file: {args.source}")

    traj_path, metadata_path, frames_path = output_paths(args.output_root)
    prepare_outputs((traj_path, metadata_path, frames_path), args.overwrite)
    print(f"Writing MP-ALOE: {args.source} -> {traj_path}", flush=True)

    count = 0
    total_atoms = 0
    elements: set[str] = set()
    pbc_counts: dict[str, int] = {}

    with Trajectory(str(traj_path), mode="w") as traj, frames_path.open("w") as frames:
        frames.write("frame\tmp_aloe_id\tformula\tnatoms\tpbc\n")
        with args.source.open() as handle:
            for row, line in enumerate(handle):
                if not line.strip():
                    continue
                if args.limit is not None and count >= args.limit:
                    break
                record = json.loads(line)
                atoms = record_to_atoms(record, row)
                pbc = "".join("T" if flag else "F" for flag in atoms.pbc)
                formula = record.get("formula_pretty") or atoms.get_chemical_formula()
                traj.write(atoms)
                frames.write(
                    f"{row}\t{record.get('mp_aloe_id', '')}\t{formula}\t"
                    f"{len(atoms)}\t{pbc}\n"
                )
                pbc_counts[pbc] = pbc_counts.get(pbc, 0) + 1
                total_atoms += len(atoms)
                elements.update(atoms.get_chemical_symbols())
                count += 1
                if args.progress_every and count % args.progress_every == 0:
                    print(f"  MP-ALOE: {count:,} frames", flush=True)

    if count == 0:
        raise SystemExit(f"No records were written from {args.source}")
    validate(traj_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format": "ASE .traj with SinglePointCalculator results",
        "dataset": "MP-ALOE",
        "functional": "r2SCAN",
        "source_file": str(args.source),
        "output": str(traj_path),
        "frames_file": str(frames_path),
        "sample_size": count,
        "total_atoms": total_atoms,
        "pbc_counts": dict(sorted(pbc_counts.items())),
        "n_elements": len(elements),
        "elements": sorted(elements),
        "energy_unit": "eV",
        "forces_unit": "eV/Angstrom",
        "stress": "not stored (source is VASP kbar/compression-positive; unused for UMA embeddings)",
        "limit": args.limit,
        "output_order": "source JSONL line order",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished MP-ALOE: {count:,} frames, {total_atoms:,} atoms", flush=True)


if __name__ == "__main__":
    main()
