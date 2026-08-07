#!/usr/bin/env python
"""Combine the MatPES mace_matpes_test_full extxyz splits into one MACE/ASE-style LMDB.

Rows are written train-then-val in file order, so LMDB ID `n` is the `n`-th frame of
the concatenation. Every row records its split, source row, MatPES id and functional in
`key_value_pairs`, and a sibling manifest CSV repeats that mapping so downstream
matrices (for example UMA latents) can be filtered by functional without reopening the
database.

The source files must already carry the `matpes_stress_convention=ase_tension_positive`
marker written by `scripts/matpes/fix_extxyz_stress_sign.py`; this script refuses to run
otherwise so a stale VASP-convention stress can never reach the LMDB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import lmdb
import numpy as np
import orjson
from ase.io.extxyz import _read_xyz_frame

ROOT = Path("/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
SOURCE_DIR = ROOT / "data/processed/matpes/r2scan/mace_matpes_test_full"

MARKER_KEY = "matpes_stress_convention"
MARKER_VALUE = "ase_tension_positive"
INFO_KEYS = ("source_index", "matpes_id", "formula_pretty", "functional")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--commit-every", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def frames(path: Path):
    with path.open() as handle:
        while line := handle.readline():
            yield _read_xyz_frame(handle, int(line))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pack(value) -> bytes:
    return zlib.compress(orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY))


def main() -> None:
    args = parse_args()
    output = args.output or args.source_dir / "matpes_r2scan.lmdb"
    manifest_path = output.with_name(f"{output.stem}_manifest.csv")
    metadata_path = output.with_name(f"{output.stem}.metadata.json")

    sources = [args.source_dir / f"{split}.extxyz" for split in args.splits]
    for path in sources:
        if not path.is_file():
            raise SystemExit(f"source does not exist: {path}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists (use --overwrite): {output}")

    total_bytes = sum(path.stat().st_size for path in sources)
    environment = lmdb.open(
        str(output),
        map_size=max(8 * total_bytes, 1 << 33),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    functional_counts: Counter[str] = Counter()
    split_ranges: dict[str, dict] = {}
    functional_ranges: list[dict] = []
    atom_counts: Counter[int] = Counter()
    row_id = 0
    transaction = environment.begin(write=True)
    transaction.put(b"deleted_ids", pack([]))

    with manifest_path.open("w", newline="") as handle:
        manifest = csv.writer(handle)
        manifest.writerow(
            ["lmdb_id", "split", "source_row", "source_index", "matpes_id", "functional", "formula_pretty", "natoms"]
        )
        for split, path in zip(args.splits, sources):
            split_start = row_id + 1
            for source_row, atoms in enumerate(frames(path)):
                info = atoms.info
                marker = info.get(MARKER_KEY)
                if marker != MARKER_VALUE:
                    raise SystemExit(
                        f"{path}: frame {source_row} has {MARKER_KEY}={marker!r}, expected {MARKER_VALUE!r}; "
                        "run scripts/matpes/fix_extxyz_stress_sign.py first"
                    )
                row_id += 1
                functional = info.get("functional")
                functional_counts[functional] += 1
                atom_counts[len(atoms)] += 1
                if not functional_ranges or functional_ranges[-1]["functional"] != functional or functional_ranges[-1]["split"] != split:
                    functional_ranges.append(
                        {"split": split, "functional": functional, "first_lmdb_id": row_id, "last_lmdb_id": row_id}
                    )
                functional_ranges[-1]["last_lmdb_id"] = row_id

                pairs = {key: info[key] for key in INFO_KEYS if key in info}
                pairs.update(
                    {
                        "split": split,
                        "source_row": source_row,
                        MARKER_KEY: MARKER_VALUE,
                        "config_type": info.get("config_type", "Default"),
                    }
                )
                record = {
                    "numbers": atoms.numbers.tolist(),
                    "positions": atoms.positions.tolist(),
                    "cell": np.asarray(atoms.cell).tolist(),
                    "pbc": atoms.pbc.tolist(),
                    "ctime": 0.0,
                    "mtime": 0.0,
                    "user": "build_matpes_test_full_lmdb",
                    "energy": float(info["mace_energy"]),
                    "forces": np.asarray(atoms.arrays["mace_forces"], dtype=float).tolist(),
                    "stress": np.asarray(info["mace_stress"], dtype=float).reshape(-1).tolist(),
                    "key_value_pairs": pairs,
                }
                if "initial_magmoms" in atoms.arrays:
                    record["initial_magmoms"] = np.asarray(atoms.arrays["initial_magmoms"], dtype=float).tolist()

                transaction.put(str(row_id).encode("ascii"), pack(record))
                manifest.writerow(
                    [
                        row_id,
                        split,
                        source_row,
                        pairs.get("source_index", ""),
                        pairs.get("matpes_id", ""),
                        functional,
                        pairs.get("formula_pretty", ""),
                        len(atoms),
                    ]
                )
                if row_id % args.commit_every == 0:
                    transaction.commit()
                    transaction = environment.begin(write=True)
                    print(f"  wrote {row_id:,} rows", flush=True)
            split_ranges[split] = {
                "first_lmdb_id": split_start,
                "last_lmdb_id": row_id,
                "rows": row_id - split_start + 1,
                "source": str(path),
            }
            print(f"finished {split}: rows {split_start:,}-{row_id:,}", flush=True)

    transaction.put(b"nextid", pack(row_id + 1))
    transaction.put(
        b"metadata",
        pack(
            {
                "format_version": 1,
                MARKER_KEY: MARKER_VALUE,
                "stress_unit": "eV/Angstrom^3",
                "stress_convention": "ASE/MACE: tension-positive",
                "energy_unit": "eV",
                "forces_unit": "eV/Angstrom",
                "rows": row_id,
                "splits": split_ranges,
                "functional_counts": dict(functional_counts),
                "functional_ranges": functional_ranges,
                "manifest": str(manifest_path),
            }
        ),
    )
    transaction.commit()
    environment.sync()
    environment.close()

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "manifest": str(manifest_path),
        "rows": row_id,
        "splits": split_ranges,
        "functional_counts": dict(functional_counts),
        "functional_ranges": functional_ranges,
        "atom_counts": {str(key): atom_counts[key] for key in sorted(atom_counts)},
        "sources": {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in sources},
        MARKER_KEY: MARKER_VALUE,
        "stress_unit": "eV/Angstrom^3",
        "energy_unit": "eV",
        "forces_unit": "eV/Angstrom",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in ("output", "rows", "functional_counts", "splits")}, indent=2))
    print(f"wrote {output}\nwrote {manifest_path}\nwrote {metadata_path}")


if __name__ == "__main__":
    main()
