#!/usr/bin/env python
"""Build the full MatPES r2SCAN LMDB from ``data/raw/matpes/all.traj``, fixing the stress sign.

``all.traj`` is the whole of MatPES-R2SCAN-2025.1: 387,897 frames, one level of
theory (r2SCAN), atom counts 1-240.  It is the set the existing MatPES artifacts
are *not*: ``mace_matpes_test_full`` is a mixed PBE+r2SCAN set capped at 4 atoms
(278,632 frames, 161,262 of them r2SCAN), and the ``r2scan_flatiron`` builds are
atom-count-limited and force-filtered.

STRESS SIGN.  ``all.traj`` was written on 2026-07-24 with the kbar -> eV/A^3
magnitude conversion but without the sign flip, so its stresses are VASP
compression-positive.  Verified directly rather than inferred: over 4,000 frames
shared with the corrected ``mace_matpes_test_full`` LMDB, the energies are equal
and the stresses are the exact negation, 4000/4000.  This script applies
``stress *= -1`` on the way in and records both conventions, so the output
matches the ASE/MACE tension-positive convention used everywhere downstream.

SINGLE FUNCTIONAL.  The whole point of this set is that it is one level of
theory, so a second functional is treated as a fatal error rather than recorded
in a column.  ``--expect-functional`` pins it.

Schema matches ``build_matpes_test_full_lmdb.py`` (ASE-db style, zlib+orjson
records keyed by 1-based id, plus ``metadata``/``nextid``/``deleted_ids``) so the
same readers and the UMA latent extractor work unchanged.

    python scripts/matpes/build_matpes_all_lmdb.py
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
from ase.io import Trajectory

ROOT = Path("/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
DEFAULT_SOURCE = ROOT / "data/raw/matpes/all.traj"
DEFAULT_OUTPUT = ROOT / "data/processed/matpes/all.lmdb"

MARKER_KEY = "matpes_stress_convention"
MARKER_VALUE = "ase_tension_positive"
INFO_KEYS = ("source_index", "matpes_id", "formula_pretty", "functional")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expect-functional", default="r2SCAN",
                        help="fail if any frame carries a different functional")
    parser.add_argument("--negate-stress", dest="negate_stress", action="store_true", default=True,
                        help="apply stress *= -1 (default; the source is VASP-convention)")
    parser.add_argument("--no-negate-stress", dest="negate_stress", action="store_false",
                        help="source is already tension-positive")
    parser.add_argument("--commit-every", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pack(value) -> bytes:
    return zlib.compress(orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"source does not exist: {source}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = output.with_name(f"{output.stem}_manifest.csv")
    metadata_path = output.with_name(f"{output.stem}.metadata.json")

    print(f"hashing {source.name} ...", flush=True)
    source_sha = sha256(source)

    trajectory = Trajectory(str(source))
    total = len(trajectory)
    print(f"{total:,} frames; negate_stress={args.negate_stress}", flush=True)

    environment = lmdb.open(
        str(output),
        map_size=max(12 * source.stat().st_size, 1 << 34),
        subdir=False,
        meminit=False,
        map_async=True,
    )
    functional_counts: Counter[str] = Counter()
    atom_counts: Counter[int] = Counter()
    row_id = 0
    transaction = environment.begin(write=True)
    transaction.put(b"deleted_ids", pack([]))

    with manifest_path.open("w", newline="") as handle:
        manifest = csv.writer(handle)
        manifest.writerow(
            ["lmdb_id", "split", "source_row", "source_index", "matpes_id",
             "functional", "formula_pretty", "natoms"]
        )
        for source_row in range(total):
            atoms = trajectory[source_row]
            info = atoms.info
            functional = info.get("functional")
            if functional != args.expect_functional:
                raise SystemExit(
                    f"frame {source_row} has functional={functional!r}, expected "
                    f"{args.expect_functional!r}; this set is not single-theory"
                )
            results = atoms.calc.results if atoms.calc is not None else {}
            for key in ("energy", "forces", "stress"):
                if key not in results:
                    raise SystemExit(f"frame {source_row} is missing {key}")

            stress = np.asarray(results["stress"], dtype=float)
            if args.negate_stress:
                stress = -stress

            row_id += 1
            functional_counts[functional] += 1
            atom_counts[len(atoms)] += 1

            pairs = {key: info[key] for key in INFO_KEYS if key in info}
            pairs.update({
                "split": "all",
                "source_row": source_row,
                MARKER_KEY: MARKER_VALUE,
                "config_type": info.get("config_type", "Default"),
            })
            record = {
                "numbers": atoms.numbers.tolist(),
                "positions": atoms.positions.tolist(),
                "cell": np.asarray(atoms.cell).tolist(),
                "pbc": atoms.pbc.tolist(),
                "ctime": 0.0,
                "mtime": 0.0,
                "user": "build_matpes_all_lmdb",
                "energy": float(results["energy"]),
                "forces": np.asarray(results["forces"], dtype=float).tolist(),
                "stress": stress.reshape(-1).tolist(),
                "key_value_pairs": pairs,
            }
            if "initial_magmoms" in atoms.arrays:
                record["initial_magmoms"] = np.asarray(
                    atoms.arrays["initial_magmoms"], dtype=float).tolist()

            transaction.put(str(row_id).encode("ascii"), pack(record))
            manifest.writerow([
                row_id, "all", source_row, pairs.get("source_index", ""),
                pairs.get("matpes_id", ""), functional,
                pairs.get("formula_pretty", ""), len(atoms),
            ])
            if row_id % args.commit_every == 0:
                transaction.commit()
                transaction = environment.begin(write=True)
                print(f"  wrote {row_id:,}/{total:,} rows", flush=True)

    trajectory.close()

    db_metadata = {
        "format_version": 1,
        MARKER_KEY: MARKER_VALUE,
        "stress_unit": "eV/Angstrom^3",
        "stress_convention": "ASE/MACE: tension-positive",
        "stress_input_convention": "VASP: compression-positive" if args.negate_stress else MARKER_VALUE,
        "stress_correction": "stress *= -1" if args.negate_stress else "none",
        "energy_unit": "eV",
        "forces_unit": "eV/Angstrom",
        "rows": row_id,
        "splits": {"all": {"first_lmdb_id": 1, "last_lmdb_id": row_id, "rows": row_id,
                           "source": str(source)}},
        "functional_counts": dict(functional_counts),
        "manifest": str(manifest_path),
    }
    transaction.put(b"nextid", pack(row_id + 1))
    transaction.put(b"metadata", pack(db_metadata))
    transaction.commit()
    environment.sync()
    environment.close()

    if row_id != total:
        raise SystemExit(f"wrote {row_id:,} rows but the trajectory holds {total:,}")

    metadata_path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": source_sha,
        "output": str(output),
        "manifest": str(manifest_path),
        "rows": row_id,
        "functional_counts": dict(functional_counts),
        "atom_counts": {str(k): v for k, v in sorted(atom_counts.items())},
        **{k: db_metadata[k] for k in
           ("stress_unit", "stress_convention", "stress_input_convention",
            "stress_correction", "energy_unit", "forces_unit")},
    }, indent=2) + "\n")

    print(f"\nwrote {output}  ({row_id:,} rows)")
    print(f"wrote {manifest_path}")
    print(f"functionals: {dict(functional_counts)}")
    print(f"atom counts: min={min(atom_counts)} max={max(atom_counts)}")


if __name__ == "__main__":
    main()
