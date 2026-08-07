#!/usr/bin/env python
"""Negate MatPES extxyz stresses once, turning VASP compression-positive into ASE tension-positive.

`data/processed/matpes/r2scan/mace_matpes_test_full` was written before the MatPES
stress-sign fix, so its `mace_stress` entries carry the VASP convention while the
corrected reference set in `data/processed/matpes/r2scan_flatiron` carries the ASE
convention. Both are already in eV/Angstrom^3, so the correction is exactly `-1`:
the full conversion from raw MatPES kbar is `-0.1 * GPa`, and only the sign is missing.

The rewrite is surgical -- only the `mace_stress` field of each comment line changes,
every other byte is preserved -- and idempotent via a `matpes_stress_convention` marker.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
DEFAULT_INPUTS = (
    ROOT / "data/processed/matpes/r2scan/mace_matpes_test_full/train.extxyz",
    ROOT / "data/processed/matpes/r2scan/mace_matpes_test_full/val.extxyz",
)

MARKER_KEY = "matpes_stress_convention"
MARKER_VALUE = "ase_tension_positive"
STRESS_KEY = "mace_stress"
JSON_PREFIX = "_JSON "

STRESS_FIELD = re.compile(rf'{STRESS_KEY}="([^"]*)"')
MARKER_FIELD = re.compile(rf"{MARKER_KEY}=(\S+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument(
        "--quarantine-dir",
        type=Path,
        default=None,
        help="Where to move the uncorrected originals (default: <input dir>/quarantine_pre_stress_sign_fix).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def negate(value: str) -> str:
    """Negate a `_JSON [[...]]` 3x3 stress, preserving round-trippable float repr."""
    if not value.startswith(JSON_PREFIX):
        raise ValueError(f"unexpected {STRESS_KEY} encoding: {value[:40]!r}")
    matrix = json.loads(value[len(JSON_PREFIX) :])
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError(f"expected a 3x3 stress, got {value[:60]!r}")
    # `+ 0.0` keeps a negated zero from serialising as "-0.0".
    flipped = [[-float(entry) + 0.0 for entry in row] for row in matrix]
    return JSON_PREFIX + json.dumps(flipped)


def rewrite(source: Path, destination: Path) -> dict:
    frames = 0
    with source.open() as reader, destination.open("w") as writer:
        while True:
            count_line = reader.readline()
            if not count_line:
                break
            count = int(count_line)
            comment = reader.readline()
            if not comment:
                raise ValueError(f"{source}: truncated after frame {frames}")

            marker = MARKER_FIELD.search(comment)
            if marker is not None:
                raise SystemExit(
                    f"{source}: frame {frames} already carries {MARKER_KEY}={marker.group(1)}; "
                    "refusing to negate an already-corrected file"
                )
            match = STRESS_FIELD.search(comment)
            if match is None:
                raise ValueError(f"{source}: frame {frames} has no {STRESS_KEY}")

            corrected = (
                comment[: match.start(1)]
                + negate(match.group(1))
                + comment[match.end(1) :].rstrip("\n")
                + f" {MARKER_KEY}={MARKER_VALUE}\n"
            )
            writer.write(count_line)
            writer.write(corrected)
            for _ in range(count):
                line = reader.readline()
                if not line:
                    raise ValueError(f"{source}: truncated inside frame {frames}")
                writer.write(line)
            frames += 1
    return {"frames": frames}


def main() -> None:
    args = parse_args()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "correction": "stress *= -1",
        "stress_unit": "eV/Angstrom^3",
        "stress_input_convention": "VASP: compression-positive",
        "stress_output_convention": "ASE/MACE: tension-positive",
        "matpes_kbar_to_ase_factor": "-0.1 * GPa",
        "files": {},
    }

    for source in args.inputs:
        if not source.is_file():
            raise SystemExit(f"input does not exist: {source}")
        quarantine = args.quarantine_dir or source.parent / "quarantine_pre_stress_sign_fix"
        target = quarantine / source.name
        if target.exists():
            raise SystemExit(f"quarantine copy already exists, refusing to overwrite: {target}")

        temporary = source.with_name(f".{source.name}.stress-sign.tmp")
        print(f"correcting {source}", flush=True)
        stats = rewrite(source, temporary)
        stats["quarantined_original"] = str(target)
        report["files"][str(source)] = stats
        print(f"  frames: {stats['frames']:,}", flush=True)

        if args.dry_run:
            temporary.unlink()
            print("  dry run: discarded rewrite", flush=True)
            continue

        quarantine.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        temporary.replace(source)
        print(f"  original preserved at {target}", flush=True)

    if not args.dry_run and args.inputs:
        audit = Path(args.inputs[0]).parent / "STRESS_SIGN_FIX.json"
        audit.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {audit}", flush=True)
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
