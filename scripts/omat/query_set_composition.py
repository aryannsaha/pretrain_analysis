#!/usr/bin/env python
"""Element presence of the query datasets themselves, not of their neighbors.

``multiset_top5_presence.py`` characterises what each query set *retrieves* out
of OMAT24.  This one characterises what each query set *is*, so the two can be
read side by side: what you have, versus what it pulls from the pretraining set.

Sources:

  mad_train   ``data/processed/mad/train/mad_train.traj``, 76,483 frames.
              MAD is split train/val/test; only the train split is used.
  mpaloe_all  ``data/processed/mpaloe/mpaloe_all.traj``, 909,792 frames.
              MP ALOE has no train/val/test split in this repo, so this is the
              whole dataset -- the name says so rather than implying a split.

Every structure counts once; there is no retrieval multiplicity here, unlike the
neighbor tables.

Both are read from the trajectory rather than from the ``formula`` column of the
frames table, because that column holds the *reduced* formula: MP ALOE frame
"ScW" is a six-atom cell, so the column identifies elements correctly but cannot
give their counts, and atom fractions derived from it would be wrong.

Three checks, all against numbers somebody else recorded: frame count, total
atoms, and element inventory from the dataset builder's metadata, plus a
per-frame comparison of the trajectory's atom counts against the ``natoms``
column of the frames table, which is what would catch a frame-ordering slip
between the trajectory and the embedding rows.

    python scripts/omat/query_set_composition.py \
        --root /scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols

MAX_Z = 118
HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.environ.get("PRETRAIN_ANALYSIS_ROOT", HERE.parents[1]))
DEFAULT_OUT = "runs/omat_knn_probe/multiset_top5_presence"

QUERY_SETS = {
    "mad_train": {
        "label": "MAD (train split)",
        "traj": "data/processed/mad/train/mad_train.traj",
        "frames": "data/processed/mad/train/mad_train_frames.tsv",
        "metadata": "data/processed/mad/train/mad_train_metadata.json",
    },
    "mpaloe_all": {
        "label": "MP ALOE (all; no train split defined)",
        "traj": "data/processed/mpaloe/mpaloe_all.traj",
        "frames": "data/processed/mpaloe/mpaloe_all_frames.tsv",
        "metadata": "data/processed/mpaloe/mpaloe_all_metadata.json",
    },
}


def log(message):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def atomic_write_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def declared_natoms(root, spec):
    """The `natoms` column of the frames table, in frame order."""
    path = root / spec["frames"]
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "natoms" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} has no `natoms` column")
        return np.array([int(row["natoms"]) for row in reader], dtype=np.int64)


def read_from_traj(root, spec, name):
    from ase.io import Trajectory

    path = root / spec["traj"]
    log(f"{name}: reading {path}")
    rows = []
    with Trajectory(str(path), "r") as traj:
        total = len(traj)
        for i, atoms in enumerate(traj):
            counts = np.bincount(
                atoms.get_atomic_numbers(), minlength=MAX_Z + 1
            )[: MAX_Z + 1]
            rows.append(counts.astype(np.int32))
            if (i + 1) % 100000 == 0 or i + 1 == total:
                log(f"  {name}: {i + 1:,}/{total:,} frames")
    zcounts = np.asarray(rows, dtype=np.int32)

    # Frame-by-frame agreement with the frames table.  An aggregate atom total
    # can survive a reordering; this cannot, and the embedding rows are indexed
    # by the same frame order.
    declared = declared_natoms(root, spec)
    if declared.size != zcounts.shape[0]:
        raise SystemExit(
            f"{name}: trajectory has {zcounts.shape[0]:,} frames, frames table has "
            f"{declared.size:,}"
        )
    mismatched = np.nonzero(zcounts.sum(1) != declared)[0]
    if mismatched.size:
        first = int(mismatched[0])
        raise SystemExit(
            f"{name}: {mismatched.size:,} frames disagree with the frames table on "
            f"atom count; first at frame {first} "
            f"(trajectory {int(zcounts[first].sum())}, table {int(declared[first])})"
        )
    log(f"  {name}: frame-by-frame natoms agrees with {spec['frames']}")
    return zcounts


def check_against_metadata(root, spec, name, zcounts):
    """Compare frame count, atom total, and element inventory with the builder's."""
    meta_path = root / spec["metadata"]
    verdict = {"metadata": str(spec["metadata"])}
    if not meta_path.exists():
        verdict["status"] = "skipped"
        return verdict
    meta = json.load(open(meta_path))

    expected_frames = meta.get("sample_size")
    if expected_frames is not None and int(expected_frames) != zcounts.shape[0]:
        raise SystemExit(
            f"{name}: read {zcounts.shape[0]:,} frames, metadata says "
            f"{int(expected_frames):,}"
        )
    expected_atoms = meta.get("total_atoms")
    if expected_atoms is not None and int(expected_atoms) != int(zcounts.sum()):
        raise SystemExit(
            f"{name}: read {int(zcounts.sum()):,} atoms, metadata says "
            f"{int(expected_atoms):,}"
        )
    present = {chemical_symbols[z] for z in np.nonzero(zcounts.sum(0))[0]}
    listed = set(meta.get("elements", []))
    if listed and present != listed:
        raise SystemExit(
            f"{name}: element inventory differs from metadata; "
            f"only here {sorted(present - listed)}, only in metadata "
            f"{sorted(listed - present)}"
        )
    verdict.update(
        status="ok",
        frames=int(zcounts.shape[0]),
        total_atoms=int(zcounts.sum()),
        n_elements=len(present),
    )
    log(
        f"{name}: metadata check ok "
        f"({zcounts.shape[0]:,} frames, {int(zcounts.sum()):,} atoms, "
        f"{len(present)} elements)"
    )
    return verdict


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--datasets", nargs="+", default=list(QUERY_SETS), choices=list(QUERY_SETS)
    )
    parser.add_argument("--force", action="store_true", help="ignore cached zcounts")
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or root / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)

    populations, meta = {}, {}
    for name in args.datasets:
        spec = QUERY_SETS[name]
        zpath = out / f"{name}_dataset_zcounts.npy"
        if zpath.exists() and not args.force:
            zcounts = np.load(zpath)
            log(f"{name}: cached ({zcounts.shape[0]:,} frames)")
        else:
            zcounts = read_from_traj(root, spec, name)
            atomic_write_npy(zpath, zcounts)
            log(f"{name}: wrote {zpath.name}")

        verdict = check_against_metadata(root, spec, name, zcounts)
        presence = (zcounts > 0).mean(0)
        atoms = zcounts.sum(0).astype(np.float64)
        populations[f"{name}_dataset"] = (presence, atoms / atoms.sum())
        meta[f"{name}_dataset"] = {
            "label": spec["label"],
            "n_structures": int(zcounts.shape[0]),
            "total_atoms": int(zcounts.sum()),
            "source": spec.get("traj") or spec.get("frames"),
            "check": verdict,
        }

    names = list(populations)
    active = sorted({
        z for name in names
        for z in np.nonzero(populations[name][0])[0]
        if 1 <= z <= MAX_Z
    })
    rows = []
    for z in active:
        row = {"atomic_number": z, "symbol": chemical_symbols[z]}
        for name in names:
            presence, fraction = populations[name]
            row[f"presence_{name}"] = round(float(presence[z]), 8)
            row[f"atomfrac_{name}"] = round(float(fraction[z]), 8)
        rows.append(row)
    fields = ["atomic_number", "symbol"] + [
        f"{kind}_{name}" for name in names for kind in ("presence", "atomfrac")
    ]
    csv_path = out / "element_presence_datasets.csv"
    write_csv(csv_path, fields, rows)
    log(f"wrote {csv_path} ({len(rows)} elements, {len(names)} populations)")

    meta_path = out / "presence_metadata_datasets.json"
    with meta_path.open("w") as handle:
        json.dump(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "root": str(root),
                "weighting": "one structure, one observation",
                "populations": meta,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    log(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
