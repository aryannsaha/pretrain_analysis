#!/usr/bin/env python
"""Element presence of the top-K nearest OMAT24 neighbors, for several query sets.

``mof_omat_neighbor_chemistry.py`` answers this for MOF-off R2SCAN-D4 only, and
its extract step is expensive because it also pulls two 50k-row OMAT24 baselines
and the full query set.  This script does the one piece needed to put several
query sets on the same periodic table:

  for each query set, take the top-K neighbor slots out of ``indices.npy``,
  read the distinct retrieved OMAT24 structures out of the ``.aselmdb`` shards,
  and reduce them to a per-element presence rate.

Query sets (all three kNN runs share one OMAT24 reference row space and one
25-PC basis -- verified by their identical ``omat_calibration_thresholds_*``
entries -- so their global rows and their presence rates are comparable):

  mof_off   MOF-off R2SCAN-D4 train      80,643 queries
  mad       MAD                          95,595 queries
  mpaloe    MP ALOE                     909,792 queries

Weighting matches ``Profile.presence_rate`` in ``mof_omat_neighbor_chemistry``:
a neighbor slot is the unit of observation, so an OMAT24 structure retrieved by
20 different queries counts 20 times.  This is a property of the *retrieved
set*, not of the distinct structures in it; ``--distinct`` switches to the
unweighted variant.

The OMAT24 background column is not recomputed -- it is the same 50k uniform
random rows already extracted by ``mof_omat_neighbor_chemistry.py extract``, and
it is query-set independent, so all datasets share one baseline.

Extraction is cached: a query set whose ``*_zcounts.npy`` already exists is
skipped unless ``--force`` is given.

    python scripts/omat/multiset_top5_presence.py \
        --root /scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis \
        --geometry pc25 --depth 5
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols

HERE = Path(__file__).resolve().parent
MAX_Z = 118

DEFAULT_ROOT = Path(
    os.environ.get("PRETRAIN_ANALYSIS_ROOT", HERE.parents[1])
)

# knn run directory per (query set, geometry), relative to root.
QUERY_SETS = {
    "mof_off": {
        "label": "MOF-off R2SCAN-D4",
        "runs": {
            "raw128": "runs/omat_knn_probe/omat_knn_raw128_mof_off/MOF_off_R2SCAN",
            "pc25": "runs/omat_knn_probe/omat_knn_pc25_mof_off/MOF_off_R2SCAN",
        },
    },
    "mad": {
        "label": "MAD",
        "runs": {
            "raw128": "runs/omat_knn_probe/omat_knn_raw128_mad/MAD_all",
            "pc25": "runs/omat_knn_probe/omat_knn_pc25_mad/MAD_all",
        },
    },
    "mpaloe": {
        "label": "MP ALOE",
        "runs": {
            "raw128": "runs/omat_knn_probe/omat_knn_raw128_mpaloe/MPALOE_all",
            "pc25": "runs/omat_knn_probe/omat_knn_pc25_mpaloe/MPALOE_all",
        },
    },
}

# Reused from the MOF-off extract; 50k uniform random OMAT24 rows, query-set
# independent, so every dataset is differenced against the same background.
OMAT_BG = "runs/omat_knn_probe/mof_omat_neighbor_chemistry/structures/omat_bg_global_zcounts.npy"
DEFAULT_OUT = "runs/omat_knn_probe/multiset_top5_presence"


def load_chemistry_module():
    """Import the sibling script for its verified OMAT24 row-mapping helpers."""
    path = HERE / "mof_omat_neighbor_chemistry.py"
    spec = importlib.util.spec_from_file_location("mof_omat_neighbor_chemistry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def log(message):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def atomic_write_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def neighbor_slots(root, dataset, geometry, depth):
    """Distinct retrieved rows and their retrieval multiplicity, for top-`depth`."""
    run = root / QUERY_SETS[dataset]["runs"][geometry]
    idx = np.load(run / "indices.npy", mmap_mode="r")
    if idx.ndim != 2:
        raise SystemExit(f"{dataset}/{geometry}: indices.npy is not 2-d ({idx.shape})")
    if depth > idx.shape[1]:
        raise SystemExit(
            f"{dataset}/{geometry}: asked for top-{depth} but only "
            f"{idx.shape[1]} neighbors were stored"
        )
    flat = np.asarray(idx[:, :depth], dtype=np.int64).ravel()
    unique_rows, inverse = np.unique(flat, return_inverse=True)
    weights = np.bincount(inverse, minlength=unique_rows.size).astype(np.float64)
    return unique_rows, weights, int(idx.shape[0])


def extract_dataset(chem, root, out, dataset, geometry, depth, force):
    """Read the distinct top-`depth` neighbors out of the shards; cache zcounts."""
    stem = f"{dataset}_{geometry}_top{depth}"
    zpath = out / f"{stem}_zcounts.npy"
    wpath = out / f"{stem}_weights.npy"
    rpath = out / f"{stem}_rows.npy"

    unique_rows, weights, n_queries = neighbor_slots(root, dataset, geometry, depth)
    if zpath.exists() and wpath.exists() and rpath.exists() and not force:
        cached_rows = np.load(rpath)
        if np.array_equal(cached_rows, unique_rows):
            log(f"{stem}: cached ({unique_rows.size:,} distinct rows)")
            return np.load(zpath), np.load(wpath), n_queries
        log(f"{stem}: cache does not match current indices, re-extracting")

    log(
        f"{stem}: {n_queries:,} queries x top-{depth} = {int(weights.sum()):,} slots, "
        f"{unique_rows.size:,} distinct OMAT24 rows"
    )
    manifest, starts, _ = chem.read_manifest(root)
    _, counts = chem.read_omat_rows(
        root, unique_rows, manifest, starts, stem, stem
    )
    zcounts = np.asarray(counts, dtype=np.int32)
    if zcounts.shape[0] != unique_rows.size:
        raise SystemExit(
            f"{stem}: extracted {zcounts.shape[0]} structures for "
            f"{unique_rows.size} requested rows"
        )

    atomic_write_npy(rpath, unique_rows)
    atomic_write_npy(wpath, weights)
    atomic_write_npy(zpath, zcounts)
    log(f"{stem}: wrote {zpath.name}")
    return zcounts, weights, n_queries


def presence_and_fraction(zcounts, weights):
    """Slot-weighted share of structures containing each Z, and atom fraction."""
    total = weights.sum()
    if total <= 0:
        raise SystemExit("empty population")
    presence = (weights[:, None] * (zcounts > 0)).sum(0) / total
    atoms = (weights[:, None] * zcounts).sum(0)
    return presence, atoms / atoms.sum()


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def crosscheck_mof_off(root, presence, geometry, depth):
    """Compare the mof_off column against the already-published MOF-off table.

    Same numbers by two independent code paths: this script re-derives them from
    indices.npy plus a fresh shard read, the reference came from the full
    neighbor-union extract.  A mismatch means the row mapping or the weighting
    drifted, so it is worth failing loudly on.
    """
    reference = (
        root / "runs/omat_knn_probe/mof_omat_neighbor_chemistry/analysis"
        / "element_presence_and_abundance.csv"
    )
    column = f"presence_{geometry}_top{depth}"
    if not reference.exists():
        return {"status": "skipped", "reason": f"missing {reference}"}
    with reference.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or column not in rows[0]:
        return {"status": "skipped", "reason": f"no column {column}"}

    worst_z, worst = None, 0.0
    for row in rows:
        z = int(row["atomic_number"])
        delta = abs(float(row[column]) - float(presence[z]))
        if delta > worst:
            worst_z, worst = z, delta
    verdict = {
        "status": "ok" if worst < 5e-6 else "MISMATCH",
        "column": column,
        "max_abs_delta": worst,
        "worst_element": chemical_symbols[worst_z] if worst_z else None,
    }
    log(
        f"crosscheck vs published {column}: max |delta| = {worst:.3e} "
        f"({verdict['status']})"
    )
    if verdict["status"] != "ok":
        raise SystemExit(
            f"mof_off {column} disagrees with the published table by {worst:.3e} "
            f"at {verdict['worst_element']}; the row mapping or weighting changed"
        )
    return verdict


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--geometry", default="pc25", choices=["pc25", "raw128"])
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument(
        "--datasets", nargs="+", default=list(QUERY_SETS),
        choices=list(QUERY_SETS),
    )
    parser.add_argument(
        "--distinct", action="store_true",
        help="weight each retrieved structure once instead of once per slot",
    )
    parser.add_argument("--force", action="store_true", help="ignore cached zcounts")
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or root / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)
    chem = load_chemistry_module()

    populations, meta = {}, {}
    for dataset in args.datasets:
        zcounts, weights, n_queries = extract_dataset(
            chem, root, out, dataset, args.geometry, args.depth, args.force
        )
        if args.distinct:
            weights = np.ones_like(weights)
        presence, fraction = presence_and_fraction(zcounts, weights)
        name = f"{dataset}_{args.geometry}_top{args.depth}"
        populations[name] = (presence, fraction)
        meta[name] = {
            "query_set": QUERY_SETS[dataset]["label"],
            "n_queries": n_queries,
            "neighbor_slots": int(weights.sum()) if not args.distinct else None,
            "distinct_rows": int(zcounts.shape[0]),
            "knn_run": QUERY_SETS[dataset]["runs"][args.geometry],
        }

    bg = np.load(root / OMAT_BG)
    bg_presence, bg_fraction = presence_and_fraction(
        bg, np.ones(bg.shape[0], dtype=np.float64)
    )
    populations["omat_bg_global"] = (bg_presence, bg_fraction)
    meta["omat_bg_global"] = {
        "query_set": "OMAT24 uniform random",
        "distinct_rows": int(bg.shape[0]),
        "source": OMAT_BG,
    }

    if "mof_off" in args.datasets and not args.distinct:
        meta["crosscheck"] = crosscheck_mof_off(
            root,
            populations[f"mof_off_{args.geometry}_top{args.depth}"][0],
            args.geometry,
            args.depth,
        )

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
    csv_path = out / f"element_presence_{args.geometry}_top{args.depth}.csv"
    write_csv(csv_path, fields, rows)
    log(f"wrote {csv_path} ({len(rows)} elements, {len(names)} populations)")

    meta_path = out / f"presence_metadata_{args.geometry}_top{args.depth}.json"
    with meta_path.open("w") as handle:
        json.dump(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "root": str(root),
                "geometry": args.geometry,
                "depth": args.depth,
                "weighting": "distinct" if args.distinct else "per-neighbor-slot",
                "populations": meta,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    log(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
