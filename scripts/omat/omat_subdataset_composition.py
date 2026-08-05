#!/usr/bin/env python
"""Element composition of a single OMAT24 subdataset, as a presence table.

``query_set_composition.py`` does this for the external query datasets (MAD,
MP ALOE) and ``multiset_top5_presence.py`` does it for OMAT24 *as a whole* --
its ``omat_bg_global`` column is 50k uniform random rows drawn from all
100,824,585 embedded rows.  Neither answers "what is in one OMAT24 subdataset",
which is the question that matters once the neighbor search turns out to land
almost entirely inside ``aimd-from-PBE-3000-nvt``.

This script produces that column.  A subdataset is far too large to read in
full (``aimd-from-PBE-3000-nvt`` alone is 7,839,846 frames), so the population
is a uniform random sample of its frames -- the same construction and the same
sample size as the ``omat_bg_global`` baseline, so the two panels are directly
comparable.

Weighting is the dataset-panel convention: one structure, one observation.
That is deliberate and differs from the neighbor panels, where the unit is a
retrieval slot.

    python scripts/omat/omat_subdataset_composition.py \
        --root /scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis \
        --subdataset aimd-from-PBE-3000-nvt

By default the nvt-3000 sample is not re-extracted: ``mof_omat_neighbor_chemistry.py
extract`` already drew and validated one as its ``omat_bg_nvt3000`` baseline, and
re-drawing it would only spend an hour of shard reads to get a different 50k
rows.  ``--sample N`` (or ``--force``) draws a fresh one.

Then render, on one shared colour scale, with the plotter used for every other
periodic table in this project:

    PYTHONPATH=$HOME/pylibs python scripts/omat/plot_multiset_top5_ptable.py \
        --presence <out>/element_presence_omat_subsets.csv
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.data import chemical_symbols

HERE = Path(__file__).resolve().parent
MAX_Z = 118

DEFAULT_ROOT = Path(os.environ.get("PRETRAIN_ANALYSIS_ROOT", HERE.parents[1]))
DEFAULT_OUT = "runs/omat_knn_probe/omat_subdataset_composition"

# The OMAT24-wide baseline the plotter differences against, and which gives this
# figure something to be read next to.  50k uniform random rows over all shards.
OMAT_BG = (
    "runs/omat_knn_probe/mof_omat_neighbor_chemistry/structures/omat_bg_global_zcounts.npy"
)

# Extractions already drawn by mof_omat_neighbor_chemistry.py, reusable as-is.
# Each is validated against its own .csv before use, so a stale or mislabelled
# cache fails loudly rather than being plotted under the wrong name.
PREEXTRACTED = {
    "aimd-from-PBE-3000-nvt": "runs/omat_knn_probe/mof_omat_neighbor_chemistry/structures/omat_bg_nvt3000",
}


def log(message):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def load_chemistry_module():
    """Import the sibling script for its verified OMAT24 row-mapping helpers."""
    path = HERE / "mof_omat_neighbor_chemistry.py"
    spec = importlib.util.spec_from_file_location("mof_omat_neighbor_chemistry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_write_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def slug(subdataset):
    """`aimd-from-PBE-3000-nvt` -> `aimd_from_pbe_3000_nvt`."""
    return re.sub(r"[^a-z0-9]+", "_", subdataset.lower()).strip("_")


def subdataset_shards(manifest, subdataset):
    """Manifest indices for one subdataset, with its total embedded row count."""
    picks = [i for i, r in enumerate(manifest) if r["subdataset"] == subdataset]
    if not picks:
        available = sorted({r["subdataset"] for r in manifest})
        raise SystemExit(
            f"no manifest shards for subdataset {subdataset!r}; available:\n  "
            + "\n  ".join(available)
        )
    total = sum(int(manifest[i]["n_rows"]) for i in picks)
    return picks, total


def sample_subdataset_rows(manifest, subdataset, n_sample, seed):
    """Uniform random global rows from one subdataset's shards.

    Mirrors the nvt-3000 draw in ``mof_omat_neighbor_chemistry.command_extract``:
    sample offsets in the subdataset's own concatenated row space, then map each
    offset back through the owning shard's ``global_start``.
    """
    picks, total = subdataset_shards(manifest, subdataset)
    if n_sample > total:
        raise SystemExit(
            f"asked for {n_sample:,} frames but {subdataset} has only {total:,}"
        )
    sizes = np.array([int(manifest[i]["n_rows"]) for i in picks], dtype=np.int64)
    stops = np.cumsum(sizes)
    prev = np.concatenate([[0], stops[:-1]])

    rng = np.random.default_rng(seed)
    offsets = np.sort(rng.choice(total, n_sample, replace=False))
    shard_pos = np.searchsorted(stops, offsets, side="right")
    rows = np.array(
        [
            int(manifest[picks[int(s)]]["global_start"]) + int(o - prev[int(s)])
            for o, s in zip(offsets, shard_pos)
        ],
        dtype=np.int64,
    )
    return rows, total, len(picks)


def validate_cached(stem_path, subdataset):
    """Load a pre-extracted population, refusing anything not purely this subdataset.

    The sidecar ``.csv`` carries a ``subdataset`` column per structure, so this is
    a real check on what was read out of the shards -- not a check on a filename.
    """
    zpath = stem_path.with_name(stem_path.name + "_zcounts.npy")
    cpath = stem_path.with_name(stem_path.name + ".csv")
    if not zpath.exists() or not cpath.exists():
        return None

    zcounts = np.load(zpath)
    with cpath.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != zcounts.shape[0]:
        raise SystemExit(
            f"{cpath.name} has {len(rows):,} structures but "
            f"{zpath.name} has {zcounts.shape[0]:,}"
        )
    seen = {r["subdataset"] for r in rows}
    if seen != {subdataset}:
        raise SystemExit(
            f"{cpath} is not purely {subdataset!r}; it contains {sorted(seen)}"
        )
    shards = len({r["shard"] for r in rows})
    systems = len({r["chemical_system"] for r in rows})
    log(
        f"reusing {zpath.name}: {zcounts.shape[0]:,} frames, {shards} shards, "
        f"{systems:,} distinct chemical systems"
    )
    return zcounts, {
        "source": str(zpath),
        "sampled_shards": shards,
        "distinct_chemical_systems": systems,
        "distinct_formulas": len({r["formula_hill"] for r in rows}),
    }


def extract_subdataset(chem, root, out, subdataset, n_sample, seed):
    """Read a fresh uniform sample of one subdataset out of the .aselmdb shards."""
    manifest, starts, _ = chem.read_manifest(root)
    rows, total, n_shards = sample_subdataset_rows(manifest, subdataset, n_sample, seed)
    name = f"omat_{slug(subdataset)}"
    log(
        f"{subdataset}: sampling {rows.size:,} of {total:,} frames "
        f"({100.0 * rows.size / total:.3f}%) across {n_shards} shards"
    )
    records, counts = chem.read_omat_rows(root, rows, manifest, starts, name, name)
    zcounts = np.asarray(counts, dtype=np.int32)
    if zcounts.shape[0] != rows.size:
        raise SystemExit(
            f"{name}: extracted {zcounts.shape[0]} structures for {rows.size} rows"
        )
    chem.write_population(out / "structures", name, records, zcounts)
    return zcounts, {
        "source": str(out / "structures" / f"{name}_zcounts.npy"),
        "sampled_shards": len({r["shard"] for r in records}),
        "distinct_chemical_systems": len({r["chemical_system"] for r in records}),
        "distinct_formulas": len({r["formula_hill"] for r in records}),
        "seed": seed,
    }


def presence_and_fraction(zcounts):
    """One structure one observation: share containing each Z, and atom fraction."""
    if zcounts.shape[0] == 0:
        raise SystemExit("empty population")
    presence = (zcounts > 0).mean(0)
    atoms = zcounts.sum(0, dtype=np.int64)
    return presence, atoms / atoms.sum()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--subdataset", default="aimd-from-PBE-3000-nvt")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="draw a fresh uniform sample of this many frames instead of reusing "
             "the pre-extracted one (default: reuse if available, else 50000)",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--force", action="store_true", help="re-extract even if a cache exists"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or root / DEFAULT_OUT).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "data/processed/omat24/uma_latents_copy/all_uma_embeddings_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest at {manifest_path}\n"
            f"--root must point at the main checkout (data/ is not in a worktree)"
        )

    chem = load_chemistry_module()
    manifest, _, _ = chem.read_manifest(root)
    _, subset_total = subdataset_shards(manifest, args.subdataset)
    omat_total = sum(int(r["n_rows"]) for r in manifest)

    zcounts, provenance = None, None
    if args.sample is None and not args.force:
        stem = PREEXTRACTED.get(args.subdataset)
        if stem:
            found = validate_cached(root / stem, args.subdataset)
            if found:
                zcounts, provenance = found
    if zcounts is None:
        zcounts, provenance = extract_subdataset(
            chem, root, out, args.subdataset, args.sample or 50000, args.seed
        )

    presence, fraction = presence_and_fraction(zcounts)
    name = f"omat_{slug(args.subdataset)}_dataset"
    n = int(zcounts.shape[0])
    populations = {name: (presence, fraction)}
    meta = {
        name: {
            "label": (
                f"OMAT24 {args.subdataset} "
                f"({n:,} uniform random frames of {subset_total:,})"
            ),
            "subdataset": args.subdataset,
            "n_structures": n,
            "subdataset_total_frames": subset_total,
            "sampled_fraction_of_subdataset": round(n / subset_total, 8),
            "subdataset_share_of_omat24": round(subset_total / omat_total, 6),
            "total_atoms": int(zcounts.sum()),
            **provenance,
        }
    }

    # The plotter requires this population as its difference denominator, and it
    # is the panel this one is meant to be read against.
    bg = np.load(root / OMAT_BG)
    bg_presence, bg_fraction = presence_and_fraction(bg)
    populations["omat_bg_global"] = (bg_presence, bg_fraction)
    meta["omat_bg_global"] = {
        "label": f"OMAT24 overall ({bg.shape[0]:,} uniform random frames)",
        "n_structures": int(bg.shape[0]),
        "distinct_rows": int(bg.shape[0]),
        "omat24_total_frames": omat_total,
        "source": OMAT_BG,
    }

    names = list(populations)
    active = sorted(
        {
            z
            for nm in names
            for z in np.nonzero(populations[nm][0])[0]
            if 1 <= z <= MAX_Z
        }
    )
    rows = []
    for z in active:
        row = {"atomic_number": z, "symbol": chemical_symbols[z]}
        for nm in names:
            p, f = populations[nm]
            row[f"presence_{nm}"] = round(float(p[z]), 8)
            row[f"atomfrac_{nm}"] = round(float(f[z]), 8)
        rows.append(row)
    fields = ["atomic_number", "symbol"] + [
        f"{kind}_{nm}" for nm in names for kind in ("presence", "atomfrac")
    ]

    csv_path = out / "element_presence_omat_subsets.csv"
    write_csv(csv_path, fields, rows)
    log(f"wrote {csv_path} ({len(rows)} elements, {len(names)} populations)")

    only_here = [
        chemical_symbols[z] for z in active if presence[z] > 0 and bg_presence[z] == 0
    ]
    only_bg = [
        chemical_symbols[z] for z in active if presence[z] == 0 and bg_presence[z] > 0
    ]
    meta["comparison"] = {
        "elements_in_subdataset": int((presence > 0).sum()),
        "elements_in_omat24_overall": int((bg_presence > 0).sum()),
        "present_only_in_subdataset": only_here,
        "absent_from_subdataset_present_overall": only_bg,
    }
    log(
        f"{args.subdataset}: {int((presence > 0).sum())} elements present vs "
        f"{int((bg_presence > 0).sum())} in the OMAT24-wide sample"
    )
    if only_bg:
        log(f"  absent here but present overall: {', '.join(only_bg)}")
    if only_here:
        log(f"  present here but absent overall: {', '.join(only_here)}")

    meta_path = out / "presence_metadata_omat_subsets.json"
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
