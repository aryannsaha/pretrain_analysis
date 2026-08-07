#!/usr/bin/env python
"""Split the MatPES UMA latent matrix by DFT functional.

``data/processed/matpes/uma_latents/matpes_r2scan.npy`` is named for the
directory it was built from, not for its contents: it holds all 278,632 frames
of ``mace_matpes_test_full``, which is 161,262 r2SCAN **and** 117,370 PBE.
Anything that treats the whole matrix as an r2SCAN query set silently mixes
functionals -- the trap documented for the parent LMDB.

This writes the single-functional subsets, each with the manifest slice that
produced it, so a downstream kNN run can name its query set honestly.

ROW CORRESPONDENCE.  The latent matrix is the LMDB read in order, so latent row
``i`` is manifest row ``i``.  That is asserted rather than assumed: ``lmdb_id``
must be exactly ``1..N`` in file order and ``N`` must equal the matrix row
count.  If the manifest is ever reordered or filtered in place, this fails
instead of silently emitting a permuted subset.

    python scripts/matpes/filter_matpes_latents_by_functional.py \
        --root /path/to/pretrain_analysis --functional r2SCAN
"""

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.environ.get("PRETRAIN_ANALYSIS_ROOT", HERE.parents[1]))
LATENTS = "data/processed/matpes/uma_latents/matpes_r2scan.npy"
MANIFEST = "data/processed/matpes/uma_latents/matpes_r2scan_manifest.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--functional", default="r2SCAN",
                        help="functional to keep, as spelled in the manifest")
    parser.add_argument("--out-stem", default=None,
                        help="output stem; defaults to matpes_<functional>_only")
    args = parser.parse_args()

    root = args.root.resolve()
    latents = np.load(root / LATENTS, mmap_mode="r")
    rows = list(csv.DictReader((root / MANIFEST).open(newline="")))

    if len(rows) != latents.shape[0]:
        raise SystemExit(
            f"manifest has {len(rows):,} rows but the latent matrix has "
            f"{latents.shape[0]:,}; they do not describe the same frames"
        )
    ids = np.array([int(r["lmdb_id"]) for r in rows], dtype=np.int64)
    if not np.array_equal(ids, np.arange(1, len(rows) + 1)):
        raise SystemExit(
            "lmdb_id is not contiguous 1..N in file order, so latent row i is not "
            "manifest row i; the row mapping must be rebuilt before filtering"
        )

    present = Counter(r["functional"] for r in rows)
    if args.functional not in present:
        raise SystemExit(
            f"functional {args.functional!r} not in manifest; present: {dict(present)}"
        )
    keep = np.array([r["functional"] == args.functional for r in rows], dtype=bool)
    print(f"manifest functionals: {dict(present)}")
    print(f"keeping {int(keep.sum()):,} / {len(rows):,} rows for {args.functional}")

    stem = args.out_stem or f"matpes_{args.functional.lower()}_only"
    out_dir = (root / LATENTS).parent
    out_npy = out_dir / f"{stem}.npy"
    out_csv = out_dir / f"{stem}_manifest.csv"

    subset = np.ascontiguousarray(latents[keep])
    if subset.shape[0] != int(keep.sum()):
        raise SystemExit("row count changed during the subset copy")

    # np.save appends ".npy" unless the path already ends in it, which would
    # leave the temp file somewhere other than where os.replace looks; writing
    # through a handle keeps the name exactly as given.
    tmp = out_npy.with_name(out_npy.name + f".partial.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, subset)
    os.replace(tmp, out_npy)

    fields = list(rows[0].keys()) + ["source_lmdb_id", "query_row"]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query_row, index in enumerate(np.flatnonzero(keep)):
            record = dict(rows[index])
            record["source_lmdb_id"] = record["lmdb_id"]
            record["query_row"] = query_row
            writer.writerow(record)

    (out_dir / f"{stem}.metadata.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_latents": str(root / LATENTS),
        "source_manifest": str(root / MANIFEST),
        "source_rows": int(latents.shape[0]),
        "source_functionals": dict(present),
        "functional": args.functional,
        "rows": int(subset.shape[0]),
        "shape": list(subset.shape),
        "dtype": str(subset.dtype),
        "row_mapping": "query row j is source lmdb_id (query_row->source_lmdb_id in the manifest)",
    }, indent=2) + "\n")

    print(f"wrote {out_npy}  {subset.shape} {subset.dtype}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
