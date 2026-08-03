#!/usr/bin/env python
"""Retrieval statistics for the MAD -> OMAT24 raw-128d nearest-neighbour run.

MAD ("Massive Atomic Diversity", Materials Cloud c4ene-0mv14; Mazitov et al.,
PET-MAD arXiv:2503.14118) is 95,595 one-shot PBEsol snapshots over 8 subsets --
not a trajectory dataset -- so every structure carries equal weight and none of
the MOF-off per-material reweighting applies.

This module answers "how does MAD sit inside OMAT24 as a retrieval problem",
using only the existing kNN arrays.  No structure extraction, no GPU.

  A  distance structure, retrieval concentration, isolation vs redundancy
  B  source attribution and outlier forensics

MOF-off R2SCAN is measured through the identical code path as an anchor, so the
two datasets are never compared across differently-computed statistics.

Calibration is deliberately rank-based: there is no raw-128d OMAT24 self-kNN,
and the existing leave-one-out calibration is in the pc25 geometry, whose units
are unrelated.  Statements therefore stay relative.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mof_omat_neighbor_chemistry import (  # noqa: E402
    DEFAULT_ROOT,
    atomic_write_json,
    log,
    read_manifest,
    utc_now,
    write_csv,
)

MAD_KNN = "runs/omat_knn_probe/omat_knn_raw128_mad"
MAD_SPLITS = [("MAD_train", "train"), ("MAD_val", "val"), ("MAD_test", "test")]
MAD_FRAMES = "data/processed/mad/{split}/mad_{split}_frames.tsv"
MOF_KNN = "runs/omat_knn_probe/omat_knn_raw128_mof_off/MOF_off_R2SCAN"
K = 5
QUANTILES = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 0.999]


def gini(counts):
    ordered = np.sort(np.asarray(counts, dtype=np.float64))
    cumulative = np.cumsum(ordered) / ordered.sum()
    return float(1 - 2 * np.trapezoid(cumulative, dx=1 / ordered.size))


def concentration(links):
    """Reuse statistics for a flat array of retrieved OMAT global rows."""
    counts = np.array(sorted(Counter(links.tolist()).values(), reverse=True))
    total = counts.sum()
    n = counts.size
    return {
        "links": int(total),
        "distinct_structures": int(n),
        "links_per_structure": round(float(total / n), 4),
        "singleton_structures": int((counts == 1).sum()),
        "singleton_fraction": round(float((counts == 1).mean()), 6),
        "share_top10_pct": round(float(counts[:10].sum() / total * 100), 4),
        "share_top100_pct": round(float(counts[:100].sum() / total * 100), 4),
        "share_top1pct_pct": round(float(counts[: max(1, n // 100)].sum() / total * 100), 4),
        "max_reuse": int(counts[0]),
        "gini_reuse": round(gini(counts), 6),
    }, counts


def lorenz(counts, points=200):
    """Lorenz curve of reuse: cumulative share of links vs share of structures."""
    ordered = np.sort(np.asarray(counts, dtype=np.float64))
    cumulative = np.concatenate([[0.0], np.cumsum(ordered) / ordered.sum()])
    x = np.linspace(0, 1, cumulative.size)
    grid = np.linspace(0, 1, points)
    return grid, np.interp(grid, x, cumulative)


def quantile_row(label, values, extra=None):
    row = {"population": label, "n": int(values.size),
           "mean": round(float(values.mean()), 4)}
    for q in QUANTILES:
        row[f"p{q * 100:g}"] = round(float(np.quantile(values, q)), 4)
    row["max"] = round(float(values.max()), 4)
    if extra:
        row.update(extra)
    return row


def load_mad(root):
    idx, dist, split_of = [], [], []
    for directory, split in MAD_SPLITS:
        base = root / MAD_KNN / directory
        i = np.load(base / "indices.npy")
        d = np.load(base / "distances.npy")
        if i.shape != d.shape:
            raise SystemExit(f"{directory}: indices/distances shape mismatch")
        idx.append(i)
        dist.append(d)
        split_of.append(np.full(i.shape[0], split, dtype=object))
    idx, dist = np.concatenate(idx), np.concatenate(dist)
    split_of = np.concatenate(split_of)

    subset, natoms, pbc = [], [], []
    for _, split in MAD_SPLITS:
        path = root / MAD_FRAMES.format(split=split)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        subset.extend(r["subset"] for r in rows)
        natoms.extend(int(r["natoms"]) for r in rows)
        pbc.extend(r["pbc"] for r in rows)
    subset = np.array(subset, dtype=object)
    if subset.size != idx.shape[0]:
        raise SystemExit(f"frames TSV rows {subset.size} != kNN rows {idx.shape[0]}")
    return idx, dist, split_of, subset, np.array(natoms), np.array(pbc, dtype=object)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--outlier-threshold", type=float, default=1000.0)
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or root / "runs/omat_knn_probe/mad_omat_neighbor_stats").resolve()
    out.mkdir(parents=True, exist_ok=True)

    log("loading MAD kNN and frame metadata")
    idx, dist, split_of, subset, natoms, pbc = load_mad(root)
    n = idx.shape[0]
    log(f"MAD: {n:,} queries, {idx.shape[1]} neighbours each")

    manifest, starts, _ = read_manifest(root)
    sub_arr = np.array([r["subdataset"] for r in manifest], dtype=object)
    shard_arr = np.array([r["shard"] for r in manifest], dtype=object)
    mrow = np.searchsorted(starts, idx.ravel(), side="right") - 1
    src = sub_arr[mrow].reshape(idx.shape)
    shard = shard_arr[mrow].reshape(idx.shape)

    # --- gate: reproduce the per-split subdataset counts in the existing summaries
    for directory, split in MAD_SPLITS:
        recorded = json.loads((root / MAD_KNN / directory / "summary.json").read_text())
        expected = recorded.get("neighbor_subdataset_counts", {})
        got = Counter(src[split_of == split].ravel().tolist())
        if {k: int(v) for k, v in expected.items()} != dict(got):
            raise SystemExit(f"{directory}: subdataset counts disagree with summary.json\n"
                             f"  summary={expected}\n  recomputed={dict(got)}")
    log("verified: recomputed subdataset counts match all three summary.json files")

    d1, d5, d10 = dist[:, 0], dist[:, 4], dist[:, 9]
    isolation = d10 / np.maximum(d1, 1e-12)
    distinct_shards = np.array([len(set(row)) for row in shard])

    # --- MOF-off anchor through the identical code path --------------------
    mof_idx = np.load(root / MOF_KNN / "indices.npy")
    mof_dist = np.load(root / MOF_KNN / "distances.npy")
    mof_mrow = np.searchsorted(starts, mof_idx.ravel(), side="right") - 1
    mof_shard = shard_arr[mof_mrow].reshape(mof_idx.shape)
    mof_iso = mof_dist[:, 9] / np.maximum(mof_dist[:, 0], 1e-12)
    mof_shards = np.array([len(set(row)) for row in mof_shard])

    # --- A1 distance structure ---------------------------------------------
    dist_rows = [quantile_row("MAD (all)", d1, {"metric": "d1"}),
                 quantile_row("MAD (all)", d5, {"metric": "d5"}),
                 quantile_row("MAD (all)", d10, {"metric": "d10"})]
    for _, split in MAD_SPLITS:
        m = split_of == split
        dist_rows.append(quantile_row(f"MAD {split}", d1[m], {"metric": "d1"}))
    for name in sorted(set(subset.tolist())):
        m = subset == name
        dist_rows.append(quantile_row(name, d1[m], {"metric": "d1"}))
    dist_rows.append(quantile_row("MOF-off R2SCAN", mof_dist[:, 0], {"metric": "d1"}))
    dist_rows.append(quantile_row("MOF-off R2SCAN", mof_dist[:, 9], {"metric": "d10"}))
    write_csv(out / "distance_quantiles.csv", list(dist_rows[0].keys()), dist_rows)

    # --- A2 concentration ---------------------------------------------------
    conc_rows, lorenz_rows = [], []
    populations = [("MAD (all)", idx[:, :K].ravel())]
    for name in sorted(set(subset.tolist())):
        populations.append((name, idx[subset == name][:, :K].ravel()))
    populations.append(("MOF-off R2SCAN", mof_idx[:, :K].ravel()))
    for label, links in populations:
        stats, counts = concentration(links)
        conc_rows.append({"population": label, **stats})
        gx, gy = lorenz(counts)
        for x, y in zip(gx, gy):
            lorenz_rows.append({"population": label,
                                "structure_fraction": round(float(x), 5),
                                "link_fraction": round(float(y), 6)})
    write_csv(out / "retrieval_concentration.csv", list(conc_rows[0].keys()), conc_rows)
    write_csv(out / "reuse_lorenz.csv",
              ["population", "structure_fraction", "link_fraction"], lorenz_rows)

    # --- A3 isolation vs redundancy ----------------------------------------
    iso_rows = []
    for label, iso, shards in (("MAD (all)", isolation, distinct_shards),
                               ("MOF-off R2SCAN", mof_iso, mof_shards)):
        iso_rows.append({
            "population": label, "n": int(iso.size),
            "isolation_median": round(float(np.median(iso)), 4),
            "isolation_p90": round(float(np.quantile(iso, 0.90)), 4),
            "isolation_p99": round(float(np.quantile(iso, 0.99)), 4),
            "dense_lt_1p1_pct": round(float((iso < 1.1).mean() * 100), 3),
            "isolated_gt_2_pct": round(float((iso > 2).mean() * 100), 3),
            "mean_distinct_shards_of_10": round(float(shards.mean()), 4),
            "all_ten_one_shard_pct": round(float((shards == 1).mean() * 100), 3),
            "ge5_shards_pct": round(float((shards >= 5).mean() * 100), 3),
        })
    for name in sorted(set(subset.tolist())):
        m = subset == name
        iso_rows.append({
            "population": name, "n": int(m.sum()),
            "isolation_median": round(float(np.median(isolation[m])), 4),
            "isolation_p90": round(float(np.quantile(isolation[m], 0.90)), 4),
            "isolation_p99": round(float(np.quantile(isolation[m], 0.99)), 4),
            "dense_lt_1p1_pct": round(float((isolation[m] < 1.1).mean() * 100), 3),
            "isolated_gt_2_pct": round(float((isolation[m] > 2).mean() * 100), 3),
            "mean_distinct_shards_of_10": round(float(distinct_shards[m].mean()), 4),
            "all_ten_one_shard_pct": round(float((distinct_shards[m] == 1).mean() * 100), 3),
            "ge5_shards_pct": round(float((distinct_shards[m] >= 5).mean() * 100), 3),
        })
    write_csv(out / "isolation_redundancy.csv", list(iso_rows[0].keys()), iso_rows)

    # --- B1 source attribution ---------------------------------------------
    src5 = src[:, :K]
    sources = sorted(set(src.ravel().tolist()))
    source_rows = []
    for name in sorted(set(subset.tolist())) + ["MAD (all)"]:
        m = np.ones(n, dtype=bool) if name == "MAD (all)" else (subset == name)
        counts = Counter(src5[m].ravel().tolist())
        total = sum(counts.values())
        row = {"population": name, "structures": int(m.sum()), "links": total}
        for s in sources:
            row[f"pct_{s}"] = round(counts.get(s, 0) / total * 100, 4)
        # a structure "reaches" a source if any of its top-5 lands there
        row["pct_structures_reaching_rattled_relax"] = round(
            float((src5[m] == "rattled-relax").any(axis=1).mean() * 100), 3)
        row["median_natoms"] = float(np.median(natoms[m]))
        row["pct_periodic_TTT"] = round(float((pbc[m] == "T T T").mean() * 100), 3)
        source_rows.append(row)
    write_csv(out / "source_attribution.csv", list(source_rows[0].keys()), source_rows)

    # Do the structures that reach rattled-relax differ?
    reaches = (src5 == "rattled-relax").any(axis=1)
    contrast = [{
        "group": label, "structures": int(g.sum()),
        "median_natoms": float(np.median(natoms[g])),
        "median_d1": round(float(np.median(d1[g])), 4),
        "pct_TTT": round(float((pbc[g] == "T T T").mean() * 100), 3),
        "pct_cell_free_FFF": round(float((pbc[g] == "F F F").mean() * 100), 3),
        "top_subset": Counter(subset[g].tolist()).most_common(1)[0][0],
    } for label, g in (("reaches rattled-relax", reaches), ("does not", ~reaches))]
    write_csv(out / "rattled_relax_contrast.csv", list(contrast[0].keys()), contrast)

    # --- B2 outlier forensics ----------------------------------------------
    far = np.nonzero(d1 > args.outlier_threshold)[0]
    order = far[np.argsort(-d1[far])]
    outliers = [{
        "rank": r + 1, "mad_row": int(i), "split": str(split_of[i]),
        "subset": str(subset[i]), "natoms": int(natoms[i]), "pbc": str(pbc[i]),
        "d1": round(float(d1[i]), 3), "d10": round(float(d10[i]), 3),
        "isolation": round(float(isolation[i]), 4),
        "nearest_omat_row": int(idx[i, 0]), "nearest_source": str(src[i, 0]),
    } for r, i in enumerate(order)]
    if outliers:
        write_csv(out / "distance_outliers.csv", list(outliers[0].keys()), outliers)

    # The most isolated queries: a close rank-1 but a far rank-10.
    top_iso = np.argsort(-isolation)[:50]
    write_csv(out / "isolation_outliers.csv",
              ["rank", "mad_row", "split", "subset", "natoms", "pbc", "d1", "d10", "isolation"],
              [{"rank": r + 1, "mad_row": int(i), "split": str(split_of[i]),
                "subset": str(subset[i]), "natoms": int(natoms[i]), "pbc": str(pbc[i]),
                "d1": round(float(d1[i]), 3), "d10": round(float(d10[i]), 3),
                "isolation": round(float(isolation[i]), 3)}
               for r, i in enumerate(top_iso)])

    # --- hub structures -----------------------------------------------------
    hub_counts = Counter(idx[:, :K].ravel().tolist())
    hubs = hub_counts.most_common(50)
    hub_rows = []
    for r, (row_id, count) in enumerate(hubs, 1):
        mi = int(np.searchsorted(starts, row_id, side="right") - 1)
        served = subset[(idx[:, :K] == row_id).any(axis=1)]
        hub_rows.append({
            "rank": r, "omat_global_row": int(row_id), "times_retrieved": int(count),
            "share_of_links_pct": round(count / (n * K) * 100, 4),
            "subdataset": str(manifest[mi]["subdataset"]), "shard": str(manifest[mi]["shard"]),
            "distinct_mad_structures_served": int(np.unique(np.nonzero(
                (idx[:, :K] == row_id).any(axis=1))[0]).size),
            "top_mad_subset": Counter(served.tolist()).most_common(1)[0][0] if served.size else "",
        })
    write_csv(out / "hub_structures.csv", list(hub_rows[0].keys()), hub_rows)

    atomic_write_json(out / "neighbor_stats_metadata.json", {
        "created_utc": utc_now(), "geometry": "raw_uma_128d_euclidean",
        "k_used_for_concentration": K, "mad_queries": int(n),
        "mad_subsets": {s: int((subset == s).sum()) for s in sorted(set(subset.tolist()))},
        "mad_splits": {s: int((split_of == s).sum()) for _, s in MAD_SPLITS},
        "outlier_threshold_d1": args.outlier_threshold,
        "n_outliers": int(far.size),
        "calibration": "rank-based only; no raw-128d OMAT24 self-kNN exists and the pc25 "
                       "leave-one-out calibration is in unrelated units",
        "anchor": "MOF_off_R2SCAN measured through the identical code path",
    })
    log(f"neighbour statistics written -> {out}")


if __name__ == "__main__":
    main()
