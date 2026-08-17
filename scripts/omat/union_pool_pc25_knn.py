#!/usr/bin/env python
"""pc25 top-5 neighbour attribution when the searched pool is OMAT24 *plus* every
finetuning/external corpus -- not OMAT24 alone.

The existing table (``pc_dimension_subdataset_table.py``) queries each external set
against OMAT24 train only, so every neighbour slot is forced to land on one of the 11
OMAT24 subdatasets.  Here the pool becomes

    OMAT24 train (100,824,585)  U  MOF-off R2SCAN  U  MOF-off PBE  U  MAD
                                U  MP-ALOE  U  MatPES r2SCAN  U  AM Full
    = 114,533,618 rows

and a query may retrieve a frame from any of them.  The geometry is unchanged: the
frozen 25-PC standardised UMA basis fitted on OMAT24 (``pca_state``), plain Euclidean.
Refitting the PCA on the union was deliberately *not* done -- the external corpora are
13.6% of the union by rows and refitting would invalidate every existing baseline
(the 1M leave-one-out calibration, the d1 percentiles, the 0.051% attribution figure)
for a basis perturbation of that order.  See ``--help`` on ``table`` for the panels.

Two panels are produced:

  A  self-excluded    every pool row is a candidate except the query's own row.
                      This is the literal "put everything in the pool" answer, and it
                      is dominated by self-retrieval the same way OMAT24's own
                      leave-one-out row is.
  B  corpus-excluded  the query's entire own corpus is removed from the pool, so the
                      question becomes "which *foreign* corpus wins".  Panel B is the
                      one that is comparable to the published OMAT-only table.

Only the external half is computed here.  The OMAT24 half of every query set already
exists on disk from the published pc25 runs (validated recall@1 = 1.0, zero distance
inflation) and is merged in, which is exact for a top-5: those runs stored 10
neighbours and a top-5 can draw at most 5 from OMAT.

Usage::

    python scripts/omat/union_pool_pc25_knn.py project
    python scripts/omat/union_pool_pc25_knn.py search --dataset all
    python scripts/omat/union_pool_pc25_knn.py validate
    python scripts/omat/union_pool_pc25_knn.py table
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PRETRAIN_ANALYSIS_ROOT", HERE.parents[1]))

PROBE = ROOT / "runs/omat_knn_probe"
OUT_ROOT = PROBE / "union_pool_pc25"
PCA_STATE = ROOT / "runs/uma_latents_copy_full_pca/uma_100824585_rapids_pca_state_11212254.npz"
OMAT_PC25 = PROBE / "omat_knn_pc25/omat_pc25_unweighted.npy"
OMAT_MANIFEST = ROOT / "data/processed/omat24/uma_latents_copy/all_uma_embeddings_manifest.csv"
OMAT_ROWS = 100_824_585
DIMENSIONS = 25
FOCUS = "aimd-from-PBE-3000-nvt"

# External corpora, in pool order.  AM Small is deliberately absent: it is an exact
# uniform 100k subsample of AM Full (seed 42, source ids 11..12,028,075, verified --
# the two embedding passes agree to max|diff| 1.1e-4), so admitting it as a separate
# pool member would give every AM Small query a duplicate of itself at ~0 distance and
# would double-count 100k AM Full rows.  It stays a query-only row whose self-exclusion
# maps through ``AM_SMALL_IDS`` into the AM Full segment.
CORPORA = [
    ("MOF-off R2SCAN-D4", "data/processed/mof_off/uma_latents/R2SCAN/train.npy", 80_643),
    ("MOF-off PBE-D3BJ", "data/processed/mof_off/uma_latents/PBE/train.npy", 106_946),
    ("MAD", "data/processed/mad/uma_latents/mad_all.npy", 95_595),
    ("MP ALOE", "data/processed/mpaloe/uma_latents/all/mpaloe_all.npy", 909_792),
    ("MatPES r2SCAN", "data/processed/matpes/uma_latents_all/matpes_all_r2scan.npy", 387_897),
    ("AM Full", "data/processed/AM/uma_latents/train.npy", 12_028_160),
]

AM_SMALL_LATENTS = "data/processed/AM/uma_latents_small/train.npy"
AM_SMALL_IDS = "data/processed/AM/train_small_sample_ids.tsv"

# Query sets: name -> (pc25 query source, owning corpus, stored OMAT-side run dir).
# ``own`` is the corpus removed for panel B; ``None`` means the query set is OMAT24
# itself, whose panel-B exclusion is the whole OMAT24 half of the pool.
QUERIES = {
    "MOF-off R2SCAN-D4": ("corpus", "MOF-off R2SCAN-D4", "omat_knn_pc25_mof_off/MOF_off_R2SCAN"),
    "MOF-off PBE-D3BJ": ("corpus", "MOF-off PBE-D3BJ", "omat_knn_pc25_mof_off/MOF_off_PBE"),
    "MAD": ("corpus", "MAD", "omat_knn_pc25_mad/MAD_all"),
    "MP ALOE": ("corpus", "MP ALOE", "omat_knn_pc25_mpaloe/MPALOE_all"),
    "MatPES r2SCAN": ("corpus", "MatPES r2SCAN", "omat_knn_pc25_matpes_all/MatPES_r2SCAN_all"),
    "AM Small": ("am_small", "AM Full", "omat_knn_pc25_final/AM_Small"),
    "AM Full": ("corpus", "AM Full", "omat_knn_pc25_final/AM_Full"),
    "OMAT24 (leave-one-out)": ("omat_loo", None, "omat_knn_pc25_final/OMAT_calibration"),
}

# Print order for the table rows.
ROW_ORDER = [
    "OMAT24 (leave-one-out)",
    f"OMAT24 rows outside {FOCUS}",
    "MOF-off R2SCAN-D4",
    "MOF-off PBE-D3BJ",
    "MAD",
    "MP ALOE",
    "MatPES r2SCAN",
    "AM Small",
    "AM Full",
]

K_STORED = 10          # neighbours kept per panel
K_SEARCH = 256         # IVF candidates pulled before masking (panel B needs depth)
NLIST = 4096
NPROBE = 1024

# Distance below which two pool rows are the *same structure*, not neighbours.  This is
# measured, not chosen: AM Small and AM Full hold the same 100,000 structures embedded in
# two independent UMA passes, and the pc25 distance between those matched pairs has
# median 1.3e-6 and maximum 1.11e-4 -- pure re-embedding noise.  1e-3 sits an order of
# magnitude above that ceiling and inside a clean valley in the observed d1 histogram
# (MOF-off R2SCAN: 79.86% of queries below 1e-3, 79.89% below 1e-2), so it separates
# relabelled copies from genuine neighbours without cutting into real structure.
DUP_TOL = 1e-3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


# --------------------------------------------------------------------------- pool


def segments() -> list[dict]:
    """External pool layout: contiguous [start, stop) per corpus."""
    rows, cursor = [], 0
    for name, path, count in CORPORA:
        rows.append({"name": name, "path": path, "rows": count,
                     "start": cursor, "stop": cursor + count})
        cursor += count
    return rows


def external_rows() -> int:
    return sum(count for _, _, count in CORPORA)


def load_pca() -> dict:
    with np.load(PCA_STATE) as state:
        return {
            "mean": np.asarray(state["mean"], dtype=np.float32),
            "scale": np.asarray(state["scale"], dtype=np.float32),
            "components": np.asarray(state["components"][:DIMENSIONS], dtype=np.float32),
        }


def project_numpy(values: np.ndarray, pca: dict) -> np.ndarray:
    centred = (np.asarray(values, dtype=np.float32) - pca["mean"]) / pca["scale"]
    return (centred @ pca["components"].T).astype(np.float32)


def command_project(args: argparse.Namespace) -> None:
    """Project every external corpus into the frozen pc25 basis, concatenated."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    target = OUT_ROOT / "external_pc25.npy"
    layout = segments()
    total = external_rows()
    meta = {
        "created_utc": utc_now(),
        "dimensions": DIMENSIONS,
        "geometry": "unweighted_standardized_pca_scores",
        "pca_state": str(PCA_STATE),
        "pca_state_sha256": sha256_file(PCA_STATE),
        "external_rows": total,
        "omat_rows": OMAT_ROWS,
        "union_rows": total + OMAT_ROWS,
        "segments": layout,
    }
    if target.is_file() and not args.force:
        existing = np.load(target, mmap_mode="r")
        if existing.shape == (total, DIMENSIONS):
            log(f"Reusing {target} {existing.shape}")
            atomic_json(OUT_ROOT / "pool_metadata.json", meta)
            return
    pca = load_pca()
    partial = target.with_suffix(".npy.partial")
    out = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                    shape=(total, DIMENSIONS))
    for seg in layout:
        source = np.load(ROOT / seg["path"], mmap_mode="r")
        if source.shape != (seg["rows"], 128):
            raise SystemExit(f"{seg['path']}: expected {(seg['rows'], 128)}, got {source.shape}")
        log(f"Projecting {seg['name']} {source.shape} -> rows [{seg['start']}, {seg['stop']})")
        step = 200_000
        for start in range(0, seg["rows"], step):
            stop = min(start + step, seg["rows"])
            out[seg["start"] + start: seg["start"] + stop] = project_numpy(source[start:stop], pca)
    out.flush()
    del out
    os.replace(partial, target)
    atomic_json(OUT_ROOT / "pool_metadata.json", meta)
    log(f"Wrote {target} ({total:,} x {DIMENSIONS})")


# ------------------------------------------------------------------------- search


def am_small_self_rows(layout: list[dict]) -> np.ndarray:
    """AM Small query row i -> its duplicate's row in the AM Full pool segment."""
    offset = next(s["start"] for s in layout if s["name"] == "AM Full")
    ids = []
    with (ROOT / AM_SMALL_IDS).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            ids.append(int(row["source_id"]))
    return np.asarray(ids, dtype=np.int64) - 1 + offset


def query_pc25(name: str, layout: list[dict], external: np.ndarray):
    """(query matrix as float32 numpy, self row per query or None)."""
    kind, own, _ = QUERIES[name]
    if kind == "corpus":
        seg = next(s for s in layout if s["name"] == own)
        rows = np.arange(seg["start"], seg["stop"], dtype=np.int64)
        return external[seg["start"]: seg["stop"]], rows
    if kind == "am_small":
        pca = load_pca()
        raw = np.load(ROOT / AM_SMALL_LATENTS, mmap_mode="r")
        return project_numpy(np.asarray(raw), pca), am_small_self_rows(layout)
    if kind == "omat_loo":
        source = np.load(PROBE / QUERIES[name][2] / "source_rows.npy")
        cache = OUT_ROOT / "omat_loo_query_pc25.npy"
        if cache.is_file():
            cached = np.load(cache, mmap_mode="r")
            if cached.shape == (source.size, DIMENSIONS):
                return cached, None
        omat = np.load(OMAT_PC25, mmap_mode="r")
        order = np.argsort(source)
        gathered = np.empty((source.size, DIMENSIONS), dtype=np.float32)
        # Sequential pass over the 10 GB memmap; random gather would thrash the cache.
        step = 4_000_000
        sorted_ids = source[order]
        cursor = 0
        for start in range(0, OMAT_ROWS, step):
            stop = min(start + step, OMAT_ROWS)
            end = cursor + int(np.searchsorted(sorted_ids[cursor:], stop))
            if end > cursor:
                block = np.asarray(omat[start:stop])
                gathered[order[cursor:end]] = block[sorted_ids[cursor:end] - start]
                cursor = end
        if cursor != source.size:
            raise SystemExit("failed to gather every OMAT calibration query row")
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        np.save(cache, gathered)
        return gathered, None
    raise SystemExit(f"unknown query kind {kind}")


def command_search(args: argparse.Namespace) -> None:
    import cupy as cp
    from cuml.neighbors import NearestNeighbors

    layout = segments()
    external = np.load(OUT_ROOT / "external_pc25.npy", mmap_mode="r")
    if external.shape != (external_rows(), DIMENSIONS):
        raise SystemExit(f"external_pc25.npy has shape {external.shape}; run `project` first")

    names = list(QUERIES) if args.dataset == "all" else [args.dataset]
    for name in names:
        if name not in QUERIES:
            raise SystemExit(f"unknown dataset {name}; choose from {list(QUERIES)}")

    log(f"Loading external pool {external.shape} to GPU")
    reference_gpu = cp.asarray(np.asarray(external), dtype=cp.float32, order="C")
    log(f"Building IVF-Flat nlist={NLIST} nprobe={NPROBE}")
    started = time.monotonic()
    model = NearestNeighbors(n_neighbors=K_SEARCH, algorithm="ivfflat", metric="sqeuclidean",
                             algo_params={"nlist": NLIST, "nprobe": NPROBE}, output_type="cupy")
    model.fit(reference_gpu, convert_dtype=False)
    log(f"Index built in {time.monotonic() - started:.1f}s")

    # AM Full is 88.4% of the external pool, so an AM-owned query cannot be relied on to
    # find 5 non-AM neighbours inside a top-256 list.  Panel B for those two query sets
    # is served by a dedicated index over the AM-free prefix of the pool instead, which
    # makes the exclusion exact rather than coverage-limited.  The small block is a
    # prefix, so its row ids are already valid external ids.
    small_stop = next(s["start"] for s in layout if s["name"] == "AM Full")
    log(f"Building AM-free block index over rows [0, {small_stop:,})")
    started = time.monotonic()
    small_gpu = reference_gpu[:small_stop]
    small_model = NearestNeighbors(n_neighbors=K_STORED, algorithm="ivfflat",
                                   metric="sqeuclidean",
                                   algo_params={"nlist": 2048, "nprobe": 512},
                                   output_type="cupy")
    small_model.fit(small_gpu, convert_dtype=False)
    log(f"AM-free index built in {time.monotonic() - started:.1f}s")

    for name in names:
        out_dir = OUT_ROOT / slug(name)
        if (out_dir / "metadata.json").is_file() and not args.force:
            log(f"Reusing completed search: {out_dir}")
            continue
        _, own, _ = QUERIES[name]
        own_lo, own_hi = -1, -1
        if own is not None:
            seg = next(s for s in layout if s["name"] == own)
            own_lo, own_hi = seg["start"], seg["stop"]
        am_owned = own == "AM Full"
        log(f"Query set {name}: preparing pc25 rows")
        queries, self_rows = query_pc25(name, layout, external)
        n = queries.shape[0]
        out_dir.mkdir(parents=True, exist_ok=True)

        keys = ("ext_all_idx", "ext_all_dist", "ext_foreign_idx", "ext_foreign_dist",
                "ext_nodup_idx", "ext_nodup_dist")
        parts = {key: np.lib.format.open_memmap(
            out_dir / f"{key}.npy.partial", mode="w+",
            dtype=(np.int64 if key.endswith("idx") else np.float32), shape=(n, K_STORED))
            for key in keys}
        foreign_available = np.zeros(n, dtype=np.int16)
        nodup_available = np.zeros(n, dtype=np.int16)

        batch = args.batch_rows
        started = time.monotonic()
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            q = cp.asarray(np.ascontiguousarray(queries[start:stop]), dtype=cp.float32)
            _, cand = model.kneighbors(q, n_neighbors=K_SEARCH, convert_dtype=False)
            cand = cp.asarray(cand, dtype=cp.int64)
            # Exact re-rank of the IVF candidates, ascending.
            gathered = reference_gpu[cp.where(cand < 0, 0, cand)]
            squared = cp.sum((gathered - q[:, None, :]) ** 2, axis=2)
            squared = cp.where(cand < 0, cp.inf, squared)
            order = cp.argsort(squared, axis=1)
            squared = cp.take_along_axis(squared, order, axis=1)
            cand = cp.take_along_axis(cand, order, axis=1)

            if self_rows is not None:
                selves = cp.asarray(self_rows[start:stop])[:, None]
                squared = cp.where(cand == selves, cp.inf, squared)
            all_d, all_i = _top(cp, squared, cand, K_STORED)

            if am_owned:
                # Exact: search the AM-free block directly instead of filtering a top-256
                # list that AM Full, at 88% of the pool, would otherwise fill.
                _, f_cand = small_model.kneighbors(q, n_neighbors=K_STORED, convert_dtype=False)
                f_cand = cp.asarray(f_cand, dtype=cp.int64)
                f_gathered = small_gpu[cp.where(f_cand < 0, 0, f_cand)]
                squared_f = cp.sum((f_gathered - q[:, None, :]) ** 2, axis=2)
                squared_f = cp.where(f_cand < 0, cp.inf, squared_f)
                for_d, for_i = _top(cp, squared_f, f_cand, K_STORED)
            else:
                if own_lo >= 0:
                    foreign = (cand < own_lo) | (cand >= own_hi)
                    squared_f = cp.where(foreign, squared, cp.inf)
                else:
                    squared_f = squared
                for_d, for_i = _top(cp, squared_f, cand, K_STORED)
            foreign_available[start:stop] = cp.asnumpy(
                cp.minimum(cp.sum(cp.isfinite(squared_f), axis=1), 32767)).astype(np.int16)

            # Panel C: same pool as panel A, minus rows that are the query restated --
            # a relabelled copy in a sibling corpus, or a re-embedded duplicate.
            squared_nd = cp.where(squared < DUP_TOL * DUP_TOL, cp.inf, squared)
            nodup_available[start:stop] = cp.asnumpy(
                cp.minimum(cp.sum(cp.isfinite(squared_nd), axis=1), 32767)).astype(np.int16)
            nd_d, nd_i = _top(cp, squared_nd, cand, K_STORED)

            parts["ext_all_dist"][start:stop] = cp.asnumpy(cp.sqrt(cp.maximum(all_d, 0.0)))
            parts["ext_all_idx"][start:stop] = cp.asnumpy(all_i)
            parts["ext_foreign_dist"][start:stop] = cp.asnumpy(cp.sqrt(cp.maximum(for_d, 0.0)))
            parts["ext_foreign_idx"][start:stop] = cp.asnumpy(for_i)
            parts["ext_nodup_dist"][start:stop] = cp.asnumpy(cp.sqrt(cp.maximum(nd_d, 0.0)))
            parts["ext_nodup_idx"][start:stop] = cp.asnumpy(nd_i)
            if start == 0 or stop == n or (start // batch) % 20 == 0:
                rate = stop / max(time.monotonic() - started, 1e-9)
                log(f"  {name}: {stop:,}/{n:,} ({rate:,.0f} rows/s)")

        for key, array in parts.items():
            array.flush()
        del parts
        for key in keys:
            os.replace(out_dir / f"{key}.npy.partial", out_dir / f"{key}.npy")
        np.save(out_dir / "foreign_available.npy", foreign_available)
        np.save(out_dir / "nodup_available.npy", nodup_available)
        atomic_json(out_dir / "metadata.json", {
            "complete": True,
            "created_utc": utc_now(),
            "dataset": name,
            "rows": int(n),
            "own_corpus": own,
            "own_segment": [int(own_lo), int(own_hi)],
            "k_search": K_SEARCH,
            "k_stored": K_STORED,
            "index": {"algorithm": "ivfflat", "nlist": NLIST, "nprobe": NPROBE},
            "panel_b_source": "am_free_block_index" if am_owned else "filtered_top_k_search",
            "duplicate_tolerance": DUP_TOL,
            "elapsed_seconds": time.monotonic() - started,
            "foreign_coverage_at_5": float(np.mean(foreign_available >= 5)),
            "foreign_coverage_at_10": float(np.mean(foreign_available >= 10)),
            "nodup_coverage_at_5": float(np.mean(nodup_available >= 5)),
            "duplicate_fraction_at_rank1": float(np.mean(nodup_available < K_SEARCH)),
        })
        log(f"  {name}: done, foreign coverage@5 = {np.mean(foreign_available >= 5):.6f}, "
            f"non-duplicate coverage@5 = {np.mean(nodup_available >= 5):.6f}")


def _top(cp, squared, candidates, k: int):
    order = cp.argsort(squared, axis=1)[:, :k]
    return cp.take_along_axis(squared, order, axis=1), cp.take_along_axis(candidates, order, axis=1)


# ----------------------------------------------------------------------- validate


def _brute_force(cp, reference_gpu, q, stop_row: int, block: int, k: int):
    """Exact top-k of q over reference_gpu[:stop_row], tiled to bound memory."""
    best_d = cp.full((q.shape[0], k), cp.inf, dtype=cp.float32)
    best_i = cp.full((q.shape[0], k), -1, dtype=cp.int64)
    query_norm = cp.sum(q * q, axis=1)[:, None]
    for start in range(0, stop_row, block):
        end = min(start + block, stop_row)
        chunk = reference_gpu[start:end]
        squared = query_norm + cp.sum(chunk * chunk, axis=1)[None, :] - 2.0 * (q @ chunk.T)
        squared = cp.maximum(squared, 0.0)
        keep = min(k, end - start)
        # cupy's argpartition materialises int64 index arrays of the same shape as
        # `squared` (2x its bytes), so block must stay small: rows=2,000 with
        # block=1e6 asked for a 16 GB allocation and OOMed an 80 GB A100.
        local = cp.argpartition(squared, keep - 1, axis=1)[:, :keep]
        merged_d = cp.concatenate((best_d, cp.take_along_axis(squared, local, axis=1)), axis=1)
        merged_i = cp.concatenate((best_i, local.astype(cp.int64) + start), axis=1)
        sel = cp.argsort(merged_d, axis=1)[:, :k]
        best_d = cp.take_along_axis(merged_d, sel, axis=1)
        best_i = cp.take_along_axis(merged_i, sel, axis=1)
        del squared, local, merged_d, merged_i, sel
    return best_d, best_i


def _compare(cp, exact_d, exact_i, got_i, got_d) -> dict:
    """Tie-aware comparison of an IVF result against the brute-force truth.

    Identity-based recall is meaningless on this pool: the external corpora share large
    numbers of literally identical structures (MOF-off R2SCAN vs PBE, AM Full vs MatPES),
    so many neighbours sit at distance ~0 and an exact tie can be broken either way.  A
    ratio-based distance inflation is meaningless for the same reason -- it divides by a
    zero true distance.  What matters is whether the retrieved neighbour is *as close*,
    so these are absolute-distance metrics.
    """
    exact_i_np = cp.asnumpy(exact_i)
    exact_d_np = np.sqrt(np.maximum(cp.asnumpy(exact_d), 0.0))
    overlap = [len(set(exact_i_np[r]).intersection(got_i[r].tolist()))
               for r in range(exact_i_np.shape[0])]
    gap = np.maximum(got_d[:, 0] - exact_d_np[:, 0], 0.0)
    tolerance = 1e-6 + exact_d_np[:, 0] * 1e-5
    return {
        "equivalent_top1_distance_fraction": float(np.mean(gap <= tolerance)),
        "identical_top1_id_fraction": float(np.mean(got_i[:, 0] == exact_i_np[:, 0])),
        "mean_recall_at_10_by_id": float(np.mean(overlap) / exact_i_np.shape[1]),
        "abs_d1_gap": {"p50": float(np.quantile(gap, 0.5)),
                       "p99": float(np.quantile(gap, 0.99)),
                       "max": float(np.max(gap))},
        "true_d1_median": float(np.median(exact_d_np[:, 0])),
    }


def _panel_a_labels(manifest, layout, omat_i, omat_d, ext_i, ext_d, n_labels):
    """Panel-A top-5 label counts for one block of rows (shared by table and validate)."""
    n_offset = len(manifest.names)
    o_lab = manifest.code_of(omat_i.ravel()).reshape(omat_i.shape).astype(np.int32)
    e_lab = np.zeros(ext_i.shape, dtype=np.int32)
    for pos, seg in enumerate(layout):
        e_lab[(ext_i >= seg["start"]) & (ext_i < seg["stop"])] = n_offset + pos
    counts = np.zeros(n_labels, dtype=np.int64)
    _pick_top5(counts, np.concatenate((omat_d, ext_d), axis=1),
               np.concatenate((o_lab, e_lab), axis=1), n_labels)
    return counts


def command_validate(args: argparse.Namespace) -> None:
    """Brute-force check that the IVF searches did not miss true external neighbours.

    Two paths are checked per query set: ``ext_all`` (the top-256 search over the whole
    external pool, which feeds panel A) and, for the AM-owned sets, ``ext_foreign``
    (the dedicated AM-free block index, which feeds panel B).
    """
    import cupy as cp

    layout = segments()
    small_stop = next(s["start"] for s in layout if s["name"] == "AM Full")
    external = np.load(OUT_ROOT / "external_pc25.npy", mmap_mode="r")
    reference_gpu = cp.asarray(np.asarray(external), dtype=cp.float32, order="C")
    manifest = OmatManifest(OMAT_MANIFEST)
    labels, _ = pool_labels(manifest, layout)
    report = {}
    rng = np.random.default_rng(args.seed)
    pool = cp.get_default_memory_pool()
    for name in QUERIES:
        out_dir = OUT_ROOT / slug(name)
        if not (out_dir / "metadata.json").is_file():
            continue
        pool.free_all_blocks()
        queries, self_rows = query_pc25(name, layout, external)
        n = queries.shape[0]
        pick = np.sort(rng.choice(n, size=min(args.rows, n), replace=False))
        q = cp.asarray(np.ascontiguousarray(queries[pick]), dtype=cp.float32)

        best_d, best_i = _brute_force(cp, reference_gpu, q, reference_gpu.shape[0],
                                      args.block_rows, K_STORED + 1)
        if self_rows is not None:
            selves = cp.asarray(self_rows[pick])[:, None]
            best_d = cp.where(best_i == selves, cp.inf, best_d)
        exact_d, exact_i = _top(cp, best_d, best_i, K_STORED)
        got_i = np.load(out_dir / "ext_all_idx.npy", mmap_mode="r")[pick]
        got_d = np.load(out_dir / "ext_all_dist.npy", mmap_mode="r")[pick]
        entry = {"rows_checked": int(pick.size),
                 "ext_all": _compare(cp, exact_d, exact_i, got_i, got_d)}

        # The metric that actually certifies the table: do the panel-A top-5 *label*
        # shares move if the external side is replaced by the brute-force truth?
        stored = PROBE / QUERIES[name][2]
        o_i = np.asarray(np.load(stored / "indices.npy", mmap_mode="r")[pick]).astype(np.int64)
        o_d = np.asarray(np.load(stored / "distances.npy", mmap_mode="r")[pick]).astype(np.float32)
        share_ivf = shares(_panel_a_labels(manifest, layout, o_i, o_d,
                                           np.asarray(got_i), np.asarray(got_d), len(labels)))
        share_exact = shares(_panel_a_labels(
            manifest, layout, o_i, o_d, cp.asnumpy(exact_i),
            np.sqrt(np.maximum(cp.asnumpy(exact_d), 0.0)), len(labels)))
        delta = np.abs(share_ivf - share_exact)
        entry["panel_a_share_agreement"] = {
            "max_abs_share_delta_pp": float(delta.max()),
            "worst_label": labels[int(delta.argmax())],
            "total_abs_share_delta_pp": float(delta.sum()),
        }

        if QUERIES[name][1] == "AM Full":
            f_d, f_i = _brute_force(cp, reference_gpu, q, small_stop, args.block_rows, K_STORED)
            entry["ext_foreign_am_free_block"] = _compare(
                cp, f_d, f_i,
                np.load(out_dir / "ext_foreign_idx.npy", mmap_mode="r")[pick],
                np.load(out_dir / "ext_foreign_dist.npy", mmap_mode="r")[pick])
        report[name] = entry
        log(f"validate {name}: {json.dumps(entry)}")
    atomic_json(OUT_ROOT / "ivf_exactness_validation.json",
                {"created_utc": utc_now(), "k_search": K_SEARCH,
                 "index": {"nlist": NLIST, "nprobe": NPROBE},
                 "am_free_index": {"nlist": 2048, "nprobe": 512, "rows": int(small_stop)},
                 "datasets": report})


# -------------------------------------------------------------------------- table


class OmatManifest:
    """global OMAT row -> subdataset code, from the shard manifest."""

    def __init__(self, path: Path):
        rows = list(csv.DictReader(path.open()))
        self.start = np.array([int(r["global_start"]) for r in rows], dtype=np.int64)
        self.stop = np.array([int(r["global_stop"]) for r in rows], dtype=np.int64)
        if self.start[0] != 0 or not np.all(self.start[1:] == self.stop[:-1]):
            raise SystemExit("manifest shards are not contiguous")
        names = [r["subdataset"] for r in rows]
        self.names = sorted(set(names))
        self.code = {name: index for index, name in enumerate(self.names)}
        self.shard_code = np.array([self.code[n] for n in names], dtype=np.int16)
        self.population = np.zeros(len(self.names), dtype=np.int64)
        for r, name in zip(rows, names):
            self.population[self.code[name]] += int(r["n_rows"])
        self.total = int(self.population.sum())

    def code_of(self, global_rows: np.ndarray) -> np.ndarray:
        shard = np.searchsorted(self.start, global_rows, side="right") - 1
        if np.any(global_rows >= self.stop[shard]) or np.any(global_rows < 0):
            raise SystemExit("global row fell outside its manifest shard")
        return self.shard_code[shard]


def pool_labels(manifest: OmatManifest, layout: list[dict]) -> tuple[list[str], np.ndarray]:
    labels = list(manifest.names) + [seg["name"] for seg in layout]
    population = np.concatenate([manifest.population,
                                 np.array([seg["rows"] for seg in layout], dtype=np.int64)])
    return labels, population


def _pick_top5(counts, merged_d, merged_lab, n_labels):
    """Accumulate the labels of the 5 smallest finite distances per row."""
    order = np.argsort(merged_d, axis=1, kind="stable")[:, :5]
    picked_d = np.take_along_axis(merged_d, order, axis=1)
    picked_lab = np.take_along_axis(merged_lab, order, axis=1)
    good = np.isfinite(picked_d)
    counts += np.bincount(picked_lab[good].ravel(), minlength=n_labels)


def merge_panels(name: str, manifest: OmatManifest, layout: list[dict], n_labels: int,
                 mask: np.ndarray | None = None):
    """Top-5 pool-label counts for panels A, B and C over the (optionally masked) rows."""
    out_dir = OUT_ROOT / slug(name)
    stored = PROBE / QUERIES[name][2]
    omat_i = np.load(stored / "indices.npy", mmap_mode="r")
    omat_d = np.load(stored / "distances.npy", mmap_mode="r")
    ext = {key: np.load(out_dir / f"ext_{key}.npy", mmap_mode="r")
           for key in ("all_idx", "all_dist", "foreign_idx", "foreign_dist",
                       "nodup_idx", "nodup_dist")}
    n = ext["all_idx"].shape[0]
    if omat_i.shape[0] != n:
        raise SystemExit(f"{name}: stored OMAT run has {omat_i.shape[0]} rows, external has {n}")
    is_omat_query = QUERIES[name][1] is None
    rows = np.arange(n, dtype=np.int64) if mask is None else np.flatnonzero(mask)

    counts = [np.zeros(n_labels, dtype=np.int64) for _ in range(3)]
    n_offset = len(manifest.names)

    def ext_labels(idx):
        lab = np.zeros(idx.shape, dtype=np.int32)
        for pos, seg in enumerate(layout):
            lab[(idx >= seg["start"]) & (idx < seg["stop"])] = n_offset + pos
        return lab

    step = 500_000
    for start in range(0, rows.size, step):
        pick = rows[start:start + step]
        o_i = np.asarray(omat_i[pick]).astype(np.int64)
        o_d = np.asarray(omat_d[pick]).astype(np.float32)
        o_lab = manifest.code_of(o_i.ravel()).reshape(o_i.shape).astype(np.int32)

        a_d = np.asarray(ext["all_dist"][pick]).astype(np.float32)
        a_lab = ext_labels(np.asarray(ext["all_idx"][pick]).astype(np.int64))
        _pick_top5(counts[0], np.concatenate((o_d, a_d), axis=1),
                   np.concatenate((o_lab, a_lab), axis=1), n_labels)

        f_d = np.asarray(ext["foreign_dist"][pick]).astype(np.float32)
        f_lab = ext_labels(np.asarray(ext["foreign_idx"][pick]).astype(np.int64))
        if is_omat_query:
            # Panel B for OMAT24 queries removes the OMAT24 half of the pool entirely.
            _pick_top5(counts[1], f_d, f_lab, n_labels)
        else:
            _pick_top5(counts[1], np.concatenate((o_d, f_d), axis=1),
                       np.concatenate((o_lab, f_lab), axis=1), n_labels)

        # Panel C drops near-duplicates on BOTH sides of the pool, not just the external
        # half -- an OMAT row can restate the query too.
        c_d = np.asarray(ext["nodup_dist"][pick]).astype(np.float32)
        c_lab = ext_labels(np.asarray(ext["nodup_idx"][pick]).astype(np.int64))
        o_d_nd = np.where(o_d < DUP_TOL, np.inf, o_d)
        _pick_top5(counts[2], np.concatenate((o_d_nd, c_d), axis=1),
                   np.concatenate((o_lab, c_lab), axis=1), n_labels)
    return counts[0], counts[1], counts[2], int(rows.size)


def shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return 100.0 * counts / total if total else np.zeros_like(counts, dtype=float)


def markdown_table(title: str, note: str, labels, population, rows) -> str:
    lines = [f"## {title}", "", note, "",
             "| query set | rows | " + " | ".join(labels) + " |",
             "|---|---:|" + "---:|" * len(labels)]
    pop = shares(population)
    lines.append("| **pool population share** | "
                 + f"{population.sum():,} | "
                 + " | ".join(f"{v:.3f}" for v in pop) + " |")
    for name, count, share in rows:
        lines.append(f"| {name} | {count:,} | " + " | ".join(f"{v:.3f}" for v in share) + " |")
    return "\n".join(lines)


def provenance_section(meta: dict) -> str:
    """How the numbers were produced and how far they can be trusted."""
    lines = ["## Provenance and accuracy", "",
             "The OMAT24 half of every query set is *reused* from the published pc25 runs "
             "(validated recall@1 = 1.0, zero distance inflation); only the external half "
             "was computed here.  Merging is exact for a top-5 because those runs stored "
             "10 neighbours and a top-5 can draw at most 5 from OMAT.", ""]
    validation = OUT_ROOT / "ivf_exactness_validation.json"
    if validation.is_file():
        report = json.loads(validation.read_text()).get("datasets", {})
        lines += [
            "Accuracy of the external IVF search, checked against brute force on 2,000 "
            "sampled rows per query set.  Identity-based recall is not meaningful on this "
            "pool -- the corpora share literally identical structures, so many neighbours "
            "tie at distance 0 -- hence absolute distance gaps, plus the metric that "
            "actually matters: how far the panel A top-5 *shares* move when the IVF "
            "external side is replaced by the exact answer.", "",
            "| query set | true median d1 | d1 gap p50 | d1 gap p99 | d1 gap max | "
            "max share shift (pp) | foreign cov@5 | non-dup cov@5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ROW_ORDER:
            entry = report.get(name)
            if entry is None:
                continue
            gap = entry["ext_all"]["abs_d1_gap"]
            search = meta.get(name, {}).get("search", {})
            lines.append(
                f"| {name} | {entry['ext_all']['true_d1_median']:.4f} | {gap['p50']:.1e} | "
                f"{gap['p99']:.1e} | {gap['max']:.1e} | "
                f"{entry['panel_a_share_agreement']['max_abs_share_delta_pp']:.3f} | "
                f"{search.get('foreign_coverage_at_5', float('nan')):.6f} | "
                f"{search.get('nodup_coverage_at_5', float('nan')):.6f} |")
        lines += ["", "`foreign cov@5` is the fraction of queries with at least 5 non-own-corpus "
                  "candidates available for panel B; `non-dup cov@5` the same for panel C.  "
                  "The AM-owned sets get panel B from a dedicated index over the AM-free "
                  "pool prefix, so their exclusion is exact rather than coverage-limited."]
    return "\n".join(lines)


def command_table(args: argparse.Namespace) -> None:
    manifest = OmatManifest(OMAT_MANIFEST)
    layout = segments()
    labels, population = pool_labels(manifest, layout)
    n_labels = len(labels)

    panel_a, panel_b, panel_c, meta = [], [], [], {}
    for name in QUERIES:
        if not (OUT_ROOT / slug(name) / "metadata.json").is_file():
            log(f"skipping {name}: no search output")
            continue
        count_a, count_b, count_c, n = merge_panels(name, manifest, layout, n_labels)
        meta[name] = {"rows": n,
                      "search": json.loads((OUT_ROOT / slug(name) / "metadata.json").read_text())}
        panel_a.append((name, n, shares(count_a)))
        panel_b.append((name, n, shares(count_b)))
        panel_c.append((name, n, shares(count_c)))
        if name == "OMAT24 (leave-one-out)":
            # The published comparator row: OMAT24 rows that are not themselves nvt-3000.
            source = np.load(PROBE / QUERIES[name][2] / "source_rows.npy")
            outside = manifest.code_of(source) != manifest.code[FOCUS]
            ca, cb, cc, rows_out = merge_panels(name, manifest, layout, n_labels, mask=outside)
            tag = f"OMAT24 rows outside {FOCUS}"
            panel_a.append((tag, rows_out, shares(ca)))
            panel_b.append((tag, rows_out, shares(cb)))
            panel_c.append((tag, rows_out, shares(cc)))

    order = {name: i for i, name in enumerate(ROW_ORDER)}
    for panel in (panel_a, panel_b, panel_c):
        panel.sort(key=lambda r: order.get(r[0], 99))

    union = population.sum()
    header = (
        f"# pc25 top-5 neighbour attribution over the **union** pool\n\n"
        f"Pool = OMAT24 train ({manifest.total:,}) plus every external corpus "
        f"({external_rows():,}) = **{union:,} rows**.  Geometry unchanged: 25-PC "
        f"standardised UMA basis fitted on OMAT24, Euclidean.  Each query contributes "
        f"its 5 nearest pool rows; each slot is attributed to the corpus (or OMAT24 "
        f"subdataset) of the row it landed on.  AM Small is an exact 100k subsample of "
        f"AM Full and is therefore a query-only row, never a separate pool member.\n"
    )
    body = [header,
            markdown_table(
                "Panel A -- self-excluded (the literal union-pool answer)",
                "Every pool row is a candidate except the query's own row.  For a corpus "
                "of AIMD frames this is dominated by the query's own trajectory twins, "
                "exactly as OMAT24's own leave-one-out row is.",
                labels, population, panel_a),
            "",
            markdown_table(
                "Panel B -- own corpus excluded (comparable to the published OMAT-only table)",
                "The query's entire own corpus is removed from the pool, so the question is "
                "which *foreign* corpus wins.  For the OMAT24 rows the whole OMAT24 half is "
                "removed, so those two rows show where an OMAT frame goes when it may only "
                "retrieve external data.",
                labels, population, panel_b),
            "",
            markdown_table(
                f"Panel C -- near-duplicates excluded (distance < {DUP_TOL:g})",
                "Panel A minus every pool row that is the query restated: a relabelled copy "
                "in a sibling corpus, or the same structure re-embedded.  The threshold is "
                "measured from 100,000 known same-structure AM Small / AM Full pairs "
                "(median 1.3e-6, max 1.11e-4), not chosen.  This is the panel to read for "
                "'what is the nearest genuinely different structure'.",
                labels, population, panel_c),
            "",
            provenance_section(meta)]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "union_pool_subdataset_shares.md").write_text("\n".join(body) + "\n")

    with (OUT_ROOT / "union_pool_subdataset_shares.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "query_set", "queries"] + labels)
        writer.writerow(["population", "pool", union] + [f"{v:.6f}" for v in shares(population)])
        for panel, rows in (("A_self_excluded", panel_a), ("B_corpus_excluded", panel_b),
                            ("C_duplicates_excluded", panel_c)):
            for name, n, share in rows:
                writer.writerow([panel, name, n] + [f"{v:.6f}" for v in share])
    atomic_json(OUT_ROOT / "table_metadata.json",
                {"created_utc": utc_now(), "labels": labels,
                 "population": population.tolist(), "union_rows": int(union),
                 "queries": meta})
    log(f"Wrote {OUT_ROOT / 'union_pool_subdataset_shares.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("project", help="project external corpora into the frozen pc25 basis")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=command_project)

    p = sub.add_parser("search", help="kNN of each query set against the external pool")
    p.add_argument("--dataset", default="all")
    p.add_argument("--batch-rows", type=int, default=16384)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=command_search)

    p = sub.add_parser("validate", help="brute-force exactness check of the IVF search")
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--block-rows", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=20260816)
    p.set_defaults(func=command_validate)

    p = sub.add_parser("table", help="merge with the stored OMAT side and write the table")
    p.set_defaults(func=command_table)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
