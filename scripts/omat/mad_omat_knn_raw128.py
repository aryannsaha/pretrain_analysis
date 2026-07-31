#!/usr/bin/env python
"""Exact MAD -> OMAT24 top-10 neighbors in raw 128-d UMA embedding space.

Brute-force GPU scan (torch): the full OMAT24 embedding matrix is held resident
on one GPU and block-scanned per query tile with d^2 = |q|^2 + |r|^2 - 2 q.r^T,
keeping a running top-`keep` candidate set that is then re-ranked with an exact
direct-difference fp32 pass. Distances are Euclidean on the raw float32
embeddings; no PCA and no standardization (geometry: raw_uma_128d_euclidean).

Subcommands:
  query  GPU scan for one shard of query rows; writes per-shard distances.npy /
         indices.npy / metadata.json with an idempotency signature. Optionally
         runs an fp64 full-reference audit on a seeded subset of its rows.
  merge  Concatenate shard outputs, run structural checks, write per-dataset
         distances.npy / indices.npy / summary.json.
  join   Write neighbors_top10_joined.csv (manifest + frame-ids join) and an
         fp64 spot recompute of stored distances.

Clone of scripts/omat/mof_omat_knn_raw128.py with the query sets switched to the
three native MAD splits; the scan/audit/merge algorithm is unchanged. This is a
raw-128-d run only -- no pc25 comparison is computed here.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY = "raw_uma_128d_euclidean"
DEFAULT_REFERENCE = ROOT / "data/processed/omat24/uma_latents_copy/all_uma_embeddings.npy"
DEFAULT_MANIFEST = ROOT / "data/processed/omat24/uma_latents_copy/all_uma_embeddings_manifest.csv"
DEFAULT_RUN_ROOT = ROOT / "runs/omat_knn_probe/omat_knn_raw128_mad"
DEFAULT_QUERIES = {
    "MAD_train": {
        "matrix": ROOT / "data/processed/mad/uma_latents/train/mad_train.npy",
        "sample_ids": ROOT / "data/processed/mad/train/mad_train_frames.tsv",
    },
    "MAD_val": {
        "matrix": ROOT / "data/processed/mad/uma_latents/val/mad_val.npy",
        "sample_ids": ROOT / "data/processed/mad/val/mad_val_frames.tsv",
    },
    "MAD_test": {
        "matrix": ROOT / "data/processed/mad/uma_latents/test/mad_test.npy",
        "sample_ids": ROOT / "data/processed/mad/test/mad_test_frames.tsv",
    },
}
JOINED_CSV_FIELDS = [
    "mad_row", "mad_subset", "mad_split", "natoms", "pbc", "rank",
    "euclidean_distance", "omat_global_row", "subdataset", "shard", "local_row", "raw_path",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def matrix_info(path):
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"{path} must be a two-dimensional .npy matrix, got {matrix.shape}")
    stat = Path(path).stat()
    info = {
        "path": str(Path(path).resolve()),
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "dtype": str(matrix.dtype),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return matrix, info


def shard_interval(total_rows, num_shards, shard_index):
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index {shard_index} outside [0, {num_shards})")
    base, remainder = divmod(total_rows, num_shards)
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + (1 if shard_index < remainder else 0)
    return start, stop


def query_signature(ref_info, query_info, args, row_start, row_stop):
    return {
        "geometry": GEOMETRY,
        "reference": ref_info,
        "query": query_info,
        "row_start": row_start,
        "row_stop": row_stop,
        "neighbors": args.neighbors,
        "keep": args.keep,
        "block_rows": args.block_rows,
        "tile_rows": args.tile_rows,
        "tf32": bool(args.tf32),
    }


def completed_output(shard_dir, signature):
    metadata_path = shard_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False
    return bool(metadata.get("complete")) and metadata.get("signature") == signature


def load_reference_gpu(torch, path, chunk_rows):
    source, info = matrix_info(path)
    log(f"Loading reference to GPU: {info['path']} shape={tuple(source.shape)}")
    started = time.monotonic()
    reference = torch.empty(source.shape, dtype=torch.float32, device="cuda")
    # Row norms are accumulated per load block. Computing them as one
    # (reference * reference).sum(dim=1) would materialize a second full-size
    # 48.08 GiB temporary next to the resident reference and cannot fit on an
    # 80 GiB card; per-block the extra peak is only chunk_rows x 128 x 4 bytes.
    reference_sq = torch.empty(source.shape[0], dtype=torch.float32, device="cuda")
    for start in range(0, source.shape[0], chunk_rows):
        stop = min(start + chunk_rows, source.shape[0])
        block = np.ascontiguousarray(source[start:stop])
        reference[start:stop].copy_(torch.from_numpy(block))
        resident = reference[start:stop]
        reference_sq[start:stop] = (resident * resident).sum(dim=1)
    torch.cuda.synchronize()
    log(f"Reference resident on GPU in {time.monotonic() - started:.1f}s")
    return reference, reference_sq, info


class Tf32Context:
    """Enable TF32 matmul only inside the block-scan GEMM."""

    def __init__(self, torch, enabled):
        self.torch = torch
        self.enabled = enabled

    def __enter__(self):
        self.previous = self.torch.backends.cuda.matmul.allow_tf32
        self.torch.backends.cuda.matmul.allow_tf32 = self.enabled

    def __exit__(self, *_):
        self.torch.backends.cuda.matmul.allow_tf32 = self.previous


def scan_shard(torch, reference, reference_sq, query_rows, args):
    rows = query_rows.shape[0]
    n_reference = reference.shape[0]
    keep = min(args.keep, n_reference)
    out_distances = np.empty((rows, args.neighbors), dtype=np.float32)
    out_indices = np.empty((rows, args.neighbors), dtype=np.int64)
    for tile_start in range(0, rows, args.tile_rows):
        tile_stop = min(tile_start + args.tile_rows, rows)
        tile = torch.from_numpy(np.ascontiguousarray(query_rows[tile_start:tile_stop])).cuda()
        tile_sq = (tile * tile).sum(dim=1, keepdim=True)
        best_d2 = torch.full((tile.shape[0], keep), float("inf"), dtype=torch.float32, device="cuda")
        best_idx = torch.full((tile.shape[0], keep), -1, dtype=torch.int64, device="cuda")
        for start in range(0, n_reference, args.block_rows):
            stop = min(start + args.block_rows, n_reference)
            block = reference[start:stop]
            with Tf32Context(torch, args.tf32):
                gram = tile @ block.T
            squared = tile_sq + reference_sq[start:stop][None, :] - 2.0 * gram
            local_keep = min(keep, stop - start)
            values, locations = torch.topk(squared, local_keep, dim=1, largest=False)
            best_d2 = torch.cat((best_d2, values), dim=1)
            best_idx = torch.cat((best_idx, locations.long() + start), dim=1)
            best_d2, selection = torch.topk(best_d2, keep, dim=1, largest=False)
            best_idx = torch.gather(best_idx, 1, selection)
        # Exact fp32 re-rank of the kept candidates (direct difference: no
        # cancellation, unaffected by any TF32 use in the scan above).
        candidates = reference[best_idx]
        squared = (candidates - tile[:, None, :]).pow(2).sum(dim=-1)
        squared, order = torch.sort(squared, dim=1)
        ranked_idx = torch.gather(best_idx, 1, order)
        distances = torch.sqrt(torch.clamp(squared[:, : args.neighbors], min=0.0))
        out_distances[tile_start:tile_stop] = distances.cpu().numpy()
        out_indices[tile_start:tile_stop] = ranked_idx[:, : args.neighbors].cpu().numpy()
        log(f"Scanned query rows {tile_stop:,}/{rows:,}")
    return out_distances, out_indices


def fp64_exact_topk(torch, reference, query_rows, neighbors, block_rows):
    queries = torch.from_numpy(query_rows.astype(np.float64)).cuda()
    query_sq = (queries * queries).sum(dim=1, keepdim=True)
    keep = max(neighbors * 4, 64)
    best_d2 = torch.full((queries.shape[0], keep), float("inf"), dtype=torch.float64, device="cuda")
    best_idx = torch.full((queries.shape[0], keep), -1, dtype=torch.int64, device="cuda")
    for start in range(0, reference.shape[0], block_rows):
        stop = min(start + block_rows, reference.shape[0])
        block = reference[start:stop].double()
        squared = query_sq + (block * block).sum(dim=1)[None, :] - 2.0 * (queries @ block.T)
        local_keep = min(keep, stop - start)
        values, locations = torch.topk(squared, local_keep, dim=1, largest=False)
        best_d2 = torch.cat((best_d2, values), dim=1)
        best_idx = torch.cat((best_idx, locations.long() + start), dim=1)
        best_d2, selection = torch.topk(best_d2, keep, dim=1, largest=False)
        best_idx = torch.gather(best_idx, 1, selection)
    candidates = reference[best_idx].double()
    squared = (candidates - queries[:, None, :]).pow(2).sum(dim=-1)
    squared, order = torch.sort(squared, dim=1)
    ranked_idx = torch.gather(best_idx, 1, order)
    distances = torch.sqrt(torch.clamp(squared[:, :neighbors], min=0.0))
    return distances.cpu().numpy(), ranked_idx[:, :neighbors].cpu().numpy()


def audit_against_fp64(torch, reference, query_rows, produced_distances, produced_indices,
                       neighbors, block_rows, sample_rows, seed):
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(query_rows.shape[0], min(sample_rows, query_rows.shape[0]), replace=False))
    exact_d, exact_i = fp64_exact_topk(torch, reference, query_rows[sample], neighbors, block_rows)
    got_d = produced_distances[sample]
    got_i = produced_indices[sample]
    matched = 0
    for row in range(sample.size):
        exact_set = set(exact_i[row].tolist())
        got_set = set(got_i[row].tolist())
        if exact_set == got_set:
            matched += 1
            continue
        # Tolerate tie permutations at the top-k boundary: every disagreeing
        # index must sit within float tolerance of the exact k-th distance.
        boundary = exact_d[row, -1]
        tolerance = 1e-6 * (1.0 + boundary)
        if np.all(np.abs(np.sort(got_d[row]) - exact_d[row]) <= tolerance):
            matched += 1
    relative_error = np.abs(got_d.astype(np.float64) - exact_d) / (1.0 + exact_d)
    return {
        "sampled_rows": sample.tolist(),
        "recall_at_k": matched / sample.size,
        "max_relative_distance_error": float(relative_error.max()),
        "mean_relative_distance_error": float(relative_error.mean()),
    }


def command_query(args):
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for the query stage")
    reference, ref_sq, ref_info = load_reference_gpu(torch, args.reference, args.load_chunk_rows)
    validations = {}
    for name, spec in DEFAULT_QUERIES.items():
        query, query_info = matrix_info(spec["matrix"])
        if query.shape[1] != reference.shape[1]:
            raise SystemExit(f"{name}: query dim {query.shape[1]} != reference dim {reference.shape[1]}")
        row_start, row_stop = shard_interval(query.shape[0], args.num_shards, args.shard_index)
        shard_dir = args.output_root / name / "shards" / f"part-{args.shard_index:05d}-of-{args.num_shards:05d}"
        signature = query_signature(ref_info, query_info, args, row_start, row_stop)
        if completed_output(shard_dir, signature) and not args.force:
            log(f"Reusing completed shard: {shard_dir}")
            if args.validation_rows > 0:
                distances = np.load(shard_dir / "distances.npy")
                indices = np.load(shard_dir / "indices.npy")
                rows = np.ascontiguousarray(query[row_start:row_stop])
                validations[name] = audit_against_fp64(
                    torch, reference, rows, distances, indices,
                    args.neighbors, args.block_rows, args.validation_rows, args.seed)
            continue
        log(f"Querying {name} rows [{row_start:,}, {row_stop:,}) of {query.shape[0]:,}")
        started = time.monotonic()
        rows = np.ascontiguousarray(query[row_start:row_stop])
        distances, indices = scan_shard(torch, reference, ref_sq, rows, args)
        atomic_npy(shard_dir / "distances.npy", distances)
        atomic_npy(shard_dir / "indices.npy", indices)
        atomic_json(shard_dir / "metadata.json", {
            "complete": True,
            "created_utc": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "signature": signature,
        })
        log(f"Wrote shard {shard_dir} in {time.monotonic() - started:.1f}s")
        if args.validation_rows > 0:
            validations[name] = audit_against_fp64(
                torch, reference, rows, distances, indices,
                args.neighbors, args.block_rows, args.validation_rows, args.seed)
    if validations:
        failed = {
            name: report for name, report in validations.items()
            if report["recall_at_k"] < 1.0 or report["max_relative_distance_error"] > 1e-5
        }
        atomic_json(args.output_root / "full_reference_fp64_validation.json", {
            "created_utc": utc_now(),
            "geometry": GEOMETRY,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "neighbors": args.neighbors,
            "tf32": bool(args.tf32),
            "gate": "recall_at_k == 1.0 and max_relative_distance_error <= 1e-5",
            "passed": not failed,
            "datasets": validations,
        })
        if failed:
            raise SystemExit(f"fp64 audit FAILED for: {sorted(failed)}; rerun with --no-tf32 or inspect")
        log("fp64 audit passed for all datasets")
    if args.shard_index == 0:
        atomic_json(args.output_root / "run_configuration.json", {
            "created_utc": utc_now(),
            "geometry": GEOMETRY,
            "reference": ref_info,
            "queries": {name: str(spec["matrix"]) for name, spec in DEFAULT_QUERIES.items()},
            "neighbors": args.neighbors,
            "keep": args.keep,
            "block_rows": args.block_rows,
            "tile_rows": args.tile_rows,
            "tf32": bool(args.tf32),
            "num_shards": args.num_shards,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        })


def merge_dataset(dataset_dir, num_shards, neighbors, reference_rows):
    parts, row_stops = [], []
    expected_start = 0
    for shard_index in range(num_shards):
        shard_dir = dataset_dir / "shards" / f"part-{shard_index:05d}-of-{num_shards:05d}"
        metadata = json.loads((shard_dir / "metadata.json").read_text())
        if not metadata.get("complete"):
            raise SystemExit(f"Incomplete shard: {shard_dir}")
        signature = metadata["signature"]
        if signature["row_start"] != expected_start:
            raise SystemExit(f"Non-contiguous shards at {shard_dir}: "
                             f"start {signature['row_start']} != expected {expected_start}")
        expected_start = signature["row_stop"]
        parts.append((np.load(shard_dir / "distances.npy"), np.load(shard_dir / "indices.npy")))
        row_stops.append(signature["row_stop"])
    distances = np.concatenate([p[0] for p in parts], axis=0)
    indices = np.concatenate([p[1] for p in parts], axis=0)
    if distances.shape != (expected_start, neighbors) or indices.shape != distances.shape:
        raise SystemExit(f"Merged shape mismatch in {dataset_dir}: {distances.shape} / {indices.shape}")
    if distances.dtype != np.float32 or indices.dtype != np.int64:
        raise SystemExit(f"Merged dtype mismatch in {dataset_dir}")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise SystemExit(f"Non-finite or negative distances in {dataset_dir}")
    if np.any(np.diff(distances, axis=1) < -1e-6 * (1.0 + distances[:, :-1])):
        raise SystemExit(f"Distances not ascending within rows in {dataset_dir}")
    if np.any(indices < 0) or np.any(indices >= reference_rows):
        raise SystemExit(f"Neighbor indices out of range in {dataset_dir}")
    atomic_npy(dataset_dir / "distances.npy", distances)
    atomic_npy(dataset_dir / "indices.npy", indices)
    return distances, indices


def load_manifest(path):
    import pandas as pd

    manifest = pd.read_csv(path)
    stops = manifest["global_stop"].to_numpy(np.int64)
    if np.any(np.diff(stops) <= 0):
        raise SystemExit(f"Manifest global_stop not strictly increasing: {path}")
    return manifest, stops


def neighbor_source_frame(manifest, stops, flat_indices):
    rows = np.searchsorted(stops, flat_indices, side="right")
    starts = manifest["global_start"].to_numpy(np.int64)[rows]
    counts = manifest["n_rows"].to_numpy(np.int64)[rows]
    local = flat_indices - starts
    if np.any(local < 0) or np.any(local >= counts):
        raise SystemExit("Manifest lookup produced out-of-range local rows")
    return {
        "subdataset": manifest["subdataset"].to_numpy(object)[rows],
        "shard": manifest["shard"].to_numpy(object)[rows],
        "local_row": local,
        "raw_path": manifest["raw_path"].to_numpy(object)[rows],
    }


def command_merge(args):
    reference, _ = matrix_info(args.reference)
    manifest, stops = load_manifest(args.manifest)
    for name in DEFAULT_QUERIES:
        dataset_dir = args.output_root / name
        distances, indices = merge_dataset(dataset_dir, args.num_shards, args.neighbors, reference.shape[0])
        sources = neighbor_source_frame(manifest, stops, indices.ravel())
        unique, counts = np.unique(sources["subdataset"], return_counts=True)
        atomic_json(dataset_dir / "summary.json", {
            "created_utc": utc_now(),
            "geometry": GEOMETRY,
            "rows": int(distances.shape[0]),
            "neighbors": int(distances.shape[1]),
            "d1": {"min": float(distances[:, 0].min()), "median": float(np.median(distances[:, 0])),
                   "max": float(distances[:, 0].max())},
            "d10": {"min": float(distances[:, -1].min()), "median": float(np.median(distances[:, -1])),
                    "max": float(distances[:, -1].max())},
            "neighbor_subdataset_counts": {str(k): int(v) for k, v in zip(unique, counts, strict=True)},
        })
        log(f"Merged {name}: {distances.shape[0]:,} rows")


def spot_recompute(reference_path, query_path, distances, indices, sample_rows, seed):
    reference = np.load(reference_path, mmap_mode="r")
    query = np.load(query_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(distances.shape[0], min(sample_rows, distances.shape[0]), replace=False))
    worst = 0.0
    for row in sample:
        q = query[row].astype(np.float64)
        neighbors = reference[indices[row]].astype(np.float64)
        exact = np.sqrt(((neighbors - q[None, :]) ** 2).sum(axis=1))
        worst = max(worst, float(np.max(np.abs(exact - distances[row].astype(np.float64)) / (1.0 + exact))))
    return {"sampled_rows": int(sample.size), "max_relative_distance_error": worst}


def command_join(args):
    import pandas as pd

    manifest, stops = load_manifest(args.manifest)
    for name, spec in DEFAULT_QUERIES.items():
        dataset_dir = args.output_root / name
        distances = np.load(dataset_dir / "distances.npy")
        indices = np.load(dataset_dir / "indices.npy")
        ids = pd.read_csv(spec["sample_ids"], sep="\t")
        if len(ids) != distances.shape[0]:
            raise SystemExit(f"{name}: frame ids rows {len(ids)} != result rows {distances.shape[0]}")
        if not np.array_equal(ids["frame"].to_numpy(), np.arange(len(ids))):
            raise SystemExit(f"{name}: frame ids are not positional (frame != row order)")
        recompute = spot_recompute(args.reference, spec["matrix"], distances, indices,
                                   args.spot_rows, args.seed)
        if recompute["max_relative_distance_error"] > 1e-5:
            raise SystemExit(f"{name}: stored distances failed fp64 spot recompute: {recompute}")
        log(f"{name}: fp64 spot recompute passed ({recompute})")
        rows, k = indices.shape
        sources = neighbor_source_frame(manifest, stops, indices.ravel())
        joined = pd.DataFrame({
            "mad_row": np.repeat(np.arange(rows, dtype=np.int64), k),
            "mad_subset": np.repeat(ids["subset"].to_numpy(object), k),
            "mad_split": np.repeat(ids["split"].to_numpy(object), k),
            "natoms": np.repeat(ids["natoms"].to_numpy(), k),
            "pbc": np.repeat(ids["pbc"].to_numpy(object), k),
            "rank": np.tile(np.arange(1, k + 1, dtype=np.int64), rows),
            "euclidean_distance": distances.ravel(),
            "omat_global_row": indices.ravel(),
            "subdataset": sources["subdataset"],
            "shard": sources["shard"],
            "local_row": sources["local_row"],
            "raw_path": sources["raw_path"],
        }, columns=JOINED_CSV_FIELDS)
        output_csv = dataset_dir / "neighbors_top10_joined.csv"
        joined.to_csv(output_csv, index=False, float_format="%.7g")
        log(f"Wrote {output_csv} ({len(joined):,} rows)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["query", "merge", "join"])
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--keep", type=int, default=128, help="Candidate depth kept during the scan before exact re-ranking.")
    parser.add_argument("--block-rows", type=int, default=262_144, help="Reference rows per scan block.")
    parser.add_argument("--tile-rows", type=int, default=8_192, help="Query rows per GPU tile.")
    parser.add_argument("--load-chunk-rows", type=int, default=2_000_000, help="Host->GPU staging chunk for the reference matrix.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False, help="Use TF32 tensor cores for the scan GEMM (exact fp32 re-rank still applies).")
    parser.add_argument("--validation-rows", type=int, default=64, help="Per-dataset fp64 full-reference audit rows (query stage; 0 disables).")
    parser.add_argument("--spot-rows", type=int, default=1_000, help="Rows for the fp64 spot recompute in the join stage.")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--force", action="store_true", help="Recompute shards even if a matching completed output exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "query":
        command_query(args)
    elif args.command == "merge":
        command_merge(args)
    else:
        command_join(args)


if __name__ == "__main__":
    main()
