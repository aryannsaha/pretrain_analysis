#!/usr/bin/env python
"""Sample OMAT24 structures of one subdataset by weighted-PC1 range and overlay them on the by-source UMA PCA backdrop.

Weighted PC1/PC2 are computed exactly as in ``plot_uma_direct_pca_by_source.py``:
standardized 128-d UMA latent @ components[:2].T * retained_variance_ratio[:2].
The backdrop is re-rendered from that script's histogram cache so it matches the
``by_source/<subdataset>.png`` figure pixel for pixel, and the sampled structures
are drawn on top.

Outputs (under --output-dir):
  <range_name>/NN_<formula>_g<global_row>.extxyz   one ASE extxyz file per sampled structure
  <range_name>/samples.csv                          row provenance + PC1/PC2 + energy per structure
  <subdataset>_pc12.npz                             cached weighted PC1/PC2 for every row of the subdataset
  samples_on_backdrop.png                           the backdrop with the sampled structures highlighted
  metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SUBDATASET = "aimd-from-PBE-3000-nvt"
# The nvt-3000 cloud has a dense island left of PC1 = 0 and a diffuse fan to the right;
# PC1 < -0.75 isolates the island, PC1 > 0 the fan.
DEFAULT_RANGES = ["pc1_lt_-0.75:-inf:-0.75", "pc1_gt_0:0:inf"]
MARKERS = ["o", "^", "s", "D", "P", "X"]
MARKER_COLORS = ["#1f77b4", "#117733", "#9467bd", "#000000", "#ff7f0e", "#17becf"]
# Label offsets in points, fanned so up to six neighbouring markers get distinct label positions.
LABEL_OFFSETS = [(-28, 26), (-34, -22), (-40, 48), (-46, -46), (-52, 72), (-58, -70)]


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def parse_range(spec: str) -> tuple[str, float, float]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"range must be NAME:LO:HI, got {spec!r}")
    name, lo, hi = parts
    return slugify(name), float(lo), float(hi)


def resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_pca(path: Path):
    with np.load(path) as p:
        mean = np.asarray(p["mean"], dtype=np.float64)
        scale = np.asarray(p["scale"], dtype=np.float64)
        components = np.asarray(p["components"], dtype=np.float64)[:2]
        weights = np.asarray(p["retained_variance_ratio"], dtype=np.float64)[:2]
    return mean, scale, components, weights


def project_pc12(chunk: np.ndarray, mean, scale, components, weights) -> np.ndarray:
    return ((np.asarray(chunk, dtype=np.float64) - mean) / scale) @ components.T * weights


def read_manifest(path: Path, subdataset: str, root: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["subdataset"] != subdataset:
                continue
            rows.append(
                {
                    "global_start": int(row["global_start"]),
                    "global_stop": int(row["global_stop"]),
                    "n_rows": int(row["n_rows"]),
                    "shard": row["shard"],
                    "raw_path": resolve(Path(row["raw_path"]), root),
                    "embedding_path": resolve(Path(row["embedding_path"]), root),
                }
            )
    rows.sort(key=lambda r: r["global_start"])
    if not rows:
        raise SystemExit(f"no manifest rows for subdataset {subdataset!r} in {path}")
    return rows


def compute_scores(rows: list[dict], pca, cache: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (global_row, pc12) for every row of the subdataset, cached on disk."""
    n_total = sum(r["n_rows"] for r in rows)
    if cache is not None and cache.exists():
        with np.load(cache) as payload:
            global_row = np.asarray(payload["global_row"], dtype=np.int64)
            pc12 = np.asarray(payload["pc12"], dtype=np.float32)
        if global_row.size == n_total:
            print(f"Loaded cached scores: {cache} ({n_total:,} rows)", flush=True)
            return global_row, pc12
        print("Score cache has a different row count; recomputing", flush=True)
    mean, scale, components, weights = pca
    global_row = np.empty(n_total, dtype=np.int64)
    pc12 = np.empty((n_total, 2), dtype=np.float32)
    offset = 0
    for i, row in enumerate(rows):
        chunk = np.load(row["embedding_path"], mmap_mode="r")
        if chunk.shape[0] != row["n_rows"]:
            raise SystemExit(f"{row['embedding_path']}: {chunk.shape[0]} rows, manifest says {row['n_rows']}")
        pc12[offset : offset + row["n_rows"]] = project_pc12(chunk, mean, scale, components, weights)
        global_row[offset : offset + row["n_rows"]] = np.arange(row["global_start"], row["global_stop"])
        offset += row["n_rows"]
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            print(f"  projected {i + 1}/{len(rows)} shards ({offset:,} rows)", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, global_row=global_row, pc12=pc12)
        print(f"Saved score cache: {cache}", flush=True)
    return global_row, pc12


def open_shard(path: Path):
    try:
        from fairchem.core.datasets import AseDBDataset
    except ImportError:  # vendored fork used by the MACE scripts
        from mace.tools.fairchem_dataset import AseDBDataset
    return AseDBDataset(config={"src": str(path)})


def locate(global_row: int, rows: list[dict]) -> tuple[dict, int]:
    starts = np.array([r["global_start"] for r in rows], dtype=np.int64)
    i = int(np.searchsorted(starts, global_row, side="right") - 1)
    row = rows[i]
    if not (row["global_start"] <= global_row < row["global_stop"]):
        raise SystemExit(f"global row {global_row} not inside any manifest range")
    return row, int(global_row - row["global_start"])


def safe_energy(atoms):
    try:
        return float(atoms.get_potential_energy())
    except Exception:  # noqa: BLE001 - no calculator attached
        return None


def write_samples(name, picks, global_row, pc12, rows, subdataset, out_dir, shards):
    from ase.io import write as ase_write

    range_dir = out_dir / name
    range_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for rank, idx in enumerate(picks, start=1):
        g = int(global_row[idx])
        row, local_row = locate(g, rows)
        shard = shards.setdefault(str(row["raw_path"]), open_shard(row["raw_path"]))
        atoms = shard.get_atoms(local_row)
        formula = atoms.get_chemical_formula("metal")
        energy = safe_energy(atoms)
        n_atoms = len(atoms)
        file_name = f"{rank:02d}_{slugify(formula)}_g{g}.extxyz"
        atoms.info.update(
            {
                "omat24_subdataset": subdataset,
                "omat24_shard": row["shard"],
                "omat24_local_row": local_row,
                "uma_bank_global_row": g,
                "weighted_pc1": float(pc12[idx, 0]),
                "weighted_pc2": float(pc12[idx, 1]),
                "sample_range": name,
                "sample_rank": rank,
            }
        )
        ase_write(range_dir / file_name, atoms, format="extxyz")
        records.append(
            {
                "rank": rank,
                "label": f"{name}#{rank}",
                "global_row": g,
                "subdataset": subdataset,
                "shard": row["shard"],
                "local_row": local_row,
                "raw_path": str(row["raw_path"]),
                "weighted_pc1": float(pc12[idx, 0]),
                "weighted_pc2": float(pc12[idx, 1]),
                "formula": formula,
                "n_atoms": n_atoms,
                "energy_eV": energy,
                "energy_per_atom_eV": None if energy is None else energy / n_atoms,
                "volume_A3": float(atoms.get_volume()),
                "file": str(range_dir / file_name),
            }
        )
    with (range_dir / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def single_source_image(hist: np.ndarray, color: np.ndarray) -> np.ndarray:
    intensity = np.log1p(hist) / np.log1p(hist.max()) if hist.max() > 0 else hist
    return 1.0 - intensity[..., None] * (1.0 - color)


def load_backdrop(cache: Path, subdataset: str):
    with np.load(cache) as payload:
        sources = [str(s) for s in payload["sources"]]
        if subdataset not in sources:
            raise SystemExit(f"{subdataset!r} not in histogram cache sources: {sources}")
        idx = sources.index(subdataset)
        hist = np.asarray(payload["histograms"][idx], dtype=np.float64)
        color = np.asarray(payload["colors"][idx], dtype=np.float64)
        x_edges = np.asarray(payload["x_edges"], dtype=np.float64)
        y_edges = np.asarray(payload["y_edges"], dtype=np.float64)
        total = int(payload["source_counts"][idx])
    return hist, color, x_edges, y_edges, total


def draw(out_path, backdrop, subdataset, ranges, samples, dpi):
    hist, color, x_edges, y_edges, total = backdrop
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    ax.imshow(
        np.swapaxes(single_source_image(hist, color), 0, 1),
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="auto",
        interpolation="nearest",
    )
    handles, listing = [], []
    for k, (name, lo, hi) in enumerate(ranges):
        marker, mcolor = MARKERS[k % len(MARKERS)], MARKER_COLORS[k % len(MARKER_COLORS)]
        tag = chr(ord("A") + k)
        for bound in (lo, hi):
            if np.isfinite(bound):
                ax.axvline(bound, color=mcolor, linestyle="--", linewidth=1.0, alpha=0.8, zorder=3)
        recs = samples[name]
        xs = [r["weighted_pc1"] for r in recs]
        ys = [r["weighted_pc2"] for r in recs]
        ax.scatter(xs, ys, s=90, marker=marker, facecolor=mcolor, edgecolor="white", linewidth=1.2, zorder=5)
        for j, r in enumerate(recs):
            # Fan the short tags out around the marker with a leader line so
            # neighbouring samples (common inside the dense lobe) stay legible.
            dx, dy = LABEL_OFFSETS[j % len(LABEL_OFFSETS)]
            if k % 2 == 1:
                dx = -dx
            ax.annotate(
                f"{tag}{r['rank']}",
                (r["weighted_pc1"], r["weighted_pc2"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color=mcolor,
                ha="center",
                va="center",
                zorder=6,
                arrowprops={"arrowstyle": "-", "color": mcolor, "linewidth": 0.6, "shrinkB": 4},
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": mcolor, "linewidth": 0.6, "alpha": 0.9},
            )
            listing.append(f"{tag}{r['rank']}  {r['formula']:<18} PC1 {r['weighted_pc1']:+.2f}  PC2 {r['weighted_pc2']:+.2f}")
        desc = describe_range(lo, hi)
        handles.append(
            Line2D([], [], marker=marker, linestyle="none", markersize=8, markerfacecolor=mcolor,
                   markeredgecolor="white", label=f"{tag}: {name} ({len(recs)} sampled, weighted PC1 {desc})")
        )
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    ax.text(
        0.99, 0.02, "\n".join(listing), transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, family="monospace", zorder=6,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.9},
    )
    ax.set(
        xlabel="Weighted PC1",
        ylabel="Weighted PC2",
        title=f"{subdataset} on full UMA PCA axes\nrows in frame: {int(hist.sum()):,} / {total:,}; sampled structures highlighted",
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def describe_range(lo: float, hi: float) -> str:
    if np.isfinite(lo) and np.isfinite(hi):
        return f"in ({lo:g}, {hi:g})"
    if np.isfinite(hi):
        return f"< {hi:g}"
    if np.isfinite(lo):
        return f"> {lo:g}"
    return "unbounded"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="checkout holding data/ and runs/")
    parser.add_argument("--subdataset", default=DEFAULT_SUBDATASET)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/omat24/uma_latents_copy/all_uma_embeddings_manifest.csv"))
    parser.add_argument("--pca-state", type=Path, default=Path("runs/uma_latents_copy_full_pca/uma_100824585_rapids_pca_state_11212254.npz"))
    parser.add_argument("--histogram-cache", type=Path, default=Path("runs/uma_100m_pca_by_source/uma_source_pca_histograms.npz"))
    parser.add_argument("--range", dest="ranges", type=parse_range, action="append", default=None,
                        help="NAME:LO:HI on weighted PC1, exclusive bounds, -inf/inf allowed; repeatable "
                             f"(default: {' '.join(DEFAULT_RANGES)})")
    parser.add_argument("--per-range", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/uma_100m_pca_by_source/nvt3000_pc1_samples"))
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    args.ranges = args.ranges or [parse_range(s) for s in DEFAULT_RANGES]
    root = args.repo_root.resolve()
    for key in ("manifest", "pca_state", "histogram_cache", "output_dir"):
        setattr(args, key, resolve(getattr(args, key), root))
    args.repo_root = root
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest, args.subdataset, args.repo_root)
    print(f"{args.subdataset}: {len(rows)} shards, {sum(r['n_rows'] for r in rows):,} rows", flush=True)
    pca = load_pca(args.pca_state)
    scores_cache = args.output_dir / f"{slugify(args.subdataset)}_pc12.npz"
    global_row, pc12 = compute_scores(rows, pca, scores_cache)
    pc1 = pc12[:, 0].astype(np.float64)

    rng = np.random.default_rng(args.seed)
    taken = np.zeros(pc1.size, dtype=bool)
    picks_by_range, counts = {}, {}
    for name, lo, hi in args.ranges:
        mask = (pc1 > lo) & (pc1 < hi)
        counts[name] = int(mask.sum())
        candidates = np.flatnonzero(mask & ~taken)
        if candidates.size < args.per_range:
            raise SystemExit(f"range {name}: only {candidates.size} candidate rows, need {args.per_range}")
        picks = np.sort(rng.choice(candidates, size=args.per_range, replace=False))
        taken[picks] = True
        picks_by_range[name] = picks
        print(f"range {name} ({describe_range(lo, hi)}): {counts[name]:,} rows "
              f"({100.0 * counts[name] / pc1.size:.2f} % of subdataset); sampled {args.per_range}", flush=True)

    shards: dict = {}
    samples = {}
    for name, _, _ in args.ranges:
        samples[name] = write_samples(name, picks_by_range[name], global_row, pc12, rows, args.subdataset, args.output_dir, shards)
        for r in samples[name]:
            e = "n/a" if r["energy_per_atom_eV"] is None else f"{r['energy_per_atom_eV']:.3f} eV/atom"
            print(f"  {r['label']:<16} {r['formula']:<24} PC1={r['weighted_pc1']:+.3f} PC2={r['weighted_pc2']:+.3f} "
                  f"{r['shard']}:{r['local_row']} {e}", flush=True)

    backdrop = load_backdrop(args.histogram_cache, args.subdataset)
    plot_path = args.output_dir / "samples_on_backdrop.png"
    draw(plot_path, backdrop, args.subdataset, args.ranges, samples, args.dpi)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subdataset": args.subdataset,
        "manifest": str(args.manifest),
        "pca_state": str(args.pca_state),
        "histogram_cache": str(args.histogram_cache),
        "weighted_pc_definition": "standardized latent @ components[:2].T * retained_variance_ratio[:2]",
        "seed": args.seed,
        "per_range": args.per_range,
        "subdataset_rows": int(pc1.size),
        "ranges": [
            {"name": name, "pc1_lo": lo, "pc1_hi": hi, "rows_in_range": counts[name],
             "fraction_of_subdataset": counts[name] / pc1.size, "samples": samples[name]}
            for name, lo, hi in args.ranges
        ],
        "plot": str(plot_path),
        "scores_cache": str(scores_cache),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Plot: {plot_path}", flush=True)
    print(f"Metadata: {args.output_dir / 'metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
