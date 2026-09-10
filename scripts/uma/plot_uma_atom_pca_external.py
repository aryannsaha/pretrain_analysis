#!/usr/bin/env python
"""External corpora atoms on the OMAT24 per-atom UMA PCA (2026-09-10).

Projects the assembled per-atom latent banks of the external 10 % subsets
(``data/processed/external_uma_atom_latents/<dataset>/atom_latents.npy`` from
``assemble_uma_atom_latents.py``) through an OMAT24 atom PCA state
(``fit_descriptor_bank_pca.py`` output) and draws them over the OMAT24 atom
density of that same PCA.  The density comes from ``--density-file`` (default:
``<pca-dir>/background_density_full.npz`` written by ``plot_uma_atom_pca.py``,
which spans every OMAT24 atom on symmetric-log axes; falls back to the fit's
linear percentile-box ``background_density.npz``).  External atoms are binned
and drawn in the same warped coordinates, so nothing is cut off.

  uma_atom_pca_external_overlay.png    one facet per corpus: gray = OMAT24 atoms,
                                       blue contours enclose 50/80/95 % of the
                                       corpus' atoms, dots = a 15k-atom subsample
  uma_atom_pca_external_by_family.png  corpus x element-family medians (text) on the density
  external_atom_pca_scores.npz         pc1/pc2/pc25, atomic numbers, source frame per corpus
  external_atom_pca_summary.json       counts, medians, inside-extent, fraction of atoms
                                       in bins that hold no OMAT24 atom (outside OMAT24 support)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atom_pca_axes import INK, INK_2, MUTED, SPINE, SURFACE, draw_density, load_density, style_warped_axis  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GRAY_RAMP = LinearSegmentedColormap.from_list("surface_greys", [SURFACE, "#e1e0d9", "#c3c2b7", "#898781", "#52514e"])
ACCENT = "#2a78d6"
ACCENT_DARK = "#104281"
DATASET_TITLES = {"mof_off_r2scan": "MOF-off R2SCAN-D4", "mad": "MAD", "mpaloe": "MP-ALOE", "matpes_r2scan": "MatPES r2SCAN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pca-dir", required=True, type=Path, help="pca_state.npz + density file")
    parser.add_argument("--density-file", type=Path,
                        help="OMAT24 atom density to draw under the corpora (default: "
                             "<pca-dir>/background_density_full.npz, else <pca-dir>/background_density.npz)")
    parser.add_argument("--external-root", type=Path, default=ROOT / "data/processed/external_uma_atom_latents")
    parser.add_argument("--datasets", nargs="+", default=["mof_off_r2scan", "mad", "mpaloe", "matpes_r2scan"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--background-label", default="OMAT24 sr_1m atoms")
    parser.add_argument("--scatter", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def load_state(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        return {"mean": npz["mean"].astype(np.float64), "scale": npz["scale"].astype(np.float64),
                "components": npz["components"].astype(np.float64),
                "explained": npz["explained_variance_ratio"].astype(np.float64), "n_rows": int(npz["n_rows"])}


def project(state: dict, values: np.ndarray, k: int, chunk: int = 2_000_000) -> np.ndarray:
    projector = (state["components"][:k] / state["scale"][None, :]).T
    out = np.empty((len(values), k), dtype=np.float32)
    for s in range(0, len(values), chunk):
        block = np.asarray(values[s: s + chunk], dtype=np.float64) - state["mean"]
        out[s: s + chunk] = (block @ projector).astype(np.float32)
    return out


def mass_levels(hist: np.ndarray, fractions=(0.95, 0.80, 0.50)) -> list[float]:
    flat = np.sort(hist.ravel())[::-1]
    cum = np.cumsum(flat) / max(flat.sum(), 1e-12)
    return [float(flat[min(int(np.searchsorted(cum, f)), len(flat) - 1)]) for f in fractions]


def style(ax, explained, xlabel=True, ylabel=True):
    ax.set_facecolor(SURFACE)
    if xlabel:
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)", color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)", color=INK_2, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=8)


def main() -> None:
    args = parse_args()
    out_dir = (args.output_dir or args.pca_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    state = load_state(args.pca_dir / "pca_state.npz")
    density_path = args.density_file
    if density_path is None:
        density_path = args.pca_dir / "background_density_full.npz"
        if not density_path.is_file():
            density_path = args.pca_dir / "background_density.npz"
    density = load_density(density_path)
    warp, hist, tx, ty = density["warp"], density["hist"], density["t_x_edges"], density["t_y_edges"]
    bins = len(tx) - 1
    explained = state["explained"]
    vmax = float(np.log10(hist.max() + 1.0))
    support = hist > 0
    txc, tyc = 0.5 * (tx[1:] + tx[:-1]), 0.5 * (ty[1:] + ty[:-1])
    rng = np.random.default_rng(args.seed)
    print(f"density {density_path}: {bins}x{bins} bins, {warp.describe()}", flush=True)

    results, bundle = {}, {}
    for name in args.datasets:
        bank_dir = args.external_root / name
        latents = np.load(bank_dir / "atom_latents.npy", mmap_mode="r")
        numbers = np.load(bank_dir / "atom_atomic_numbers.npy")
        frames = np.load(bank_dir / "atom_frame_index.npy")
        grp_labels = np.load(bank_dir / "labels_element_group/row_labels.npy")
        grp_names = json.loads((bank_dir / "labels_element_group/row_labels.json").read_text())["names"]
        scores = project(state, latents, 25)
        xy = scores[:, :2]
        txy = warp.forward(xy)
        own_hist, _, _ = np.histogram2d(txy[:, 0], txy[:, 1], bins=[tx, ty])
        inside = own_hist.sum() / len(xy)
        ix = np.clip(np.searchsorted(tx, txy[:, 0], side="right") - 1, 0, bins - 1)
        iy = np.clip(np.searchsorted(ty, txy[:, 1], side="right") - 1, 0, bins - 1)
        in_box = (txy[:, 0] >= tx[0]) & (txy[:, 0] <= tx[-1]) & (txy[:, 1] >= ty[0]) & (txy[:, 1] <= ty[-1])
        outside_support = float(np.mean(~in_box | ~support[ix, iy]))
        fam_medians = {}
        for g, gname in enumerate(grp_names):
            mask = grp_labels == g
            if mask.sum() >= 50:
                fam_medians[gname] = [float(v) for v in np.median(xy[mask], axis=0)] + [int(mask.sum())]
        results[name] = {"atoms": int(len(xy)), "frames": int(len(np.unique(frames))), "inside_extent": float(inside),
                         "outside_omat_support": outside_support,
                         "median_pc1_pc2": [float(v) for v in np.median(xy, axis=0)],
                         "pc1_range": [float(xy[:, 0].min()), float(xy[:, 0].max())],
                         "pc2_range": [float(xy[:, 1].min()), float(xy[:, 1].max())],
                         "family_medians": fam_medians, "own_hist": own_hist, "txy": txy}
        bundle[f"{name}_pc25"] = scores
        bundle[f"{name}_atomic_number"] = numbers.astype(np.int16)
        bundle[f"{name}_frame_source_index"] = frames.astype(np.int32)
        print(f"{name}: {len(xy):,} atoms of {results[name]['frames']:,} frames; median {results[name]['median_pc1_pc2']}; "
              f"PC1 {results[name]['pc1_range']}, PC2 {results[name]['pc2_range']}; "
              f"{inside:.2%} inside extent, {outside_support:.2%} outside OMAT24 support", flush=True)

    # --- overlay facets --------------------------------------------------------
    n = len(args.datasets)
    figure, axes = plt.subplots(1, n, figsize=(4.2 * n + 0.6, 5.2), facecolor=SURFACE, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    halo = [path_effects.withStroke(linewidth=2.0, foreground=SURFACE)]
    for i, (ax, name) in enumerate(zip(axes, args.datasets)):
        r = results[name]
        draw_density(ax, hist, tx, ty, GRAY_RAMP, vmax=vmax, x_edges=density["x_edges"], y_edges=density["y_edges"])
        txy = r["txy"]
        pick = rng.choice(len(txy), size=min(args.scatter, len(txy)), replace=False)
        ax.scatter(txy[pick, 0], txy[pick, 1], s=2.5, c=ACCENT, alpha=0.25, linewidths=0, rasterized=True)
        smooth = gaussian_filter(r["own_hist"], sigma=2.0)
        levels = mass_levels(smooth)
        if levels[0] < levels[-1]:
            ax.contour(txc, tyc, smooth.T, levels=levels, colors=[ACCENT_DARK], linewidths=[0.8, 1.1, 1.5])
        style(ax, explained, ylabel=i == 0)
        style_warped_axis(ax, warp, (tx[0], tx[-1]), (ty[0], ty[-1]), nbins=5, compact=True)
        ax.set_title(f"{DATASET_TITLES.get(name, name)}\n{r['atoms']:,} atoms of {r['frames']:,} frames\n"
                     f"{r['inside_extent']:.1%} inside extent · {r['outside_omat_support']:.1%} outside OMAT24 support",
                     loc="left", fontsize=8.5, color=INK)
    figure.suptitle("External corpora atoms on the OMAT24 per-atom UMA PCA", x=0.01, y=0.99, ha="left", va="top",
                    fontsize=12.5, fontweight="bold", color=INK)
    figure.text(0.01, 0.925, f"Gray = log10 density of {args.background_label} ({state['n_rows']:,} atoms fitted the PCA; "
                "atoms per core-sized cell, every non-empty bin at least the one-atom tint); "
                "blue dots = 15k random atoms of the corpus; contours enclose 50 / 80 / 95 % of the corpus' atoms.\n"
                f"Extent: {warp.describe()}; \"outside OMAT24 support\" = share of corpus atoms whose "
                f"{bins}x{bins} bin holds no OMAT24 atom (or lies off the extent).",
                fontsize=8.5, color=INK_2, va="top")
    figure.subplots_adjust(left=0.05, right=0.99, top=0.77, bottom=0.11, wspace=0.08)
    out_overlay = out_dir / "uma_atom_pca_external_overlay.png"
    figure.savefig(out_overlay, dpi=args.dpi, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out_overlay}", flush=True)

    # --- element-family medians per corpus --------------------------------------
    figure, axes = plt.subplots(1, n, figsize=(4.2 * n + 0.6, 4.9), facecolor=SURFACE, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    all_medians = np.array([m[:2] for r in results.values() for m in r["family_medians"].values()])
    zoom_lo = warp.forward(all_medians.min(0) - 3.0)   # box holding every family median
    zoom_hi = warp.forward(all_medians.max(0) + 3.0)
    zoom_x = (max(zoom_lo[0], tx[0]), min(zoom_hi[0], tx[-1]))
    zoom_y = (max(zoom_lo[1], ty[0]), min(zoom_hi[1], ty[-1]))
    for i, (ax, name) in enumerate(zip(axes, args.datasets)):
        draw_density(ax, hist, tx, ty, GRAY_RAMP, vmax=vmax, x_edges=density["x_edges"], y_edges=density["y_edges"])
        for gname, (mx, my, count) in results[name]["family_medians"].items():
            tmx, tmy = warp.forward([mx, my])
            ax.text(tmx, tmy, gname.replace(" metals", "").replace("reactive ", ""), fontsize=7, color=INK, ha="center",
                    va="center", clip_on=True, path_effects=halo)
        style(ax, explained, ylabel=i == 0)
        style_warped_axis(ax, warp, zoom_x, zoom_y, nbins=6)
        ax.set_title(f"{DATASET_TITLES.get(name, name)}: element-family medians", loc="left", fontsize=8.5, color=INK)
    figure.suptitle("Where each corpus' element families sit on the OMAT24 per-atom UMA PCA", x=0.01, y=0.99,
                    ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    figure.text(0.01, 0.925, "Text = median (PC1, PC2) of that family's atoms in the corpus (families with >= 50 atoms); "
                "axes zoomed to the box holding every median.",
                fontsize=8.5, color=INK_2, va="top")
    figure.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.11, wspace=0.08)
    out_family = out_dir / "uma_atom_pca_external_by_family.png"
    figure.savefig(out_family, dpi=args.dpi, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out_family}", flush=True)

    np.savez(out_dir / "external_atom_pca_scores.npz", pca_state=str((args.pca_dir / "pca_state.npz").resolve()),
             datasets=np.array(args.datasets), **bundle)
    (out_dir / "external_atom_pca_summary.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pca_state": str((args.pca_dir / "pca_state.npz").resolve()),
        "density_file": str(density_path.resolve()),
        "extent": {"axes": warp.to_json(), "bins": bins,
                   "x_range": [float(density["x_edges"][0]), float(density["x_edges"][-1])],
                   "y_range": [float(density["y_edges"][0]), float(density["y_edges"][-1])]},
        "background_rows": state["n_rows"],
        "explained_variance_ratio_pc1_pc2": [float(explained[0]), float(explained[1])],
        "datasets": {k: {kk: vv for kk, vv in v.items() if kk not in ("own_hist", "txy")} for k, v in results.items()},
        "figures": {"overlay": str(out_overlay), "by_family": str(out_family)},
    }, indent=2) + "\n")
    print(f"wrote external_atom_pca_scores.npz and external_atom_pca_summary.json to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
