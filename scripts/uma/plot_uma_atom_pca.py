#!/usr/bin/env python
"""Figures for the PER-ATOM UMA latent PCA of an OMAT24 sample (2026-09-09, full
extent 2026-09-10).

Consumes what ``scripts/mace/fit_descriptor_bank_pca.py`` wrote for the atom
bank (``pca_state.npz``, ``pc12_all.npy``) plus the per-atom label arrays from
``scripts/uma/assemble_uma_atom_latents.py``.

Extent.  By default (``--extent full``) every atom is inside the axes: the
PC1/PC2 scores are binned on a uniform grid in symmetric-log coordinates
(linear between -linthresh and +linthresh, one unit per decade beyond; see
``atom_pca_axes.py``) that spans the whole data range, cached as
``<pca-dir>/background_density_full.npz`` (one ``hist_<name>`` per subdataset /
element / element family).  ``--extent percentile-box`` reproduces the earlier
figures from the fit's ``background_density.npz`` (0.02-99.98 percentile box on
linear axes), which drops the ~0.04 % of atoms in the tails.

Figures (--output-dir, default = --pca-dir):
  uma_atom_pca_all.png                 log10 atom density of the whole bank
  uma_atom_pca_by_subdataset.png       all + one panel per OMAT24 subdataset
  uma_atom_pca_by_element_group.png    all + one panel per periodic-table family
  uma_atom_pca_by_element.png          one panel per element present, by Z (no
                                       truncation: every element is shown)
  uma_atom_pca_medians.png             element symbols / subdataset names placed
                                       at their median (PC1, PC2) over the density
Also uma_atom_pca_scores.npz (pc1/pc2 + every per-atom label, self-contained;
skip with --skip-scores-bundle when it already exists) and uma_atom_pca_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as path_effects  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atom_pca_axes import (  # noqa: E402
    GRID, INK, INK_2, MUTED, SPINE, SURFACE, IdentityWarp, SymLogWarp, core_cell, draw_density, load_density,
    load_or_compute_density, style_warped_axis,
)

ROOT = Path(__file__).resolve().parents[2]

# Same tokens as scripts/mace/plot_official_mace_pca_overlay.py (reference palette).
GRAY_RAMP = LinearSegmentedColormap.from_list(
    "surface_greys", [SURFACE, "#e1e0d9", "#c3c2b7", "#898781", "#52514e"]
)
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "palette_blues",
    [SURFACE, "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
     "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
)
ACCENT = "#2a78d6"  # categorical slot 1; the only hue used on top of the density


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pca-dir", required=True, type=Path)
    parser.add_argument("--bank-dir", required=True, type=Path,
                        help="assemble_uma_atom_latents.py output (labels_* dirs, atom arrays)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--sample-label", default="sr_1m (1,000,000 stratified random OMAT24 training frames)")
    parser.add_argument("--extent", choices=("full", "percentile-box"), default="full",
                        help="full: every atom inside symmetric-log axes (default); "
                             "percentile-box: the fit's linear 0.02-99.98 percentile box")
    parser.add_argument("--linthresh", type=float, default=40.0,
                        help="full extent: |PC| below this is linear, beyond it logarithmic")
    parser.add_argument("--linscale", type=float, default=2.0,
                        help="full extent: decade-widths given to each linear half-range (2 = core ~55-60 %% of the axis)")
    parser.add_argument("--bins", type=int, default=800, help="full extent: bins per axis in warped coordinates")
    parser.add_argument("--density-file", type=Path,
                        help="full extent: where the warped densities are cached "
                             "(default <pca-dir>/background_density_full.npz)")
    parser.add_argument("--recompute-density", action="store_true")
    parser.add_argument("--skip-scores-bundle", action="store_true",
                        help="do not rewrite uma_atom_pca_scores.npz (3.7 GB for sr_10m; unchanged by the extent)")
    return parser.parse_args()


def load_labels(bank_dir: Path, name: str) -> tuple[np.ndarray, list[str], dict]:
    labels = np.load(bank_dir / name / "row_labels.npy", mmap_mode="r")
    meta = json.loads((bank_dir / name / "row_labels.json").read_text())
    return labels, list(meta["names"]), meta


def style_axis(ax, explained, show_xlabel=True, show_ylabel=True, fontsize=9):
    ax.set_facecolor(SURFACE)
    if show_xlabel:
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)", color=INK_2, fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)", color=INK_2, fontsize=fontsize)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(SPINE)
    ax.tick_params(colors=MUTED, labelsize=max(fontsize - 1, 6))


def panel_grid(panels, density, explained, title, subtitle, out, dpi, columns=4,
               panel_size=3.3, label_fontsize=8.5, title_fontsize=12.5):
    warp, tx, ty = density["warp"], density["t_x_edges"], density["t_y_edges"]
    rows = int(np.ceil(len(panels) / columns))
    width = panel_size * columns + 0.4
    # Wrap the caption to the figure width (~13 characters per inch at 8.5 pt) and reserve room for it.
    subtitle = "\n".join(textwrap.fill(line, max(60, int(width * 13))) for line in subtitle.split("\n"))
    block = 0.55 + 0.15 * (subtitle.count("\n") + 1) + 0.05
    figure, axes = plt.subplots(
        rows, columns, figsize=(width, (panel_size - 0.2) * rows + 0.5 + block),
        facecolor=SURFACE, sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for index, (ax, (name, hist)) in enumerate(zip(axes, panels)):
        draw_density(ax, hist, tx, ty, BLUE_RAMP, x_edges=density["x_edges"], y_edges=density["y_edges"])
        style_axis(ax, explained, show_xlabel=index >= len(panels) - columns, show_ylabel=index % columns == 0,
                   fontsize=label_fontsize)
        style_warped_axis(ax, warp, (tx[0], tx[-1]), (ty[0], ty[-1]), nbins=5, compact=True)
        ax.set_title(name, loc="left", fontsize=label_fontsize, color=INK)
    figure.suptitle(title, x=0.01, ha="left", fontsize=title_fontsize, fontweight="bold", color=INK)
    figure.text(0.01, 1 - 0.55 / figure.get_figheight(), subtitle, fontsize=8.5, color=INK_2, va="top")
    figure.tight_layout(rect=(0, 0, 1, 1 - block / figure.get_figheight()))
    figure.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out}", flush=True)


def group_quantiles(scores: np.ndarray, labels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(n, 3, 2) array of the 25/50/75 percentiles of (PC1, PC2) per label id, plus counts."""
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=n)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    quantiles = np.full((n, 3, 2), np.nan)
    for i in range(n):
        if counts[i]:
            quantiles[i] = np.percentile(scores[order[bounds[i]: bounds[i + 1]]], [25, 50, 75], axis=0)
    return quantiles, counts


def main() -> None:
    args = parse_args()
    out_dir = (args.output_dir or args.pca_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.pca_dir / "pca_state.npz", allow_pickle=False) as npz:
        explained = np.asarray(npz["explained_variance_ratio"], dtype=np.float64)
        n_rows = int(npz["n_rows"])
    scores_mm = np.load(args.pca_dir / "pc12_all.npy", mmap_mode="r")
    if len(scores_mm) != n_rows:
        raise SystemExit(f"pc12_all has {len(scores_mm):,} rows, pca_state says {n_rows:,}")

    sub_labels, sub_names, _ = load_labels(args.bank_dir, "labels_subdataset")
    el_labels, el_names, el_meta = load_labels(args.bank_dir, "labels_element")
    grp_labels, grp_names, grp_meta = load_labels(args.bank_dir, "labels_element_group")
    atomic_numbers = np.load(args.bank_dir / "atom_atomic_numbers.npy")
    frame_index = np.load(args.bank_dir / "atom_frame_index.npy")
    for arr, what in ((sub_labels, "subdataset"), (el_labels, "element"), (grp_labels, "element group"),
                      (atomic_numbers, "atomic numbers"), (frame_index, "frame index")):
        if len(arr) != n_rows:
            raise SystemExit(f"{what} labels: {len(arr):,} rows for {n_rows:,} atoms")
    n_frames = int(frame_index.max()) + 1

    if args.extent == "full":
        warp = SymLogWarp(args.linthresh, linscale=args.linscale)
        density_path = args.density_file or (args.pca_dir / "background_density_full.npz")
        density = load_or_compute_density(
            density_path, scores_mm,
            [("subdataset", sub_labels, sub_names), ("element", el_labels, el_names),
             ("element_group", grp_labels, grp_names)],
            warp, bins=args.bins, recompute=args.recompute_density,
        )
    else:
        density_path = args.pca_dir / "background_density.npz"
        density = load_density(density_path)
        warp = density["warp"]
    hist, tx, ty, hists = density["hist"], density["t_x_edges"], density["t_y_edges"], density["hists"]
    x_edges, y_edges = density["x_edges"], density["y_edges"]
    inside, core = float(density["inside_fraction"]), float(density["core_fraction"])
    bins = len(tx) - 1
    scores = np.asarray(scores_mm, dtype=np.float32)
    print(f"{n_rows:,} atoms from {n_frames:,} frames; PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%}, "
          f"PC1-10 {explained[:10].sum():.1%}; {inside:.2%} inside the plotted extent "
          f"({args.extent}, {warp.describe()})", flush=True)

    cell_x, cell_y = core_cell(x_edges, y_edges)
    shading = f"log10(1 + atoms per {cell_x:.2f} x {cell_y:.2f} cell)"
    if isinstance(warp, IdentityWarp):
        extent_note = (f"{inside:.2%} of atoms fall inside the plotted {bins}x{bins}-bin extent "
                       f"(0.02-99.98 percentile box, linear axes); the rest are cut off.")
    else:
        d_lo, d_hi = density["data_min"], density["data_max"]
        extent_note = (f"Every atom is inside the plotted {bins}x{bins}-bin extent ({inside:.2%}): {warp.describe()}. "
                       f"PC1 spans [{d_lo[0]:.0f}, {d_hi[0]:.0f}], PC2 [{d_lo[1]:.0f}, {d_hi[1]:.0f}]; "
                       f"{core:.2%} of atoms have |PC1| and |PC2| <= {warp.linthresh:g}. Beyond that the bins widen "
                       f"with the log axis, so counts are scaled to the core cell (a density, continuous across the "
                       f"switch) and every non-empty bin is drawn at least at the one-atom tint.")
    base_subtitle = (
        f"Standardized PCA of uma-s-1p2 (omat task) energy-head last-layer input, one row per ATOM "
        f"({n_rows:,} atoms of {n_frames:,} frames); {args.sample_label}.\n"
        f"Each panel is {shading} scaled to its own peak (empty bins white): panels show where a "
        f"group lives, not its share of the population.\n{extent_note}"
    )

    # --- 1: everything --------------------------------------------------------
    figure, ax = plt.subplots(figsize=(8.6, 7.3), facecolor=SURFACE)
    image = draw_density(ax, hist, tx, ty, BLUE_RAMP, x_edges=x_edges, y_edges=y_edges)
    style_axis(ax, explained)
    style_warped_axis(ax, warp, (tx[0], tx[-1]), (ty[0], ty[-1]), nbins=7)
    vmax = float(np.log10(hist.max() + 1.0))
    colorbar = figure.colorbar(image, ax=ax, pad=0.02, fraction=0.04, ticks=np.arange(0, np.ceil(vmax) + 1))
    colorbar.set_label(shading, color=INK_2, fontsize=9)
    figure.suptitle(f"Per-atom UMA latent PCA of OMAT24 ({args.sample_label.split(' ')[0]} sample)",
                    x=0.02, y=0.99, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    figure.text(0.02, 0.945, textwrap.fill(base_subtitle.split("\n")[0], 118) + "\n"
                + textwrap.fill(extent_note, 118), fontsize=8.5, color=INK_2, va="top")
    figure.tight_layout(rect=(0, 0, 1, 0.845))
    out_all = out_dir / "uma_atom_pca_all.png"
    figure.savefig(out_all, dpi=args.dpi, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out_all}", flush=True)

    # --- 2: by subdataset -------------------------------------------------------
    sub_counts = np.bincount(np.asarray(sub_labels), minlength=len(sub_names))
    panels = [("all atoms", hist)] + [
        (f"{name}\n{sub_counts[i]:,} atoms ({sub_counts[i] / n_rows:.1%})", hists[name])
        for i, name in enumerate(sub_names)
    ]
    panel_grid(panels, density, explained,
               "Per-atom UMA latent PCA by OMAT24 source subdataset", base_subtitle,
               out_dir / "uma_atom_pca_by_subdataset.png", args.dpi)

    # --- 3: by element group ----------------------------------------------------
    grp_counts = np.bincount(np.asarray(grp_labels), minlength=len(grp_names))
    panels = [("all atoms", hist)] + [
        (f"{name}\n{grp_counts[i]:,} atoms ({grp_counts[i] / n_rows:.1%})", hists[name])
        for i, name in enumerate(grp_names) if grp_counts[i]
    ]
    panel_grid(panels, density, explained,
               "Per-atom UMA latent PCA by element family", base_subtitle,
               out_dir / "uma_atom_pca_by_element_group.png", args.dpi)

    # --- 4: by element, every element present, ordered by Z ---------------------
    el_counts = np.bincount(np.asarray(el_labels), minlength=len(el_names))
    el_z = el_meta["atomic_numbers"]
    panels = [
        (f"{name} (Z={z}) · {el_counts[i]:,}", hists[name])
        for i, (name, z) in enumerate(zip(el_names, el_z)) if el_counts[i]
    ]
    panel_grid(panels, density, explained,
               f"Per-atom UMA latent PCA by element ({len(panels)} elements present, ordered by Z)",
               base_subtitle, out_dir / "uma_atom_pca_by_element.png", args.dpi,
               columns=10, panel_size=2.1, label_fontsize=7, title_fontsize=14)

    # --- 5: medians ---------------------------------------------------------------
    el_q, _ = group_quantiles(scores, np.asarray(el_labels), len(el_names))
    sub_q, _ = group_quantiles(scores, np.asarray(sub_labels), len(sub_names))
    grp_q, _ = group_quantiles(scores, np.asarray(grp_labels), len(grp_names))
    el_medians, sub_medians, grp_medians = el_q[:, 1], sub_q[:, 1], grp_q[:, 1]
    halo = [path_effects.withStroke(linewidth=2.2, foreground=SURFACE)]
    present = el_counts > 0
    in_extent = present & (el_medians[:, 0] >= x_edges[0]) & (el_medians[:, 0] <= x_edges[-1]) \
        & (el_medians[:, 1] >= y_edges[0]) & (el_medians[:, 1] <= y_edges[-1])
    # The zoom box is set by the medians inside the linear core (|PC| <= linthresh); the few
    # elements whose median lies beyond it (noble gases) are listed with their coordinates.
    L = warp.linthresh if np.isfinite(warp.linthresh) else np.inf
    in_core = in_extent & (np.abs(el_medians[:, 0]) <= L) & (np.abs(el_medians[:, 1]) <= L)
    if not in_core.any():
        in_core = in_extent
    pad = 2.0
    zoom_lo = warp.forward([el_medians[in_core, 0].min() - pad, el_medians[in_core, 1].min() - pad])
    zoom_hi = warp.forward([el_medians[in_core, 0].max() + pad, el_medians[in_core, 1].max() + pad])
    zoom_x = (max(zoom_lo[0], tx[0]), min(zoom_hi[0], tx[-1]))
    zoom_y = (max(zoom_lo[1], ty[0]), min(zoom_hi[1], ty[-1]))
    el_medians_t = warp.forward(el_medians)
    in_zoom = in_extent & (el_medians_t[:, 0] >= zoom_x[0]) & (el_medians_t[:, 0] <= zoom_x[1]) \
        & (el_medians_t[:, 1] >= zoom_y[0]) & (el_medians_t[:, 1] <= zoom_y[1])
    outside = [f"{el_names[i]} ({el_medians[i, 0]:.0f}, {el_medians[i, 1]:.0f})"
               for i in np.flatnonzero(present & ~in_zoom)]
    figure = plt.figure(figsize=(15, 6.4), facecolor=SURFACE)
    # Column 1 is an empty spacer so the long subdataset tick labels never reach the map.
    grid = figure.add_gridspec(1, 4, width_ratios=[1.6, 0.42, 0.72, 0.72], wspace=0.12)
    ax_el = figure.add_subplot(grid[0, 0])
    ax_pc1 = figure.add_subplot(grid[0, 2])
    ax_pc2 = figure.add_subplot(grid[0, 3], sharey=ax_pc1)
    # Left: element symbols at their median, zoomed to the box that holds them (warped coordinates).
    draw_density(ax_el, hist, tx, ty, GRAY_RAMP, x_edges=x_edges, y_edges=y_edges)
    style_axis(ax_el, explained)
    style_warped_axis(ax_el, warp, zoom_x, zoom_y, nbins=7)
    for i in np.flatnonzero(in_zoom):
        ax_el.text(el_medians_t[i, 0], el_medians_t[i, 1], el_names[i], fontsize=7.5, color=INK, ha="center",
                   va="center", clip_on=True, path_effects=halo)
    omitted = f"\nbeyond this zoom (median PC1, PC2): {', '.join(outside)}" if outside else ""
    ax_el.set_title(f"median (PC1, PC2) of every element, zoomed to the medians inside |PC| <= {L:g}{omitted}",
                    loc="left", fontsize=9, color=INK)
    # Right: per-subdataset median with the 25-75% span, one axis per PC (data units).
    y_pos = np.arange(len(sub_names))[::-1]
    for ax, pc, label in ((ax_pc1, 0, "PC1"), (ax_pc2, 1, "PC2")):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=GRID, linewidth=1, zorder=1)
        ax.hlines(y_pos, sub_q[:, 0, pc], sub_q[:, 2, pc], color=SPINE, linewidth=2.5, zorder=2)
        ax.scatter(sub_medians[:, pc], y_pos, s=30, c=ACCENT, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        ax.set_xlabel(f"{label} per atom: median and 25-75% span", color=INK_2, fontsize=8.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(SPINE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_ylim(-0.7, len(sub_names) - 0.3)
    ax_pc1.set_yticks(y_pos)
    ax_pc1.set_yticklabels(sub_names, fontsize=8, color=INK)
    plt.setp(ax_pc2.get_yticklabels(), visible=False)
    ax_pc1.set_title("OMAT24 subdatasets", loc="left", fontsize=9, color=INK)
    figure.suptitle("Where elements and subdatasets sit in the per-atom UMA PCA", x=0.01, y=0.99, ha="left",
                    va="top", fontsize=12.5, fontweight="bold", color=INK)
    figure.text(0.01, 0.935, "Left: gray = log10 atom density of the whole bank, each symbol at the median position "
                "of that element's atoms (spread shown in the faceted figures); the view is zoomed to the box "
                "holding every median.\nRight: subdataset medians differ by under 2 units while the 25-75% spans "
                "are several units wide, i.e. the source subdataset barely moves an atom in this space.",
                fontsize=8.5, color=INK_2, va="top")
    figure.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.11)
    out_medians = out_dir / "uma_atom_pca_medians.png"
    figure.savefig(out_medians, dpi=args.dpi, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out_medians}", flush=True)

    # --- bundle + summary --------------------------------------------------------
    if not args.skip_scores_bundle:
        np.savez(
            out_dir / "uma_atom_pca_scores.npz",
            pc1=scores[:, 0], pc2=scores[:, 1],
            atomic_number=atomic_numbers.astype(np.int16), frame_index=frame_index.astype(np.int32),
            subdataset=np.asarray(sub_labels).astype(np.int16), subdataset_names=np.array(sub_names),
            element=np.asarray(el_labels).astype(np.int16), element_names=np.array(el_names),
            element_group=np.asarray(grp_labels).astype(np.int16), element_group_names=np.array(grp_names),
            explained_variance_ratio=explained,
            pca_state=str((args.pca_dir / "pca_state.npz").resolve()),
            note="one row per atom of the sample's train.lmdb (frame_index = zero-based LMDB row); "
                 "scores are standardized-PCA projections",
        )
    (out_dir / "uma_atom_pca_summary.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "atoms": n_rows, "frames": n_frames,
                "explained_variance_ratio_pc1_pc2": [float(explained[0]), float(explained[1])],
                "explained_variance_ratio_first10_cumulative": float(explained[:10].sum()),
                "extent": {
                    "mode": args.extent, "axes": warp.to_json(), "bins": bins,
                    "density_file": str(density_path.resolve()),
                    "inside_extent": inside,
                    "inside_linear_core": None if np.isnan(core) else core,
                    "pc1_data_range": [float(v) for v in (density["data_min"][0], density["data_max"][0])],
                    "pc2_data_range": [float(v) for v in (density["data_min"][1], density["data_max"][1])],
                },
                "inside_extent": inside,
                "x_range": [float(x_edges[0]), float(x_edges[-1])],
                "y_range": [float(y_edges[0]), float(y_edges[-1])],
                "subdatasets": {
                    name: {"atoms": int(sub_counts[i]), "median_pc1_pc2": [float(v) for v in sub_medians[i]]}
                    for i, name in enumerate(sub_names)
                },
                "element_groups": {
                    name: {"atoms": int(grp_counts[i]), "median_pc1_pc2": [float(v) for v in grp_medians[i]]}
                    for i, name in enumerate(grp_names) if grp_counts[i]
                },
                "elements": {
                    name: {"Z": int(z), "atoms": int(el_counts[i]), "median_pc1_pc2": [float(v) for v in el_medians[i]]}
                    for i, (name, z) in enumerate(zip(el_names, el_z)) if el_counts[i]
                },
                "figures": {
                    "all": str(out_all),
                    "by_subdataset": str(out_dir / "uma_atom_pca_by_subdataset.png"),
                    "by_element_group": str(out_dir / "uma_atom_pca_by_element_group.png"),
                    "by_element": str(out_dir / "uma_atom_pca_by_element.png"),
                    "medians": str(out_medians),
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote uma_atom_pca_summary.json{'' if args.skip_scores_bundle else ' and uma_atom_pca_scores.npz'} "
          f"to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
