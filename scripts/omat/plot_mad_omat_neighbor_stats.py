#!/usr/bin/env python
"""Figures for the MAD -> OMAT24 neighbour-statistics analysis.

  1  distances.png      d1 spread per MAD subset, MOF-off overlaid as the anchor
  2  concentration.png  Lorenz curves of neighbour reuse + links-per-structure
  3  isolation.png      isolation vs trajectory redundancy, one point per subset
  4  sources.png        which OMAT24 subdataset each MAD subset retrieves from

MOF-off is drawn as a neutral dashed reference throughout rather than as a ninth
categorical series: it plays the role of an anchor, not a peer, and keeping it
out of the categorical slots leaves the eight MAD subsets on the eight
documented hues.
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#75746f"
GRID = "#e4e3df"
ANCHOR = "#75746f"

# The eight documented categorical hues, in order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Ordered largest-first so the biggest subsets take the leading, most separable hues.
SUBSETS = ["MC3D", "MC3D-rattled", "MC3D-clusters", "SHIFTML-molcrys",
           "MC3D-surfaces", "SHIFTML-molfrags", "MC3D-random", "MC2D"]
COLOR = dict(zip(SUBSETS, SLOTS))
ANCHOR_LABEL = "MOF-off R2SCAN"


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def style(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold",
                     loc="left", pad=10)


def save(fig, path, note=None):
    if note:
        fig.text(0.01, -0.02, note, fontsize=8.5, color=TEXT_MUTED, wrap=True)
    fig.patch.set_facecolor(SURFACE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def plot_distances(analysis, out):
    rows = {r["population"]: r for r in read_csv(analysis / "distance_quantiles.csv")
            if r["metric"] == "d1"}
    order = sorted(SUBSETS, key=lambda s: float(rows[s]["p50"]))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for i, name in enumerate(order):
        r = rows[name]
        y = len(order) - 1 - i
        lo, hi = float(r["p10"]), float(r["p90"])
        q1, q3 = float(r["p25"]), float(r["p75"])
        ax.plot([lo, hi], [y, y], color=COLOR[name], linewidth=2.0, alpha=0.45,
                solid_capstyle="round", zorder=3)
        ax.plot([q1, q3], [y, y], color=COLOR[name], linewidth=7.0,
                solid_capstyle="round", zorder=4)
        ax.plot([float(r["p50"])], [y], marker="o", markersize=8.5, color=COLOR[name],
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5)
        ax.text(hi + 2, y, f"n={int(r['n']):,}", va="center", fontsize=8,
                color=TEXT_MUTED)
    anchor = rows[ANCHOR_LABEL]
    ax.axvline(float(anchor["p50"]), color=ANCHOR, linestyle="--", linewidth=1.6, zorder=2)
    ax.text(float(anchor["p50"]) + 1.5, len(order) - 0.35,
            f"MOF-off median {float(anchor['p50']):.1f}", fontsize=9, color=ANCHOR)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)), fontsize=9.5)
    ax.set_ylim(-0.6, len(order) - 0.15)
    style(ax, xlabel="distance to the nearest OMAT24 structure (raw-128d, d1)",
          title="Every MAD subset sits closer to OMAT24 than MOF-off does")
    ax.yaxis.grid(False)
    save(fig, out / "mad_distances.png",
         "Dot = median, thick bar = interquartile range, thin bar = 10th-90th percentile. "
         "The long tail is not shown: 7 of 95,595 MAD queries exceed d1 = 1000, up to 14,924.")


def plot_concentration(analysis, out):
    lor = read_csv(analysis / "reuse_lorenz.csv")
    conc = {r["population"]: r for r in read_csv(analysis / "retrieval_concentration.csv")}
    curves = {}
    for r in lor:
        curves.setdefault(r["population"], ([], []))
        curves[r["population"]][0].append(float(r["structure_fraction"]))
        curves[r["population"]][1].append(float(r["link_fraction"]))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, zorder=1)
    for name in SUBSETS:
        x, y = curves[name]
        ax.plot(x, y, color=COLOR[name], linewidth=1.9, label=name, zorder=3)
    x, y = curves[ANCHOR_LABEL]
    ax.plot(x, y, color=ANCHOR, linewidth=2.2, linestyle="--", label=ANCHOR_LABEL, zorder=4)
    style(ax, xlabel="fraction of distinct OMAT24 structures (least-used first)",
          ylabel="fraction of all neighbour links",
          title="How concentrated is retrieval?")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")

    ax = axes[1]
    order = sorted(SUBSETS, key=lambda s: float(conc[s]["links_per_structure"]))
    vals = [float(conc[s]["links_per_structure"]) for s in order]
    ax.barh(range(len(order)), vals, color=[COLOR[s] for s in order], height=0.68, zorder=3)
    ax.axvline(float(conc[ANCHOR_LABEL]["links_per_structure"]), color=ANCHOR,
               linestyle="--", linewidth=1.6, zorder=4)
    ax.text(float(conc[ANCHOR_LABEL]["links_per_structure"]) + 0.5, 0.1,
            f"MOF-off {float(conc[ANCHOR_LABEL]['links_per_structure']):.1f}",
            fontsize=9, color=ANCHOR)
    for i, (s, v) in enumerate(zip(order, vals)):
        ax.text(v + 0.4, i, f"{v:.1f}   {float(conc[s]['singleton_fraction']) * 100:.0f}% used once",
                va="center", fontsize=8, color=TEXT_MUTED)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9.5)
    ax.set_xlim(0, max(vals) * 1.55)
    style(ax, xlabel="neighbour links per distinct OMAT24 structure",
          title="Reuse intensity spans 26x across subsets")
    ax.yaxis.grid(False)
    save(fig, out / "mad_concentration.png",
         "Left: a curve hugging the diagonal means every OMAT24 structure is used about equally; "
         "a curve pinned to the bottom-right means a few structures serve most links. "
         "MOF-off and MAD have almost identical Gini (0.765 vs 0.753) yet very different shapes, "
         "which is why the summary statistic alone is misleading.")


def plot_isolation(analysis, out):
    rows = {r["population"]: r for r in read_csv(analysis / "isolation_redundancy.csv")}
    # The five bulk-crystal subsets pile up near (2.15, 1.11); fixed per-subset
    # label offsets keep them legible without moving the points.
    offsets = {
        "MC3D": (-14, -20, "right"), "MC3D-rattled": (16, 12, "left"),
        "MC3D-clusters": (-16, 6, "right"), "MC2D": (-14, 20, "right"),
        "MC3D-surfaces": (14, -14, "left"), "SHIFTML-molcrys": (0, 15, "center"),
        "SHIFTML-molfrags": (0, 15, "center"), "MC3D-random": (0, -20, "center"),
    }
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    for name in SUBSETS:
        r = rows[name]
        x, yv = float(r["mean_distinct_shards_of_10"]), float(r["isolation_median"])
        ax.scatter(x, yv, s=np.sqrt(int(r["n"])) * 3.2, color=COLOR[name], alpha=0.85,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name)
        dx, dy, ha = offsets[name]
        ax.annotate(name, (x, yv), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=8.5, color=TEXT_SECONDARY, zorder=6)
    a = rows[ANCHOR_LABEL]
    ax.scatter(float(a["mean_distinct_shards_of_10"]), float(a["isolation_median"]),
               s=np.sqrt(int(a["n"])) * 3.2, facecolor="none", edgecolor=ANCHOR,
               linewidth=2.0, zorder=4)
    ax.annotate(ANCHOR_LABEL, (float(a["mean_distinct_shards_of_10"]),
                               float(a["isolation_median"])),
                textcoords="offset points", xytext=(0, 13), ha="center",
                fontsize=8.5, color=ANCHOR, fontweight="bold")
    style(ax, xlabel="distinct OMAT24 shards among a query's 10 neighbours  (redundancy →)",
          ylabel="median d10 / d1  (isolation →)",
          title="Redundant one-trajectory matches vs genuinely spread neighbourhoods")
    save(fig, out / "mad_isolation.png",
         "Marker area is proportional to sqrt(number of structures). Bottom-left = each query's ten "
         "neighbours are ten frames of one OMAT24 trajectory sitting almost equidistant "
         "(a redundant match); upper-right = the ten neighbours are spread across distinct "
         "materials at varying distance. MOF-off is the open circle.")


def plot_sources(analysis, out):
    rows = [r for r in read_csv(analysis / "source_attribution.csv")
            if r["population"] != "MAD (all)"]
    rows.sort(key=lambda r: -float(r["pct_rattled-relax"]))
    names = [r["population"] for r in rows]
    y = np.arange(len(names))
    rattled = [float(r["pct_rattled-relax"]) for r in rows]
    # Plot the quantity that varies, on a zero-based axis.  A stacked 100% bar
    # truncated to 85-108% would encode magnitude by length on a clipped scale,
    # which is exactly the axis-truncation anti-pattern.
    cell_free = {"MC3D-clusters", "SHIFTML-molfrags"}
    colors = ["#eb6834" if n in cell_free else "#9ec5f4" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(y, rattled, color=colors, height=0.66, zorder=3)
    overall = 3.7887
    ax.axvline(overall, color=ANCHOR, linestyle="--", linewidth=1.5, zorder=4)
    ax.text(overall + 0.25, len(names) - 0.4, f"MAD overall {overall:.1f}%",
            fontsize=9, color=ANCHOR)
    for i, (n, v) in enumerate(zip(names, rattled)):
        tag = "  no cell" if n in cell_free else ""
        ax.text(v + 0.2, i, f"{v:.2f}%{tag}", va="center", fontsize=8.5,
                color=TEXT_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(rattled) * 1.42)
    style(ax, xlabel="share of top-5 neighbour links landing in rattled-relax (%)",
          title="The two cell-free subsets are the ones that reach rattled-relax")
    ax.yaxis.grid(False)
    save(fig, out / "mad_sources.png",
         "The remaining links go to aimd-from-PBE-3000-nvt, except 31 of 955,950 that reach "
         "aimd-from-PBE-1000-npt. MC3D-clusters and SHIFTML-molfrags are the only two MAD subsets "
         "with no periodic cell (orange), and they reach rattled-relax at ~12% against 0.01-3.6% "
         "for every periodic subset.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    out = (args.out or analysis / "figures").resolve()
    plot_distances(analysis, out)
    plot_concentration(analysis, out)
    plot_isolation(analysis, out)
    plot_sources(analysis, out)


if __name__ == "__main__":
    main()
