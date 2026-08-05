#!/usr/bin/env python
"""Figures for the pc25 top-5 neighbour chemistry comparison.

Reads the CSV tables written by ``pc25_top5_chemistry_report.py`` and renders:

  1. dimensionality.png           0D/1D/2D/3D split of every population
  2. coordination.png             CN distribution + per-element mean CN
  3. bonding.png                  bond-type composition and the organic channels
  4. descriptor_ranges.png        median with p25-p75 range for each descriptor
  5. family_affinity_<set>.png    query family x OMAT family log2 lift
  6. dimensionality_confusion.png query dimensionality vs neighbour dimensionality

Colour is fixed by entity, never by value rank: each query set keeps one hue and
its retrieved neighbours the paired accent, OMAT24 baselines stay neutral grey so
they read as reference rather than as another series.
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
from matplotlib.colors import TwoSlopeNorm

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#75746f"
GRID = "#e4e3df"

ORDER = [
    "MOF-off R2SCAN",
    "MOF-off pc25 top-5",
    "MAD",
    "MAD pc25 top-5",
    "OMAT24 bg (nvt-3000)",
    "OMAT24 bg (all)",
]
COLOR = {
    "MOF-off R2SCAN": "#2a78d6",
    "MOF-off pc25 top-5": "#eb6834",
    "MAD": "#7b4fc9",
    "MAD pc25 top-5": "#1baf7a",
    "OMAT24 bg (nvt-3000)": "#9a9993",
    "OMAT24 bg (all)": "#c5c4be",
}
DIM_SHADES = ["#dfdedb", "#b5c9e4", "#6a9bd8", "#1f5fa8"]


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5, length=3)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, pad=9, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    return ax


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def to_float(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


def plot_dimensionality(analysis, figures):
    rows = {r["population"]: r for r in read_csv(analysis / "dimensionality.csv")}
    names = [n for n in ORDER if n in rows]
    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(names) + 1.9))
    style(ax, "Connectivity dimensionality of the largest bonded component",
          xlabel="share of structures (%)")
    left = np.zeros(len(names))
    for d in range(4):
        values = np.array([to_float(rows[n][f"dim{d}_pct"]) for n in names])
        bars = ax.barh(names, values, left=left, color=DIM_SHADES[d],
                       edgecolor=SURFACE, linewidth=0.8,
                       label=f"{d}D" + (" (molecular)" if d == 0 else
                                        " (framework)" if d == 3 else ""))
        for bar, value in zip(bars, values):
            if value >= 6:
                ax.text(bar.get_x() + value / 2, bar.get_y() + bar.get_height() / 2,
                        f"{value:.0f}", ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if d >= 2 else TEXT_PRIMARY)
        left += values
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.28 if len(names) > 4 else -0.4))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    save(fig, figures / "dimensionality.png")


def plot_coordination(analysis, figures):
    cn_rows = {r["population"]: r for r in read_csv(analysis / "cn_distribution.csv")}
    elements = read_csv(analysis / "element_presence_and_cn.csv")
    names = [n for n in ORDER if n in cn_rows]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.0),
                             gridspec_kw={"height_ratios": [1, 1.25]})
    ax = style(axes[0], "Per-atom coordination number distribution",
               xlabel="coordination number", ylabel="share of atoms")
    bins = [k for k in cn_rows[names[0]] if k.startswith("cn")]
    x = np.arange(len(bins))
    for name in names:
        y = np.array([to_float(cn_rows[name][b]) for b in bins])
        ax.plot(x, y, color=COLOR[name], linewidth=2.0, label=name,
                marker="o", markersize=3.2)
    ax.set_xticks(x)
    ax.set_xticklabels([b[2:] if b != bins[-1] else f"{b[2:]}+" for b in bins])
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    # Rank elements by the largest mean CN spread across populations so the rows
    # that actually differ lead; never by one population's prevalence.
    def spread(row):
        values = [to_float(row[f"{n} mean_CN"]) for n in names
                  if row.get(f"{n} mean_CN") not in ("", None)]
        values = [v for v in values if np.isfinite(v)]
        return max(values) - min(values) if len(values) > 1 else 0.0

    top = sorted(elements, key=lambda r: -to_float(r["max_presence"]))[:22]
    top.sort(key=lambda r: -spread(r))
    ax = style(axes[1], "Mean coordination number by element "
                        "(elements ordered by spread across populations)",
               ylabel="mean CN")
    width = 0.8 / len(names)
    x = np.arange(len(top))
    for k, name in enumerate(names):
        y = [to_float(r.get(f"{name} mean_CN", "")) for r in top]
        ax.bar(x + k * width - 0.4 + width / 2, y, width, color=COLOR[name],
               label=name, edgecolor=SURFACE, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r["element"] for r in top], fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, ncol=3)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(h_pad=2.4)
    save(fig, figures / "coordination.png")


def plot_bonding(analysis, figures):
    rows = {r["descriptor"]: r for r in read_csv(analysis / "descriptor_comparison.csv")}
    names = [n for n in ORDER if f"{n} median" in next(iter(rows.values()))]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    ax = style(axes[0], "Bond-type composition (median share of bonds)",
               xlabel="share of bonds")
    channels = [("bonds metal-metal", "#4a4945"), ("bonds metal-nonmetal", "#7fa8d4"),
                ("bonds nonmetal-nonmetal", "#e8b04b")]
    left = np.zeros(len(names))
    for label, colour in channels:
        values = np.array([to_float(rows[label][f"{n} mean"]) for n in names])
        ax.barh(names, values, left=left, color=colour, edgecolor=SURFACE,
                linewidth=0.8, label=label.replace("bonds ", ""))
        left += values
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.34))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    ax = style(axes[1], "Organic and metal-ligand bond channels (mean share)",
               ylabel="share of bonds")
    channels = ["bonds C-H", "bonds C-C", "bonds C-O", "bonds metal-O"]
    x = np.arange(len(channels))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        y = [to_float(rows[c][f"{name} mean"]) for c in channels]
        ax.bar(x + k * width - 0.4 + width / 2, y, width, color=COLOR[name],
               label=name, edgecolor=SURFACE, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("bonds ", "") for c in channels])
    ax.legend(frameon=False, fontsize=7.6, ncol=2)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout(w_pad=3.0)
    save(fig, figures / "bonding.png")


def plot_descriptor_ranges(analysis, figures):
    rows = read_csv(analysis / "descriptor_comparison.csv")
    names = [n for n in ORDER if f"{n} median" in rows[0]]
    selected = ["density (g/cm3)", "packing fraction", "mean CN (all atoms)",
                "mean CN (metals)", "mean CN (non-metals)", "electronegativity spread",
                "metal atom fraction", "H per C", "energy/atom (eV)",
                "connected components", "largest component frac", "atoms"]
    keep = [r for r in rows if r["descriptor"] in selected]
    keep.sort(key=lambda r: selected.index(r["descriptor"]))

    ncols = 3
    nrows = int(np.ceil(len(keep) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 2.5 * nrows))
    for ax, row in zip(axes.ravel(), keep):
        style(ax, row["descriptor"])
        for k, name in enumerate(names):
            p25 = to_float(row[f"{name} p25"])
            p50 = to_float(row[f"{name} median"])
            p75 = to_float(row[f"{name} p75"])
            ax.plot([p25, p75], [k, k], color=COLOR[name], linewidth=3.2,
                    solid_capstyle="round", alpha=0.55)
            ax.plot([p50], [k], marker="o", color=COLOR[name], markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
    for ax in axes.ravel()[len(keep):]:
        ax.axis("off")
    fig.suptitle("Median (dot) and p25-p75 range (bar) per population",
                 color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left", y=1.005)
    fig.tight_layout(h_pad=2.0, w_pad=2.2)
    save(fig, figures / "descriptor_ranges.png")


def plot_family_affinity(analysis, figures, query, title, max_columns=16):
    path = analysis / f"family_affinity_{query}_lift.csv"
    share_path = analysis / f"family_affinity_{query}_share.csv"
    if not path.exists() or not share_path.exists():
        return
    rows = read_csv(path)
    shares = read_csv(share_path)
    cols = [c for c in rows[0] if c not in ("query_family", "row_mass")]

    # There are ~80 OMAT families and most carry almost no mass; showing all of
    # them makes the panel unreadable and hides the ones that matter.  Keep the
    # columns that actually receive links, ranked by total weighted share.
    mass = np.array([sum(to_float(s[c]) * to_float(s["row_mass"]) for s in shares)
                     for c in cols])
    keep = np.argsort(-mass)[:max_columns]
    keep = keep[np.argsort(-mass[keep])]
    cols = [cols[i] for i in keep]
    matrix = np.array([[to_float(r[c]) for c in cols] for r in rows])
    labels = [f"{r['query_family']}  ({to_float(r['row_mass']):.0f})" for r in rows]

    # Empty cells give log2(~0); clip to a robust range so a handful of extreme
    # low-support cells cannot flatten the whole colour scale.
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    limit = float(np.nanpercentile(np.abs(matrix), 98))
    limit = limit if limit > 0 else 1.0
    matrix = np.clip(matrix, -limit, limit)

    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 6.0, 0.38 * len(rows) + 3.0))
    style(ax, title)
    ax.grid(False)
    mesh = ax.pcolormesh(matrix, cmap="RdBu_r",
                         norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                         edgecolors=SURFACE, linewidth=0.6)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) >= 0.5 * limit:
                ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:+.1f}", ha="center",
                        va="center", fontsize=7,
                        color="#ffffff" if abs(matrix[i, j]) > 0.75 * limit
                        else TEXT_PRIMARY)
    ax.set_xticks(np.arange(len(cols)) + 0.5)
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    bar = fig.colorbar(mesh, ax=ax, pad=0.015)
    bar.set_label(f"log2 lift vs the overall OMAT family mix (clipped at +-{limit:.1f})",
                  color=TEXT_SECONDARY, fontsize=8.5)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    save(fig, figures / f"family_affinity_{query}.png")


def plot_dimensionality_confusion(analysis, figures):
    rows = read_csv(analysis / "dimensionality_confusion.csv")
    queries = sorted({r["query_set"] for r in rows})
    fig, axes = plt.subplots(1, len(queries), figsize=(5.4 * len(queries), 4.4))
    axes = np.atleast_1d(axes)
    for ax, query in zip(axes, queries):
        subset = [r for r in rows if r["query_set"] == query]
        matrix = np.array([[to_float(r[f"neighbour_dim{k}"]) for k in range(4)]
                           for r in subset])
        style(ax, f"{query}: query vs neighbour dimensionality",
              xlabel="neighbour dimensionality", ylabel="query dimensionality")
        mesh = ax.pcolormesh(matrix, cmap="Blues", vmin=0, vmax=1,
                             edgecolors=SURFACE, linewidth=1.2)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:.2f}", ha="center",
                        va="center", fontsize=9,
                        color="#ffffff" if matrix[i, j] > 0.55 else TEXT_PRIMARY)
        ax.set_xticks(np.arange(4) + 0.5)
        ax.set_xticklabels([f"{k}D" for k in range(4)])
        ax.set_yticks(np.arange(4) + 0.5)
        ax.set_yticklabels([f"{k}D" for k in range(4)])
        ax.invert_yaxis()
        fig.colorbar(mesh, ax=ax, pad=0.02).ax.tick_params(
            colors=TEXT_SECONDARY, labelsize=8)
    fig.tight_layout(w_pad=2.6)
    save(fig, figures / "dimensionality_confusion.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args()

    analysis = args.analysis.resolve()
    figures = (args.figures or analysis.parent / "figures").resolve()
    figures.mkdir(parents=True, exist_ok=True)

    plot_dimensionality(analysis, figures)
    plot_coordination(analysis, figures)
    plot_bonding(analysis, figures)
    plot_descriptor_ranges(analysis, figures)
    plot_family_affinity(analysis, figures, "mof_off",
                         "MOF family x OMAT24 family, pc25 top-5")
    plot_family_affinity(analysis, figures, "mad",
                         "MAD subset x OMAT24 family, pc25 top-5")
    plot_dimensionality_confusion(analysis, figures)


if __name__ == "__main__":
    main()
