#!/usr/bin/env python
"""Figures for the MOF-off R2SCAN -> OMAT24 nearest-neighbor chemistry analysis.

Reads the CSV tables written by ``mof_omat_neighbor_chemistry.py analyze`` and
renders the panels that carry the argument:

  1. element_presence.png     which elements the retrieved OMAT24 structures
                              contain, against the MOF-off queries themselves
  2. element_enrichment.png   log2 enrichment of the neighbor set over a matched
                              random baseline from the same OMAT24 subdataset
  3. motifs.png               organic/inorganic character and the MOF motif
  4. distributions.png        atom count, volume per atom, density, energy/atom
  5. pairing.png              per-rank composition agreement between each query
                              and its retrieved neighbors
  6. mof_element_coverage.png per MOF-off element, how often the nearest OMAT24
                              neighbor contains that element too

Colour roles are fixed by entity, never by rank: MOF-off queries are blue, the
raw-128d neighbor set orange, the pc25 neighbor set aqua, and OMAT24 baselines
stay neutral grey so they read as reference rather than as a fourth series.
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
SERIES = {
    "mof_off_r2scan": "#2a78d6",
    "raw128": "#eb6834",
    "pc25": "#1baf7a",
    "baseline": "#9a9993",
    "baseline_alt": "#c5c4be",
}
LABELS = {
    "mof_off_r2scan": "MOF-off R2SCAN queries",
    "raw128": "raw-128d neighbors",
    "pc25": "pc25 neighbors",
    "omat_bg_nvt3000": "OMAT24 random (nvt-3000)",
    "omat_bg_global": "OMAT24 random (all)",
}


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold",
                     loc="left", pad=10)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def plot_element_presence(analysis, out, top_n=26):
    rows = read_csv(analysis / "element_presence_and_abundance.csv")
    rows.sort(key=lambda r: -float(r["presence_mof_off_r2scan"]))
    rows = [r for r in rows if float(r["presence_mof_off_r2scan"]) > 0][:top_n]
    symbols = [r["symbol"] for r in rows]
    y = np.arange(len(rows))

    series = [
        ("mof_off_r2scan", [float(r["presence_mof_off_r2scan"]) for r in rows]),
        ("raw128", [float(r["presence_raw128_top10"]) for r in rows]),
        ("pc25", [float(r["presence_pc25_top10"]) for r in rows]),
    ]
    baseline = [float(r["presence_omat_bg_nvt3000"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    height = 0.24
    for offset, (key, values) in zip((height, 0.0, -height), series):
        ax.barh(y + offset, values, height=height * 0.92, color=SERIES[key],
                label=LABELS[key], zorder=3)
    ax.scatter(baseline, y, marker="|", s=90, color=SERIES["baseline"],
               linewidths=1.8, label=LABELS["omat_bg_nvt3000"], zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(symbols, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    style_axes(ax, xlabel="fraction of structures containing the element",
               title="Element content: MOF-off queries vs the OMAT24 structures retrieved as their neighbors")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    fig.text(0.01, -0.01,
             "Elements ordered by how common they are in the MOF-off R2SCAN train set. "
             "Neighbor bars use all ten retrieved neighbors per query.",
             fontsize=8.5, color=TEXT_MUTED)
    save(fig, out / "element_presence.png")


def plot_enrichment(analysis, out, top_n=18):
    rows = read_csv(analysis / "element_enrichment.csv")
    presence = {r["symbol"]: r for r in read_csv(
        analysis / "element_presence_and_abundance.csv")}

    # Only elements that actually appear in the neighbor set are informative.
    kept = [r for r in rows
            if float(presence[r["symbol"]]["presence_raw128_top10"]) > 0.005
            or float(presence[r["symbol"]]["presence_pc25_top10"]) > 0.005]
    kept.sort(key=lambda r: float(r["log2_raw128_vs_omat_bg_nvt3000"]))
    if len(kept) > top_n * 2:
        kept = kept[:top_n] + kept[-top_n:]
    symbols = [r["symbol"] for r in kept]
    y = np.arange(len(kept))

    fig, ax = plt.subplots(figsize=(9, max(5.5, 0.30 * len(kept))))
    height = 0.38
    for offset, key in ((height / 2, "raw128"), (-height / 2, "pc25")):
        values = [float(r[f"log2_{key}_vs_omat_bg_nvt3000"]) for r in kept]
        ax.barh(y + offset, values, height=height * 0.9, color=SERIES[key],
                label=LABELS[key], zorder=3)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=1.0, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(symbols, fontsize=9.5)
    style_axes(ax, xlabel="log2(neighbor presence / random nvt-3000 presence)",
               title="Which elements the neighbor search over- and under-selects")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    fig.text(0.01, -0.02,
             "Positive = the element is more common among retrieved neighbors than in a size-matched "
             "random sample of the same OMAT24 subdataset.",
             fontsize=8.5, color=TEXT_MUTED)
    save(fig, out / "element_enrichment.png")


def plot_motifs(analysis, out):
    rows = {r["population"]: r for r in read_csv(analysis / "element_classes_and_motifs.csv")}
    motifs = [
        ("has_metal", "contains\na metal"),
        ("has_C_and_H", "contains\nC and H"),
        ("has_C_H_O", "contains\nC, H and O"),
        ("has_C_H_N_O", "contains\nC, H, N and O"),
        ("mof_motif_metal_C_H_O", "MOF motif:\nmetal + C + H + O"),
        ("purely_inorganic_no_C_no_H", "no C and\nno H at all"),
    ]
    populations = [
        ("mof_off_r2scan", SERIES["mof_off_r2scan"], LABELS["mof_off_r2scan"]),
        ("raw128_top10", SERIES["raw128"], LABELS["raw128"]),
        ("pc25_top10", SERIES["pc25"], LABELS["pc25"]),
        ("omat_bg_nvt3000", SERIES["baseline"], LABELS["omat_bg_nvt3000"]),
        ("omat_bg_global", SERIES["baseline_alt"], LABELS["omat_bg_global"]),
    ]
    x = np.arange(len(motifs))
    width = 0.16

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for i, (population, color, label) in enumerate(populations):
        offset = (i - (len(populations) - 1) / 2) * width
        values = [float(rows[population][key]) for key, _ in motifs]
        bars = ax.bar(x + offset, values, width=width * 0.9, color=color, label=label, zorder=3)
        # Direct-label the reference series only; the rest are read off the axis.
        if population == "mof_off_r2scan":
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}",
                        ha="center", fontsize=8, color=SERIES["mof_off_r2scan"],
                        fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in motifs], fontsize=9)
    ax.set_ylim(0, 1.10)
    style_axes(ax, ylabel="fraction of structures",
               title="Organic/inorganic character: MOF-off queries vs their OMAT24 neighbors")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY, ncol=5,
              loc="lower center", bbox_to_anchor=(0.5, 1.06), handlelength=1.2,
              columnspacing=1.4)
    save(fig, out / "motifs.png")


def plot_distributions(analysis, out):
    rows = read_csv(analysis / "size_and_energy_stats.csv")
    table = {(r["population"], r["field"]): r for r in rows}
    fields = [
        ("n_atoms", "atoms per structure"),
        ("volume_per_atom", "volume per atom (A^3)"),
        ("density_g_cm3", "density (g/cm3)"),
        ("energy_per_atom", "energy per atom (eV)"),
    ]
    populations = [
        ("mof_off_r2scan", SERIES["mof_off_r2scan"], LABELS["mof_off_r2scan"]),
        ("raw128_top10", SERIES["raw128"], LABELS["raw128"]),
        ("pc25_top10", SERIES["pc25"], LABELS["pc25"]),
        ("omat_bg_nvt3000", SERIES["baseline"], LABELS["omat_bg_nvt3000"]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))
    for ax, (field, label) in zip(axes, fields):
        for i, (population, color, series_label) in enumerate(populations):
            row = table.get((population, field))
            if row is None:
                continue
            median = float(row["p50"])
            lo25, hi75 = float(row["p25"]), float(row["p75"])
            lo05, hi95 = float(row["p05"]), float(row["p95"])
            y = len(populations) - 1 - i
            ax.plot([lo05, hi95], [y, y], color=color, linewidth=2.0, solid_capstyle="round",
                    alpha=0.45, zorder=3)
            ax.plot([lo25, hi75], [y, y], color=color, linewidth=6.0, solid_capstyle="round",
                    zorder=4)
            ax.plot([median], [y], marker="o", markersize=8, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=5,
                    label=series_label if field == "n_atoms" else None)
        ax.set_yticks(range(len(populations)))
        ax.set_yticklabels([])
        ax.set_ylim(-0.6, len(populations) - 0.4)
        style_axes(ax, xlabel=label)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)

    axes[0].set_title("Size, packing and energetics", color=TEXT_PRIMARY, fontsize=12,
                      fontweight="bold", loc="left", pad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.text(0.01, -0.13, "Dot = median, thick bar = interquartile range, thin bar = 5th-95th percentile.",
             fontsize=8.5, color=TEXT_MUTED)
    save(fig, out / "distributions.png")


def plot_pairing(analysis, out):
    rows = read_csv(analysis / "query_neighbor_pairing.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    metrics = [
        ("mean_jaccard_elements", "mean Jaccard of element sets"),
        ("frac_share_any_element", "fraction sharing >=1 element"),
        ("frac_share_any_metal", "fraction sharing >=1 metal"),
    ]
    for ax, (field, label) in zip(axes, metrics):
        for geometry in ("raw128", "pc25"):
            subset = [r for r in rows if r["geometry"] == geometry]
            subset.sort(key=lambda r: int(r["rank"]))
            ranks = [int(r["rank"]) for r in subset]
            values = [float(r[field]) for r in subset]
            ax.plot(ranks, values, color=SERIES[geometry], linewidth=2.0, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=LABELS[geometry], zorder=4)
        ax.set_xticks(range(1, 11))
        ax.set_ylim(0, max(0.25, ax.get_ylim()[1]))
        style_axes(ax, xlabel="neighbor rank", ylabel=label)
    axes[0].set_title("Composition agreement between each MOF-off query and its OMAT24 neighbors",
                      color=TEXT_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    save(fig, out / "pairing.png")


def plot_coverage(analysis, out, min_share=0.01):
    rows = read_csv(analysis / "mof_element_coverage_by_nearest.csv")
    by_symbol = {}
    for row in rows:
        by_symbol.setdefault(row["symbol"], {})[row["geometry"]] = row
    kept = [(symbol, data) for symbol, data in by_symbol.items()
            if float(next(iter(data.values()))["share_of_mof_structures"]) >= min_share]
    kept.sort(key=lambda item: -float(next(iter(item[1].values()))["share_of_mof_structures"]))
    symbols = [symbol for symbol, _ in kept]
    y = np.arange(len(kept))

    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(kept))))
    height = 0.38
    for offset, geometry in ((height / 2, "raw128"), (-height / 2, "pc25")):
        values = [float(data[geometry]["frac_nearest_neighbor_also_has"]) for _, data in kept]
        ax.barh(y + offset, values, height=height * 0.9, color=SERIES[geometry],
                label=LABELS[geometry], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(symbols, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    style_axes(ax, xlabel="fraction whose nearest OMAT24 neighbor also contains the element",
               title="Element-level coverage of the MOF-off set by its nearest OMAT24 structures")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    fig.text(0.01, -0.02,
             "Restricted to elements present in at least 1% of MOF-off R2SCAN train structures.",
             fontsize=8.5, color=TEXT_MUTED)
    save(fig, out / "mof_element_coverage.png")


# Tightening the neighbor window (top-10 -> top-5 -> top-2) is an ordinal
# progression, not five independent categories, so those three take a one-hue
# ordinal ramp (the documented blue steps 250/450/650, light = loosest cut) while
# the two reference populations keep distinct categorical identities.  Verified
# against the light surface: lightest ramp step 2.06:1, adjacent ramp separation
# OKLab dE 20.0 / 19.5, and every ramp-vs-reference pair >=15 normal / >=8 CVD.
RANK_RAMP = {"top10": "#86b6ef", "top5": "#2a78d6", "top2": "#104281"}
REFERENCE = {"mof_off_r2scan": "#eb6834", "omat_bg_global": "#75746f"}


def plot_presence_by_rank_depth(analysis, out, geometry, min_presence=0.03, top_n=30):
    """Element presence across OMAT24, MOF-off, and tightening neighbor windows."""
    rows = read_csv(analysis / "element_presence_and_abundance.csv")
    series = [
        ("omat_bg_global", REFERENCE["omat_bg_global"], "OMAT24 (uniform random)"),
        ("mof_off_r2scan", REFERENCE["mof_off_r2scan"], "MOF-off R2SCAN"),
        (f"{geometry}_top10", RANK_RAMP["top10"], "all 10 neighbors"),
        (f"{geometry}_top5", RANK_RAMP["top5"], "top 5 neighbors"),
        (f"{geometry}_top2", RANK_RAMP["top2"], "top 2 neighbors"),
    ]

    def value(row, population):
        return float(row[f"presence_{population}"]) * 100.0

    kept = [r for r in rows
            if max(value(r, population) for population, _, _ in series) >= min_presence * 100]
    kept.sort(key=lambda r: -value(r, "mof_off_r2scan"))
    kept = kept[:top_n]
    symbols = [r["symbol"] for r in kept]
    y = np.arange(len(kept))

    fig, ax = plt.subplots(figsize=(10.5, max(6.5, 0.46 * len(kept))))
    height = 0.16
    for i, (population, color, label) in enumerate(series):
        offset = ((len(series) - 1) / 2 - i) * height
        ax.barh(y + offset, [value(r, population) for r in kept], height=height * 0.92,
                color=color, label=label, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(symbols, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    style_axes(ax, xlabel="percentage of structures containing the element (%)",
               title=f"Element presence as the neighbor window tightens ({geometry})")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right")
    fig.text(0.01, -0.015,
             f"Neighbors are the OMAT24 structures retrieved for the 80,643 MOF-off R2SCAN train "
             f"frames under the {geometry} metric; the three blue shades are nested cuts of the same "
             f"ranking (darker = closer). Elements shown: present in >={min_presence:.0%} of at least "
             f"one population, ordered by MOF-off prevalence.",
             fontsize=8.5, color=TEXT_MUTED, wrap=True)
    save(fig, out / f"element_presence_by_rank_depth_{geometry}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="directory written by `mof_omat_neighbor_chemistry.py analyze`")
    parser.add_argument("--out", type=Path, default=None, help="figure output directory")
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    out = (args.out or analysis.parent / "figures").resolve()

    plot_element_presence(analysis, out)
    plot_enrichment(analysis, out)
    plot_motifs(analysis, out)
    plot_distributions(analysis, out)
    plot_pairing(analysis, out)
    plot_coverage(analysis, out)
    for geometry in ("raw128", "pc25"):
        plot_presence_by_rank_depth(analysis, out, geometry)
    print(f"figures written to {out}", flush=True)


if __name__ == "__main__":
    main()
