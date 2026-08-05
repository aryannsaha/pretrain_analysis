#!/usr/bin/env python
"""Figures for the MOF-off -> OMAT24 chemical-relationship controls.

  1. composition_baseline.png   how much of the retrieval a fractional element
                                vector reproduces, and the sum-pooling artifact
  2. metal_node.png             per-metal agreement and the CN difference
  3. distance_calibration.png   MOF d1 against OMAT24's own leave-one-out NN
                                distribution, with and without trajectory twins
  4. pc_interpretation.png      R^2 of every retained PC against each descriptor
  5. rdf_similarity.png         independent similarity check on the retrieved pairs

Colour is fixed by entity, matching plot_pc25_top5_chemistry.py: MOF-off blue,
its retrieved neighbours orange, OMAT24 references neutral grey.
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
GRID = "#e4e3df"
MOF = "#2a78d6"
NEIGHBOUR = "#eb6834"
REFERENCE = "#9a9993"
REFERENCE_ALT = "#c5c4be"
ACCENT = "#1baf7a"


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value):
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return np.nan


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
    ax.grid(color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    return ax


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def plot_composition(analysis, figures):
    rows = read_csv(analysis / "composition_baseline.csv")
    norms = read_csv(analysis / "embedding_norm_extensivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    ax = style(axes[0], "Composition cosine similarity achieved by the retrieval",
               xlabel="mean cosine similarity to the MOF composition")
    wanted = ["composition cosine to the composition-optimal top-5",
              "composition cosine to embedding top-5 (higher = more similar)",
              "composition cosine to random pool structures"]
    labels = ["composition-optimal top-5\n(ceiling)", "embedding top-5\n(what pc25 returns)",
              "random OMAT24\n(floor)"]
    colours = [ACCENT, NEIGHBOUR, REFERENCE]
    values, lookup = [], {r["quantity"]: r for r in rows}
    for key in wanted:
        values.append(to_float(lookup[key]["mean"]) if key in lookup else np.nan)
    bars = ax.barh(labels, values, color=colours, edgecolor=SURFACE, linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlim(0, max(1.0, max(values) * 1.2))

    ax = style(axes[1], "Embedding norm vs extensive quantities (sum-pooling artifact)",
               ylabel="R^2")
    populations = sorted({r["population"] for r in norms})
    fields = ["n_atoms", "total_z"]
    width = 0.8 / len(fields)
    x = np.arange(len(populations))
    palette = {"n_atoms": MOF, "total_z": NEIGHBOUR}
    for k, field in enumerate(fields):
        y = [to_float(next((r["r_squared"] for r in norms
                            if r["population"] == p and r["against"] == field), "nan"))
             for p in populations]
        ax.bar(x + k * width - 0.4 + width / 2, y, width, color=palette[field],
               label=field, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace(" ", "\n", 1) for p in populations], fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(w_pad=3.0)
    save(fig, figures / "composition_baseline.png")


def plot_metal_node(analysis, figures):
    metals = read_csv(analysis / "metal_node_agreement.csv")
    per_mof = read_csv(analysis / "per_mof_summary.csv")
    metals.sort(key=lambda r: -int(r["n_mofs"]))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = style(axes[0], "Does the neighbour reproduce the MOF metal node?",
               ylabel="% of MOFs")
    channels = [("pct_neighbour_shares_metal", "same metal present", MOF),
                ("pct_neighbour_shares_donor", "same donor type", NEIGHBOUR),
                ("pct_neighbour_same_cn", "same CN (+-0.5)", ACCENT)]
    x = np.arange(len(metals))
    width = 0.8 / len(channels)
    for k, (field, label, colour) in enumerate(channels):
        y = [to_float(r[field]) for r in metals]
        ax.bar(x + k * width - 0.4 + width / 2, y, width, color=colour, label=label,
               edgecolor=SURFACE, linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['metal']}\n({r['n_mofs']})" for r in metals], fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, ncol=3)

    ax = style(axes[1], "Neighbour CN minus MOF CN (same metal)",
               xlabel="CN difference", ylabel="MOFs")
    deltas = np.array([to_float(r["cn_delta"]) for r in per_mof])
    deltas = deltas[np.isfinite(deltas)]
    ax.hist(deltas, bins=40, color=NEIGHBOUR, edgecolor=SURFACE, linewidth=0.4)
    ax.axvline(0, color=TEXT_PRIMARY, linewidth=1.2, linestyle="--")
    ax.text(0.02, 0.95, f"median {np.median(deltas):+.2f}\nn = {deltas.size:,}",
            transform=ax.transAxes, va="top", fontsize=9, color=TEXT_PRIMARY)
    fig.tight_layout(w_pad=3.0)
    save(fig, figures / "metal_node.png")


def plot_calibration(analysis, figures, root):
    calibration = np.load(root / "runs/omat_knn_probe/omat_knn_pc25/OMAT_calibration"
                                 "/distances.npy")[:, 0]
    indices = np.load(root / "runs/omat_knn_probe/omat_knn_pc25/OMAT_calibration"
                             "/indices.npy")
    sources = np.load(root / "runs/omat_knn_probe/omat_knn_pc25/OMAT_calibration"
                             "/source_rows.npy")
    full = np.load(root / "runs/omat_knn_probe/omat_knn_pc25/OMAT_calibration"
                          "/distances.npy")
    far = np.abs(indices - sources[:, None]) > 1000
    has_far = far.any(axis=1)
    calibration_far = full[np.arange(full.shape[0]), np.argmax(far, axis=1)][has_far]
    mof = np.load(root / "runs/omat_knn_probe/omat_knn_pc25_mof_off/MOF_off_R2SCAN"
                         "/distances.npy")[:, 0]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    style(ax, "pc25 nearest-neighbour distance: MOF-off against OMAT24's own scale",
          xlabel="distance to nearest OMAT24 structure (pc25 Euclidean)",
          ylabel="cumulative fraction")
    for values, label, colour in (
        (calibration, "OMAT24 leave-one-out NN (all, 1M rows)", REFERENCE_ALT),
        (calibration_far, "OMAT24 leave-one-out NN (different trajectory)", REFERENCE),
        (mof, "MOF-off R2SCAN d1", MOF),
    ):
        ordered = np.sort(values)
        ax.plot(ordered, np.arange(1, ordered.size + 1) / ordered.size,
                color=colour, linewidth=2.2, label=f"{label}  (median {np.median(values):.3f})")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    save(fig, figures / "distance_calibration.png")


def plot_pcs(analysis, figures):
    rows = read_csv(analysis / "pc_interpretation.csv")
    populations = sorted({r["population"] for r in rows})
    fields = [k[3:] for k in rows[0] if k.startswith("r2_")]
    fig, axes = plt.subplots(len(populations), 1,
                             figsize=(1.0 * len(fields) + 3.5, 4.2 * len(populations)))
    axes = np.atleast_1d(axes)
    for ax, population in zip(axes, populations):
        subset = [r for r in rows if r["population"] == population]
        matrix = np.array([[to_float(r[f"r2_{f}"]) for f in fields] for r in subset])
        style(ax, f"{population}: R^2 of each PC against physical descriptors")
        ax.grid(False)
        mesh = ax.pcolormesh(matrix, cmap="Blues", vmin=0, vmax=1,
                             edgecolors=SURFACE, linewidth=1.0)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]) and matrix[i, j] >= 0.10:
                    ax.text(j + 0.5, i + 0.5, f"{matrix[i, j]:.2f}", ha="center",
                            va="center", fontsize=7.5,
                            color="#ffffff" if matrix[i, j] > 0.55 else TEXT_PRIMARY)
        ax.set_xticks(np.arange(len(fields)) + 0.5)
        ax.set_xticklabels(fields, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(subset)) + 0.5)
        ax.set_yticklabels([f"PC{r['PC']} ({to_float(r['explained_variance_ratio']):.1%})"
                            for r in subset], fontsize=8)
        ax.invert_yaxis()
        fig.colorbar(mesh, ax=ax, pad=0.015).ax.tick_params(
            colors=TEXT_SECONDARY, labelsize=8)
    fig.tight_layout(h_pad=2.5)
    save(fig, figures / "pc_interpretation.png")


def plot_rdf(analysis, figures):
    rows = read_csv(analysis / "rdf_similarity.csv")
    keep = [r for r in rows if not r["comparison"].startswith("paired")]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    style(ax, "Independent similarity check: RDF cosine similarity",
          xlabel="RDF cosine similarity")
    colours = [NEIGHBOUR, REFERENCE]
    for k, (row, colour) in enumerate(zip(keep, colours)):
        median = to_float(row["median"])
        ax.plot([to_float(row["p10"]), to_float(row["p90"])], [k, k], color=colour,
                linewidth=4.0, solid_capstyle="round", alpha=0.55)
        ax.plot([median], [k], marker="o", color=colour, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=1.3)
        ax.text(median, k - 0.25, f"{median:.3f}", ha="center", fontsize=9,
                color=TEXT_PRIMARY)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([r["comparison"] for r in keep], fontsize=9)
    ax.invert_yaxis()
    save(fig, figures / "rdf_similarity.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--root", type=Path,
                        default=Path("/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis"))
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args()

    analysis = args.analysis.resolve()
    figures = (args.figures or analysis.parent / "figures").resolve()
    figures.mkdir(parents=True, exist_ok=True)

    for name, fn in (("composition", lambda: plot_composition(analysis, figures)),
                     ("metal node", lambda: plot_metal_node(analysis, figures)),
                     ("calibration", lambda: plot_calibration(analysis, figures, args.root)),
                     ("PCs", lambda: plot_pcs(analysis, figures)),
                     ("RDF", lambda: plot_rdf(analysis, figures))):
        try:
            fn()
        except FileNotFoundError as error:
            print(f"skipping {name}: {error}")


if __name__ == "__main__":
    main()
