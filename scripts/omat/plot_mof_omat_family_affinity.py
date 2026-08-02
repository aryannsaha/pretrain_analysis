#!/usr/bin/env python
"""Two-panel MOF-family x OMAT-family affinity heatmaps.

Panel A  row-normalised flow: of a MOF family's top-5 neighbour links, what
         share lands in each OMAT family.  Rows sum to 100%.
Panel B  log2 lift against independence, with the neutral grey pinned to
         exactly zero.  This is the panel that answers the question: the flow
         panel is dominated by the shared column marginal (`oxide / alkali`
         alone is ~28% of all mass), so almost every row's largest flow cell
         is the same column whether or not there is any real affinity.

Both panels use one clustered order, computed on the lift matrix, so a cell at
(i, j) is the same family pair in both.  Cells are gated three ways:
  * fewer than the minimum distinct MOFs      -> flat grey, unscored
  * bootstrap sign agreement below the floor  -> hatched, unresolved
  * resting on <= 2 distinct OMAT trajectories -> marked with a dot, meaning
    "one specific OMAT material", not a family
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
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.patches import Rectangle

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#75746f"
GRID = "#e4e3df"
UNSCORED = "#eceae6"

FLOW_CMAP = LinearSegmentedColormap.from_list(
    "flow_blue",
    ["#cde2fb", "#9ec5f4", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"],
)
# Diverging: documented red pole and blue ramp, documented neutral grey at 0.5.
LIFT_CMAP = LinearSegmentedColormap.from_list(
    "lift_div",
    [(0.000, "#6b1615"), (0.125, "#ab2b2a"), (0.250, "#e34948"), (0.375, "#f4a5a4"),
     (0.500, "#f0efec"),
     (0.625, "#cde2fb"), (0.750, "#9ec5f4"), (0.875, "#3987e5"), (1.000, "#0d366b")],
)


def read_matrix(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    R = max(int(r["display_row"]) for r in rows) + 1
    C = max(int(r["display_col"]) for r in rows) + 1
    grids = {k: np.zeros((R, C)) for k in
             ("row_share_pct", "log2_lift", "scored", "n_distinct_mofs",
              "n_distinct_omat_trajectories", "bootstrap_sign_agreement",
              "close_decile_weighted_mofs", "close_decile_n_mofs", "raw_links")}
    row_lab, col_lab = [""] * R, [""] * C
    for r in rows:
        i, j = int(r["display_row"]), int(r["display_col"])
        row_lab[i], col_lab[j] = r["mof_family"], r["omat_family"]
        for k in grids:
            grids[k][i, j] = float(r[k])
    return grids, row_lab, col_lab


def style_grid(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)


def draw_panel(ax, values, mask, cmap, norm, annotate=None, hatch=None, dots=None):
    data = np.ma.masked_where(mask, values)
    shaded = cmap.copy()
    shaded.set_bad(UNSCORED)
    R, C = values.shape
    mesh = ax.pcolormesh(np.arange(C + 1), np.arange(R + 1), data,
                         cmap=shaded, norm=norm, edgecolors=SURFACE, linewidth=1.2)
    if hatch is not None:
        for i, j in zip(*np.nonzero(hatch)):
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, hatch="///",
                                   edgecolor="#c3c2b7", linewidth=0.0))
    if dots is not None:
        for i, j in zip(*np.nonzero(dots)):
            ax.plot(j + 0.82, i + 0.20, marker="o", markersize=2.1,
                    color=TEXT_SECONDARY, alpha=0.75)
    if annotate is not None:
        for i, j in zip(*np.nonzero(annotate)):
            v = values[i, j]
            frac = norm(v)
            ax.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center",
                    fontsize=7.0,
                    color=SURFACE if frac > 0.55 else TEXT_PRIMARY)
    ax.set_xlim(0, C)
    ax.set_ylim(R, 0)
    style_grid(ax)
    return mesh


def build_figure(grids, row_lab, col_lab, row_n_mofs, out_path, title, subtitle, caption,
                 flow_key="row_share_pct", support_key="n_distinct_mofs",
                 min_mofs=5, stability_floor=0.90):
    flow = grids[flow_key]
    lift = grids["log2_lift"]
    scored = (grids[support_key] >= min_mofs)
    stable = grids["bootstrap_sign_agreement"] >= stability_floor
    traj_thin = scored & (grids["n_distinct_omat_trajectories"] <= 2)

    R, C = flow.shape
    col_share = grids["raw_links"].sum(axis=0) / max(grids["raw_links"].sum(), 1) * 100

    # Explicit inch budgets; bbox_inches="tight" is deliberately NOT used because
    # it reflows manually placed colourbars and captions.
    left_in, right_pad = 3.05, 0.30
    bottom_in, top_in = 1.85, 3.30
    panel_w, panel_h = C * 0.40, R * 0.32
    fig_w = left_in + 2 * panel_w + 0.35 + right_pad
    fig_h = panel_h + bottom_in + top_in
    axes_top = 1 - top_in / fig_h
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), sharey=True,
        gridspec_kw={"wspace": 0.06, "left": left_in / fig_w,
                     "right": 1 - right_pad / fig_w,
                     "top": axes_top, "bottom": bottom_in / fig_h},
    )

    vmax = min(100.0, np.ceil(flow.max() / 5) * 5)
    mesh_a = draw_panel(axes[0], flow, ~np.isfinite(flow), FLOW_CMAP,
                        PowerNorm(0.5, vmin=0, vmax=vmax), annotate=(flow >= 5.0))

    lift_masked = ~scored
    lim = 3.0
    mesh_b = draw_panel(axes[1], lift, lift_masked, LIFT_CMAP,
                        Normalize(vmin=-lim, vmax=lim),
                        hatch=(scored & ~stable), dots=traj_thin)

    # Outline each row's largest-flow cell in BOTH panels: this is what lets the
    # reader locate "where this family actually goes" inside the lift field.
    for i in range(R):
        j = int(np.argmax(flow[i]))
        for ax in axes:
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False,
                                   edgecolor=TEXT_PRIMARY, linewidth=1.25))

    axes[0].set_yticks(np.arange(R) + 0.5)
    axes[0].set_yticklabels(
        [f"{lab}   n={int(n)}" for lab, n in zip(row_lab, row_n_mofs)], fontsize=8.3)
    for ax in axes:
        ax.set_xticks(np.arange(C) + 0.5)
        ax.tick_params(labeltop=True, labelbottom=False, top=False, bottom=False)
        ax.set_xticklabels([f"{lab}  {s:.0f}%" for lab, s in zip(col_lab, col_share)],
                           rotation=40, ha="left", rotation_mode="anchor", fontsize=8.3)
    axes[1].tick_params(labelleft=False)

    panel_title_y = axes_top + 1.62 / fig_h
    for ax, text in zip(axes, ("A · where each MOF family's neighbours land  (row %)",
                               "B · over/under-representation vs independence  (log2)")):
        x0 = ax.get_position().x0
        fig.text(x0, panel_title_y, text, fontsize=11.5, fontweight="bold",
                 color=TEXT_PRIMARY, va="bottom")

    bar_y, bar_h = 1.02 / fig_h, 0.30 / fig_h
    for ax, mesh, label, ticks, labels in (
        (axes[0], mesh_a, "share of this MOF family's top-5 links (%)  ·  sqrt scale",
         None, None),
        (axes[1], mesh_b, "observed / expected  ·  grey = exactly 1x",
         [-3, -2, -1, 0, 1, 2, 3], ["1/8x", "1/4x", "1/2x", "1x", "2x", "4x", "8x"]),
    ):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0, bar_y, pos.width * 0.70, bar_h * 0.16])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                          extend="both" if ticks else "neither")
        if ticks:
            cb.set_ticks(ticks)
            cb.set_ticklabels(labels)
        cb.set_label(label, fontsize=8.5, color=TEXT_SECONDARY, labelpad=4)
        cb.outline.set_edgecolor(GRID)
        cb.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)

    fig.text(0.006, 1 - 0.32 / fig_h, title, fontsize=14, fontweight="bold",
             color=TEXT_PRIMARY, va="top")
    fig.text(0.006, 1 - 0.66 / fig_h, subtitle, fontsize=9.5, color=TEXT_MUTED, va="top")
    fig.text(0.006, 0.10 / fig_h, caption, fontsize=8.2, color=TEXT_MUTED, va="bottom",
             wrap=True)

    fig.patch.set_facecolor(SURFACE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}  ({int(scored.sum())} scored cells, "
          f"{int((scored & stable).sum())} bootstrap-stable)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="the family_affinity directory")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    analysis = args.analysis.resolve()
    out = (args.out or analysis.parent / "figures").resolve()
    grids, row_lab, col_lab = read_matrix(analysis / "family_affinity_matrix.csv")

    # Row totals come from the marginals table, not from a max over cells.
    with (analysis / "mof_family_marginals.csv").open(newline="") as handle:
        marg = {r["mof_family"]: int(r["distinct_mofs"]) for r in csv.DictReader(handle)}
    row_n_mofs = np.array([marg[lab] for lab in row_lab])
    if row_n_mofs.sum() != 3269:
        print(f"warning: row MOF counts sum to {row_n_mofs.sum()}, expected 3269", flush=True)

    caption = (
        "MOF-off R2SCAN, raw-128d metric, top-5 neighbours: 403,215 links from 80,643 frames of "
        "3,269 distinct MOFs. Each MOF carries equal total weight (split evenly over its "
        "temperatures, then its frames), so row totals are counts of distinct MOFs and a "
        "90-frame MOF cannot outvote a 1-frame MOF; the effective sample size is MOFs, not links. "
        "Rows in panel A sum to 100%. Panel B is log2(observed/expected) with the row profile "
        "shrunk toward the column marginal by 5 MOF-equivalents. Flat grey = fewer than 5 "
        "distinct MOFs in the cell (not scored); hatched = bootstrap sign agreement below 90% "
        "(unresolved); a dot marks cells resting on <=2 distinct OMAT trajectories, i.e. one "
        "specific material rather than a family. The black outline is each row's largest-flow "
        "cell, shown in both panels. Rows and columns share one order, clustered on the lift "
        "matrix (correlation distance, average linkage)."
    )
    build_figure(
        grids, row_lab, col_lab, row_n_mofs, out / "family_affinity_raw128.png",
        "Which MOF families land on which OMAT24 families",
        "Panel A is dominated by the shared column marginal; panel B is where the chemistry is.",
        caption,
    )

    # Closest-decile companion: same families, same clustered order.
    close = grids["close_decile_weighted_mofs"]
    totals = close.sum(axis=1, keepdims=True)
    close_share = np.divide(close * 100, totals, out=np.zeros_like(close), where=totals > 0)
    grids_close = dict(grids)
    grids_close["close_share"] = close_share
    caption_close = (
        "Same families and same clustered order as the main figure, restricted to the closest "
        "decile of queries by d1 (raw-128d d1 <= 28.83): 8,065 frames from only 322 of the 3,269 "
        "MOFs. Most rows therefore have little or no support here - flat grey means fewer than 5 "
        "distinct MOFs contributed, which is the common case rather than the exception. Read this "
        "panel only where support is present; it asks whether the affinity structure survives when "
        "genuine proximity is demanded."
    )
    build_figure(
        grids_close, row_lab, col_lab, row_n_mofs, out / "family_affinity_closest_decile.png",
        "Closest decile only: does the affinity structure survive?",
        "Restricted to the 10% of queries whose nearest OMAT24 neighbour is genuinely close.",
        caption_close, flow_key="close_share", support_key="close_decile_n_mofs",
    )


if __name__ == "__main__":
    main()
