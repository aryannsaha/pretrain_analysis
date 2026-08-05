#!/usr/bin/env python3
"""Periodic-group and period (row) composition of OMAT24 pc25 nearest neighbours.

Answers: do the OMAT24 structures retrieved as nearest neighbours of an external
dataset correlate with periodic-table *group* or *row*?

For each query set (MOF-off R2SCAN, MAD, MP-ALOE) we compare three populations by
atom fraction:

  query       - the external dataset's own atoms
  neighbours  - the OMAT24 pc25 top-5 neighbours, weighted by retrieval
                multiplicity (an OMAT row retrieved 12 times counts 12x)
  OMAT24 pool - a 50k random sample of the OMAT24 pool being searched

Enrichment panels plot log2(neighbours / OMAT24 pool): the searched pool is the
denominator, so a non-flat profile means retrieval prefers certain groups/rows.

Group and row are badly confounded (d- and f-block elements are exactly the heavy
ones), so the third figure holds the row fixed and asks whether any group
structure survives. It does - but it reverses sign between periods 4 and 5, which
is why row is the coherent axis and group is not.

Inputs are the stored per-structure element-count matrices (`*_zcounts.npy`,
column j == atomic number Z), so nothing is recomputed from structures.

Usage:
    python scripts/omat/plot_neighbor_group_row.py \
        --runs-root /scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/runs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── design tokens (dataviz skill reference palette, light mode) ────────────────
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
C_QUERY, C_NEIGH, C_BG = "#2a78d6", "#eb6834", "#898781"  # slot 1, slot 2, de-emphasis
C_UP, C_DOWN = "#2a78d6", "#e34948"  # diverging poles (blue<->red), gray zero rule

NZ = 119        # zcounts columns: Z = 0..118
MIN_FRAC = 1e-4  # drop a bucket only if BELOW this in EVERY population (symmetric)


def periodic_tables() -> tuple[np.ndarray, np.ndarray]:
    """Return (period, group) arrays indexed by Z. Ln/An get group 0 == f-block."""
    period = np.zeros(NZ, dtype=int)
    for lo, hi, p in [(1, 2, 1), (3, 10, 2), (11, 18, 3), (19, 36, 4),
                      (37, 54, 5), (55, 86, 6), (87, 118, 7)]:
        period[lo:hi + 1] = p

    g: dict[int, int] = {1: 1, 2: 18}
    for z, grp in zip(range(3, 11), [1, 2, 13, 14, 15, 16, 17, 18]):
        g[z] = grp
    for z in range(11, 19):
        g[z] = g[z - 8]
    for z in range(19, 37):
        g[z] = {19: 1, 20: 2}.get(z, z - 18)
    for z in range(37, 55):
        g[z] = g[z - 18]
    for z in range(55, 87):
        g[z] = {55: 1, 56: 2}.get(z, 0 if 57 <= z <= 71 else g[z - 32])
    for z in range(87, 119):
        g[z] = {87: 1, 88: 2}.get(z, 0 if 89 <= z <= 103 else g[z - 32])

    group = np.zeros(NZ, dtype=int)
    for z, grp in g.items():
        group[z] = grp
    group[57:72] = 0   # f-block gets its own bucket, never folded into group 3
    group[89:104] = 0
    return period, group


PERIOD, GROUP = periodic_tables()
GROUPS = [0] + list(range(1, 19))   # 0 == f-block
PERIODS = list(range(1, 8))
GLABEL = {0: "f"}


def glabels(vals):
    return [GLABEL.get(v, str(v)) for v in vals]


def atom_fractions(zcounts: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Total atoms per Z, optionally weighted per row, normalised to fractions."""
    if weights is None:
        totals = zcounts.sum(axis=0, dtype=np.float64)
    else:
        totals = (zcounts.astype(np.float64) * weights[:, None]).sum(axis=0)
    total = totals.sum()
    return totals / total if total else totals


def load_populations(runs: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Load query / neighbour / background element distributions per dataset."""
    struct = runs / "omat_knn_probe" / "pc25_top5_chemistry" / "structures"
    multi = runs / "omat_knn_probe" / "multiset_top5_presence"

    bg = atom_fractions(np.load(struct / "omat_bg_global_zcounts.npy"))
    bg_nvt = atom_fractions(np.load(struct / "omat_bg_nvt3000_zcounts.npy"))

    spec = {
        "MOF-off R2SCAN": (struct / "query_mof_off_zcounts.npy", "mof_off"),
        "MAD": (struct / "query_mad_zcounts.npy", "mad"),
        "MP-ALOE": (multi / "mpaloe_all_dataset_zcounts.npy", "mpaloe"),
    }
    out = {}
    for label, (qpath, stem) in spec.items():
        nz = np.load(multi / f"{stem}_pc25_top5_zcounts.npy")
        w = np.load(multi / f"{stem}_pc25_top5_weights.npy")
        out[label] = {
            "query": atom_fractions(np.load(qpath)),
            "neigh": atom_fractions(nz, w),
            "n_queries": int(round(w.sum() / 5)),
            "n_unique_neighbours": int(len(w)),
        }
    return out, bg, bg_nvt


def compute_axis(pop, bg, key, values):
    """Bucket the three populations and return survivors + log2 lift."""
    q = np.array([pop["query"][key == v].sum() for v in values])
    n = np.array([pop["neigh"][key == v].sum() for v in values])
    b = np.array([bg[key == v].sum() for v in values])
    keep = np.maximum.reduce([q, n, b]) >= MIN_FRAC   # symmetric across populations
    q, n, b = q[keep], n[keep], b[keep]
    vals = [v for v, k in zip(values, keep) if k]
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where((n > 0) & (b > 0), np.log2(n / np.where(b > 0, b, 1)), np.nan)
    return vals, q, n, b, lift


def spread(lift, weights) -> float:
    """Pool-weighted std of log2 lift - a comparable 'how much structure' measure."""
    v, w = np.asarray(lift, float), np.asarray(weights, float)
    m = np.isfinite(v) & (w > 0)
    v, w = v[m], w[m]
    if not len(v):
        return float("nan")
    w = w / w.sum()
    return float(np.sqrt((w * (v - (v * w).sum()) ** 2).sum()))


# ── drawing ───────────────────────────────────────────────────────────────────
def style_axes(ax, ylabel):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)


def draw_distribution(ax, cats, series, title):
    """Grouped bars: query / neighbours / pool atom fraction."""
    x = np.arange(len(cats), dtype=float)
    slot = 0.86 / len(series)
    for i, (label, vals, color) in enumerate(series):
        ax.bar(x - 0.43 + slot * (i + 0.5), vals, slot * 0.82, label=label,
               color=color, linewidth=0, zorder=3)
    style_axes(ax, "atom fraction")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_xlim(-0.7, len(cats) - 0.3)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


def draw_enrichment(ax, cats, lift, has_pool, title, ylabel, n_label=3, cap=None):
    """Diverging bars: log2 lift. NaN + pool -> 'none' sentinel; NaN + no pool -> blank."""
    x = np.arange(len(cats), dtype=float)
    lift = np.asarray(lift, float)
    finite = np.isfinite(lift)
    if cap is None:
        cap = cap_for(lift)
    vals = np.where(finite, lift, 0.0)
    ax.bar(x[finite], vals[finite], 0.66,
           color=[C_UP if v >= 0 else C_DOWN for v in vals[finite]], linewidth=0, zorder=3)

    sentinel = (~finite) & np.asarray(has_pool, bool)
    if sentinel.any():
        ax.bar(x[sentinel], np.full(sentinel.sum(), -cap * 0.92), 0.66, color=SURFACE,
               edgecolor=C_DOWN, linewidth=0.9, linestyle=":", zorder=3)
        for i in np.where(sentinel)[0]:
            ax.annotate("none", (x[i], -cap * 0.92), ha="center", va="bottom",
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=7.5, color=MUTED, style="italic")
    ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=4)

    order = np.argsort(np.where(finite, lift, np.nan))
    order = [i for i in order if finite[i]]
    chosen = sorted(set(order[:n_label] + order[-n_label:]))
    # Stagger adjacent same-side labels. Push the SHALLOWER bar's label outward:
    # the extreme bar sits near the axis limit and has no room to spare.
    step = {i: 0 for i in chosen}
    for a, b in zip(chosen, chosen[1:]):
        if b == a + 1 and (lift[a] >= 0) == (lift[b] >= 0):
            step[a if abs(lift[a]) < abs(lift[b]) else b] = 1
    for i in chosen:
        v = lift[i]
        pad = (3 if v >= 0 else -4) + (13 if v >= 0 else -13) * step[i]
        ax.annotate(f"{2 ** v:.2g}x", (x[i], v), ha="center",
                    va="bottom" if v >= 0 else "top",
                    xytext=(0, pad), textcoords="offset points",
                    fontsize=8, color=INK2)

    style_axes(ax, ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_xlim(-0.7, len(cats) - 0.3)
    ax.set_ylim(-cap, cap)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)


LIFT_LABEL = "log$_2$ (neighbours / OMAT24 pool)"
LABEL_HEADROOM = 1.9  # data units reserved past the extreme bar so value labels never clip


def cap_for(lift) -> float:
    """Symmetric y-limit leaving room for the staggered extreme-value labels."""
    v = np.asarray(lift, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return 1.0
    return float(np.ceil((np.abs(v).max() + LABEL_HEADROOM) * 2) / 2)


def figure_headline(pops, bg, out, dpi):
    head = "MOF-off R2SCAN"
    p = pops[head]
    gv, gq, gn, gb, gl = compute_axis(p, bg, GROUP, GROUPS)
    pv, pq, pn, pb, pl = compute_axis(p, bg, PERIOD, PERIODS)
    gcats, pcats = glabels(gv), [str(v) for v in pv]

    fig = plt.figure(figsize=(13.2, 8.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[len(gcats), len(pcats) + 1.5],
                          hspace=0.44, wspace=0.17,
                          left=0.055, right=0.985, top=0.845, bottom=0.085)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    series_g = [("MOF-off query", gq, C_QUERY),
                ("pc25 top-5 neighbours", gn, C_NEIGH),
                ("OMAT24 pool", gb, C_BG)]
    series_p = [(lbl, v, c) for (lbl, _, c), v in zip(series_g, [pq, pn, pb])]
    draw_distribution(ax_a, gcats, series_g, "a  Composition by periodic group")
    draw_distribution(ax_b, pcats, series_p, "b  Composition by period (row)")
    draw_enrichment(ax_c, gcats, gl, gb > 0, "c  Retrieval enrichment by group", LIFT_LABEL)
    draw_enrichment(ax_d, pcats, pl, pb > 0, "d  Retrieval enrichment by period", LIFT_LABEL)

    for ax in (ax_a, ax_c):
        ax.set_xlabel("periodic group   (f = lanthanides + actinides)", color=INK2, fontsize=9.5)
    for ax in (ax_b, ax_d):
        ax.set_xlabel("period", color=INK2, fontsize=9.5)

    handles = [Patch(facecolor=c, label=l) for l, _, c in series_g]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.997),
               frameon=False, fontsize=10, ncol=3, labelcolor=INK2,
               handlelength=1.1, handleheight=1.1, columnspacing=1.6)
    fig.text(0.055, 0.962, "Retrieved OMAT24 neighbours are light main-group matter",
             fontsize=15.5, color=INK, ha="left", va="center")
    fig.text(0.055, 0.920,
             f"{head}: {p['n_queries']:,} queries -> {p['n_unique_neighbours']:,} unique OMAT24 rows "
             f"(pc25 top-5, multiplicity-weighted). 1x = composition of the searched pool.",
             fontsize=9.5, color=INK2, ha="left", va="center")
    fig.text(0.055, 0.893,
             "Period falls off monotonically (5.0x at row 2 -> 0.08x at row 6). The group profile in c is "
             "largely that same effect: d- and f-block elements are the heavy ones.",
             fontsize=9.5, color=INK2, ha="left", va="center")
    fig.savefig(out / "neighbor_group_row_mof_off.png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return {"group_spread": spread(gl, gb), "period_spread": spread(pl, pb)}


def figure_all_datasets(pops, bg, out, dpi):
    names = list(pops)
    fig = plt.figure(figsize=(13.2, 9.4), facecolor=SURFACE)
    gcats_ref = glabels(compute_axis(pops[names[0]], bg, GROUP, GROUPS)[0])
    gs = fig.add_gridspec(len(names), 2, width_ratios=[len(gcats_ref), 8.5],
                          hspace=0.55, wspace=0.17,
                          left=0.055, right=0.985, top=0.875, bottom=0.065)
    rows_csv, spreads = [], {}
    for r, name in enumerate(names):
        pp = pops[name]
        gv, gq, gn, gb, gl = compute_axis(pp, bg, GROUP, GROUPS)
        pv, pq, pn, pb, pl = compute_axis(pp, bg, PERIOD, PERIODS)
        draw_enrichment(fig.add_subplot(gs[r, 0]), glabels(gv), gl, gb > 0,
                        f"{'ace'[r]}  {name} - by group", LIFT_LABEL, n_label=2)
        draw_enrichment(fig.add_subplot(gs[r, 1]), [str(v) for v in pv], pl, pb > 0,
                        f"{'bdf'[r]}  {name} - by period", LIFT_LABEL, n_label=2)
        spreads[name] = {"group": round(spread(gl, gb), 3), "period": round(spread(pl, pb), 3)}
        for axis, vals, q_, n_, b_ in [("group", gv, gq, gn, gb), ("period", pv, pq, pn, pb)]:
            for v, qq, nn, bb in zip(vals, q_, n_, b_):
                rows_csv.append({
                    "dataset": name, "axis": axis,
                    "bucket": "f" if (axis == "group" and v == 0) else v,
                    "query_atom_frac": round(float(qq), 6),
                    "neighbour_atom_frac": round(float(nn), 6),
                    "omat_pool_atom_frac": round(float(bb), 6),
                    "log2_lift": (round(float(np.log2(nn / bb)), 4) if nn > 0 and bb > 0 else ""),
                })
    fig.text(0.055, 0.962, "The same row signature appears for every query set",
             fontsize=15.5, color=INK, ha="left", va="center")
    fig.text(0.055, 0.921,
             "log$_2$ enrichment of pc25 top-5 OMAT24 neighbours over the searched pool. Light rows are "
             "retrieved far above pool rate and heavy rows far below, for MOF-off, MAD and MP-ALOE alike.",
             fontsize=9.5, color=INK2, ha="left", va="center")
    fig.savefig(out / "neighbor_group_row_all_datasets.png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return rows_csv, spreads


def figure_within_period(pops, bg, out, dpi, head="MOF-off R2SCAN"):
    """Hold the row fixed: does any group structure survive, and is it consistent?"""
    p = pops[head]
    n, b = p["neigh"], bg
    periods = [2, 3, 4, 5, 6]
    cats = [g for g in GROUPS
            if any((b[(PERIOD == per) & (GROUP == g)].sum() > 0) for per in periods)]
    # one shared, data-driven cap so panels are comparable and nothing is truncated
    per_lift, detail = {}, {}
    for per in periods:
        m = PERIOD == per
        nt, bt = n[m].sum(), b[m].sum()
        lift, has_pool = [], []
        for g in cats:
            sel = m & (GROUP == g)
            nn, bb = n[sel].sum(), b[sel].sum()
            has_pool.append(bb > 0)
            lift.append(np.log2((nn / nt) / (bb / bt)) if (nn > 0 and bb > 0) else np.nan)
        per_lift[per] = (np.array(lift, float), np.array(has_pool, bool))
        detail[f"period_{per}"] = {
            GLABEL.get(g, str(g)): (round(float(v), 3) if np.isfinite(v) else None)
            for g, v in zip(cats, lift)}
    cap = cap_for(np.concatenate([v for v, _ in per_lift.values()]))

    fig = plt.figure(figsize=(12.0, 11.6), facecolor=SURFACE)
    gs = fig.add_gridspec(len(periods), 1, hspace=0.60,
                          left=0.085, right=0.985, top=0.855, bottom=0.05)
    for r, per in enumerate(periods):
        lift, has_pool = per_lift[per]
        draw_enrichment(fig.add_subplot(gs[r, 0]), glabels(cats), lift, has_pool,
                        f"{'abcde'[r]}  Period {per}", "log$_2$ lift (within row)",
                        n_label=2, cap=cap)
    fig.text(0.085, 0.968, "Holding the row fixed, the group preference reverses",
             fontsize=15.5, color=INK, ha="left", va="center")
    fig.text(0.085, 0.936,
             "Group enrichment against each period's OWN retrieval rate, so the row effect is divided out.",
             fontsize=9.5, color=INK2, ha="left", va="center")
    fig.text(0.085, 0.913,
             "Structure survives - but in period 4 the d-block is favoured 1.3x and main-group "
             "disfavoured 0.8x, while in period 5 that flips (d-block 0.7x, main-group 1.4x).",
             fontsize=9.5, color=INK2, ha="left", va="center")
    fig.text(0.085, 0.890,
             "A consistent rule would hold the same sign down a column. This one does not, so group is "
             "not a stable predictor of what gets retrieved; row is.",
             fontsize=9.5, color=INK2, ha="left", va="center")
    fig.savefig(out / "neighbor_group_row_within_period.png", dpi=dpi, facecolor=SURFACE)
    plt.close(fig)

    block = {}
    for per in periods:
        m = PERIOD == per
        nt, bt = n[m].sum(), b[m].sum()
        row = {}
        for lbl, sel in [("main_group", m & ((GROUP <= 2) | (GROUP >= 13)) & (GROUP != 0)),
                         ("d_block", m & (GROUP >= 3) & (GROUP <= 12)),
                         ("f_block", m & (GROUP == 0))]:
            nn, bb = n[sel].sum(), b[sel].sum()
            row[lbl] = round(float((nn / nt) / (bb / bt)), 3) if bb > 0 and nt > 0 else None
        block[f"period_{per}"] = row
    return detail, block


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, required=True,
                    help="path to the repository's runs/ directory")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <runs-root>/omat_knn_probe/neighbor_group_row")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    out = args.out_dir or args.runs_root / "omat_knn_probe" / "neighbor_group_row"
    out.mkdir(parents=True, exist_ok=True)

    pops, bg, bg_nvt = load_populations(args.runs_root)
    head_spreads = figure_headline(pops, bg, out, args.dpi)
    rows_csv, spreads = figure_all_datasets(pops, bg, out, args.dpi)
    within, block = figure_within_period(pops, bg, out, args.dpi)

    with open(out / "neighbor_group_row.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
        wtr.writeheader()
        wtr.writerows(rows_csv)

    # control: does the pattern survive a subdataset-matched background?
    ctrl = {}
    for name in pops:
        gv, _, gn, gb, gl = compute_axis(pops[name], bg_nvt, GROUP, GROUPS)
        pv, _, pn, pb, pl = compute_axis(pops[name], bg_nvt, PERIOD, PERIODS)
        ctrl[name] = {
            "group": {GLABEL.get(v, str(v)): (round(float(x), 3) if np.isfinite(x) else None)
                      for v, x in zip(gv, gl)},
            "period": {str(v): (round(float(x), 3) if np.isfinite(x) else None)
                       for v, x in zip(pv, pl)},
        }

    meta = {
        "geometry": "pc25 (standardized 25-PC UMA), top-5 neighbours, multiplicity-weighted",
        "background": "omat_bg_global_zcounts.npy (50k random OMAT24 sample)",
        "background_control": "omat_bg_nvt3000_zcounts.npy (50k from aimd-from-PBE-3000-nvt)",
        "f_block_note": "Z 57-71 and 89-103 bucketed as 'f', never folded into group 3",
        "category_filter": f"bucket dropped only if below {MIN_FRAC} in EVERY population (symmetric)",
        "datasets": {k: {"n_queries": v["n_queries"],
                         "n_unique_neighbours": v["n_unique_neighbours"]}
                     for k, v in pops.items()},
        "structure_spread_pool_weighted_std_log2_lift": {
            "mof_off_headline": {k: round(v, 3) for k, v in head_spreads.items()},
            "per_dataset": spreads,
            "note": "group and period spreads are comparable in size; the group value is "
                    "mostly the row effect leaking through (d/f-block == heavy elements)",
        },
        "within_period_group_lift": within,
        "within_period_block_lift": block,
        "nvt3000_matched_control": ctrl,
    }
    (out / "neighbor_group_row_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote 3 figures + neighbor_group_row.csv + metadata to {out}")


if __name__ == "__main__":
    main()
