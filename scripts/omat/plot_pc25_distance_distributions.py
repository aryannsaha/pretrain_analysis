#!/usr/bin/env python
"""How far each query set sits from OMAT24 in the pc25 space, in three views.

The distance tables answer this numerically; this is the same content as a
figure, built so the reader can see *where* a dataset sits rather than compare
seven-column rows.  OMAT24 is drawn as the reference in neutral grey and the
three query sets take categorical slots, because the question is "where do these
sit relative to OMAT24", not "compare four peers".

  A  Cumulative distribution of d1.  Where a curve sits left-to-right is how
     close that dataset is; the vertical rules are OMAT24's own p95 and p99, so
     the share of a curve to their right is the share of structures that are
     further from OMAT24 than a typical OMAT24 row is.

  B  The same thing normalised.  Each query's d1 is expressed as a percentile of
     OMAT24's own d1 distribution, so a dataset statistically identical to
     OMAT24 traces the diagonal exactly.  Curves that bow *below* the diagonal
     carry more mass at high percentiles -- further out than OMAT24 is from
     itself.  This removes the units, which is what makes the three query sets
     comparable to each other and not just to the reference.

  C  Share beyond OMAT24's own p95, as the neighbour rank K grows.  d1 asks "is
     there any close match"; d10 asks "is there a whole neighbourhood".  A curve
     that falls steeply is a dataset with no single close twin but plenty of
     near-misses, which is a different kind of novelty from one that stays flat.

Panels A and C use the raw distances; panel B uses the novelty percentiles that
``compute_omat_knn_novelty.py`` already wrote next to them.

    python scripts/omat/plot_pc25_distance_distributions.py \
        --root /scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.environ.get("PRETRAIN_ANALYSIS_ROOT", HERE.parents[1]))
PROBE = "runs/omat_knn_probe"
CALIBRATION = f"{PROBE}/omat_knn_pc25_final/OMAT_calibration"
DEFAULT_OUT = f"{PROBE}/multiset_top5_presence/figures/pc25_distance_distributions.png"

# Reference palette, categorical slots 1-3 unchanged, plus neutral ink for the
# reference series.  Text wears ink, never the series colour; the adjacent line
# carries identity.
SERIES = [
    ("MOF-off R2SCAN-D4", f"{PROBE}/omat_knn_pc25_mof_off/MOF_off_R2SCAN", "#2a78d6"),
    ("MAD", f"{PROBE}/omat_knn_pc25_mad/MAD_all", "#eb6834"),
    ("MP ALOE", f"{PROBE}/omat_knn_pc25_mpaloe/MPALOE_all", "#1baf7a"),
]
REFERENCE = "#767570"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e2e1dc"

POINTS = 2000


def ecdf(values, points=POINTS):
    """Thinned empirical CDF: x ascending, y the share at or below x, in %."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = ordered.size
    take = np.unique(np.linspace(0, n - 1, min(points, n)).astype(np.int64))
    return ordered[take], 100.0 * (take + 1) / n


def x_at(xs, ys, target):
    """The x where a monotone ECDF first reaches `target` percent."""
    index = int(np.searchsorted(ys, target))
    return xs[min(max(index, 0), xs.size - 1)]


# Each series is labelled at its own height so the labels cannot collide,
# whatever the curves do.
LABEL_HEIGHTS = [35.0, 62.0, 88.0]


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=3)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or root / DEFAULT_OUT).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    calibration = np.load(root / CALIBRATION / "distances.npy")
    omat_d1 = np.asarray(calibration[:, 0], dtype=np.float64)
    # OMAT24's own p95 at every neighbour rank, for panel C.
    omat_p95_by_k = np.array(
        [np.quantile(calibration[:, k], 0.95) for k in range(calibration.shape[1])]
    )
    p95, p99 = float(np.quantile(omat_d1, 0.95)), float(np.quantile(omat_d1, 0.99))

    loaded = []
    for label, rel, color in SERIES:
        distances = np.load(root / rel / "distances.npy", mmap_mode="r")
        percentile = np.load(root / rel / "omat_novelty_percentile_d1.npy", mmap_mode="r")
        summary = json.load(open(root / rel / "summary.json"))
        loaded.append({
            "label": label, "color": color,
            "d1": np.asarray(distances[:, 0], dtype=np.float64),
            "distances": distances,
            "percentile": np.asarray(percentile, dtype=np.float64),
            "n": int(summary["rows"]),
        })

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.6))
    fig.patch.set_facecolor("white")

    # ---- A: cumulative distribution of d1 -----------------------------------
    ax = axes[0]
    style(ax)
    x, y = ecdf(omat_d1)
    ax.plot(x, y, color=REFERENCE, linewidth=2.0, linestyle=(0, (5, 2)),
            zorder=3, label="OMAT24 (leave-one-out)")
    for entry, height in zip(loaded, LABEL_HEIGHTS):
        x, y = ecdf(entry["d1"])
        ax.plot(x, y, color=entry["color"], linewidth=2.0, zorder=4,
                label=entry["label"])
        ax.text(x_at(x, y, height) * 1.18, height, entry["label"],
                va="center", ha="left", zorder=5)
    for value, name, height in ((p95, "OMAT24 p95", 14), (p99, "p99", 46)):
        ax.axvline(value, color=INK_SOFT, linewidth=1.0, linestyle=":", zorder=2)
        ax.text(value * 0.93, height, name, rotation=90, va="center", ha="center")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("distance (via $d_1$, dimensionless, standardised embedding units)\n"
                  "further from OMAT24 →")
    ax.set_ylabel("share of structures at or below (%)")
    ax.set_title("Percentage of dataset with 1 neighbor within distance", loc="left", pad=10)

    # ---- B: novelty percentile against the uniform diagonal -----------------
    ax = axes[1]
    style(ax)
    ax.plot([0, 100], [0, 100], color=REFERENCE, linewidth=2.0,
            linestyle=(0, (5, 2)), zorder=3)
    # Kept low-left, where no query-set curve reaches, so it cannot collide.
    ax.text(26, 30, "OMAT24 = diagonal", rotation=38, ha="center", va="center")
    for entry, height in zip(loaded, LABEL_HEIGHTS):
        x, y = ecdf(entry["percentile"])
        ax.plot(x, y, color=entry["color"], linewidth=2.0, zorder=4)
        # Labelled at a fixed height per series, left of the curve, so the
        # labels stay separated no matter how tightly the curves bunch.
        ax.text(x_at(x, y, height) - 2.5, height, entry["label"],
                va="center", ha="right", zorder=5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("$d_1$ as a percentile of OMAT24's own $d_1$\n"
                  "below the diagonal = further out than OMAT24 is from itself")
    ax.set_ylabel("share of structures at or below (%)")
    ax.set_title("Normalised", loc="left", pad=10)

    # ---- C: share beyond OMAT24 p95, by neighbour rank ----------------------
    ax = axes[2]
    style(ax)
    ks = np.arange(1, 11)
    ax.axhline(5.0, color=REFERENCE, linewidth=2.0, linestyle=(0, (5, 2)), zorder=3)
    ax.text(1.0, 7.2, "OMAT24 = 5% (by construction)", va="bottom", ha="left")
    ends = []
    for entry in loaded:
        shares = [
            100.0 * float(np.mean(np.asarray(entry["distances"][:, k]) > omat_p95_by_k[k]))
            for k in range(10)
        ]
        ax.plot(ks, shares, color=entry["color"], linewidth=2.0, marker="o",
                markersize=5, zorder=4)
        ends.append([shares[-1], entry["label"]])
    # Labels sit at the right end; nudge them apart where two curves land close
    # together (MAD and MP ALOE finish about one point apart).
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 2.6)
    for height, label in ends:
        ax.text(10.4, height, label, va="center", ha="left", zorder=5)
    ax.set_xticks(ks)
    ax.set_xlim(0.4, 15.8)
    ax.set_ylim(0, 45)
    ax.set_xlabel("neighbour rank $K$\n")
    ax.set_ylabel("structures beyond OMAT24's own p95 (%)")
    # Wrapped by hand: one line of this does not fit the panel width.  "95% of
    # OMAT24 rows fall below" rather than "is d_K for 95% of rows" -- the
    # threshold is a percentile, not a value those rows share.
    ax.set_title("% of structures beyond $d_K$-at-p95, where $d_K$-at-p95 is\n"
                 "the $d_K$ that 95% of OMAT24 rows fall below",
                 loc="left", pad=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.115))

    counts = " · ".join(f"{e['label']} {e['n']:,}" for e in loaded)
    fig.suptitle(
        "How far each dataset sits from OMAT24 in the 25-PC UMA space",
        x=0.5, y=0.99,
    )
    fig.text(
        0.5, 0.935,
        f"OMAT24 reference is 1,000,000 leave-one-out rows against all "
        f"100,824,585 · {counts}",
        ha="center",
    )
    # What the axis actually measures, since a distance in a learned embedding
    # space is easy to misread as a physical separation.
    fig.text(
        0.5, 0.015,
        "$d_K$ = Euclidean distance from a structure to its $K$-th nearest OMAT24 "
        "row, in the 25-PC projection of the 128-d UMA embedding.",
        ha="center", va="bottom",
    )

    fig.tight_layout(rect=(0, 0.23, 1, 0.91))
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
