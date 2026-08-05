#!/usr/bin/env python
"""Figures for the LogME transferability study.

Produces:

``logme_vs_scale``
    LogME against pretraining scale, one line per feature set, with the
    across-subsample spread as a band.  If the band swamps the between-model
    differences, the ranking is noise and nothing downstream is meaningful --
    which is the first thing to check, so it is the first figure.
``per_layer``
    Which layer's features score highest, per downstream target.
``target_contrast``
    MOF-OFF (out of distribution) against AM/MPtrj (nearer in distribution).
``predictor_grid``
    All predictors, z-scored within a feature set, against scale -- so
    predictors that merely track ``log N_pretrain`` are visible as such.

Colors are the validated categorical palette, assigned in fixed slot order and
never cycled.  Series are direct-labeled as well as legended, so identity never
rests on color alone.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA_ROOT = Path(
    os.environ.get("PRETRAIN_ANALYSIS_ROOT", "/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
)

LOGGER = logging.getLogger("plot")

# Validated categorical palette, light mode, fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e5e4e0"

SCALE_ORDER = ["omat_100k", "omat_500k", "omat_1m", "omat_2m", "omat_5m", "omat_10m"]
SCALE_LABEL = {
    "omat_100k": "100k",
    "omat_500k": "500k",
    "omat_1m": "1M",
    "omat_2m": "2M",
    "omat_5m": "5M",
    "omat_10m": "10M",
}


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
        }
    )


def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _feature_sets(df):
    """Distinct feature sets, with exact duplicates collapsed.

    MACE's last product block is scalars only, so ``layer1/normed`` is
    bit-identical to ``layer1/inv``.  Plotting both would imply two independent
    measurements where there is one.
    """
    present = list(dict.fromkeys(df["feature_set"]))
    keep = []
    for name in present:
        if name.endswith("/normed"):
            twin = name.replace("/normed", "/inv")
            if twin in present:
                a = df[df.feature_set == name].sort_values("checkpoint")
                b = df[df.feature_set == twin].sort_values("checkpoint")
                col = "energy_logme"
                if col in a and np.allclose(
                    a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float),
                    equal_nan=True,
                ):
                    LOGGER.info("collapsing %s (identical to %s)", name, twin)
                    continue
        keep.append(name)
    return keep


def plot_logme_vs_scale(df, out_dir, pooling="mean"):
    sub = df[df.pooling == pooling]
    targets = sorted(sub["target"].unique())
    fsets = _feature_sets(sub)

    fig, axes = plt.subplots(
        1, len(targets), figsize=(6.2 * len(targets), 4.6), squeeze=False
    )
    for ax, target in zip(axes[0], targets, strict=True):
        tsub = sub[sub.target == target]
        for i, fset in enumerate(fsets):
            g = tsub[tsub.feature_set == fset].set_index("checkpoint").reindex(SCALE_ORDER)
            x = np.arange(len(SCALE_ORDER))
            y = g["energy_logme"].to_numpy(dtype=float)
            sd = g.get("energy_logme_subsample_std")
            color = SERIES[i % len(SERIES)]
            ax.plot(x, y, color=color, marker="o", label=fset, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            if sd is not None:
                sd = sd.to_numpy(dtype=float)
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.16,
                                linewidth=0, zorder=2)
            # Direct label at the right end, so identity is not color-alone.
            if np.isfinite(y[-1]):
                ax.annotate(fset, (x[-1], y[-1]), xytext=(6, 0),
                            textcoords="offset points", color=color,
                            fontsize=8.5, va="center")
        ax.set_xticks(np.arange(len(SCALE_ORDER)))
        ax.set_xticklabels([SCALE_LABEL[s] for s in SCALE_ORDER])
        ax.set_xlabel("OMat24 pretraining structures")
        ax.set_ylabel("LogME (energy)   nats / structure")
        ax.set_title(f"{target}   ({pooling}-pool, per-atom energy)")
        ax.set_xlim(-0.3, len(SCALE_ORDER) - 0.3 + 1.4)
        _despine(ax)

    fig.suptitle(
        "LogME on frozen features vs pretraining scale\n"
        "band = ±1 SD across random subsamples",
        fontsize=13, fontweight="semibold", y=1.02,
    )
    fig.tight_layout()
    path = out_dir / f"logme_vs_scale_{pooling}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def plot_per_layer(df, out_dir, pooling="mean"):
    """Which layer scores highest, per target -- grouped bars."""
    sub = df[df.pooling == pooling]
    fsets = _feature_sets(sub)
    targets = sorted(sub["target"].unique())

    fig, axes = plt.subplots(
        len(targets), 1, figsize=(9.0, 3.6 * len(targets)), squeeze=False
    )
    for ax, target in zip(axes[:, 0], targets, strict=True):
        tsub = sub[sub.target == target]
        width = 0.8 / max(len(fsets), 1)
        x = np.arange(len(SCALE_ORDER))
        for i, fset in enumerate(fsets):
            g = tsub[tsub.feature_set == fset].set_index("checkpoint").reindex(SCALE_ORDER)
            y = g["energy_logme"].to_numpy(dtype=float)
            ax.bar(
                x + i * width - 0.4 + width / 2, y, width * 0.88,
                color=SERIES[i % len(SERIES)], label=fset, zorder=3,
                linewidth=1.2, edgecolor=SURFACE,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([SCALE_LABEL[s] for s in SCALE_ORDER])
        ax.set_ylabel("LogME (energy)")
        ax.set_title(f"{target}")
        ax.legend(ncol=len(fsets), fontsize=8.5, loc="upper left",
                  bbox_to_anchor=(0, 1.02))
        _despine(ax)
    axes[-1, 0].set_xlabel("OMat24 pretraining structures")

    fig.suptitle("LogME by layer and feature variant", fontsize=13,
                 fontweight="semibold", y=1.0)
    fig.tight_layout()
    path = out_dir / f"per_layer_{pooling}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def plot_predictor_grid(df, out_dir, pooling="mean"):
    """Every predictor against scale, z-scored within feature set.

    Predictors whose curve is a straight line are, at n = 6, indistinguishable
    from ``log N_pretrain`` -- which is the baseline LogME has to beat.
    """
    sub = df[(df.pooling == pooling)]
    cols = [
        c
        for c in (
            "energy_logme", "forcemag_logme", "energy_ridge_r2",
            "forcemag_ridge_r2", "energy_h_score", "gmm_target_loglik",
            "mahalanobis_mean", "participation_ratio",
        )
        if c in sub.columns and sub[c].notna().any()
    ]
    if not cols:
        return
    targets = sorted(sub["target"].unique())
    fsets = _feature_sets(sub)

    fig, axes = plt.subplots(
        len(targets), len(fsets),
        figsize=(3.5 * len(fsets), 3.2 * len(targets)),
        squeeze=False, sharex=True,
    )
    for r, target in enumerate(targets):
        for c, fset in enumerate(fsets):
            ax = axes[r][c]
            g = (
                sub[(sub.target == target) & (sub.feature_set == fset)]
                .set_index("checkpoint")
                .reindex(SCALE_ORDER)
            )
            x = np.arange(len(SCALE_ORDER))
            for i, col in enumerate(cols):
                y = g[col].to_numpy(dtype=float)
                if not np.isfinite(y).any() or np.nanstd(y) == 0:
                    continue
                z = (y - np.nanmean(y)) / np.nanstd(y)
                ax.plot(x, z, color=SERIES[i % len(SERIES)], marker="o",
                        markersize=4, linewidth=1.6,
                        label=col if (r == 0 and c == 0) else None)
            ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels([SCALE_LABEL[s] for s in SCALE_ORDER],
                               rotation=45, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{target}\nz-score", fontsize=9)
            if r == 0:
                ax.set_title(fset, fontsize=10)
            _despine(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "All predictors vs pretraining scale (z-scored within panel)",
        fontsize=13, fontweight="semibold", y=1.01,
    )
    fig.tight_layout()
    path = out_dir / f"predictor_grid_{pooling}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def plot_stability(df, out_dir, pooling="mean"):
    """Between-model spread against within-model subsample noise.

    The diagnostic that decides whether any of this is measurable: if a point
    sits below the diagonal, subsample noise exceeds the signal the ranking
    would have to resolve.
    """
    sub = df[df.pooling == pooling]
    if "energy_logme_subsample_std" not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    targets = sorted(sub["target"].unique())
    for i, target in enumerate(targets):
        pts_x, pts_y, labels = [], [], []
        for fset in _feature_sets(sub):
            g = sub[(sub.target == target) & (sub.feature_set == fset)]
            spread = float(np.nanstd(g["energy_logme"].to_numpy(dtype=float)))
            noise = float(np.nanmean(g["energy_logme_subsample_std"].to_numpy(dtype=float)))
            if np.isfinite(spread) and np.isfinite(noise):
                pts_x.append(noise)
                pts_y.append(spread)
                labels.append(fset)
        ax.scatter(pts_x, pts_y, s=80, color=SERIES[i % len(SERIES)],
                   label=target, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        for xx, yy, lab in zip(pts_x, pts_y, labels, strict=True):
            ax.annotate(lab, (xx, yy), xytext=(6, 4), textcoords="offset points",
                        fontsize=7.5, color=INK_2)
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate("signal = noise", (lim[1] * 0.62, lim[1] * 0.66), color=MUTED,
                fontsize=8.5, rotation=38)
    ax.set_xlabel("within-model subsample SD of LogME")
    ax.set_ylabel("between-model SD of LogME (across the 6 scales)")
    ax.set_title("Is the between-checkpoint signal above subsample noise?")
    ax.legend(fontsize=9)
    _despine(ax)
    fig.tight_layout()
    path = out_dir / f"stability_{pooling}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logme-dir", default=str(DATA_ROOT / "outputs/logme"))
    ap.add_argument("--targets", nargs="*", default=["mof_off", "am"])
    ap.add_argument("--out", default=str(DATA_ROOT / "outputs/logme/figures"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _style()

    frames = []
    for t in args.targets:
        p = Path(args.logme_dir) / t / f"logme_scores_{t}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            LOGGER.warning("missing %s", p)
    if not frames:
        LOGGER.error("no score files found")
        return 1
    df = pd.concat(frames, ignore_index=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pooling in sorted(df["pooling"].unique()):
        plot_logme_vs_scale(df, out_dir, pooling)
        plot_per_layer(df, out_dir, pooling)
        plot_predictor_grid(df, out_dir, pooling)
        plot_stability(df, out_dir, pooling)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
