#!/usr/bin/env python
"""Shared helpers for the per-atom UMA PCA figures (2026-09-10): a symmetric-log
axis warp, full-extent density binning on that warped grid, tick placement and
the density renderer used by plot_uma_atom_pca.py and plot_uma_atom_pca_external.py.

Why a warp: the PC1/PC2 scores of the OMAT24 atom bank span roughly -2200..+950
while 99.96 % of the atoms sit inside |PC| < 50.  A linear full-range plot
collapses the population into a handful of bins and a percentile box (what
fit_descriptor_bank_pca.py writes to background_density.npz) cuts the tails off.
The figures therefore bin and draw in warped coordinates: linear between
-linthresh and +linthresh (mapped to [-1, 1]), then one warped unit per decade
beyond, so every atom is inside the axes and the dense core keeps its shape.
Ticks are placed at warp(value) and labelled with the data value.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SPINE = "#c3c2b7"


class SymLogWarp:
    """Linear inside [-linthresh, linthresh] -> [-linscale, linscale]; one unit per decade beyond.

    linscale = how many decade-widths the linear half-range occupies (2 keeps the dense
    core at roughly 55-60 % of each axis for the OMAT24 atom bank)."""

    kind = "symlog"

    def __init__(self, linthresh: float = 40.0, base: float = 10.0, linscale: float = 2.0):
        if not linthresh > 0 or not linscale > 0:
            raise ValueError("linthresh and linscale must be positive")
        self.linthresh, self.base, self.linscale = float(linthresh), float(base), float(linscale)

    def forward(self, values):
        v = np.asarray(values, dtype=np.float64)
        a = np.abs(v)
        lin = self.linscale * v / self.linthresh
        log = np.sign(v) * (self.linscale + np.log(np.maximum(a, self.linthresh) / self.linthresh) / np.log(self.base))
        return np.where(a > self.linthresh, log, lin)

    def inverse(self, warped):
        t = np.asarray(warped, dtype=np.float64)
        a = np.abs(t)
        return np.where(a > self.linscale, np.sign(t) * self.linthresh * self.base ** (a - self.linscale),
                        t * self.linthresh / self.linscale)

    def describe(self) -> str:
        L = self.linthresh
        return f"axes are linear between -{L:g} and {L:g} and logarithmic beyond (dotted lines mark the switch)"

    def ticks(self, t_lo: float, t_hi: float, nbins: int = 6, compact: bool = False):
        """(tick positions in warped units, labels in data units) for a warped-axis range.
        compact: drop the +-linthresh tick where a decade tick follows it (small panels; the
        dotted switch line still marks linthresh)."""
        lo, hi = float(self.inverse(t_lo)), float(self.inverse(t_hi))
        L = self.linthresh
        values = set()
        lin_lo, lin_hi = max(lo, -L), min(hi, L)
        if lin_lo < lin_hi:
            for v in MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(lin_lo, lin_hi):
                if lin_lo <= v <= lin_hi:
                    values.add(float(v))
        first = int(np.ceil(np.log(L) / np.log(self.base)))
        decades = [self.base ** e for e in range(first, first + 8) if self.base ** e > L]
        for sign in (-1.0, 1.0):
            beyond = [sign * v for v in decades if lo <= sign * v <= hi]
            values.update(beyond)
            if lo <= sign * L <= hi and not (compact and beyond):
                values.add(sign * L)
            elif compact and beyond:
                values.discard(sign * L)
        values = sorted(values)
        labels = [f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:g}" for v in values]
        return self.forward(values), labels

    def switch_positions(self):
        return (-self.linscale, self.linscale)

    def to_json(self) -> dict:
        return {"kind": self.kind, "linthresh": self.linthresh, "base": self.base, "linscale": self.linscale}


class IdentityWarp:
    """Plain linear axes (the percentile-box densities from fit_descriptor_bank_pca.py)."""

    kind = "linear"
    linthresh = float("inf")

    def forward(self, values):
        return np.asarray(values, dtype=np.float64)

    def inverse(self, warped):
        return np.asarray(warped, dtype=np.float64)

    def describe(self) -> str:
        return "linear axes"

    def ticks(self, t_lo: float, t_hi: float, nbins: int = 6, compact: bool = False):
        values = [float(v) for v in MaxNLocator(nbins=nbins).tick_values(t_lo, t_hi) if t_lo <= v <= t_hi]
        return np.asarray(values), [f"{v:g}" for v in values]

    def switch_positions(self):
        return ()

    def to_json(self) -> dict:
        return {"kind": self.kind}


def warp_from_json(meta: dict):
    if meta.get("kind") == "symlog":
        return SymLogWarp(meta["linthresh"], meta.get("base", 10.0), meta.get("linscale", 1.0))
    return IdentityWarp()


def style_warped_axis(ax, warp, t_xlim, t_ylim, mark_switch: bool = True, nbins: int = 6,
                      compact: bool = False) -> None:
    """Limits + ticks in warped units with data-unit labels; dotted lines where the scale switches."""
    for which, (lo, hi) in (("x", t_xlim), ("y", t_ylim)):
        pos, labels = warp.ticks(lo, hi, nbins=nbins, compact=compact)
        if which == "x":
            ax.set_xlim(lo, hi)
            ax.set_xticks(pos)
            ax.set_xticklabels(labels)
        else:
            ax.set_ylim(lo, hi)
            ax.set_yticks(pos)
            ax.set_yticklabels(labels)
        if mark_switch:
            for s in warp.switch_positions():
                if lo < s < hi:
                    (ax.axvline if which == "x" else ax.axhline)(
                        s, color=SPINE, linewidth=0.6, linestyle=(0, (2, 3)), zorder=1.5)


def core_cell(x_edges, y_edges) -> tuple[float, float]:
    """Data-unit size of the narrowest (core) bin on each axis."""
    return float(np.diff(x_edges).min()), float(np.diff(y_edges).min())


def draw_density(ax, hist, t_x_edges, t_y_edges, cmap, vmax=None, vmin=-0.5, x_edges=None, y_edges=None):
    """Shading = log10(1 + atoms per core-sized cell).  Bins are uniform in warped coordinates, so
    beyond the linear range they widen in data units; when the data-unit edges are given the
    counts are scaled by core_cell_area / bin_area (a density that is continuous across the
    linear/log switch).  Every non-empty bin is drawn at least at the one-atom tint so single
    atoms in the tails stay visible; empty bins keep the surface colour."""
    counts = np.asarray(hist, dtype=np.float64)
    if x_edges is not None and y_edges is not None:
        dx, dy = np.diff(np.asarray(x_edges, dtype=np.float64)), np.diff(np.asarray(y_edges, dtype=np.float64))
        counts = counts * np.outer(dx.min() / dx, dy.min() / dy)
    image = np.where(np.asarray(hist) > 0, np.maximum(np.log10(counts + 1.0), np.log10(2.0)), np.nan).T
    cmap = cmap.copy()
    cmap.set_bad(SURFACE)
    return ax.imshow(
        image, origin="lower", extent=(t_x_edges[0], t_x_edges[-1], t_y_edges[0], t_y_edges[-1]),
        cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax, interpolation="antialiased",
    )


def compute_full_density(scores, label_groups, warp, bins: int = 800, chunk: int = 10_000_000, log=print) -> dict:
    """Bin every row of ``scores`` (n, 2) on a uniform grid in warped coordinates that spans the
    full data range.  ``label_groups`` = [(group, per-row int labels, names), ...]; one
    ``hist_<name>`` per name.  Returns the dict that is saved as the density file."""
    n = len(scores)
    started = time.time()
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for s in range(0, n, chunk):
        block = np.asarray(scores[s: s + chunk], dtype=np.float32)
        lo, hi = np.minimum(lo, block.min(0)), np.maximum(hi, block.max(0))
    t_lo, t_hi = warp.forward(lo), warp.forward(hi)
    pad = 0.015 * (t_hi - t_lo)
    t_x_edges = np.linspace(t_lo[0] - pad[0], t_hi[0] + pad[0], bins + 1)
    t_y_edges = np.linspace(t_lo[1] - pad[1], t_hi[1] + pad[1], bins + 1)
    dx, dy = t_x_edges[1] - t_x_edges[0], t_y_edges[1] - t_y_edges[0]
    cells = bins * bins
    hist = np.zeros(cells, dtype=np.int64)
    per_group = {group: np.zeros(len(names) * cells, dtype=np.int64) for group, _, names in label_groups}
    n_core, L = 0, warp.linthresh
    for s in range(0, n, chunk):
        block = np.asarray(scores[s: s + chunk], dtype=np.float32)
        tx, ty = warp.forward(block[:, 0]), warp.forward(block[:, 1])
        ix = np.clip(((tx - t_x_edges[0]) / dx).astype(np.int64), 0, bins - 1)
        iy = np.clip(((ty - t_y_edges[0]) / dy).astype(np.int64), 0, bins - 1)
        flat = ix * bins + iy
        hist += np.bincount(flat, minlength=cells)
        n_core += int(np.count_nonzero((np.abs(block[:, 0]) <= L) & (np.abs(block[:, 1]) <= L)))
        for group, labels, names in label_groups:
            ids = np.asarray(labels[s: s + chunk], dtype=np.int64)
            if ids.min() < 0 or ids.max() >= len(names):
                raise SystemExit(f"{group}: label ids outside 0..{len(names) - 1} in rows {s}..{s + len(ids)}")
            per_group[group] += np.bincount(ids * cells + flat, minlength=len(names) * cells)
        log(f"  density: {min(s + chunk, n):,}/{n:,} rows binned ({time.time() - started:.0f} s)")
    names, groups_meta = [], {}
    density = {
        "hist": hist.reshape(bins, bins).astype(np.int32),
        "t_x_edges": t_x_edges, "t_y_edges": t_y_edges,
        "x_edges": warp.inverse(t_x_edges), "y_edges": warp.inverse(t_y_edges),
        "bins": bins, "n_rows": n, "inside_fraction": hist.sum() / n, "core_fraction": n_core / n,
        "data_min": lo, "data_max": hi, "warp": json.dumps(warp.to_json()),
    }
    for group, _, group_names in label_groups:
        counts = per_group[group].reshape(len(group_names), bins, bins)
        groups_meta[group] = list(group_names)
        for index, name in enumerate(group_names):
            if f"hist_{name}" in density:
                raise SystemExit(f"duplicate density name {name!r}")
            density[f"hist_{name}"] = counts[index].astype(np.int32)
            names.append(name)
    density["names"] = np.array(names)
    density["label_groups"] = json.dumps(groups_meta)
    log(f"  density: {bins}x{bins} warped bins, PC1 [{lo[0]:.1f}, {hi[0]:.1f}], PC2 [{lo[1]:.1f}, {hi[1]:.1f}], "
        f"{n_core / n:.4%} of atoms inside |PC| <= {L:g}; {time.time() - started:.0f} s")
    return density


def load_density(path: Path) -> dict:
    """Load a density file (full-extent or the percentile-box one) into a dict with a ``warp``."""
    with np.load(path, allow_pickle=False) as npz:
        out = {key: npz[key] for key in npz.files}
    if "t_x_edges" in out:
        out["warp"] = warp_from_json(json.loads(str(out["warp"])))
    else:  # fit_descriptor_bank_pca.py output: linear percentile box
        out["warp"] = IdentityWarp()
        out["t_x_edges"], out["t_y_edges"] = out["x_edges"], out["y_edges"]
        out["core_fraction"] = float("nan")
        out["data_min"] = out["data_max"] = np.full(2, np.nan)
    out["hists"] = {str(name): out[f"hist_{name}"] for name in out["names"]}
    return out


def load_or_compute_density(path: Path, scores, label_groups, warp, bins: int, recompute: bool = False,
                            log=print) -> dict:
    if path.is_file() and not recompute:
        out = load_density(path)
        same = (out["warp"].to_json() == warp.to_json() and int(out["bins"]) == bins
                and int(out["n_rows"]) == len(scores))
        if same:
            log(f"  density: reusing {path}")
            return out
        log(f"  density: {path} has a different warp/bins/row count, recomputing")
    density = compute_full_density(scores, label_groups, warp, bins=bins, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **density)
    log(f"  density: wrote {path}")
    return load_density(path)
