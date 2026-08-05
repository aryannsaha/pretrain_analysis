#!/usr/bin/env python
"""Periodic-table heatmaps of top-K OMAT24 neighbor chemistry, across query sets.

Companion to ``plot_mof_omat_ptable_heatmaps.py``, which renders the MOF-off
rank-depth series for one query set.  This one renders the same quantity for
several query sets side by side, from the table written by
``multiset_top5_presence.py``:

  <dataset>_<geometry>_top<K>   the K nearest OMAT24 neighbors of each query
  omat_bg_global                OMAT24 itself, 50k uniform random rows

Two figure families, both on scales shared across every dataset so that a colour
means the same thing in every panel:

  presence    share of retrieved structures containing each element, log scale.
  difference  that share minus the OMAT24 background, in percentage points,
              on a diverging scale with the neutral colour pinned to zero.

Unlike the single-query-set script, nothing here is hardcoded to one data range:
both scales are derived from the values actually plotted, and the captions quote
the derived numbers.  That matters because the three query sets have visibly
different spans (MOF-off tops out near +38 pp, MP ALOE near +57 pp), and reusing
MOF-off's hand-tuned colourbar would misreport the others.

pymatviz is not part of the project conda environment.  Install it somewhere
private and point PYTHONPATH at it rather than mutating the shared env, which
has training jobs running against it:

    PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
    "$PY" -m pip install --target=$HOME/pylibs pymatviz "kaleido==0.2.1"
    PYTHONPATH=$HOME/pylibs "$PY" scripts/omat/plot_multiset_top5_ptable.py --presence ...

kaleido must be 0.2.1: 1.x shells out to Google Chrome for static export, which
is not installed on the cluster.
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mplconfig")

COLORSCALE = "Blues"
BASELINE = "omat_bg_global"

# Display names, in the order the panels should be read.
LABELS = {
    "mof_off": "MOF-off R2SCAN-D4",
    "mad": "MAD",
    "mpaloe": "MP ALOE",
    BASELINE: "OMAT24 (uniform random)",
}
ORDER = ["mof_off", "mad", "mpaloe"]


def read_presence(path):
    """population -> {element symbol: presence percentage}, zeros dropped."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty presence table: {path}")
    populations = [c[len("presence_"):] for c in rows[0] if c.startswith("presence_")]
    table = {}
    for population in populations:
        values = {}
        for row in rows:
            percent = float(row[f"presence_{population}"]) * 100.0
            if percent > 0:
                values[row["symbol"]] = percent
        table[population] = values
    return table


def split_name(population):
    """`mad_pc25_top5` -> ('mad', 'pc25', 5); the baseline has no such parts."""
    if population == BASELINE:
        return BASELINE, None, None
    dataset, geometry, depth = population.rsplit("_", 2)
    return dataset, geometry, int(depth.removeprefix("top"))


def decade_ticks(low, high):
    ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    return [t for t in ticks if low / 2 <= t <= high * 2]


def diverging_ticks(low, high):
    """Zero, one negative anchor, and evenly spaced positive milestones."""
    step = 5 if high <= 30 else 10 if high <= 70 else 20
    ticks = []
    if low < -1:
        ticks.append(-int(abs(low)))
    ticks.append(0)
    value = step
    while value <= high:
        ticks.append(value)
        value += step
    return ticks


def diverging_colorscale(zero):
    """Red pole -> neutral grey pinned at `zero` -> blue ramp.

    The two arms are scaled independently because the data are lopsided: a
    symmetric range would render the depleted majority as near-white and hide
    most of the table.  The arms are therefore NOT comparable in magnitude to
    each other -- the colourbar ticks are the reference.
    """
    return [
        (0.0, "#7d1d1b"),
        (round(zero * 0.55, 6), "#e34948"),
        (round(zero, 6), "#f0efec"),
        (round(zero + (1 - zero) * 0.25, 6), "#9ec5f4"),
        (round(zero + (1 - zero) * 0.55, 6), "#3987e5"),
        (1.0, "#0d366b"),
    ]


def presence_figure(pmv, values, label, depth, low, high, scale, n_queries):
    ticks = decade_ticks(low, high)

    def percent(value):
        """Two decimals, but never collapse a non-zero value to 0.00."""
        if value != value:
            return "-"
        if value >= 0.01:
            return f"{value:.2f}"
        return f"{value:.3f}".rstrip("0") or "0"

    fig = pmv.ptable_heatmap(
        values,
        colorscale=COLORSCALE,
        log=True,
        cscale_range=(low, high),
        fmt=percent,
        show_values=True,
        nan_color="#f2f1ee",
        scale=scale,
        colorbar=dict(
            orientation="h", len=0.34, thickness=13, x=0.45, y=0.90,
            tickvals=ticks,
            ticktext=[f"{t:g}%" for t in ticks],
            tickangle=0,
            title="share of structures containing the element",
            tickfont=dict(size=12),
        ),
    )
    queries = f" &#183; {n_queries:,} queries" if n_queries else ""
    fig.update_layout(
        title=dict(
            text=(f"<b>{label}: top {depth} nearest OMAT24 neighbors</b><br>"
                  f"<sup>percentage of structures containing each element"
                  f"{queries} &#183; {len(values)} of 118 elements present "
                  f"&#183; blank = absent from this set<br>"
                  f"shared log colour scale ({low:.4g}% to {high:.3g}%) "
                  f"across every panel in this set</sup>"),
            x=0.42, y=0.95, font=dict(size=17),
        ),
        margin=dict(t=130),
    )
    return fig


def baseline_figure(pmv, values, low, high, scale):
    ticks = decade_ticks(low, high)

    def percent(value):
        if value != value:
            return "-"
        if value >= 0.01:
            return f"{value:.2f}"
        return f"{value:.3f}".rstrip("0") or "0"

    fig = pmv.ptable_heatmap(
        values, colorscale=COLORSCALE, log=True, cscale_range=(low, high),
        fmt=percent, show_values=True, nan_color="#f2f1ee", scale=scale,
        colorbar=dict(
            orientation="h", len=0.34, thickness=13, x=0.45, y=0.90,
            tickvals=ticks, ticktext=[f"{t:g}%" for t in ticks], tickangle=0,
            title="share of structures containing the element",
            tickfont=dict(size=12),
        ),
    )
    fig.update_layout(
        title=dict(
            text=(f"<b>{LABELS[BASELINE]}</b><br>"
                  f"<sup>the reference every difference panel is measured against "
                  f"&#183; percentage of structures containing each element "
                  f"&#183; {len(values)} of 118 elements present<br>"
                  f"shared log colour scale ({low:.4g}% to {high:.3g}%) "
                  f"across every panel in this set</sup>"),
            x=0.42, y=0.95, font=dict(size=17),
        ),
        margin=dict(t=130),
    )
    return fig


def difference_figure(pmv, values, label, depth, low, high, scale):
    zero = (0.0 - low) / (high - low)
    ticks = diverging_ticks(low, high)
    fig = pmv.ptable_heatmap(
        values,
        colorscale=diverging_colorscale(zero),
        log=False,
        cscale_range=(low, high),
        # nan reaches here for elements absent from both sets; render a dash.
        fmt=lambda v: "-" if v != v else f"{v:+.1f}",
        show_values=True,
        nan_color="#f2f1ee",
        scale=scale,
        colorbar=dict(
            orientation="h", len=0.34, thickness=13, x=0.45, y=0.90,
            tickvals=ticks,
            ticktext=[f"{t:+g}" if t else "0" for t in ticks],
            title="percentage-point change vs OMAT24 (zero = grey)",
            tickfont=dict(size=12),
        ),
    )
    fig.update_layout(
        title=dict(
            text=(f"<b>{label}: top {depth} nearest OMAT24 neighbors "
                  f"minus OMAT24 overall</b><br>"
                  f"<sup>percentage-point difference in the share of structures "
                  f"containing each element &#183; blue = over-selected by the "
                  f"neighbor search &#183; red = avoided &#183; grey = no change<br>"
                  f"arms are scaled independently (shared range {low:+.1f} to "
                  f"{high:+.1f} points across all query sets), so compare against "
                  f"the colourbar, not between arms</sup>"),
            x=0.42, y=0.95, font=dict(size=17),
        ),
        margin=dict(t=130),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--presence", type=Path, required=True,
                        help="CSV written by multiset_top5_presence.py")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=1.4)
    args = parser.parse_args()

    presence_path = args.presence.resolve()
    out = (args.out or presence_path.parent / "figures").resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        import pymatviz as pmv
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"{error}. pymatviz is not installed in this interpreter; see the module "
            f"docstring for the PYTHONPATH-based install that leaves the shared conda "
            f"environment untouched."
        ) from error

    table = read_presence(presence_path)
    if BASELINE not in table:
        raise SystemExit(f"{presence_path} has no {BASELINE} column")

    meta_path = presence_path.parent / presence_path.name.replace(
        "element_presence_", "presence_metadata_"
    ).replace(".csv", ".json")
    meta = json.load(open(meta_path))["populations"] if meta_path.exists() else {}

    datasets = [p for p in table if p != BASELINE]
    datasets.sort(key=lambda p: ORDER.index(split_name(p)[0])
                  if split_name(p)[0] in ORDER else len(ORDER))

    # One shared log range across every presence panel, baseline included.
    everything = [v for values in table.values() for v in values.values()]
    low, high = min(everything), max(everything)
    print(f"shared presence range: {low:.4g}% to {high:.3g}% (log)", flush=True)

    # One shared diverging range across every difference panel, so the three
    # query sets can be compared to each other and not just to OMAT24.
    differences = {}
    for population in datasets:
        values = {}
        for symbol in set(table[population]) | set(table[BASELINE]):
            values[symbol] = (table[population].get(symbol, 0.0)
                              - table[BASELINE].get(symbol, 0.0))
        differences[population] = values
    pooled = [v for values in differences.values() for v in values.values()]
    dlow, dhigh = min(pooled), max(pooled)
    print(f"shared difference range: {dlow:+.2f} to {dhigh:+.2f} pp", flush=True)

    for population in datasets:
        dataset, _, depth = split_name(population)
        label = LABELS.get(dataset, dataset)
        n_queries = meta.get(population, {}).get("n_queries")

        fig = presence_figure(pmv, table[population], label, depth,
                              low, high, args.scale, n_queries)
        path = out / f"ptable_presence_{population}.png"
        fig.write_image(str(path), scale=2)
        print(f"wrote {path}  ({len(table[population])} elements present)", flush=True)

        fig = difference_figure(pmv, differences[population], label, depth,
                                dlow, dhigh, args.scale)
        path = out / f"ptable_difference_{population}_minus_{BASELINE}.png"
        fig.write_image(str(path), scale=2)
        span = differences[population]
        print(f"wrote {path}  (this set {min(span.values()):+.2f} to "
              f"{max(span.values()):+.2f} pp)", flush=True)

    fig = baseline_figure(pmv, table[BASELINE], low, high, args.scale)
    path = out / f"ptable_presence_{BASELINE}.png"
    fig.write_image(str(path), scale=2)
    print(f"wrote {path}  ({len(table[BASELINE])} elements present)", flush=True)

    print(f"periodic-table heatmaps written to {out}", flush=True)


if __name__ == "__main__":
    main()
