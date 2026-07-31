#!/usr/bin/env python
"""Periodic-table heatmaps of element presence, one per population.

Renders the same numbers as ``element_presence_by_rank_depth_raw128.png`` onto
the periodic table via ``pymatviz.ptable_heatmap``, one figure per population:

  omat_bg_global    OMAT24, uniform random reference
  mof_off_r2scan    the MOF-off R2SCAN-D4 train queries
  raw128_top10      all ten retrieved neighbors per query
  raw128_top5       the five nearest neighbors per query
  raw128_top2       the two nearest neighbors per query

The cell value is the percentage of structures in that population containing the
element.  All five panels share one colour scale so they are directly
comparable; the scale is logarithmic because presence spans roughly four
decades (Xe at 0.004% to C at 99.7%), which a linear ramp would flatten into
"dark for the top four elements, indistinguishable for everything else".

Elements with zero presence in a given population are left uncoloured, so a
blank cell reads as "absent from this set" rather than as a very small value.

pymatviz is not part of the project conda environment.  Install it somewhere
private and point PYTHONPATH at it rather than mutating the shared env, which
has training jobs running against it:

    PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
    "$PY" -m pip install --target=$HOME/pylibs pymatviz "kaleido==0.2.1"
    PYTHONPATH=$HOME/pylibs "$PY" scripts/omat/plot_mof_omat_ptable_heatmaps.py --analysis ...

kaleido must be 0.2.1: 1.x shells out to Google Chrome for static export, which
is not installed on the cluster.
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mplconfig")

POPULATIONS = [
    ("omat_bg_global", "OMAT24 (uniform random)"),
    ("mof_off_r2scan", "MOF-off R2SCAN queries"),
    ("raw128_top10", "all 10 nearest OMAT24 neighbors"),
    ("raw128_top5", "top 5 nearest OMAT24 neighbors"),
    ("raw128_top2", "top 2 nearest OMAT24 neighbors"),
]
COLORSCALE = "Blues"


def read_presence(analysis):
    """population -> {element symbol: presence percentage}, zeros dropped."""
    with (analysis / "element_presence_and_abundance.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    table = {}
    for population, _ in POPULATIONS:
        values = {}
        for row in rows:
            percent = float(row[f"presence_{population}"]) * 100.0
            if percent > 0:
                values[row["symbol"]] = percent
        table[population] = values
    return table


def plot_difference(pmv, table, out, scale, target="raw128_top5", baseline="omat_bg_global"):
    """Signed enrichment of one population over another, in percentage points.

    A signed quantity needs a diverging scale with the neutral colour pinned to
    exactly zero, otherwise "no change" reads as a colour.  The two arms are
    scaled independently because the data are lopsided (about -6 to +54 points):
    a symmetric range would render all 64 depleted elements as near-white and
    hide three quarters of the table.  The arms are therefore NOT comparable in
    magnitude to each other - the colourbar ticks are the reference.
    """
    values = {}
    for symbol in set(table[target]) | set(table[baseline]):
        values[symbol] = table[target].get(symbol, 0.0) - table[baseline].get(symbol, 0.0)

    low, high = min(values.values()), max(values.values())
    zero = (0.0 - low) / (high - low)
    # Diverging: documented red pole -> documented neutral grey at zero ->
    # documented blue ramp.  The deep red end is a darkened step of the red pole.
    colorscale = [
        (0.0, "#7d1d1b"),
        (round(zero * 0.55, 6), "#e34948"),
        (round(zero, 6), "#f0efec"),
        (round(zero + (1 - zero) * 0.25, 6), "#9ec5f4"),
        (round(zero + (1 - zero) * 0.55, 6), "#3987e5"),
        (1.0, "#0d366b"),
    ]

    fig = pmv.ptable_heatmap(
        values,
        colorscale=colorscale,
        log=False,
        cscale_range=(low, high),
        # nan reaches here for elements absent from both sets; render a dash.
        fmt=lambda v: "-" if v != v else f"{v:+.1f}",
        show_values=True,
        nan_color="#f2f1ee",
        scale=scale,
        colorbar=dict(
            orientation="h", len=0.34, thickness=13, x=0.45, y=0.90,
            tickvals=[-5, 0, 10, 25, 40, 55],
            ticktext=["-5", "0", "+10", "+25", "+40", "+55"],
            title="percentage-point change vs OMAT24 (zero = grey)",
            tickfont=dict(size=12),
        ),
    )
    fig.update_layout(
        title=dict(
            text=("<b>Top 5 nearest OMAT24 neighbors minus OMAT24 overall</b><br>"
                  "<sup>percentage-point difference in the share of structures containing "
                  "each element &#183; blue = over-selected by the neighbor search "
                  "&#183; red = avoided &#183; grey = no change<br>"
                  "arms are scaled independently (data run -6.4 to +54.4 points), "
                  "so compare against the colourbar, not between arms</sup>"),
            x=0.42, y=0.95, font=dict(size=17),
        ),
        margin=dict(t=130),
    )
    path = out / f"ptable_difference_{target}_minus_{baseline}.png"
    fig.write_image(str(path), scale=2)
    print(f"wrote {path}  (range {low:+.2f} to {high:+.2f} pp)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="directory written by `mof_omat_neighbor_chemistry.py analyze`")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=1.4)
    args = parser.parse_args()

    analysis = args.analysis.resolve()
    out = (args.out or analysis.parent / "figures" / "ptable").resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        import pymatviz as pmv
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"{error}. pymatviz is not installed in this interpreter; see the module "
            f"docstring for the PYTHONPATH-based install that leaves the shared conda "
            f"environment untouched."
        ) from error

    table = read_presence(analysis)
    # One shared colour range across every panel, so a colour means the same
    # thing in all five.  Anchored at the global min/max of the plotted values.
    everything = [v for values in table.values() for v in values.values()]
    low, high = min(everything), max(everything)
    print(f"shared colour range: {low:.4f}% to {high:.2f}% (log scale)", flush=True)

    # Decade ticks across the plotted span; plain "%" labels instead of the
    # default SI-prefixed ones ("4.0m" for 0.004% is unreadable here).
    ticks = [0.001, 0.01, 0.1, 1, 10, 100]
    ticks = [t for t in ticks if low / 2 <= t <= high * 2]

    def percent(value):
        """Two decimals, but never collapse a non-zero value to 0.00."""
        if value >= 0.01:
            return f"{value:.2f}"
        return f"{value:.3f}".rstrip("0") or "0"

    for population, label in POPULATIONS:
        values = table[population]
        fig = pmv.ptable_heatmap(
            values,
            colorscale=COLORSCALE,
            log=True,
            cscale_range=(low, high),
            fmt=percent,
            show_values=True,
            nan_color="#f2f1ee",
            scale=args.scale,
            # pymatviz stringifies a dict passed as `title`, so keep it a plain str.
            colorbar=dict(
                orientation="h", len=0.34, thickness=13, x=0.45, y=0.90,
                tickvals=ticks,
                ticktext=[f"{t:g}%" for t in ticks],
                tickangle=0,
                title="share of structures containing the element",
                tickfont=dict(size=12),
            ),
        )
        fig.update_layout(
            title=dict(
                text=(f"<b>{label}</b><br>"
                      f"<sup>percentage of structures containing each element "
                      f"&#183; {len(values)} of 118 elements present "
                      f"&#183; blank = absent from this set "
                      f"&#183; shared log colour scale across all five panels</sup>"),
                x=0.42, y=0.95, font=dict(size=17),
            ),
            margin=dict(t=120),
        )
        path = out / f"ptable_presence_{population}.png"
        fig.write_image(str(path), scale=2)
        print(f"wrote {path}  ({len(values)} elements present)", flush=True)

    plot_difference(pmv, table, out, args.scale)

    print(f"periodic-table heatmaps written to {out}", flush=True)


if __name__ == "__main__":
    main()
