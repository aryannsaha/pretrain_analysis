#!/usr/bin/env python
"""Periodic-table heatmaps of top-K OMAT24 neighbor chemistry, across query sets.

Companion to ``plot_mof_omat_ptable_heatmaps.py``, which renders the MOF-off
rank-depth series for one query set.  This one renders several query sets side
by side, from the tables written by ``multiset_top5_presence.py`` and
``query_set_composition.py``.  Three kinds of population, distinguished by name:

  <dataset>_<geometry>_top<K>   the K nearest OMAT24 neighbors of each query
  <dataset>_dataset             the query dataset itself, no retrieval involved
  omat_bg_global                OMAT24 itself, 50k uniform random rows

and three figure families:

  presence    share of structures containing each element, log scale.  Drawn for
              every population, so "what this dataset is" and "what it retrieves"
              are directly comparable.
  difference  neighbor presence minus the OMAT24 background, in percentage
              points, diverging with the neutral colour pinned to zero.  Only
              meaningful for retrieved sets, so only those get one.

All scales are shared across every panel drawn in one run, so a colour means the
same thing everywhere.  Nothing is hardcoded to one data range, unlike the
single-query-set plotter: the sets span visibly different ranges (MOF-off
neighbors top out near +38 pp, MP ALOE near +57 pp), so reusing MOF-off's
hand-tuned colourbar would misreport the others.

Pass every table you want on one scale in a single call:

    python scripts/omat/plot_multiset_top5_ptable.py \
        --presence .../element_presence_pc25_top5.csv \
                   .../element_presence_datasets.csv

pymatviz is not part of the project conda environment.  Install it somewhere
private and point PYTHONPATH at it rather than mutating the shared env, which
has training jobs running against it:

    PY=/scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env/bin/python
    "$PY" -m pip install --target=$HOME/pylibs pymatviz "kaleido==0.2.1"

kaleido must be 0.2.1: 1.x shells out to Google Chrome for static export, which
is not installed on the cluster.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mplconfig")

COLORSCALE = "Blues"
BASELINE = "omat_bg_global"
TOP_K = re.compile(r"^(?P<dataset>.+)_(?P<geometry>raw128|pc25)_top(?P<depth>\d+)$")

# Fallbacks; a population's own metadata label wins when one is available.
LABELS = {
    "mof_off": "MOF-off R2SCAN-D4",
    "mad": "MAD",
    "mpaloe": "MP ALOE",
    BASELINE: "OMAT24 (uniform random)",
}
# Reading order for the panels.
ORDER = ["mof_off", "mad", "mad_train", "mpaloe", "mpaloe_all"]


def read_tables(paths):
    """Merge several presence CSVs into population -> {symbol: percent}, plus meta."""
    table, meta = {}, {}
    for path in paths:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise SystemExit(f"empty presence table: {path}")
        for column in rows[0]:
            if not column.startswith("presence_"):
                continue
            population = column[len("presence_"):]
            if population in table and population != BASELINE:
                raise SystemExit(f"population {population!r} appears in two tables")
            values = {}
            for row in rows:
                percent = float(row[column]) * 100.0
                if percent > 0:
                    values[row["symbol"]] = percent
            table[population] = values

        meta_path = path.parent / path.name.replace(
            "element_presence_", "presence_metadata_"
        ).replace(".csv", ".json")
        if meta_path.exists():
            meta.update(json.load(open(meta_path)).get("populations", {}))
    return table, meta


def classify(population):
    """('neighbors'|'dataset'|'baseline', dataset key, depth or None)."""
    if population == BASELINE:
        return "baseline", BASELINE, None
    match = TOP_K.match(population)
    if match:
        return "neighbors", match["dataset"], int(match["depth"])
    if population.endswith("_dataset"):
        return "dataset", population[: -len("_dataset")], None
    raise SystemExit(f"cannot classify population {population!r}")


def label_for(population, meta):
    """Prefer the label the producing script recorded; fall back to the key."""
    recorded = meta.get(population, {}).get("label")
    if recorded:
        return recorded
    _, dataset, _ = classify(population)
    return LABELS.get(dataset, dataset)


def sort_key(population):
    kind, dataset, depth = classify(population)
    rank = ORDER.index(dataset) if dataset in ORDER else len(ORDER)
    return (rank, kind != "dataset", depth or 0)


def decade_ticks(low, high):
    ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    return [t for t in ticks if low / 2 <= t <= high * 2]


def linear_ticks(high):
    """Round milestones from zero up to `high`, for the linear presence scale."""
    step = next(s for s in (0.1, 0.25, 0.5, 1, 2, 5, 10, 20) if high / s <= 8)
    ticks, value = [0.0], step
    while value <= high:
        ticks.append(round(value, 4))
        value += step
    return ticks


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


def percent(value):
    """Two decimals, but never collapse a non-zero value to 0.00."""
    if value != value:
        return "-"
    if value >= 0.01:
        return f"{value:.2f}"
    return f"{value:.3f}".rstrip("0") or "0"


def presence_figure(pmv, values, headline, subtitle, low, high, scale, log=True):
    """Log by default; linear when the plotted values span little more than a decade.

    Log is the right default for the neighbor panels, whose presence rates run
    from ~1e-4% to ~60%.  A single subdataset is the opposite case: its rates sit
    in one narrow band, so a log scale spends almost the whole colourbar on a few
    rare-earth outliers and renders the entire body of the table the same blue.
    """
    ticks = decade_ticks(low, high) if log else linear_ticks(high)
    span = (
        f"log colour scale ({low:.4g}% to {high:.3g}%)" if log
        else f"linear colour scale (0% to {high:.3g}%)"
    )
    fig = pmv.ptable_heatmap(
        values,
        colorscale=COLORSCALE,
        log=log,
        cscale_range=(low, high) if log else (0.0, high),
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
    fig.update_layout(
        title=dict(
            text=(f"<b>{headline}</b><br>"
                  f"<sup>{subtitle} &#183; {len(values)} of 118 elements present "
                  f"&#183; blank = absent from this set<br>"
                  f"shared {span} across every panel in this set</sup>"),
            x=0.42, y=0.95, font=dict(size=17),
        ),
        margin=dict(t=130),
    )
    return fig


def difference_figure(pmv, values, headline, low, high, scale):
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
            text=(f"<b>{headline}</b><br>"
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


def subtitle_for(population, kind, depth, meta):
    entry = meta.get(population, {})
    if kind == "dataset":
        n = entry.get("n_structures")
        count = f"{n:,} structures &#183; " if n else ""
        return f"the dataset itself, no retrieval &#183; {count}" \
               f"percentage of structures containing each element"
    if kind == "baseline":
        n = entry.get("distinct_rows")
        count = f"{n:,} rows &#183; " if n else ""
        return (f"the reference every difference panel is measured against "
                f"&#183; {count}percentage of structures containing each element")
    n = entry.get("n_queries")
    count = f"{n:,} queries &#183; " if n else ""
    return (f"top {depth} nearest OMAT24 neighbors &#183; {count}"
            f"percentage of structures containing each element")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--presence", type=Path, nargs="+", required=True,
                        help="one or more presence CSVs; all share one scale")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=1.4)
    parser.add_argument(
        "--linear", action="store_true",
        help="linear presence colour scale from zero instead of log; use when the "
             "plotted populations span roughly a decade or less, as a single OMAT24 "
             "subdataset does (a log scale renders those panels almost flat)",
    )
    args = parser.parse_args()

    paths = [p.resolve() for p in args.presence]
    out = (args.out or paths[0].parent / "figures").resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        import pymatviz as pmv
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"{error}. pymatviz is not installed in this interpreter; see the module "
            f"docstring for the PYTHONPATH-based install that leaves the shared conda "
            f"environment untouched."
        ) from error

    table, meta = read_tables(paths)
    if BASELINE not in table:
        raise SystemExit(f"no {BASELINE} column in {[str(p) for p in paths]}")

    populations = sorted(table, key=sort_key)

    # One shared log range across every presence panel, baseline included.
    everything = [v for values in table.values() for v in values.values()]
    low, high = min(everything), max(everything)
    kind_note = "linear from 0" if args.linear else "log"
    print(
        f"shared presence range: {low:.4g}% to {high:.3g}% ({kind_note})", flush=True
    )

    # One shared diverging range across every difference panel, so the query
    # sets can be compared to each other and not only to OMAT24.
    differences = {}
    for population in populations:
        if classify(population)[0] != "neighbors":
            continue
        values = {}
        for symbol in set(table[population]) | set(table[BASELINE]):
            values[symbol] = (table[population].get(symbol, 0.0)
                              - table[BASELINE].get(symbol, 0.0))
        differences[population] = values
    if differences:
        pooled = [v for values in differences.values() for v in values.values()]
        dlow, dhigh = min(pooled), max(pooled)
        print(f"shared difference range: {dlow:+.2f} to {dhigh:+.2f} pp", flush=True)

    for population in populations:
        kind, _, depth = classify(population)
        label = label_for(population, meta)
        headline = {
            "dataset": f"{label}: dataset composition",
            "baseline": label,
            "neighbors": f"{label}: top {depth} nearest OMAT24 neighbors",
        }[kind]

        fig = presence_figure(pmv, table[population], headline,
                              subtitle_for(population, kind, depth, meta),
                              low, high, args.scale, log=not args.linear)
        path = out / f"ptable_presence_{population}.png"
        fig.write_image(str(path), scale=2)
        print(f"wrote {path}  ({len(table[population])} elements present)", flush=True)

        if kind != "neighbors":
            continue
        span = differences[population]
        fig = difference_figure(
            pmv, span, f"{headline} minus OMAT24 overall", dlow, dhigh, args.scale
        )
        path = out / f"ptable_difference_{population}_minus_{BASELINE}.png"
        fig.write_image(str(path), scale=2)
        print(f"wrote {path}  (this set {min(span.values()):+.2f} to "
              f"{max(span.values()):+.2f} pp)", flush=True)

    print(f"periodic-table heatmaps written to {out}", flush=True)


if __name__ == "__main__":
    main()
