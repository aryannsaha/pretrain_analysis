#!/usr/bin/env python
"""MathJax-renderable tables of the pc25 kNN distance statistics.

Puts OMAT24 and the three query sets on one table, in the 25-PC standardised
space, using the conventions of ``compute_omat_knn_novelty.py``: dK is the
distance to the K-th closest OMAT24 row, columns ``{d1: 0, d5: 4, d10: 9}``, and
quantiles ``[0, .5, .9, .95, .99, .999, 1]`` of the raw Euclidean distance.

OMAT24's own row is the leave-one-out calibration: 1,000,000 sampled OMAT24 rows
queried against all 100,824,585, with the self-match dropped before ranking.  It
is what the ``> p95`` columns are measured against, so OMAT24's own shares are
5/1/0.1% by construction rather than by measurement.

Two calibration runs exist under ``runs/omat_knn_probe``.  This uses the
``omat_knn_pc25_final`` one (nprobe=1024), which is the run whose quantiles
reproduce the ``omat_calibration_thresholds_euclidean`` block recorded in every
query-set summary exactly; the earlier ``omat_knn_pc25`` run used nprobe=16 and
differs by up to 2e-2.  The script asserts that agreement rather than assuming
it, so pointing it at the wrong calibration fails instead of quietly producing a
table whose two halves disagree.

Output is Markdown with ``$$``-delimited ``array`` environments, which is what
MathJax renders -- ``tabular`` is not part of MathJax.

    python scripts/omat/pc25_distance_table.py --root /path/to/pretrain_analysis
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.environ.get("PRETRAIN_ANALYSIS_ROOT", HERE.parents[1]))
PROBE = "runs/omat_knn_probe"
CALIBRATION = f"{PROBE}/omat_knn_pc25_final/OMAT_calibration"
DEFAULT_OUT = f"{PROBE}/multiset_top5_presence/pc25_distance_statistics.md"

POSITIONS = {"d1": 0, "d5": 4, "d10": 9}
QUANTILES = [0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0]
KEYS = ["q0", "q50", "q90", "q95", "q99", "q99.9", "q100"]

QUERY_SETS = [
    ("MOF-off R2SCAN-D4", f"{PROBE}/omat_knn_pc25_mof_off/MOF_off_R2SCAN"),
    ("MAD", f"{PROBE}/omat_knn_pc25_mad/MAD_all"),
    ("MP ALOE", f"{PROBE}/omat_knn_pc25_mpaloe/MPALOE_all"),
]


def distance_summary(values):
    measured = np.quantile(np.asarray(values, dtype=np.float64), QUANTILES)
    return {f"q{q * 100:g}": float(v) for q, v in zip(QUANTILES, measured)}


def thousands(value):
    return f"{value:,}".replace(",", "{,}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or root / DEFAULT_OUT).resolve()

    calibration = np.load(root / CALIBRATION / "distances.npy")

    rows = [{
        "label": "OMAT24 (leave-one-out)",
        "n": int(calibration.shape[0]),
        "stats": {d: distance_summary(calibration[:, c]) for d, c in POSITIONS.items()},
        "frac": {"d1": {"p95": 0.05, "p99": 0.01, "p99_9": 0.001}},
    }]
    for label, rel in QUERY_SETS:
        summary = json.load(open(root / rel / "summary.json"))
        rows.append({
            "label": label,
            "n": int(summary["rows"]),
            "stats": {d: summary["distances"][d]["euclidean"] for d in POSITIONS},
            "frac": {
                d: {
                    "p95": summary["distances"][d]["fraction_above_omat_p95"],
                    "p99": summary["distances"][d]["fraction_above_omat_p99"],
                    "p99_9": summary["distances"][d]["fraction_above_omat_p99_9"],
                }
                for d in POSITIONS
            },
        })

    # The recorded thresholds must be reproducible from the calibration array,
    # otherwise the two halves of the table describe different reference runs.
    recorded = json.load(open(root / QUERY_SETS[0][1] / "summary.json"))["distances"]
    thresholds, worst = {}, 0.0
    for d, column in POSITIONS.items():
        thresholds[d] = {
            "p95": float(np.quantile(calibration[:, column], 0.95)),
            "p99": float(np.quantile(calibration[:, column], 0.99)),
            "p99_9": float(np.quantile(calibration[:, column], 0.999)),
        }
        expected = recorded[d]["omat_calibration_thresholds_euclidean"]
        worst = max(worst, max(abs(thresholds[d][k] - expected[k]) for k in expected))
    if worst > 1e-9:
        raise SystemExit(
            f"{CALIBRATION} does not reproduce the thresholds recorded in the "
            f"query-set summaries (max |delta| = {worst:.3e}); this is the wrong "
            f"calibration run"
        )
    print(f"calibration check: max |delta| vs recorded thresholds = {worst:.3e}")

    lines = []
    lines.append("# pc25 nearest-neighbour distance statistics\n")
    lines.append(
        "Euclidean distance in the 25-PC standardised UMA space. $d_K$ is the "
        "distance from a query to its $K$-th closest OMAT24 row. OMAT24's own row "
        "is the leave-one-out calibration (self-match dropped), which is also the "
        "reference the threshold columns are measured against.\n"
    )

    lines.append("## Distance quantiles\n")
    lines.append("$$")
    lines.append(r"\begin{array}{llrrrrrrr}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{query set} & \textbf{d} & \textbf{min} & \textbf{median} & "
        r"\textbf{p90} & \textbf{p95} & \textbf{p99} & \textbf{p99.9} & "
        r"\textbf{max} \\"
    )
    lines.append(r"\hline")
    for entry in rows:
        for i, d in enumerate(POSITIONS):
            head = rf"\text{{{entry['label']}}}" if i == 0 else ""
            cells = " & ".join(f"{entry['stats'][d][k]:.3f}" for k in KEYS)
            lines.append(rf"{head} & d_{{{d[1:]}}} & {cells} \\")
        lines.append(r"\hline")
    lines.append(r"\end{array}")
    lines.append("$$\n")

    lines.append("## Share of queries beyond the OMAT24 thresholds\n")
    lines.append("$$")
    lines.append(r"\begin{array}{llrrr}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{query set} & \textbf{d} & \textbf{> p95} & \textbf{> p99} & "
        r"\textbf{> p99.9} \\"
    )
    lines.append(r"\hline")
    for entry in rows:
        printed = False
        for d in POSITIONS:
            frac = entry["frac"].get(d)
            if frac is None:
                continue
            head = rf"\text{{{entry['label']}}}" if not printed else ""
            printed = True
            cells = " & ".join(
                rf"{100 * frac[k]:.2f}\%" for k in ("p95", "p99", "p99_9")
            )
            lines.append(rf"{head} & d_{{{d[1:]}}} & {cells} \\")
        lines.append(r"\hline")
    lines.append(r"\end{array}")
    lines.append("$$\n")
    lines.append(
        "OMAT24's shares are $5/1/0.1\\%$ by construction: the thresholds are its "
        "own quantiles.\n"
    )

    lines.append("## Thresholds and row counts\n")
    lines.append("$$")
    lines.append(r"\begin{array}{lrrr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{d} & \textbf{p95} & \textbf{p99} & \textbf{p99.9} \\")
    lines.append(r"\hline")
    for d in POSITIONS:
        t = thresholds[d]
        lines.append(
            rf"d_{{{d[1:]}}} & {t['p95']:.3f} & {t['p99']:.3f} & {t['p99_9']:.3f} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{array}")
    lines.append("$$\n")
    lines.append("$$")
    lines.append(r"\begin{array}{lr}")
    lines.append(r"\hline")
    lines.append(r"\textbf{query set} & \textbf{rows} \\")
    lines.append(r"\hline")
    for entry in rows:
        lines.append(rf"\text{{{entry['label']}}} & {thousands(entry['n'])} \\")
    lines.append(r"\hline")
    lines.append(r"\end{array}")
    lines.append("$$")

    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".partial.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
