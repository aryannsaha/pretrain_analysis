#!/usr/bin/env python
"""MathJax-renderable tables of which OMAT24 subdataset the pc25 neighbours come from.

The canonical unit is a **top-5 neighbour slot in the 25-PC standardised space**:
each query contributes its 5 nearest OMAT24 rows, and a slot is attributed to the
subdataset of the row it landed on.  ``summary.json`` already records
``nearest_omat_subdataset_counts``, but only for rank 1; the top-5 attribution
needs the manifest join and is computed here.  The rank-1 column is recomputed
anyway and asserted against every summary, so a manifest/run mismatch fails loudly
instead of producing a plausible wrong table.

THE POINT OF THE OMAT24 ROWS
----------------------------
Every external query set funnels into ``aimd-from-PBE-3000-nvt`` -- 87% for the
mildest (MatPES r2SCAN) up to 100% for MOF-off.  That number means nothing
without a reference for what *should* land there, and the obvious reference --
the subdataset's 7.776% share of the corpus -- is the wrong one, because the
pc25 geometry is not neutral.  Three reference rows are built from the
leave-one-out calibration (1,000,000 OMAT24 rows queried against all
100,824,585, self-match dropped), each answering a different question:

  population share      7.776%   what a subdataset-blind retriever would return.
  OMAT24, all rows      7.774%   what OMAT24 actually returns for itself.  It
                                 matches the population share to three decimals,
                                 but not because the geometry is neutral -- it is
                                 the sum of eleven subdatasets each retrieving
                                 ~99% itself (see the internal matrix).  Every
                                 subdataset is its own island; the islands just
                                 happen to sum back to the population.
  OMAT24 outside        0.051%   what an OMAT24 row that is *not already*
  nvt-3000                       nvt-3000 retrieves.  This is the comparator for
                                 an external dataset, which is likewise foreign
                                 to nvt-3000, and it is three orders of magnitude
                                 below what any external set shows.

So the external sets are not merely enriched in nvt-3000 by ~10x over its corpus
share -- they enter a region that OMAT24's own non-nvt-3000 rows essentially
never reach.  nvt-3000 (AIMD at 3000 K) is acting as the catch-all basin for
structures unlike anything in the reference set, not as a genuine chemical match.

The spread across query sets is itself informative: MatPES r2SCAN is the mildest
case (87.1% of top-5 slots, with 12.1% in ``rattled-relax``) and MOF-off the most
extreme (100.0%, nothing anywhere else).  The runner-up is always
``rattled-relax``, the only other subdataset that is not a fixed-temperature MD
trajectory.

ATOM-COUNT SENSITIVITY
----------------------
The MatPES row moved substantially when its query set was corrected, and in the
direction opposite to the obvious guess.  The earlier run (``omat_knn_pc25_matpes``)
used the ``mace_matpes_test_full`` r2SCAN subset: 161,262 frames, every one of
them 4 atoms or fewer.  The full set is 387,897 frames spanning 1-240 atoms.

    <=4 atoms   79.3% nvt-3000, 19.7% rattled-relax, d1 median 0.338
    full        88.6% nvt-3000, 10.8% rattled-relax, d1 median 0.359  (rank 1)

Restricting to small cells nearly doubles the ``rattled-relax`` share.  That is
consistent with ``rattled-relax`` being where OMAT24's small relaxation cells
live, so a small-cell query set finds real neighbours there; once large cells are
included the nvt-3000 basin reasserts itself.  The practical lesson is that this
statistic is sensitive to the atom-count distribution of the query set, so a
query set that is a size-restricted subset of its nominal dataset will
under-report nvt-3000 dominance.

WHY THERE IS NO TRAJECTORY-EXCLUDED HEADLINE NUMBER
---------------------------------------------------
OMAT24's own nearest neighbours are overwhelmingly adjacent frames of the same
AIMD trajectory, so the sibling ``mof_omat_composition_baseline.py`` excludes
neighbours within ``--trajectory-gap`` global rows of the query.  That exclusion
cannot carry the subdataset table: only 10 neighbours were stored per query, and
for the AIMD subdatasets the trajectory twins consume nearly all of them.  Slot
survival is 3.5-18% for the AIMD subdatasets but ~99.9% for the rattled ones, so
the surviving slots are ~88% rattled queries and the pooled share describes that
subsample rather than OMAT24.  The numbers are reported as a diagnostic with
their coverage attached, and the conditional row above -- which uses the full
top-5 of all 922,239 non-nvt-3000 queries -- is the comparator to quote.

Output is Markdown with ``$$``-delimited ``array`` environments, matching
``pc25_distance_table.py``; ``tabular`` is not part of MathJax.

    python scripts/omat/pc25_subdataset_table.py --root /path/to/pretrain_analysis
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.environ.get("PRETRAIN_ANALYSIS_ROOT", HERE.parents[1]))
PROBE = "runs/omat_knn_probe"
CALIBRATION = f"{PROBE}/omat_knn_pc25_final/OMAT_calibration"
MANIFEST = "data/processed/omat24/uma_latents_copy/all_uma_embeddings_manifest.csv"
DEFAULT_OUT = f"{PROBE}/multiset_top5_presence/pc25_subdataset_shares.md"

FOCUS = "aimd-from-PBE-3000-nvt"

QUERY_SETS = [
    ("MOF-off R2SCAN-D4", f"{PROBE}/omat_knn_pc25_mof_off/MOF_off_R2SCAN"),
    ("MOF-off PBE", f"{PROBE}/omat_knn_pc25_mof_off/MOF_off_PBE"),
    ("MAD", f"{PROBE}/omat_knn_pc25_mad/MAD_all"),
    ("MP ALOE", f"{PROBE}/omat_knn_pc25_mpaloe/MPALOE_all"),
    # The whole of MatPES-R2SCAN-2025.1: 387,897 frames, 100% r2SCAN, atoms
    # 1-240, via data/processed/matpes/all.lmdb.  NOT the earlier
    # omat_knn_pc25_matpes run, which used the mace_matpes_test_full r2SCAN
    # subset -- mixed-functional parent, capped at 4 atoms.  See ATOM-COUNT
    # SENSITIVITY below for why that distinction changes the number.
    ("MatPES r2SCAN", f"{PROBE}/omat_knn_pc25_matpes_all/MatPES_r2SCAN_all"),
    ("AM Small", f"{PROBE}/omat_knn_pc25_final/AM_Small"),
    ("AM Full", f"{PROBE}/omat_knn_pc25_final/AM_Full"),
]


def short(name):
    """Compact subdataset label for a table header."""
    return name.replace("aimd-from-PBE-", "aimd").replace("rattled-", "rat")


def thousands(value):
    """Thousands separator for math mode; a bare comma sets wrong spacing there."""
    return f"{value:,}".replace(",", "{,}")


def ratio(value):
    """Enrichment factors span 1e-3 to 1e1, so fixed precision loses the small end."""
    return f"{value:.1f}" if value >= 0.1 else f"{value:.4f}"


class Manifest:
    """Contiguous global-row space of the OMAT24 embedding matrix."""

    def __init__(self, path):
        rows = list(csv.DictReader(path.open(newline="")))
        if not rows:
            raise SystemExit(f"empty manifest: {path}")
        self.starts = np.array([int(r["global_start"]) for r in rows], dtype=np.int64)
        stops = np.array([int(r["global_stop"]) for r in rows], dtype=np.int64)
        self.n_rows = np.array([int(r["n_rows"]) for r in rows], dtype=np.int64)
        if not np.all(self.starts[1:] == stops[:-1]):
            raise SystemExit("manifest shards are not contiguous")
        subdataset = [r["subdataset"] for r in rows]
        self.names = sorted(set(subdataset))
        self.code = {name: i for i, name in enumerate(self.names)}
        self.shard_code = np.array([self.code[s] for s in subdataset], dtype=np.int16)
        self.total = int(stops[-1])
        self.population = np.zeros(len(self.names), dtype=np.int64)
        np.add.at(self.population, self.shard_code, self.n_rows)

    def subdataset_of(self, global_rows):
        index = np.searchsorted(self.starts, global_rows, side="right") - 1
        if index.min() < 0 or np.any(global_rows - self.starts[index] >= self.n_rows[index]):
            raise SystemExit("global row fell outside its manifest shard")
        return self.shard_code[index]

    def shares(self, codes):
        counts = np.bincount(codes, minlength=len(self.names)).astype(np.float64)
        return counts / counts.sum(), int(counts.sum())


def gate_rank1(manifest, indices, summary_path):
    """Rank-1 attribution must reproduce what the kNN run recorded."""
    recorded = json.loads(summary_path.read_text()).get("nearest_omat_subdataset_counts")
    if not recorded:
        return False
    got = np.bincount(manifest.subdataset_of(np.asarray(indices[:, 0])),
                      minlength=len(manifest.names))
    expected = np.array([recorded.get(name, 0) for name in manifest.names], dtype=np.int64)
    if not np.array_equal(got, expected):
        raise SystemExit(
            f"{summary_path}: recomputed rank-1 subdataset counts disagree with the "
            f"summary; the manifest and the kNN run describe different row spaces"
        )
    return True


def array_table(lines, spec, header, body, note=None):
    lines.append("$$")
    lines.append(rf"\begin{{array}}{{{spec}}}")
    lines.append(r"\hline")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\hline")
    lines.extend(body)
    lines.append(r"\hline")
    lines.append(r"\end{array}")
    lines.append("$$\n")
    if note:
        lines.append(note + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5,
                        help="neighbour slots per query; 5 is the canonical table")
    parser.add_argument("--trajectory-gap", type=int, default=1000,
                        help="global-row separation treated as a different trajectory")
    args = parser.parse_args()

    root = args.root.resolve()
    out = (args.out or root / DEFAULT_OUT).resolve()
    k = args.k

    manifest = Manifest(root / MANIFEST)
    focus = manifest.code[FOCUS]
    base = manifest.population[focus] / manifest.total
    order = np.argsort(-manifest.population)
    print(f"manifest: {manifest.total:,} rows, {len(manifest.names)} subdatasets")

    # --- OMAT24 leave-one-out reference rows --------------------------------
    indices = np.load(root / CALIBRATION / "indices.npy")
    source = np.load(root / CALIBRATION / "source_rows.npy")
    if indices.shape[1] < k:
        raise SystemExit(f"calibration stored {indices.shape[1]} neighbours, need {k}")
    query_code = manifest.subdataset_of(source)
    neighbour_code = manifest.subdataset_of(indices[:, :k].ravel()).reshape(-1, k)

    rows = []
    share, _ = manifest.shares(neighbour_code.ravel())
    rows.append(("OMAT24 (leave-one-out)", int(indices.shape[0]), share))
    outside = query_code != focus
    share_out, _ = manifest.shares(neighbour_code[outside].ravel())
    rows.append((rf"OMAT24 rows outside {short(FOCUS)}", int(outside.sum()), share_out))

    # Internal matrix: P(neighbour subdataset | query subdataset).
    internal = np.zeros((len(manifest.names), len(manifest.names)), dtype=np.int64)
    for column in range(k):
        np.add.at(internal, (query_code, neighbour_code[:, column]), 1)

    # --- external query sets -------------------------------------------------
    gated = 0
    for label, rel in QUERY_SETS:
        run = root / rel
        query_indices = np.load(run / "indices.npy", mmap_mode="r")
        gated += gate_rank1(manifest, query_indices, run / "summary.json")
        codes = manifest.subdataset_of(np.asarray(query_indices[:, :k]).ravel())
        share, _ = manifest.shares(codes)
        rows.append((label, int(query_indices.shape[0]), share))
    print(f"gate: rank-1 recomputation matches {gated}/{len(QUERY_SETS)} summaries")

    # --- trajectory-excluded diagnostic --------------------------------------
    gap = args.trajectory_gap
    far = np.abs(indices - source[:, None]) > gap
    # stable argsort on ~far keeps the surviving neighbours in distance order
    keep = np.argsort(~far, axis=1, kind="stable")[:, :k]
    taken = np.take_along_axis(indices, keep, axis=1)
    valid = np.take_along_axis(far, keep, axis=1)
    taken_code = manifest.subdataset_of(taken.ravel()).reshape(taken.shape)
    survival = valid.sum(axis=1)
    diagnostic = {
        "full_topk_fraction": float(valid.all(axis=1).mean()),
        "slot_survival": float(valid.mean()),
        "pooled": float((taken_code[valid] == focus).sum() / valid.sum()),
        "by_query": [
            (manifest.names[i],
             float(valid[query_code == i].mean()),
             float(survival[query_code == i].sum() / survival.sum()))
            for i in order
        ],
    }

    # --- render ---------------------------------------------------------------
    lines = [f"# pc25 neighbour attribution by OMAT24 subdataset (top-{k})\n"]
    lines.append(
        f"Every query contributes its {k} nearest OMAT24 rows in the 25-PC "
        f"standardised UMA space; each slot is attributed to the subdataset of the "
        f"row it landed on. OMAT24's own rows come from the leave-one-out "
        f"calibration ({indices.shape[0]:,} rows queried against all "
        f"{manifest.total:,}, self-match dropped).\n"
    )

    lines.append(f"## Share of top-{k} neighbour slots by subdataset\n")
    body = [
        r"\text{OMAT24 corpus (population)} & "
        + thousands(manifest.total) + " & "
        + " & ".join(rf"{100 * manifest.population[i] / manifest.total:.3f}" for i in order)
        + r" \\",
        r"\hline",
    ]
    for label, n, share in rows:
        body.append(
            rf"\text{{{label}}} & {thousands(n)} & "
            + " & ".join(f"{100 * share[i]:.3f}" for i in order)
            + r" \\"
        )
    array_table(
        lines,
        "lr" + "r" * len(manifest.names),
        [r"\textbf{population}", r"\textbf{queries}"]
        + [rf"\textbf{{{short(manifest.names[i])}}}" for i in order],
        body,
        note="All values are percentages of neighbour slots. Columns are ordered by "
             "corpus size; the first data row is the subdataset-blind null.",
    )

    lines.append(f"## {FOCUS} column, against its own baselines\n")
    body = [
        r"\text{OMAT24 corpus (population)} & "
        + thousands(manifest.total) + rf" & {100 * base:.3f}\% & 1.0 \\",
        r"\hline",
    ]
    for label, n, share in rows:
        body.append(
            rf"\text{{{label}}} & {thousands(n)} & {100 * share[focus]:.3f}\% & "
            rf"{ratio(share[focus] / base)} \\"
        )
    anchor = next(share for label, _, share in rows if label == QUERY_SETS[0][0])
    array_table(
        lines,
        "lrrr",
        [r"\textbf{population}", r"\textbf{queries}",
         rf"\textbf{{{short(FOCUS)}}}", r"\textbf{enrichment}"],
        body,
        note=f"Enrichment is the share divided by the {100 * base:.3f}\\% corpus share. "
             f"The external sets sit at ~12x that, but the corpus share is the weaker "
             f"of the two baselines. The comparator to quote is the *OMAT24 rows "
             f"outside {short(FOCUS)}* row: {QUERY_SETS[0][0]} reaches "
             f"{short(FOCUS)} {anchor[focus] / share_out[focus]:,.0f}x more often than "
             f"an OMAT24 row that is not already {short(FOCUS)} does.",
    )

    lines.append("## Why OMAT24's own share equals the population share\n")
    lines.append(
        "Each subdataset overwhelmingly retrieves itself, so the eleven islands sum "
        "back to the corpus composition. The diagonal is the self-retrieval rate and "
        f"the last column is the leak into {short(FOCUS)}.\n"
    )
    body = []
    for i in order:
        total = internal[i].sum()
        body.append(
            rf"\text{{{short(manifest.names[i])}}} & {thousands(int(total // k))} & "
            rf"{100 * internal[i, i] / total:.3f}\% & "
            rf"{100 * internal[i, focus] / total:.3f}\% \\"
        )
    array_table(
        lines,
        "lrrr",
        [r"\textbf{query subdataset}", r"\textbf{queries}",
         r"\textbf{self-retrieval}", rf"\textbf{{into {short(FOCUS)}}}"],
        body,
        note=f"Self-retrieval is inflated by within-trajectory near-duplicates for the "
             f"AIMD subdatasets; the {short(FOCUS)} column is not, because a trajectory "
             f"twin of a non-{short(FOCUS)} query is never in {short(FOCUS)}.",
    )

    lines.append("## Diagnostic: excluding within-trajectory neighbours\n")
    lines.append(
        f"Neighbours within {gap:,} global rows of the query are dropped as "
        f"probable same-trajectory frames. Only {indices.shape[1]} neighbours were "
        f"stored per query, so this exhausts the budget unevenly: "
        f"{100 * diagnostic['slot_survival']:.2f}% of slots survive overall and only "
        f"{100 * diagnostic['full_topk_fraction']:.2f}% of queries keep a full top-{k}. "
        f"Survival is near-total for the rattled subdatasets and near-zero for the AIMD "
        f"ones, so the surviving pool is not OMAT24.\n"
    )
    body = [
        rf"\text{{{short(name)}}} & {100 * surv:.2f}\% & {100 * weight:.2f}\% \\"
        for name, surv, weight in diagnostic["by_query"]
    ]
    array_table(
        lines,
        "lrr",
        [r"\textbf{query subdataset}", r"\textbf{slot survival}",
         r"\textbf{share of surviving slots}"],
        body,
        note=f"Pooled over the survivors, the {short(FOCUS)} share is "
             f"{100 * diagnostic['pooled']:.3f}\\%. That number describes a "
             f"rattled-dominated subsample, which is why the conditional row of the "
             f"main table -- full top-{k}, all "
             f"{int(outside.sum()):,} non-{short(FOCUS)} queries -- is the "
             f"reference this analysis quotes.",
    )

    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".partial.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
