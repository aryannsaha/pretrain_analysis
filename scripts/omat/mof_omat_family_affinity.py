#!/usr/bin/env python
"""Which chemical families of MOF-off map onto which families of OMAT24.

The neighbour-chemistry analysis answered "what does OMAT24 return in aggregate".
This answers the paired question: for each *kind* of MOF, what *kind* of OMAT24
structure comes back as its nearest neighbours.

Rows    MOF family  = primary metal group x linker-donor class
Columns OMAT family = anion class x dominant cation class
Cells   top-5 raw128 neighbour links, MOF-weighted (see `Weighting`)

Weighting
---------
The 80,643 query frames are only 3,269 distinct MOFs, so counting raw links
would let a 90-frame MOF outvote a 1-frame MOF 90:1.  Composition -- hence
family -- is constant within a MOF (asserted), so each MOF is given total
weight 1 and split two ways:

    w(link) = 1 / (n_temperatures(m) * n_frames(m, T) * 5)

Frames inside one (MOF, temperature) trajectory are near-replicates, but the
temperatures are not: a large minority of MOFs change their modal OMAT family
between 300 K and 1000 K.  Averaging over frames within a temperature and then
over temperatures keeps the thermal spread without letting frame count leak in.
The matrix then sums to 3,269 and every row total is literally a count of
distinct MOFs.

Lift
----
`log2(observed / expected-under-independence)`, with the row profile shrunk
toward the global column marginal by kappa MOF-equivalents:

    p~[i,j] = (n_i * p[i,j] + kappa * c[j]) / (n_i + kappa)

A flat pseudo-count would be wrong here: it floors every empty cell at the same
value, whereas a zero in a 467-MOF row is much stronger evidence of depletion
than a zero in a 30-MOF row.  Shrinkage keeps that asymmetry and regularises
only the small rows.

Support gating
--------------
A cell is scored only if at least `--min-cell-mofs` distinct MOFs contribute to
it.  Separately, the number of distinct OMAT *trajectories* behind each cell is
recorded: the retrieved structures collapse onto far fewer trajectories than
structures, so a cell can rest on one specific material rather than a family.
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mof_omat_neighbor_chemistry import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_ROOT,
    MAX_Z,
    NONMETAL_Z,
    atomic_write_json,
    load_indices,
    load_table,
    log,
    utc_now,
    write_csv,
)

GEOMETRY = "raw128"
K_NEIGHBOURS = 5

# --- element groupings -------------------------------------------------------
# La is deliberately grouped with the lanthanides.  The sibling module's
# ELEMENT_CLASSES puts La in transition_metal_5d and starts `lanthanide` at Ce;
# 43 MOFs here have La as their primary metal, so that choice would scatter a
# whole row's worth of lanthanide chemistry into the 5d bucket.
LANTHANIDE = set(range(57, 72))
ACTINIDE = set(range(89, 104))
ALKALI = {3, 11, 19, 37, 55, 87}
ALKALINE_EARTH = {4, 12, 20, 38, 56, 88}
TM_3D = set(range(21, 31))
TM_4D_5D = set(range(39, 49)) | set(range(72, 81))

METAL_MASK = np.ones(MAX_Z + 1, dtype=bool)
METAL_MASK[list(NONMETAL_Z)] = False
METAL_MASK[0] = False

NAMED_MOF_METALS = {30: "Zn", 29: "Cu", 27: "Co", 48: "Cd", 28: "Ni",
                    25: "Mn", 47: "Ag", 26: "Fe"}

# Tie-break when two metals are equally abundant.  Alkali/alkaline-earth atoms
# in a MOF are almost always charge-balancing counter-ions rather than nodes, so
# they lose ties by construction; the d-block is taken as the node metal.
TIER = {"tm3d": 0, "tm45d": 1, "pblock": 2, "ln": 3, "an": 4, "ae": 5, "alkali": 6}


def metal_tier(z):
    if z in TM_3D:
        return TIER["tm3d"]
    if z in TM_4D_5D:
        return TIER["tm45d"]
    if z in LANTHANIDE:
        return TIER["ln"]
    if z in ACTINIDE:
        return TIER["an"]
    if z in ALKALINE_EARTH:
        return TIER["ae"]
    if z in ALKALI:
        return TIER["alkali"]
    return TIER["pblock"]


METAL_TIEBREAK = np.array(
    [metal_tier(z) if METAL_MASK[z] else 99 for z in range(MAX_Z + 1)], dtype=np.int64
)

# Anion priority, most-electronegative first.  H is considered ONLY if none of
# these is present: with H in the argmax, H24F12Ge2Ni2O12 -- one of the most
# retrieved neighbours -- classifies as a hydride instead of a fluoride.
ANION_ORDER = [(8, "oxide"), (9, "fluoride"), (7, "nitride"), (6, "carbide"),
               (5, "boride"), (17, "halide"), (35, "halide"), (53, "halide"),
               (16, "sulfide"), (34, "chalcogenide"), (52, "chalcogenide"),
               (15, "phosphide"), (33, "pnictide"), (51, "pnictide"),
               (14, "silicide"), (32, "silicide")]


def metal_group(z):
    if z in NAMED_MOF_METALS:
        return NAMED_MOF_METALS[z]
    if z in LANTHANIDE or z in ACTINIDE:
        return "Ln/An"
    if z in ALKALI or z in ALKALINE_EARTH:
        return "group1-2"
    if z in TM_3D or z in TM_4D_5D:
        return "other-TM"
    return "main-group"


def cation_group(z):
    if z in ALKALI:
        return "alkali"
    if z in ALKALINE_EARTH:
        return "alk-earth"
    if z in TM_3D:
        return "3d TM"
    if z in TM_4D_5D:
        return "4d/5d TM"
    if z in LANTHANIDE or z in ACTINIDE:
        return "Ln/An"
    return "p-block"


def dominant_element(zcounts, candidate_mask, tiebreak):
    """Index of the most abundant candidate element per row; ties by tiebreak."""
    key = np.where(candidate_mask[None, :], zcounts, -1).astype(np.int64) * 1000
    key -= tiebreak[None, :]
    winner = key.argmax(axis=1)
    has_any = zcounts[:, candidate_mask].sum(axis=1) > 0
    return winner, has_any


def mof_row_labels(zcounts):
    """(metal_label, donor_label, n_metals, tie_broken) per MOF."""
    present = zcounts > 0
    n_metals = (present & METAL_MASK[None, :]).sum(axis=1)
    winner, has_metal = dominant_element(zcounts, METAL_MASK, METAL_TIEBREAK)

    counts = np.where(METAL_MASK[None, :], zcounts, -1)
    top = counts.max(axis=1)
    tie_broken = (counts == top[:, None]).sum(axis=1) > 1

    metal = []
    for i in range(zcounts.shape[0]):
        if not has_metal[i]:
            metal.append("main-group")          # Si-only frameworks
        elif n_metals[i] >= 2:
            metal.append("multi-metal")
        else:
            metal.append(metal_group(int(winner[i])))

    donor = []
    for i in range(zcounts.shape[0]):
        has = present[i]
        n, s, p = has[7], has[16], has[15]
        halide = has[9] or has[17] or has[35] or has[53]
        if n and p:
            donor.append("N+P")
        elif n and s:
            donor.append("N+S")
        elif n and halide:
            donor.append("N+halide")
        elif n:
            donor.append("N-donor")
        elif s:
            donor.append("S-donor")
        elif has[8]:
            donor.append("O-only")
        else:
            donor.append("other")
    return np.array(metal, dtype=object), np.array(donor, dtype=object), n_metals, tie_broken


def omat_col_labels(zcounts):
    present = zcounts > 0
    anion = np.empty(zcounts.shape[0], dtype=object)
    anion[:] = None
    for z, label in ANION_ORDER:
        fill = (anion == None) & present[:, z]  # noqa: E711
        anion[fill] = label
    anion[(anion == None) & present[:, 1]] = "hydride"  # noqa: E711
    anion[anion == None] = "intermetallic"  # noqa: E711

    winner, has_metal = dominant_element(zcounts, METAL_MASK, METAL_TIEBREAK)
    cation = np.array(
        [cation_group(int(winner[i])) if has_metal[i] else "covalent (no metal)"
         for i in range(zcounts.shape[0])],
        dtype=object,
    )
    return anion, cation


def backoff_rows(labels_metal, labels_donor, threshold):
    """Collapse rare (metal, donor) cells within their own metal, then globally."""
    joint = np.array([f"{m} / {d}" for m, d in zip(labels_metal, labels_donor)], dtype=object)
    counts = Counter(joint.tolist())
    stage1 = np.array(
        [j if counts[j] >= threshold else f"{m} / other-linker"
         for j, m in zip(joint, labels_metal)],
        dtype=object,
    )
    counts1 = Counter(stage1.tolist())
    stage2 = np.array(
        [s if counts1[s] >= threshold else "rare metal / mixed linker" for s in stage1],
        dtype=object,
    )
    return joint, stage2


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--row-threshold", type=int, default=40,
                        help="minimum distinct MOFs to keep a row family")
    parser.add_argument("--col-min-share", type=float, default=0.0025,
                        help="minimum share of weighted mass to keep a column family")
    parser.add_argument("--col-min-mofs", type=int, default=50,
                        help="minimum distinct supporting MOFs to keep a column family")
    parser.add_argument("--min-cell-mofs", type=int, default=5,
                        help="cells below this many distinct MOFs are not scored")
    parser.add_argument("--kappa", type=float, default=5.0,
                        help="shrinkage strength in MOF-equivalents")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--close-quantile", type=float, default=0.10)
    args = parser.parse_args()

    root = args.root.resolve()
    out_root = (args.out or root / DEFAULT_OUT).resolve()
    structures = out_root / "structures"
    out = out_root / "family_affinity"
    out.mkdir(parents=True, exist_ok=True)

    log("loading structure tables")
    mof = load_table(structures, "mof_off_r2scan")
    omat = load_table(structures, "omat_neighbors")
    mof_z = np.load(structures / "mof_off_r2scan_zcounts.npy")
    omat_z = np.load(structures / "omat_neighbors_zcounts.npy")
    idx, dist = load_indices(root, GEOMETRY)
    idx, dist = idx[:, :K_NEIGHBOURS], dist[:, :K_NEIGHBOURS]

    n_queries = len(mof["n_atoms"])
    if idx.shape[0] != n_queries:
        raise SystemExit(f"indices rows {idx.shape[0]} != MOF rows {n_queries}")
    if not np.array_equal(mof["global_row"], np.arange(n_queries)):
        raise SystemExit("mof_off_r2scan.csv is not in query-row order")

    mof_id = mof["subdataset"].astype(str)
    temperature = mof["shard"].astype(str)

    # --- MOF-level dedupe; composition must be constant within a MOF ---------
    uniq_ids, first_idx, inverse = np.unique(mof_id, return_index=True, return_inverse=True)
    n_mofs = uniq_ids.size
    log(f"{n_queries:,} frames -> {n_mofs:,} distinct MOFs")
    ref = mof_z[first_idx][inverse]
    if not np.array_equal(ref, mof_z):
        raise SystemExit("composition is NOT constant within a base MOF; weighting invalid")
    log("verified: composition constant within every base MOF")

    metal_lab, donor_lab, n_metals, tie_broken = mof_row_labels(mof_z[first_idx])
    joint_raw, row_family_mof = backoff_rows(metal_lab, donor_lab, args.row_threshold)
    row_family = row_family_mof[inverse]

    # --- OMAT labels ---------------------------------------------------------
    omat_rows = omat["global_row"].astype(np.int64)
    flat = idx.ravel()
    pos = np.searchsorted(omat_rows, flat)
    if not np.array_equal(omat_rows[pos], flat):
        raise SystemExit("some neighbour rows are missing from omat_neighbors.csv")
    anion, cation = omat_col_labels(omat_z)
    col_family_all = np.array([f"{a} / {c}" for a, c in zip(anion, cation)], dtype=object)
    log(f"top-5 links touch {np.unique(flat).size:,} distinct OMAT structures")

    # --- two-stage weights ---------------------------------------------------
    pair = np.char.add(np.char.add(mof_id.astype(str), "|"), temperature)
    _, pair_inv, pair_counts = np.unique(pair, return_inverse=True, return_counts=True)
    frames_at_T = pair_counts[pair_inv]
    temps_per_mof = np.zeros(n_mofs, dtype=np.int64)
    seen = {}
    for m, t in zip(inverse, temperature):
        seen.setdefault(int(m), set()).add(t)
    for m, ts in seen.items():
        temps_per_mof[m] = len(ts)
    n_temps = temps_per_mof[inverse]
    frame_weight = 1.0 / (n_temps * frames_at_T)
    if not np.isclose(frame_weight.sum(), n_mofs):
        raise SystemExit(f"frame weights sum to {frame_weight.sum():.6f}, expected {n_mofs}")
    link_weight = np.repeat(frame_weight / K_NEIGHBOURS, K_NEIGHBOURS)
    log(f"weights conserve: sum = {link_weight.sum():.6f} == {n_mofs} MOFs")

    link_row = np.repeat(row_family, K_NEIGHBOURS)
    link_col = col_family_all[pos]
    link_mof = np.repeat(inverse, K_NEIGHBOURS)
    link_traj = np.array(
        [f"{s}|{f}" for s, f in zip(omat["shard"][pos], omat["formula_hill"][pos])],
        dtype=object,
    )

    # --- collapse rare columns ----------------------------------------------
    col_mass = {}
    col_mofs = {}
    for c, w, m in zip(link_col, link_weight, link_mof):
        col_mass[c] = col_mass.get(c, 0.0) + w
        col_mofs.setdefault(c, set()).add(int(m))
    total_mass = sum(col_mass.values())
    keep_cols = {c for c, mass in col_mass.items()
                 if mass / total_mass >= args.col_min_share
                 and len(col_mofs[c]) >= args.col_min_mofs}
    link_col = np.array([c if c in keep_cols else "other family" for c in link_col], dtype=object)
    kept_share = sum(col_mass[c] for c in keep_cols) / total_mass
    log(f"columns: {len(keep_cols)} kept (+ residue), covering {kept_share:.2%} of mass")

    rows = sorted(set(row_family_mof.tolist()))
    cols = sorted(set(link_col.tolist()))
    rows = [r for r in rows if r != "rare metal / mixed linker"] + (
        ["rare metal / mixed linker"] if "rare metal / mixed linker" in rows else [])
    cols = [c for c in cols if c != "other family"] + (
        ["other family"] if "other family" in cols else [])
    ri = {r: i for i, r in enumerate(rows)}
    ci = {c: j for j, c in enumerate(cols)}
    R, C = len(rows), len(cols)
    log(f"matrix: {R} rows x {C} columns")

    rr = np.array([ri[r] for r in link_row])
    cc = np.array([ci[c] for c in link_col])

    # per-MOF profile matrix A (n_mofs x C); each row sums to 1
    A = np.zeros((n_mofs, C))
    np.add.at(A, (link_mof, cc), link_weight)
    if not np.allclose(A.sum(axis=1), 1.0):
        raise SystemExit("per-MOF profiles do not sum to 1")

    N = np.zeros((R, C))
    np.add.at(N, (rr, cc), link_weight)
    n_i = N.sum(axis=1)
    if not np.isclose(N.sum(), n_mofs):
        raise SystemExit(f"matrix sums to {N.sum():.4f}, expected {n_mofs}")

    raw_links = np.zeros((R, C), dtype=np.int64)
    np.add.at(raw_links, (rr, cc), 1)
    if raw_links.sum() != flat.size:
        raise SystemExit("raw link conservation failed")

    cell_mofs = np.zeros((R, C), dtype=np.int64)
    cell_traj = np.zeros((R, C), dtype=np.int64)
    seen_m, seen_t = {}, {}
    for r, c, m, t in zip(rr, cc, link_mof, link_traj):
        seen_m.setdefault((r, c), set()).add(int(m))
        seen_t.setdefault((r, c), set()).add(t)
    for (r, c), s in seen_m.items():
        cell_mofs[r, c] = len(s)
    for (r, c), s in seen_t.items():
        cell_traj[r, c] = len(s)

    # --- lift with shrinkage -------------------------------------------------
    p = N / np.maximum(n_i, 1e-12)[:, None]
    c_marg = N.sum(axis=0) / N.sum()
    shrunk = (n_i[:, None] * p + args.kappa * c_marg[None, :]) / (n_i[:, None] + args.kappa)
    lift = np.log2(shrunk / np.maximum(c_marg, 1e-12)[None, :])
    expected = n_i[:, None] * c_marg[None, :]
    scored = cell_mofs >= args.min_cell_mofs
    log(f"scored cells: {scored.sum()} of {R * C} "
        f"({raw_links[scored].sum() / raw_links.sum():.2%} of links)")

    # --- MOF-level stratified bootstrap -------------------------------------
    log(f"bootstrap: {args.bootstrap} stratified resamples over {n_mofs} MOFs")
    rng = np.random.default_rng(args.seed)
    members = [np.nonzero(np.array([ri[r] for r in row_family_mof]) == i)[0] for i in range(R)]
    agree = np.zeros((R, C))
    sign0 = np.sign(lift)
    for _ in range(args.bootstrap):
        Nb = np.empty((R, C))
        for i, mem in enumerate(members):
            draw = rng.choice(mem, size=mem.size, replace=True)
            Nb[i] = A[draw].sum(axis=0)
        nb = Nb.sum(axis=1)
        cb = Nb.sum(axis=0) / Nb.sum()
        pb = Nb / np.maximum(nb, 1e-12)[:, None]
        sb = (nb[:, None] * pb + args.kappa * cb[None, :]) / (nb[:, None] + args.kappa)
        agree += (np.sign(np.log2(sb / np.maximum(cb, 1e-12)[None, :])) == sign0)
    agree /= args.bootstrap

    # --- clustered order (on the lift matrix, residues excluded) -------------
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    core_r = [i for i, r in enumerate(rows) if r != "rare metal / mixed linker"]
    core_c = [j for j, c in enumerate(cols) if c != "other family"]
    L = np.clip(np.where(scored, lift, 0.0), -3, 3)
    sub = L[np.ix_(core_r, core_c)]
    ro = [core_r[k] for k in leaves_list(
        linkage(pdist(sub, metric="correlation"), method="average", optimal_ordering=True))]
    co = [core_c[k] for k in leaves_list(
        linkage(pdist(sub.T, metric="correlation"), method="average", optimal_ordering=True))]
    ro += [i for i in range(R) if i not in core_r]
    co += [j for j in range(C) if j not in core_c]

    # --- closest-decile companion -------------------------------------------
    cutoff = float(np.quantile(dist[:, 0], args.close_quantile))
    close_frame = dist[:, 0] <= cutoff
    close_link = np.repeat(close_frame, K_NEIGHBOURS)
    Nc = np.zeros((R, C))
    np.add.at(Nc, (rr[close_link], cc[close_link]), link_weight[close_link])
    close_mofs = np.zeros((R, C), dtype=np.int64)
    seen_c = {}
    for r, c, m in zip(rr[close_link], cc[close_link], link_mof[close_link]):
        seen_c.setdefault((r, c), set()).add(int(m))
    for (r, c), s in seen_c.items():
        close_mofs[r, c] = len(s)
    log(f"closest decile: d1 <= {cutoff:.3f}, {close_frame.sum():,} frames, "
        f"{np.unique(inverse[close_frame]).size:,} distinct MOFs")

    # --- outputs -------------------------------------------------------------
    matrix_rows = []
    for i in range(R):
        for j in range(C):
            matrix_rows.append({
                "mof_family": rows[i], "omat_family": cols[j],
                "display_row": ro.index(i), "display_col": co.index(j),
                "weighted_mofs": round(float(N[i, j]), 6),
                "raw_links": int(raw_links[i, j]),
                "row_share_pct": round(float(p[i, j] * 100), 4),
                "expected_mofs": round(float(expected[i, j]), 6),
                "log2_lift": round(float(lift[i, j]), 4),
                "scored": int(scored[i, j]),
                "n_distinct_mofs": int(cell_mofs[i, j]),
                "n_distinct_omat_trajectories": int(cell_traj[i, j]),
                "bootstrap_sign_agreement": round(float(agree[i, j]), 4),
                "close_decile_weighted_mofs": round(float(Nc[i, j]), 6),
                "close_decile_n_mofs": int(close_mofs[i, j]),
            })
    write_csv(out / "family_affinity_matrix.csv", list(matrix_rows[0].keys()), matrix_rows)

    frames_per = Counter(row_family.tolist())
    write_csv(out / "mof_family_marginals.csv",
              ["mof_family", "display_order", "distinct_mofs", "frames", "raw_links",
               "share_of_mofs_pct", "top_omat_family", "top_row_share_pct", "effective_n_families"],
              [{"mof_family": rows[i], "display_order": ro.index(i),
                "distinct_mofs": int(round(n_i[i])), "frames": int(frames_per[rows[i]]),
                "raw_links": int(raw_links[i].sum()),
                "share_of_mofs_pct": round(float(n_i[i] / n_mofs * 100), 3),
                "top_omat_family": cols[int(p[i].argmax())],
                "top_row_share_pct": round(float(p[i].max() * 100), 2),
                "effective_n_families": round(float(2 ** (
                    -(p[i][p[i] > 0] * np.log2(p[i][p[i] > 0])).sum())), 2)}
               for i in range(R)])

    col_struct = {}
    for c, pp in zip(link_col, pos):
        col_struct.setdefault(c, set()).add(int(pp))
    write_csv(out / "omat_family_marginals.csv",
              ["omat_family", "display_order", "raw_links", "share_of_links_pct",
               "weighted_share_pct", "distinct_structures", "top_mof_family", "example_formula"],
              [{"omat_family": cols[j], "display_order": co.index(j),
                "raw_links": int(raw_links[:, j].sum()),
                "share_of_links_pct": round(float(raw_links[:, j].sum() / raw_links.sum() * 100), 3),
                "weighted_share_pct": round(float(c_marg[j] * 100), 3),
                "distinct_structures": len(col_struct.get(cols[j], ())),
                "top_mof_family": rows[int(N[:, j].argmax())],
                "example_formula": str(omat["formula_hill"][
                    Counter(pos[link_col == cols[j]].tolist()).most_common(1)[0][0]])
                if (link_col == cols[j]).any() else ""}
               for j in range(C)])

    top3 = []
    for i in range(R):
        for rank, j in enumerate(np.argsort(-p[i])[:3], 1):
            top3.append({"mof_family": rows[i], "distinct_mofs": int(round(n_i[i])),
                         "rank": rank, "omat_family": cols[j],
                         "row_share_pct": round(float(p[i, j] * 100), 2),
                         "log2_lift": round(float(lift[i, j]), 3),
                         "n_distinct_mofs_in_cell": int(cell_mofs[i, j]),
                         "n_distinct_trajectories": int(cell_traj[i, j]),
                         "bootstrap_sign_agreement": round(float(agree[i, j]), 3)})
    write_csv(out / "top_omat_families_per_mof_family.csv", list(top3[0].keys()), top3)

    write_csv(out / "family_assignments_mof.csv",
              ["mof_id", "n_frames", "formula_hill", "chemical_system", "n_metal_elements",
               "metal_label", "donor_class", "metal_tie_broken", "family_raw", "family_collapsed"],
              [{"mof_id": str(uniq_ids[m]), "n_frames": int((inverse == m).sum()),
                "formula_hill": str(mof["formula_hill"][first_idx[m]]),
                "chemical_system": str(mof["chemical_system"][first_idx[m]]),
                "n_metal_elements": int(n_metals[m]), "metal_label": str(metal_lab[m]),
                "donor_class": str(donor_lab[m]), "metal_tie_broken": int(tie_broken[m]),
                "family_raw": str(joint_raw[m]), "family_collapsed": str(row_family_mof[m])}
               for m in range(n_mofs)])

    atomic_write_json(out / "family_affinity_metadata.json", {
        "created_utc": utc_now(), "geometry": GEOMETRY, "k_neighbours": K_NEIGHBOURS,
        "n_frames": int(n_queries), "n_distinct_mofs": int(n_mofs),
        "n_links": int(flat.size),
        "n_distinct_omat_structures": int(np.unique(flat).size),
        "n_distinct_omat_trajectories": int(np.unique(link_traj).size),
        "rows": rows, "cols": cols, "row_display_order": ro, "col_display_order": co,
        "row_threshold_mofs": args.row_threshold,
        "col_min_share": args.col_min_share, "col_min_mofs": args.col_min_mofs,
        "min_cell_mofs": args.min_cell_mofs, "kappa": args.kappa,
        "bootstrap": args.bootstrap, "seed": args.seed,
        "column_mass_kept": round(float(kept_share), 6),
        "scored_cells": int(scored.sum()), "total_cells": int(R * C),
        "close_decile_cutoff_d1": cutoff,
        "close_decile_frames": int(close_frame.sum()),
        "close_decile_distinct_mofs": int(np.unique(inverse[close_frame]).size),
        "weighting": "1/(n_temperatures * n_frames_at_T * k); matrix sums to n_distinct_mofs",
        "lift": "log2(shrunk_row_profile / column_marginal), kappa in MOF-equivalents",
    })
    log(f"family affinity written -> {out}")


if __name__ == "__main__":
    main()
