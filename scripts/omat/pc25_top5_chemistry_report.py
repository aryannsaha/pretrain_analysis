#!/usr/bin/env python
"""Aggregate pc25 top-5 descriptors into comparison tables and a markdown report.

Consumes the per-structure tables written by ``pc25_top5_structure_descriptors.py``
and answers, for MOF-off R2SCAN and MAD side by side: how do the OMAT24 structures
UMA retrieves compare to the query set on bonding, coordination, connectivity and
composition -- and which *kinds* of query map onto which *kinds* of OMAT24 material.

Weighting
---------
Neighbour profiles are link-weighted: a structure retrieved by 40 queries counts
40 times, because that is what the query set actually "sees".  Unique-structure
profiles are reported alongside wherever concentration could mislead.

MOF-off's 80,643 frames are only 3,269 distinct MOFs, so the family-affinity
matrix (and only that matrix) uses the MOF-equivalent weighting of the sibling
raw128 analysis, keeping the two matrices directly comparable.  MAD rows are
distinct structures, so MAD uses plain link weighting throughout.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pc25_top5_structure_descriptors import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_ROOT,
    K_NEIGHBOURS,
    MAX_Z,
    METAL_MASK,
    QUERY_SETS,
    atomic_write_json,
    log,
    utc_now,
)

from ase.data import chemical_symbols  # noqa: E402

# Populations in report order: query, its link-weighted neighbours, baselines.
POPULATIONS = [
    ("query_mof_off", "MOF-off R2SCAN", None),
    ("omat_nb_mof_off", "MOF-off pc25 top-5", "mof_off"),
    ("query_mad", "MAD", None),
    ("omat_nb_mad", "MAD pc25 top-5", "mad"),
    ("omat_bg_nvt3000", "OMAT24 bg (nvt-3000)", None),
    ("omat_bg_global", "OMAT24 bg (all)", None),
]

# Descriptors carried into the headline comparison table, with display names and
# the number of decimals each is reported to.
DESCRIPTOR_TABLE = [
    ("n_atoms", "atoms", 0),
    ("n_elements", "distinct elements", 0),
    ("density_g_cm3", "density (g/cm3)", 2),
    ("packing_fraction", "packing fraction", 3),
    ("volume_per_atom", "volume/atom (A^3)", 1),
    ("energy_per_atom", "energy/atom (eV)", 2),
    ("en_mean", "mean electronegativity", 2),
    ("en_range", "electronegativity spread", 2),
    ("metal_atom_frac", "metal atom fraction", 3),
    ("cn_mean", "mean CN (all atoms)", 2),
    ("cn_metal_mean", "mean CN (metals)", 2),
    ("cn_nonmetal_mean", "mean CN (non-metals)", 2),
    ("bond_len_mean", "mean bond length (A)", 2),
    ("frac_bond_metal_metal", "bonds metal-metal", 3),
    ("frac_bond_metal_nonmetal", "bonds metal-nonmetal", 3),
    ("frac_bond_nonmetal_nonmetal", "bonds nonmetal-nonmetal", 3),
    ("frac_bond_CH", "bonds C-H", 3),
    ("frac_bond_CC", "bonds C-C", 3),
    ("frac_bond_CO", "bonds C-O", 3),
    ("frac_bond_MO", "bonds metal-O", 3),
    ("h_per_c", "H per C", 2),
    ("n_components", "connected components", 0),
    ("largest_component_frac", "largest component frac", 2),
    ("frac_cn0", "isolated-atom fraction", 3),
]

NUMERIC_FIELDS = {name for name, _, _ in DESCRIPTOR_TABLE} | {
    "cn_C_mean", "cn_H_mean", "cn_N_mean", "cn_O_mean", "cn_max", "cn_mean_wide",
    "dimensionality", "total_z", "force_rms", "pressure_gpa", "n_bonds",
}


def load_population(structures, population):
    """(columns dict, zcounts, cnhist, cnsum) for one population."""
    path = structures / f"{population}.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty population table: {path}")
    columns = {}
    for key in rows[0]:
        raw = [r[key] for r in rows]
        if key in NUMERIC_FIELDS:
            columns[key] = np.array(
                [float(v) if v not in ("", "None") else np.nan for v in raw]
            )
        else:
            columns[key] = np.array(raw, dtype=object)
    arrays = tuple(
        np.load(structures / f"{population}_{suffix}.npy")
        for suffix in ("zcounts", "cnhist", "cnsum")
    )
    return (columns,) + arrays


def link_weights(root, query, neighbour_rows):
    """Per-neighbour-structure link counts under the pc25 top-5 retrieval."""
    idx = np.load(root / QUERY_SETS[query]["indices"] / "indices.npy")[:, :K_NEIGHBOURS]
    pos = np.searchsorted(neighbour_rows, idx.ravel())
    if np.any(neighbour_rows[pos] != idx.ravel()):
        raise SystemExit(f"{query}: neighbour table does not cover every retrieved row")
    weights = np.bincount(pos, minlength=neighbour_rows.size).astype(np.float64)
    return idx, weights


def wq(values, weights, quantiles=(0.25, 0.5, 0.75)):
    """Weighted quantiles, NaN-aware."""
    finite = np.isfinite(values)
    v, w = values[finite], weights[finite]
    if v.size == 0 or w.sum() == 0:
        return [np.nan] * len(quantiles)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= w.sum()
    return [float(np.interp(q, cum, v)) for q in quantiles]


def wmean(values, weights):
    finite = np.isfinite(values)
    v, w = values[finite], weights[finite]
    return float((v * w).sum() / w.sum()) if w.sum() else np.nan


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    log(f"wrote {path.name} ({len(rows):,} rows)")


def fmt(value, decimals):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    return f"{value:,.0f}" if decimals == 0 else f"{value:.{decimals}f}"


def md_table(header, rows):
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


# --- family taxonomy ---------------------------------------------------------

def omat_family_labels(zcounts):
    from mof_omat_family_affinity import omat_col_labels

    anion, cation = omat_col_labels(zcounts)
    return np.array([f"{a} / {c}" for a, c in zip(anion, cation)], dtype=object)


def mof_family_labels(zcounts, threshold=40):
    from mof_omat_family_affinity import backoff_rows, mof_row_labels

    metal, donor, _, _ = mof_row_labels(zcounts)
    _, collapsed = backoff_rows(metal, donor, threshold)
    return collapsed


def affinity_matrix(row_labels, col_labels_per_link, weights_per_link,
                    min_row_mass, kappa=5.0):
    """Weighted row-family x column-family table plus log2 lift vs the marginal."""
    rows = sorted(set(row_labels.tolist()))
    cols = sorted(set(col_labels_per_link.tolist()))
    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: i for i, c in enumerate(cols)}
    matrix = np.zeros((len(rows), len(cols)))
    np.add.at(
        matrix,
        (
            np.array([row_index[r] for r in row_labels]),
            np.array([col_index[c] for c in col_labels_per_link]),
        ),
        weights_per_link,
    )
    keep_rows = matrix.sum(axis=1) >= min_row_mass
    matrix, rows = matrix[keep_rows], [r for r, k in zip(rows, keep_rows) if k]
    marginal = matrix.sum(axis=0) / matrix.sum()
    totals = matrix.sum(axis=1, keepdims=True)
    profile = matrix / np.maximum(totals, 1e-12)
    # Shrink each row toward the global column marginal by kappa row-equivalents
    # so a zero in a small row is not read as strongly as a zero in a large one.
    shrunk = (totals * profile + kappa * marginal) / (totals + kappa)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.log2(shrunk / np.maximum(marginal, 1e-12))
    return rows, cols, matrix, profile, lift, marginal


def command_analyze(args):
    root = args.root.resolve()
    out_root = (args.out or root / DEFAULT_OUT).resolve()
    structures = out_root / "structures"
    analysis = out_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    log("loading populations")
    data, weights, rows_present = {}, {}, {}
    for population, _, query in POPULATIONS:
        cols, zc, ch, cs = load_population(structures, population)
        data[population] = (cols, zc, ch, cs)
        if query:
            global_rows = cols["global_row"].astype(np.int64) if cols["global_row"].dtype != object \
                else np.array([int(v) for v in cols["global_row"]], dtype=np.int64)
            idx, w = link_weights(root, query, global_rows)
            weights[population] = w
            rows_present[query] = (idx, global_rows)
        else:
            weights[population] = np.ones(len(cols["n_atoms"]))
        log(f"  {population}: {len(cols['n_atoms']):,} structures, "
            f"{weights[population].sum():,.0f} weighted")

    # --- 1. headline descriptor comparison ----------------------------------
    descriptor_rows = []
    for field, label, decimals in DESCRIPTOR_TABLE:
        row = {"descriptor": label, "field": field}
        for population, name, _ in POPULATIONS:
            cols = data[population][0]
            values = cols.get(field)
            if values is None:
                continue
            p25, p50, p75 = wq(values, weights[population])
            row[f"{name} p25"] = fmt(p25, decimals)
            row[f"{name} median"] = fmt(p50, decimals)
            row[f"{name} p75"] = fmt(p75, decimals)
            row[f"{name} mean"] = fmt(wmean(values, weights[population]), decimals + 2)
        descriptor_rows.append(row)
    write_csv(analysis / "descriptor_comparison.csv",
              list(descriptor_rows[0]), descriptor_rows)

    # --- 2. dimensionality / connectivity -----------------------------------
    dim_rows = []
    for population, name, _ in POPULATIONS:
        cols, _, _, _ = data[population]
        dim, w = cols["dimensionality"], weights[population]
        total = w.sum()
        row = {"population": name, "weighted_structures": round(float(total))}
        for d in range(4):
            row[f"dim{d}_pct"] = round(100.0 * w[dim == d].sum() / total, 3)
        row["single_component_pct"] = round(
            100.0 * w[cols["n_components"] == 1].sum() / total, 3)
        row["median_components"] = round(wq(cols["n_components"], w)[1], 2)
        dim_rows.append(row)
    write_csv(analysis / "dimensionality.csv", list(dim_rows[0]), dim_rows)

    # --- 3. coordination-number distribution --------------------------------
    cn_rows = []
    for population, name, _ in POPULATIONS:
        _, _, cnhist, _ = data[population]
        w = weights[population]
        hist = (cnhist * w[:, None]).sum(axis=0)
        hist = hist / hist.sum()
        row = {"population": name}
        row.update({f"cn{b}": round(float(hist[b]), 5) for b in range(cnhist.shape[1])})
        cn_rows.append(row)
    write_csv(analysis / "cn_distribution.csv", list(cn_rows[0]), cn_rows)

    # --- 4. per-element coordination + presence -----------------------------
    presence, mean_cn, atom_frac = {}, {}, {}
    for population, name, _ in POPULATIONS:
        _, zc, _, cs = data[population]
        w = weights[population]
        presence[name] = (w[:, None] * (zc > 0)).sum(axis=0) / w.sum()
        atoms_of_z = (w[:, None] * zc).sum(axis=0)
        atom_frac[name] = atoms_of_z / atoms_of_z.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_cn[name] = np.where(
                atoms_of_z > 0, (w[:, None] * cs).sum(axis=0) / np.maximum(atoms_of_z, 1e-12), np.nan
            )

    names = [name for _, name, _ in POPULATIONS]
    # Select and order elements by the LARGEST presence across all populations,
    # never by one population's prevalence: the asymmetric elements (common among
    # neighbours, absent from the query set) are exactly the point of comparison.
    max_presence = np.max([presence[n] for n in names], axis=0)
    keep = [z for z in range(1, MAX_Z + 1) if max_presence[z] >= args.element_min_presence]
    keep.sort(key=lambda z: -max_presence[z])
    element_rows = []
    for z in keep:
        row = {"element": chemical_symbols[z], "Z": z,
               "is_metal": bool(METAL_MASK[z]),
               "max_presence": round(float(max_presence[z]), 5)}
        for name in names:
            row[f"{name} presence"] = round(float(presence[name][z]), 5)
            row[f"{name} atom_frac"] = round(float(atom_frac[name][z]), 5)
            cn = mean_cn[name][z]
            row[f"{name} mean_CN"] = round(float(cn), 3) if np.isfinite(cn) else ""
        element_rows.append(row)
    write_csv(analysis / "element_presence_and_cn.csv",
              list(element_rows[0]), element_rows)

    # --- 5. paired query <-> neighbour agreement ----------------------------
    pair_rows, dim_confusion = [], {}
    for query, (query_pop, nb_pop) in (("mof_off", ("query_mof_off", "omat_nb_mof_off")),
                                       ("mad", ("query_mad", "omat_nb_mad"))):
        qcols = data[query_pop][0]
        ncols, nzc = data[nb_pop][0], data[nb_pop][1]
        idx, global_rows = rows_present[query]
        pos = np.searchsorted(global_rows, idx)          # (n_queries, 5)
        qzc = data[query_pop][1]

        # Element-set agreement, per query, over its 5 neighbours.
        q_present = qzc > 0
        n_present = nzc > 0
        shared_metal, identical, jaccard = [], [], []
        for k in range(K_NEIGHBOURS):
            npres = n_present[pos[:, k]]
            inter = (q_present & npres).sum(axis=1)
            union = (q_present | npres).sum(axis=1)
            jaccard.append(inter / np.maximum(union, 1))
            identical.append((inter == union) & (union > 0))
            shared_metal.append(
                ((q_present & npres) & METAL_MASK[None, : MAX_Z + 1]).any(axis=1)
            )
        jaccard = np.stack(jaccard, axis=1)
        identical = np.stack(identical, axis=1)
        shared_metal = np.stack(shared_metal, axis=1)

        qdim = qcols["dimensionality"].astype(int)
        ndim = ncols["dimensionality"].astype(int)[pos]
        dim_match = (ndim == qdim[:, None])
        conf = np.zeros((4, 4))
        for d in range(4):
            rows_d = qdim == d
            if rows_d.any():
                conf[d] = np.bincount(ndim[rows_d].ravel(), minlength=4)[:4] / rows_d.sum() / K_NEIGHBOURS
        dim_confusion[query] = conf

        for field in ("cn_mean", "density_g_cm3", "packing_fraction", "en_mean",
                      "metal_atom_frac", "n_atoms", "energy_per_atom"):
            qv, nv = qcols[field], ncols[field][pos]
            delta = np.nanmean(nv, axis=1) - qv
            finite = np.isfinite(delta)
            correlation = (
                float(np.corrcoef(qv[finite], np.nanmean(nv, axis=1)[finite])[0, 1])
                if finite.sum() > 2 else np.nan
            )
            pair_rows.append({
                "query_set": query,
                "descriptor": field,
                "query_median": round(float(np.nanmedian(qv)), 4),
                "neighbour_median": round(float(np.nanmedian(np.nanmean(nv, axis=1))), 4),
                "median_delta": round(float(np.nanmedian(delta[finite])), 4),
                "correlation": round(correlation, 4),
            })
        pair_rows.append({
            "query_set": query,
            "descriptor": "element_set_jaccard",
            "query_median": "", "neighbour_median": "",
            "median_delta": round(float(np.median(jaccard.mean(axis=1))), 4),
            "correlation": "",
        })
        for label, value in (("frac_identical_element_set", identical.mean()),
                             ("frac_sharing_any_metal", shared_metal.mean()),
                             ("frac_dimensionality_match", dim_match.mean())):
            pair_rows.append({
                "query_set": query, "descriptor": label,
                "query_median": "", "neighbour_median": "",
                "median_delta": round(float(value), 4), "correlation": "",
            })
    write_csv(analysis / "query_neighbour_pairing.csv", list(pair_rows[0]), pair_rows)

    conf_rows = []
    for query, conf in dim_confusion.items():
        for d in range(4):
            row = {"query_set": query, "query_dimensionality": d}
            row.update({f"neighbour_dim{k}": round(float(conf[d, k]), 4) for k in range(4)})
            conf_rows.append(row)
    write_csv(analysis / "dimensionality_confusion.csv", list(conf_rows[0]), conf_rows)

    # --- 6. family affinity --------------------------------------------------
    family_payload = {}
    for query, (query_pop, nb_pop) in (("mof_off", ("query_mof_off", "omat_nb_mof_off")),
                                       ("mad", ("query_mad", "omat_nb_mad"))):
        qcols, qzc = data[query_pop][0], data[query_pop][1]
        nzc = data[nb_pop][1]
        idx, global_rows = rows_present[query]
        pos = np.searchsorted(global_rows, idx)
        omat_family = omat_family_labels(nzc)

        if query == "mof_off":
            row_labels = mof_family_labels(qzc)
            # MOF-equivalent weighting: each distinct MOF carries total weight 1,
            # split evenly over its temperatures, frames and 5 links, so a MOF
            # sampled at 90 frames cannot outvote one sampled at 1.  The base MOF
            # id lives in `subdataset` (cif_name prefix); `key` is the per-frame
            # record id and would make every frame its own MOF.
            mof_id = qcols["subdataset"]
            temperature = qcols["shard"]
            per_mof_T = Counter(zip(mof_id.tolist(), temperature.tolist()))
            n_temps = Counter()
            for mof, _T in set(zip(mof_id.tolist(), temperature.tolist())):
                n_temps[mof] += 1
            frame_weight = np.array([
                1.0 / (n_temps[m] * per_mof_T[(m, t)] * K_NEIGHBOURS)
                for m, t in zip(mof_id, temperature)
            ])
            min_row_mass = args.min_row_mofs
        else:
            row_labels = qcols["subdataset"]
            frame_weight = np.full(len(row_labels), 1.0 / K_NEIGHBOURS)
            min_row_mass = 0.0

        link_rows = np.repeat(row_labels, K_NEIGHBOURS)
        link_cols = omat_family[pos.ravel()]
        link_w = np.repeat(frame_weight, K_NEIGHBOURS)
        rows_kept, cols_kept, matrix, profile, lift, marginal = affinity_matrix(
            link_rows, link_cols, link_w, min_row_mass, kappa=args.kappa
        )
        for suffix, array in (("share", profile), ("lift", lift)):
            table = []
            for i, r in enumerate(rows_kept):
                entry = {"query_family": r, "row_mass": round(float(matrix[i].sum()), 3)}
                entry.update({c: round(float(array[i, j]), 4)
                              for j, c in enumerate(cols_kept)})
                table.append(entry)
            write_csv(analysis / f"family_affinity_{query}_{suffix}.csv",
                      list(table[0]), table)
        family_payload[query] = {
            "rows": rows_kept,
            "cols": cols_kept,
            "column_marginal": {c: round(float(m), 5) for c, m in zip(cols_kept, marginal)},
            "top_column_per_row": {
                r: cols_kept[int(profile[i].argmax())] for i, r in enumerate(rows_kept)
            },
        }

        counts = Counter()
        for label, weight in zip(link_cols, link_w):
            counts[label] += weight
        total = sum(counts.values())
        write_csv(
            analysis / f"omat_family_marginal_{query}.csv",
            ["omat_family", "weighted_share_pct", "link_share_pct"],
            [{"omat_family": f,
              "weighted_share_pct": round(100.0 * c / total, 3),
              "link_share_pct": round(100.0 * (link_cols == f).sum() / link_cols.size, 3)}
             for f, c in counts.most_common()],
        )

    # --- 7. distances --------------------------------------------------------
    dist_rows = []
    for query, query_pop in (("mof_off", "query_mof_off"), ("mad", "query_mad")):
        distances = np.load(
            root / QUERY_SETS[query]["indices"] / "distances.npy")[:, :K_NEIGHBOURS]
        neighbours = np.load(
            root / QUERY_SETS[query]["indices"] / "indices.npy")[:, :K_NEIGHBOURS]
        # For MOF-off `subdataset` holds the base MOF id (3,269 of them), which is
        # not a useful breakdown; temperature is. MAD keeps its subsets.
        groups = data[query_pop][0]["shard" if query == "mof_off" else "subdataset"]
        labels = sorted(set(groups.tolist()),
                        key=lambda s: (len(s), s))[: args.max_subsets]
        for label, mask in [("all", np.ones(len(groups), dtype=bool))] + [
            (s, groups == s) for s in labels
        ]:
            if mask.sum() < 50:
                continue
            d1, d5 = distances[mask, 0], distances[mask, K_NEIGHBOURS - 1]
            # Consecutive frames of one OMAT AIMD trajectory occupy consecutive
            # global rows, so counting row-runs separated by more than the gap
            # says how many distinct *materials* a top-5 actually contains.
            ordered = np.sort(neighbours[mask], axis=1)
            distinct = 1 + (np.diff(ordered, axis=1) > args.trajectory_gap).sum(axis=1)
            dist_rows.append({
                "query_set": query, "subset": label, "n": int(mask.sum()),
                "d1_p10": round(float(np.percentile(d1, 10)), 4),
                "d1_median": round(float(np.median(d1)), 4),
                "d1_p90": round(float(np.percentile(d1, 90)), 4),
                "d5_median": round(float(np.median(d5)), 4),
                "isolation_d5_over_d1": round(float(np.median(d5 / np.maximum(d1, 1e-9))), 4),
                "mean_distinct_trajectories_in_top5": round(float(distinct.mean()), 3),
                "pct_top5_all_one_trajectory": round(
                    100.0 * float((distinct == 1).mean()), 2),
            })
    write_csv(analysis / "pc25_distances.csv", list(dist_rows[0]), dist_rows)

    # --- 8. markdown rendering of every table --------------------------------
    # The prose report embeds these verbatim, so no figure in it is ever
    # hand-transcribed from a CSV.
    blocks = ["# Generated tables (pc25, top-5)\n"]

    def block(title, header, body):
        blocks.append(f"\n## {title}\n\n{md_table(header, body)}\n")

    query_names = [name for _, name, _ in POPULATIONS]
    block(
        "Descriptor comparison (median, p25-p75 in brackets)",
        ["descriptor"] + query_names,
        [[r["descriptor"]] + [f'{r[f"{n} median"]} [{r[f"{n} p25"]}-{r[f"{n} p75"]}]'
                              for n in query_names] for r in descriptor_rows],
    )
    block(
        "Dimensionality of the largest bonded component (% of structures)",
        ["population", "0D", "1D", "2D", "3D", "single component %", "median components"],
        [[r["population"]] + [f'{r[f"dim{d}_pct"]:.1f}' for d in range(4)]
         + [f'{r["single_component_pct"]:.1f}', f'{r["median_components"]:.1f}']
         for r in dim_rows],
    )
    block(
        "Per-atom coordination number distribution (share of atoms)",
        ["population"] + [f"CN {b}" for b in range(9)] + ["CN 9+"],
        [[r["population"]] + [f'{r[f"cn{b}"]:.3f}' for b in range(9)]
         + [f'{sum(r[f"cn{b}"] for b in range(9, len(cn_rows[0]) - 1)):.3f}']
         for r in cn_rows],
    )
    element_top = element_rows[: args.report_elements]
    block(
        "Element presence (fraction of structures) and mean coordination number",
        ["element"] + [f"{n} pres / CN" for n in query_names],
        [[r["element"]] + [f'{r[f"{n} presence"]:.3f} / {r[f"{n} mean_CN"] or "-"}'
                           for n in query_names] for r in element_top],
    )
    block(
        "Query <-> neighbour agreement",
        ["query set", "quantity", "query median", "neighbour median",
         "median delta", "correlation"],
        [[r["query_set"], r["descriptor"], r["query_median"], r["neighbour_median"],
          r["median_delta"], r["correlation"]] for r in pair_rows],
    )
    block(
        "Query dimensionality vs neighbour dimensionality (row-normalised)",
        ["query set", "query dim", "-> 0D", "-> 1D", "-> 2D", "-> 3D"],
        [[r["query_set"], f'{r["query_dimensionality"]}D']
         + [f'{r[f"neighbour_dim{k}"]:.3f}' for k in range(4)] for r in conf_rows],
    )
    block(
        "pc25 distances and how many distinct materials a top-5 contains",
        ["query set", "subset", "n", "d1 p10", "d1 median", "d1 p90", "d5 median",
         "d5/d1", "distinct trajectories in top-5", "% top-5 all one trajectory"],
        [[r["query_set"], r["subset"], f'{r["n"]:,}', r["d1_p10"], r["d1_median"],
          r["d1_p90"], r["d5_median"], r["isolation_d5_over_d1"],
          r["mean_distinct_trajectories_in_top5"], r["pct_top5_all_one_trajectory"]]
         for r in dist_rows],
    )
    for query in ("mof_off", "mad"):
        path = analysis / f"omat_family_marginal_{query}.csv"
        with path.open(newline="") as handle:
            marginal = list(csv.DictReader(handle))[:12]
        block(
            f"OMAT24 families retrieved by {query} (top 12)",
            ["OMAT24 family", "weighted share %", "raw link share %"],
            [[r["omat_family"], r["weighted_share_pct"], r["link_share_pct"]]
             for r in marginal],
        )
    (analysis / "tables.md").write_text("\n".join(blocks) + "\n")
    log("wrote tables.md")

    atomic_write_json(analysis / "analysis_metadata.json", {
        "created_utc": utc_now(),
        "geometry": "pc25",
        "k_neighbours": K_NEIGHBOURS,
        "weighting": "link-weighted neighbour profiles; MOF-equivalent for the MOF family matrix",
        "element_min_presence": args.element_min_presence,
        "kappa": args.kappa,
        "populations": {p: int(len(data[p][0]["n_atoms"])) for p, _, _ in POPULATIONS},
        "weighted_totals": {p: float(weights[p].sum()) for p, _, _ in POPULATIONS},
        "family_affinity": family_payload,
    })
    log("analysis complete")
    return out_root


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--element-min-presence", type=float, default=0.03)
    parser.add_argument("--min-row-mofs", type=float, default=40.0)
    parser.add_argument("--kappa", type=float, default=5.0)
    parser.add_argument("--max-subsets", type=int, default=12)
    parser.add_argument("--trajectory-gap", type=int, default=1000,
                        help="global-row separation treated as a different trajectory")
    parser.add_argument("--report-elements", type=int, default=24,
                        help="elements carried into the markdown element table")
    args = parser.parse_args()
    command_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
