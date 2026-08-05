#!/usr/bin/env python
"""Controls for the MOF-off -> OMAT24 retrieval: is the embedding doing more than stoichiometry?

Three questions that have to be answered before any chemical claim about the
retrieved neighbours can be trusted:

1. COMPOSITION BASELINE.  If a trivial nearest neighbour in composition space
   returns the same OMAT24 structures, the embedding is not contributing
   anything a fractional element vector does not already give.  Composition is
   constant along an AIMD trajectory, so the composition side is computed once
   per distinct MOF and compared against every frame's embedding retrieval.
   The sum-pooled embedding is also extensive, so ||x|| is regressed on atom
   count and total Z to size that artifact directly.

2. DISTANCE CALIBRATION.  A rank-1 neighbour always exists.  The question is
   whether its distance is small on the scale OMAT24 sets for itself, which is
   what the 1M-row leave-one-out calibration provides.  That baseline is
   dominated by within-trajectory near-duplicates, so a trajectory-excluded
   variant is computed alongside it -- otherwise "MOF-off is far" merely
   restates "MOF-off has no AIMD twin in the reference set".

3. PC INTERPRETATION.  Each retained principal component is regressed against
   physical descriptors so the geometry the kNN runs in can be named rather
   than treated as opaque.

Every section writes a CSV; ``--report`` renders the markdown tables.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pc25_top5_structure_descriptors import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_ROOT,
    K_NEIGHBOURS,
    MAX_Z,
    METAL_MASK,
    atomic_write_json,
    atomic_write_npy,
    log,
    utc_now,
)
from pc25_top5_chemistry_report import load_population, md_table, write_csv  # noqa: E402

PCA_STATE = "runs/uma_latents_copy_full_pca/uma_100824585_rapids_pca_state_11212254.npz"
# index_configuration.json records this under runs/omat_knn_pc25/, which is where
# it was written before the probe outputs were reorganised; the live copy is here.
OMAT_PC25 = "runs/omat_knn_probe/omat_knn_pc25/omat_pc25_unweighted.npy"
CALIBRATION = "runs/omat_knn_probe/omat_knn_pc25/OMAT_calibration"
MOF_LATENTS = "data/processed/mof_off/uma_latents/R2SCAN/train.npy"
MOF_INDICES = "runs/omat_knn_probe/omat_knn_pc25_mof_off/MOF_off_R2SCAN"

# Descriptors regressed against each principal component.
PC_DESCRIPTORS = [
    "n_atoms", "total_z", "density_g_cm3", "packing_fraction", "volume_per_atom",
    "en_mean", "en_range", "metal_atom_frac", "cn_mean", "energy_per_atom",
    "mean_cov_radius",
]


def project_pc25(latents, state):
    """Raw 128-d UMA vectors -> the 25 standardised PC scores the kNN runs use."""
    return ((latents - state["mean"]) / state["scale"]) @ state["components"].T


def fractional_composition(zcounts):
    totals = zcounts.sum(axis=1, keepdims=True)
    return zcounts / np.maximum(totals, 1)


def cosine_topk(queries, reference, k, block=512):
    """Top-k cosine-nearest reference rows for each query row."""
    q = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    r = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
    out_idx = np.empty((q.shape[0], k), dtype=np.int64)
    out_sim = np.empty((q.shape[0], k), dtype=np.float32)
    for start in range(0, q.shape[0], block):
        stop = min(start + block, q.shape[0])
        similarity = q[start:stop] @ r.T
        part = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        order = np.argsort(-similarity[rows, part], axis=1)
        out_idx[start:stop] = part[rows, order]
        out_sim[start:stop] = similarity[rows, part[rows, order]]
    return out_idx, out_sim


def euclidean_topk(queries, reference, k, block=512):
    out_idx = np.empty((queries.shape[0], k), dtype=np.int64)
    out_dist = np.empty((queries.shape[0], k), dtype=np.float32)
    r2 = (reference**2).sum(axis=1)
    for start in range(0, queries.shape[0], block):
        stop = min(start + block, queries.shape[0])
        chunk = queries[start:stop]
        d2 = (chunk**2).sum(axis=1)[:, None] + r2[None, :] - 2.0 * chunk @ reference.T
        np.maximum(d2, 0, out=d2)
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        order = np.argsort(d2[rows, part], axis=1)
        out_idx[start:stop] = part[rows, order]
        out_dist[start:stop] = np.sqrt(d2[rows, part[rows, order]])
    return out_idx, out_dist


def r_squared(x, y):
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return np.nan, np.nan
    r = float(np.corrcoef(x[finite], y[finite])[0, 1])
    return r, r * r


def command_run(args):
    root = args.root.resolve()
    out_root = (args.out or root / DEFAULT_OUT).resolve()
    structures = out_root / "structures"
    analysis = out_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    state = np.load(root / PCA_STATE)
    state = {k: state[k] for k in ("mean", "scale", "components",
                                   "explained_variance_ratio", "retained_variance_ratio")}

    log("loading extracted populations")
    mof_cols, mof_zc, _, _ = load_population(structures, "query_mof_off")
    nb_cols, nb_zc, _, _ = load_population(structures, "omat_nb_mof_off")
    bg_pops = {name: load_population(structures, name)
               for name in ("omat_bg_nvt3000", "omat_bg_global")}

    nb_rows = np.array([int(v) for v in nb_cols["global_row"]], dtype=np.int64)
    idx = np.load(root / MOF_INDICES / "indices.npy")[:, :K_NEIGHBOURS]
    dist = np.load(root / MOF_INDICES / "distances.npy")[:, :K_NEIGHBOURS]
    n_queries = len(mof_cols["n_atoms"])

    mof_latents = np.load(root / MOF_LATENTS)
    if mof_latents.shape[0] != n_queries:
        raise SystemExit(f"MOF latents {mof_latents.shape} vs {n_queries} frames")
    mof_pc = project_pc25(mof_latents.astype(np.float64), state)

    omat_pc = np.load(root / OMAT_PC25, mmap_mode="r")

    # Gate: the locally projected MOF scores must reproduce the stored kNN
    # distances, otherwise every pc25 number below is measured in the wrong frame.
    sample = np.linspace(0, n_queries - 1, 400, dtype=np.int64)
    neighbour_pc = np.asarray(omat_pc[idx[sample, 0]], dtype=np.float64)
    recomputed = np.linalg.norm(mof_pc[sample] - neighbour_pc, axis=1)
    projection_error = float(np.abs(recomputed - dist[sample, 0]).max())
    log(f"pc25 projection gate: max |recomputed d1 - stored d1| = {projection_error:.3e}")
    if projection_error > args.projection_tolerance:
        raise SystemExit(
            f"pc25 projection mismatch {projection_error:.3e} exceeds "
            f"{args.projection_tolerance:.1e}; the PCA state does not match the kNN run"
        )

    # --- 1. composition baseline --------------------------------------------
    log("section 1: composition baseline")
    pool_rows = np.concatenate([nb_rows] + [
        np.array([int(v) for v in bg_pops[name][0]["global_row"]], dtype=np.int64)
        for name in bg_pops
    ])
    pool_zc = np.concatenate([nb_zc] + [bg_pops[name][1] for name in bg_pops])
    pool_rows, unique_pos = np.unique(pool_rows, return_index=True)
    pool_zc = pool_zc[unique_pos]
    pool_comp = fractional_composition(pool_zc.astype(np.float64))
    pool_pc = np.asarray(omat_pc[pool_rows], dtype=np.float64)
    log(f"  shared pool: {pool_rows.size:,} OMAT structures with composition + pc25")

    # Composition is constant along a trajectory, so work per distinct MOF.
    mof_id = mof_cols["subdataset"]
    _, first_frame, inverse = np.unique(mof_id, return_index=True, return_inverse=True)
    mof_comp = fractional_composition(mof_zc[first_frame].astype(np.float64))
    log(f"  {first_frame.size:,} distinct MOFs, {n_queries:,} frames")

    comp_idx, comp_sim = cosine_topk(mof_comp, pool_comp, K_NEIGHBOURS)
    comp_neighbour_rows = pool_rows[comp_idx]                       # per distinct MOF

    # Pool-restricted embedding retrieval makes the two searches see the same
    # candidate set; the unrestricted run searched all 100.8M rows.
    emb_pool_idx, _ = euclidean_topk(mof_pc, pool_pc, K_NEIGHBOURS)
    emb_pool_rows = pool_rows[emb_pool_idx]                          # per frame

    true_rows = idx                                                  # per frame
    comp_rows_per_frame = comp_neighbour_rows[inverse]

    def overlap(a, b, k):
        return np.array([
            len(set(x[:k].tolist()) & set(y[:k].tolist())) / k for x, y in zip(a, b)
        ])

    overlap_true = overlap(true_rows, comp_rows_per_frame, K_NEIGHBOURS)
    overlap_pool = overlap(emb_pool_rows, comp_rows_per_frame, K_NEIGHBOURS)
    overlap_true_1 = overlap(true_rows, comp_rows_per_frame, 1)
    overlap_pool_1 = overlap(emb_pool_rows, comp_rows_per_frame, 1)

    # Composition distance the embedding actually achieves, against the
    # composition distance of a random OMAT structure.
    frame_comp = mof_comp[inverse]
    frame_comp_n = frame_comp / np.maximum(
        np.linalg.norm(frame_comp, axis=1, keepdims=True), 1e-12)
    pool_comp_n = pool_comp / np.maximum(
        np.linalg.norm(pool_comp, axis=1, keepdims=True), 1e-12)
    pool_lookup = {int(r): i for i, r in enumerate(pool_rows)}
    true_pool_pos = np.array([[pool_lookup[int(v)] for v in row] for row in true_rows])
    cos_to_true = (frame_comp_n[:, None, :] * pool_comp_n[true_pool_pos]).sum(axis=2)
    rng = np.random.default_rng(args.seed)
    random_pos = rng.integers(0, pool_rows.size, size=(n_queries, K_NEIGHBOURS))
    cos_to_random = (frame_comp_n[:, None, :] * pool_comp_n[random_pos]).sum(axis=2)
    best_possible = comp_sim[inverse]

    composition_rows = [
        {"quantity": "overlap@5, true embedding top-5 vs composition top-5",
         "mean": round(float(overlap_true.mean()), 4),
         "median": round(float(np.median(overlap_true)), 4),
         "frac_nonzero": round(float((overlap_true > 0).mean()), 4)},
        {"quantity": "overlap@5, pool-restricted embedding vs composition",
         "mean": round(float(overlap_pool.mean()), 4),
         "median": round(float(np.median(overlap_pool)), 4),
         "frac_nonzero": round(float((overlap_pool > 0).mean()), 4)},
        {"quantity": "overlap@1, true embedding rank-1 vs composition rank-1",
         "mean": round(float(overlap_true_1.mean()), 4), "median": "",
         "frac_nonzero": round(float((overlap_true_1 > 0).mean()), 4)},
        {"quantity": "overlap@1, pool-restricted embedding vs composition",
         "mean": round(float(overlap_pool_1.mean()), 4), "median": "",
         "frac_nonzero": round(float((overlap_pool_1 > 0).mean()), 4)},
        {"quantity": "composition cosine to embedding top-5 (higher = more similar)",
         "mean": round(float(cos_to_true.mean()), 4),
         "median": round(float(np.median(cos_to_true)), 4), "frac_nonzero": ""},
        {"quantity": "composition cosine to random pool structures",
         "mean": round(float(cos_to_random.mean()), 4),
         "median": round(float(np.median(cos_to_random)), 4), "frac_nonzero": ""},
        {"quantity": "composition cosine to the composition-optimal top-5",
         "mean": round(float(best_possible.mean()), 4),
         "median": round(float(np.median(best_possible)), 4), "frac_nonzero": ""},
    ]
    write_csv(analysis / "composition_baseline.csv",
              list(composition_rows[0]), composition_rows)

    # Sum-pooling artifact: ||x|| against extensive quantities.
    norm_rows = []
    mof_norm = np.linalg.norm(mof_latents.astype(np.float64), axis=1)
    embeddings = np.load(root / "data/processed/omat24/uma_latents_copy/all_uma_embeddings.npy",
                         mmap_mode="r")
    for label, norms, cols in (
        ("MOF-off R2SCAN", mof_norm, mof_cols),
        ("OMAT24 pc25 top-5 neighbours",
         np.linalg.norm(np.asarray(embeddings[nb_rows], dtype=np.float64), axis=1), nb_cols),
    ):
        for field in ("n_atoms", "total_z"):
            r, r2 = r_squared(norms, cols[field])
            norm_rows.append({"population": label, "against": field,
                              "pearson_r": round(r, 4), "r_squared": round(r2, 4)})
    for name in bg_pops:
        cols, _, _, _ = bg_pops[name]
        rows = np.array([int(v) for v in cols["global_row"]], dtype=np.int64)
        norms = np.linalg.norm(np.asarray(embeddings[rows], dtype=np.float64), axis=1)
        for field in ("n_atoms", "total_z"):
            r, r2 = r_squared(norms, cols[field])
            norm_rows.append({"population": name, "against": field,
                              "pearson_r": round(r, 4), "r_squared": round(r2, 4)})
    write_csv(analysis / "embedding_norm_extensivity.csv",
              list(norm_rows[0]), norm_rows)

    # --- 2. distance calibration --------------------------------------------
    log("section 2: distance calibration")
    calib_d = np.load(root / CALIBRATION / "distances.npy")
    calib_i = np.load(root / CALIBRATION / "indices.npy")
    calib_src = np.load(root / CALIBRATION / "source_rows.npy")

    # OMAT's own nearest neighbour is usually the adjacent frame of the same AIMD
    # trajectory.  Excluding rows within `--trajectory-gap` of the source gives the
    # distance to a genuinely different material, which is the honest yardstick.
    gap = args.trajectory_gap
    far = np.abs(calib_i - calib_src[:, None]) > gap
    has_far = far.any(axis=1)
    first_far = np.argmax(far, axis=1)
    calib_far = calib_d[np.arange(calib_d.shape[0]), first_far][has_far]

    quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    calibration_rows = []
    for label, values in (
        ("OMAT24 leave-one-out NN (all)", calib_d[:, 0]),
        (f"OMAT24 leave-one-out NN (>{gap} rows away)", calib_far),
        ("MOF-off R2SCAN d1", dist[:, 0]),
        ("MOF-off R2SCAN d5", dist[:, K_NEIGHBOURS - 1]),
    ):
        row = {"population": label, "n": int(values.size)}
        row.update({f"p{q}": round(float(np.percentile(values, q)), 4) for q in quantiles})
        calibration_rows.append(row)
    write_csv(analysis / "distance_calibration.csv",
              list(calibration_rows[0]), calibration_rows)

    reference_all = np.sort(calib_d[:, 0])
    reference_far = np.sort(calib_far)
    percentile_all = np.searchsorted(reference_all, dist[:, 0]) / reference_all.size
    percentile_far = np.searchsorted(reference_far, dist[:, 0]) / reference_far.size
    stored = np.load(root / MOF_INDICES / "omat_novelty_percentile_d1.npy")
    calibration_summary = {
        "trajectory_gap_rows": gap,
        "calibration_rows": int(calib_d.shape[0]),
        "fraction_with_far_neighbour_in_top10": round(float(has_far.mean()), 5),
        "mof_d1_median_percentile_vs_all": round(float(np.median(percentile_all)), 5),
        "mof_d1_median_percentile_vs_far": round(float(np.median(percentile_far)), 5),
        "mof_frac_above_omat_p95_all": round(float((dist[:, 0] > np.percentile(calib_d[:, 0], 95)).mean()), 5),
        "mof_frac_above_omat_p95_far": round(float((dist[:, 0] > np.percentile(calib_far, 95)).mean()), 5),
        "stored_percentile_median": round(float(np.median(stored)), 5),
        "stored_vs_recomputed_max_abs_diff": round(
            float(np.abs(stored / 100.0 - percentile_all).max()), 5),
    }
    atomic_write_npy(analysis / "mof_d1_percentile_vs_far_baseline.npy",
                     percentile_far.astype(np.float32))

    # --- 3. PC interpretation ------------------------------------------------
    log("section 3: PC interpretation")
    pc_rows = []
    populations = {
        "OMAT24 bg (all)": bg_pops["omat_bg_global"],
        "OMAT24 bg (nvt-3000)": bg_pops["omat_bg_nvt3000"],
    }
    for label, (cols, zc, _, _) in populations.items():
        rows = np.array([int(v) for v in cols["global_row"]], dtype=np.int64)
        scores = np.asarray(omat_pc[rows], dtype=np.float64)
        extra = {
            "C_fraction": zc[:, 6] / np.maximum(zc.sum(axis=1), 1),
            "mean_atomic_number": (zc * np.arange(MAX_Z + 1)[None, :]).sum(axis=1)
            / np.maximum(zc.sum(axis=1), 1),
        }
        for component in range(args.n_components):
            best, best_r2, best_r = None, -1.0, np.nan
            entry = {"population": label, "PC": component + 1,
                     "explained_variance_ratio":
                         round(float(state["retained_variance_ratio"][component]), 5)}
            for field in PC_DESCRIPTORS + list(extra):
                values = extra[field] if field in extra else cols[field]
                r, r2 = r_squared(scores[:, component], np.asarray(values, dtype=np.float64))
                entry[f"r2_{field}"] = round(r2, 4) if np.isfinite(r2) else ""
                if np.isfinite(r2) and r2 > best_r2:
                    best, best_r2, best_r = field, r2, r
            entry["best_descriptor"] = best
            entry["best_r2"] = round(best_r2, 4)
            entry["best_r"] = round(best_r, 4)
            pc_rows.append(entry)
    write_csv(analysis / "pc_interpretation.csv", list(pc_rows[0]), pc_rows)

    atomic_write_json(analysis / "composition_baseline_metadata.json", {
        "created_utc": utc_now(),
        "geometry": "pc25",
        "k_neighbours": K_NEIGHBOURS,
        "pool_structures": int(pool_rows.size),
        "distinct_mofs": int(first_frame.size),
        "frames": int(n_queries),
        "pc25_projection_max_abs_error": projection_error,
        "calibration": calibration_summary,
        "seed": args.seed,
    })

    if args.report:
        blocks = ["# Composition baseline, distance calibration, PC interpretation\n"]
        blocks.append("\n## Composition baseline\n\n" + md_table(
            ["quantity", "mean", "median", "fraction > 0"],
            [[r["quantity"], r["mean"], r["median"], r["frac_nonzero"]]
             for r in composition_rows]))
        blocks.append("\n## Embedding norm vs extensive quantities\n\n" + md_table(
            ["population", "against", "Pearson r", "R2"],
            [[r["population"], r["against"], r["pearson_r"], r["r_squared"]]
             for r in norm_rows]))
        blocks.append("\n## Distance calibration (pc25 Euclidean)\n\n" + md_table(
            ["population", "n"] + [f"p{q}" for q in quantiles],
            [[r["population"], f'{r["n"]:,}'] + [r[f"p{q}"] for q in quantiles]
             for r in calibration_rows]))
        blocks.append("\n```\n" + json.dumps(calibration_summary, indent=2) + "\n```\n")
        blocks.append("\n## Principal component interpretation\n\n" + md_table(
            ["population", "PC", "variance ratio", "best descriptor", "R2", "r"],
            [[r["population"], r["PC"], r["explained_variance_ratio"],
              r["best_descriptor"], r["best_r2"], r["best_r"]] for r in pc_rows]))
        (analysis / "tables_composition_calibration_pcs.md").write_text(
            "\n".join(blocks) + "\n")
        log("wrote tables_composition_calibration_pcs.md")
    log("done")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--trajectory-gap", type=int, default=1000,
                        help="global-row separation treated as a different trajectory")
    parser.add_argument("--projection-tolerance", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    command_run(args)


if __name__ == "__main__":
    sys.exit(main())
