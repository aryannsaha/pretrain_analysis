#!/usr/bin/env python
"""Structural + bonding descriptors for pc25 top-5 OMAT24 neighbours.

The earlier neighbour-chemistry run characterised *composition* (which elements,
which chemical systems).  Composition alone cannot distinguish a porous framework
from a dense binary with the same element list, so this module reads the actual
coordinates and derives bonding-level descriptors:

  coordination   per-atom CN under a covalent-radius bond criterion, split by
                 element class, plus the full CN histogram.
  bond types     metal-metal / metal-nonmetal / nonmetal-nonmetal fractions and
                 the specific C-H, C-C, C-O, C-N, M-O, M-N channels.
  connectivity   number of connected components under PBC, largest-component
                 fraction, and the *dimensionality* (0/1/2/3) of the largest
                 component -- the descriptor that actually separates a molecular
                 crystal from a 3D framework.
  electronics    Pauling electronegativity mean/spread, packing fraction.

Bond criterion: atoms i,j are bonded iff d(i,j) < scale * (r_cov[i] + r_cov[j]).
The neighbour search runs once at scale 1.30 and the primary scale 1.15 is
recovered by filtering, so both are available for one search cost.

Populations are written as one CSV + three count matrices each:
  <pop>.csv           one row per structure, DESCRIPTOR_FIELDS columns
  <pop>_zcounts.npy   (n, 119) int32 per-element atom counts
  <pop>_cnhist.npy    (n,  17) int32 CN histogram, last bin is CN >= 16
  <pop>_cnsum.npy     (n, 119) int32 summed CN over the atoms of each element,
                      so mean CN of element Z = cnsum[:, Z].sum() / zcounts[:, Z].sum()
                      and the same ratio survives arbitrary per-structure weighting

Subcommands:
  extract   read structures and write the tables (parallel over shards/chunks).
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, chemical_symbols, covalent_radii
from ase.db import connect
from ase.neighborlist import natural_cutoffs, neighbor_list

MAX_Z = 118
DEFAULT_ROOT = Path(
    os.environ.get("PRETRAIN_ANALYSIS_ROOT", Path(__file__).resolve().parents[2])
)
DEFAULT_OUT = "runs/omat_knn_probe/pc25_top5_chemistry"

K_NEIGHBOURS = 5
BOND_SCALE = 1.15          # primary bond criterion
BOND_SCALE_WIDE = 1.30     # sensitivity criterion; also the search cutoff
CN_BINS = 17               # 0..15 plus a >=16 overflow bin

MANIFEST = "data/processed/omat24/uma_latents_copy/all_uma_embeddings_manifest.csv"

QUERY_SETS = {
    "mof_off": {
        "indices": "runs/omat_knn_probe/omat_knn_pc25_mof_off/MOF_off_R2SCAN",
        "traj": ["data/processed/mof_off/R2SCAN/R2SCAN_train.traj"],
        "meta_tsv": ["data/processed/mof_off/R2SCAN/R2SCAN_train_sample_ids.tsv"],
        "n_expected": 80643,
    },
    "mad": {
        "indices": "runs/omat_knn_probe/omat_knn_pc25_mad/MAD_all",
        "traj": [
            "data/processed/mad/train/mad_train.traj",
            "data/processed/mad/val/mad_val.traj",
            "data/processed/mad/test/mad_test.traj",
        ],
        "meta_tsv": [
            "data/processed/mad/train/mad_train_frames.tsv",
            "data/processed/mad/val/mad_val_frames.tsv",
            "data/processed/mad/test/mad_test_frames.tsv",
        ],
        "n_expected": 95595,
    },
}

# Standard inorganic/organic split; everything outside this set counts as a
# metal-ish centre.  Matches mof_omat_neighbor_chemistry.NONMETAL_Z so the two
# reports classify the same structure the same way.
NONMETAL_Z = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18,
              33, 34, 35, 36, 52, 53, 54, 85, 86}
METAL_MASK = np.ones(MAX_Z + 1, dtype=bool)
METAL_MASK[list(NONMETAL_Z)] = False
METAL_MASK[0] = False

# Pauling electronegativity by Z; NaN where undefined (noble gases, heavy Z).
_PAULING = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55,
    26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01,
    33: 2.18, 34: 2.55, 35: 2.96, 36: 3.00,
    37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90,
    44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
    51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60,
    55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13,
    62: 1.17, 63: 1.20, 64: 1.20, 65: 1.10, 66: 1.22, 67: 1.23, 68: 1.24,
    69: 1.25, 70: 1.10, 71: 1.27, 72: 1.30, 73: 1.50, 74: 2.36, 75: 1.90,
    76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33,
    83: 2.02, 84: 2.00, 85: 2.20, 86: 2.20,
    87: 0.70, 88: 0.90, 89: 1.10, 90: 1.30, 91: 1.50, 92: 1.38, 93: 1.36,
    94: 1.28, 95: 1.13, 96: 1.28, 97: 1.30, 98: 1.30, 99: 1.30, 100: 1.30,
    101: 1.30, 102: 1.30, 103: 1.30,
}
ELECTRONEGATIVITY = np.full(MAX_Z + 1, np.nan)
for _z, _v in _PAULING.items():
    ELECTRONEGATIVITY[_z] = _v

DESCRIPTOR_FIELDS = [
    # provenance
    "population", "key", "global_row", "subdataset", "shard", "local_row",
    # composition
    "formula_hill", "chemical_system", "n_atoms", "n_elements", "total_z", "pbc",
    # cell / packing
    "volume_ang3", "volume_per_atom", "density_g_cm3", "packing_fraction",
    # energetics (OMAT + MOF-off carry a calculator; MAD does not)
    "energy_ev", "energy_per_atom", "force_max", "force_rms", "pressure_gpa",
    # electronic / elemental
    "en_mean", "en_std", "en_range", "metal_atom_frac", "mean_cov_radius",
    # coordination
    "cn_mean", "cn_mean_wide", "cn_max", "cn_metal_mean", "cn_nonmetal_mean",
    "cn_H_mean", "cn_C_mean", "cn_N_mean", "cn_O_mean",
    "frac_cn0", "frac_cn_le2",
    # bonding
    "n_bonds", "bond_len_mean", "bond_len_min",
    "frac_bond_metal_metal", "frac_bond_metal_nonmetal", "frac_bond_nonmetal_nonmetal",
    "frac_bond_CH", "frac_bond_CC", "frac_bond_CO", "frac_bond_CN",
    "frac_bond_MO", "frac_bond_MN", "frac_C_with_H", "h_per_c",
    # connectivity
    "n_components", "largest_component_frac", "dimensionality",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def atomic_write_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


# --- connectivity ------------------------------------------------------------

def _find(parent, offset, x):
    """Union-find root of x with the accumulated lattice translation x -> root."""
    path = []
    while parent[x] != x:
        path.append(x)
        x = parent[x]
    root = x
    # Second pass compresses the path and folds the translations in.
    total = np.zeros(3, dtype=np.int64)
    for node in reversed(path):
        total = total + offset[node]
        parent[node] = root
        offset[node] = total.copy()
    return root


def connectivity(n_atoms, bond_i, bond_j, bond_shift):
    """(n_components, largest_component_frac, dimensionality of largest component).

    Each atom x carries T[x], the lattice translation of its unrolled image.  A
    bond (i, j, S) -- j sits at ``pos[j] + S @ cell`` relative to i -- constrains
    T[j] = T[i] + S.  When an edge closes a loop inside a component the residual
    ``(T[j] - T[i]) - S`` is a lattice vector along which that component is
    periodic; the rank of those residuals is its dimensionality.
    """
    parent = np.arange(n_atoms)
    offset = np.zeros((n_atoms, 3), dtype=np.int64)
    # (atom, residual) rather than (root, residual): roots are merged away as the
    # scan proceeds, so a residual keyed by the root current at discovery time
    # would be orphaned by any later union.  The atom is stable, so the residual
    # is attributed to its component only after all unions are done.
    cycles = []

    for i, j, shift in zip(bond_i, bond_j, bond_shift):
        if j < i:
            continue  # each undirected bond appears twice in the neighbour list
        ri = _find(parent, offset, i)
        rj = _find(parent, offset, j)
        # offset[x] after _find is T[x] - T[root(x)]
        if ri != rj:
            # attach rj under ri:  T[rj] - T[ri] = offset[i] + S - offset[j]
            parent[rj] = ri
            offset[rj] = offset[i] + shift - offset[j]
        else:
            residual = (offset[j] - offset[i]) - shift
            if np.any(residual):
                cycles.append((int(i), residual))

    roots = np.array([_find(parent, offset, x) for x in range(n_atoms)])
    labels, sizes = np.unique(roots, return_counts=True)
    biggest = labels[sizes.argmax()]
    vectors = [r for atom, r in cycles if roots[atom] == biggest]
    dim = int(np.linalg.matrix_rank(np.array(vectors, dtype=np.float64))) if vectors else 0
    return int(labels.size), float(sizes.max() / n_atoms), dim


# --- descriptors -------------------------------------------------------------

def structure_descriptors(atoms, population, key, extra):
    numbers = atoms.get_atomic_numbers()
    n_atoms = int(len(atoms))
    counts = np.bincount(numbers, minlength=MAX_Z + 1)[: MAX_Z + 1]
    present = np.nonzero(counts)[0]
    is_metal = METAL_MASK[numbers]

    record = {field: None for field in DESCRIPTOR_FIELDS}
    record.update(
        population=population,
        key=key,
        global_row=extra.get("global_row", ""),
        subdataset=extra.get("subdataset", ""),
        shard=extra.get("shard", ""),
        local_row=extra.get("local_row", ""),
        formula_hill=atoms.get_chemical_formula(mode="hill"),
        chemical_system="-".join(chemical_symbols[z] for z in present),
        n_atoms=n_atoms,
        n_elements=int(present.size),
        total_z=int(numbers.sum()),
        pbc="".join("T" if p else "F" for p in atoms.pbc),
        metal_atom_frac=round(float(is_metal.mean()), 6),
        mean_cov_radius=round(float(covalent_radii[numbers].mean()), 6),
    )

    # Volume needs three lattice vectors; MAD carries molecular frames with none.
    cell_rank = int(np.linalg.matrix_rank(atoms.cell.array))
    volume = float(atoms.get_volume()) if cell_rank == 3 else None
    if volume is not None and volume > 0:
        mass = float(atomic_masses[numbers].sum())
        sphere = float((4.0 / 3.0 * np.pi * covalent_radii[numbers] ** 3).sum())
        record.update(
            volume_ang3=round(volume, 6),
            volume_per_atom=round(volume / n_atoms, 6),
            density_g_cm3=round(mass / volume * 1.6605390666, 6),
            packing_fraction=round(sphere / volume, 6),
        )

    en = ELECTRONEGATIVITY[numbers]
    finite = en[np.isfinite(en)]
    if finite.size:
        record.update(
            en_mean=round(float(finite.mean()), 6),
            en_std=round(float(finite.std()), 6),
            en_range=round(float(finite.max() - finite.min()), 6),
        )

    calc = atoms.calc
    results = calc.results if calc is not None else {}
    if "energy" in results:
        energy = float(results["energy"])
        record["energy_ev"] = energy
        record["energy_per_atom"] = energy / n_atoms
    if "forces" in results:
        forces = np.asarray(results["forces"], dtype=np.float64)
        magnitudes = np.linalg.norm(forces, axis=1)
        record["force_max"] = float(magnitudes.max()) if magnitudes.size else None
        record["force_rms"] = float(np.sqrt((forces**2).sum() / max(forces.size, 1)))
    if "stress" in results:
        stress = np.asarray(results["stress"], dtype=np.float64).ravel()
        if stress.size >= 3:
            # ASE stress is eV/A^3 in Voigt order; positive pressure = compression.
            record["pressure_gpa"] = float(-stress[:3].mean() * 160.21766208)

    # One neighbour search at the wide cutoff serves both bond scales.
    cn_hist = np.zeros(CN_BINS, dtype=np.int32)
    idx_i, idx_j, dists, shifts = neighbor_list(
        "ijdS", atoms, natural_cutoffs(atoms, mult=BOND_SCALE_WIDE)
    )
    radii_sum = covalent_radii[numbers[idx_i]] + covalent_radii[numbers[idx_j]]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(radii_sum > 0, dists / radii_sum, np.inf)
    keep = ratio < BOND_SCALE

    cn_wide = np.bincount(idx_i, minlength=n_atoms) if idx_i.size else np.zeros(n_atoms, int)
    record["cn_mean_wide"] = round(float(cn_wide.mean()), 6)

    bi, bj, bd, bs = idx_i[keep], idx_j[keep], dists[keep], shifts[keep]
    cn = np.bincount(bi, minlength=n_atoms) if bi.size else np.zeros(n_atoms, dtype=int)
    np.add.at(cn_hist, np.minimum(cn, CN_BINS - 1), 1)
    cn_sum = np.bincount(numbers, weights=cn, minlength=MAX_Z + 1)[: MAX_Z + 1]

    record.update(
        cn_mean=round(float(cn.mean()), 6),
        cn_max=int(cn.max()),
        frac_cn0=round(float((cn == 0).mean()), 6),
        frac_cn_le2=round(float((cn <= 2).mean()), 6),
        n_bonds=int(bi.size // 2),
    )
    if is_metal.any():
        record["cn_metal_mean"] = round(float(cn[is_metal].mean()), 6)
    if (~is_metal).any():
        record["cn_nonmetal_mean"] = round(float(cn[~is_metal].mean()), 6)
    for symbol, z in (("H", 1), ("C", 6), ("N", 7), ("O", 8)):
        mask = numbers == z
        if mask.any():
            record[f"cn_{symbol}_mean"] = round(float(cn[mask].mean()), 6)

    if bi.size:
        record["bond_len_mean"] = round(float(bd.mean()), 6)
        record["bond_len_min"] = round(float(bd.min()), 6)
        zi, zj = numbers[bi], numbers[bj]
        mi, mj = METAL_MASK[zi], METAL_MASK[zj]
        n_dir = float(bi.size)

        def pair_frac(mask):
            return round(float(mask.sum() / n_dir), 6)

        def elem_pair(za, zb):
            return ((zi == za) & (zj == zb)) | ((zi == zb) & (zj == za))

        record.update(
            frac_bond_metal_metal=pair_frac(mi & mj),
            frac_bond_metal_nonmetal=pair_frac(mi ^ mj),
            frac_bond_nonmetal_nonmetal=pair_frac(~mi & ~mj),
            frac_bond_CH=pair_frac(elem_pair(6, 1)),
            frac_bond_CC=pair_frac((zi == 6) & (zj == 6)),
            frac_bond_CO=pair_frac(elem_pair(6, 8)),
            frac_bond_CN=pair_frac(elem_pair(6, 7)),
            frac_bond_MO=pair_frac(mi & (zj == 8)),
            frac_bond_MN=pair_frac(mi & (zj == 7)),
        )
        if counts[6]:
            ch = (zi == 6) & (zj == 1)
            record["h_per_c"] = round(float(ch.sum() / counts[6]), 6)
            record["frac_C_with_H"] = round(
                float(np.unique(bi[ch]).size / counts[6]), 6
            )
        n_comp, largest, dim = connectivity(n_atoms, bi, bj, bs)
    else:
        if counts[6]:
            record["h_per_c"] = 0.0
            record["frac_C_with_H"] = 0.0
        n_comp, largest, dim = n_atoms, 1.0 / n_atoms, 0

    record.update(
        n_components=n_comp, largest_component_frac=round(largest, 6), dimensionality=dim
    )
    return record, counts.astype(np.int32), cn_hist, cn_sum.astype(np.int32)


# --- population readers ------------------------------------------------------

def read_manifest(root):
    with (root / MANIFEST).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty manifest: {root / MANIFEST}")
    starts = np.array([int(r["global_start"]) for r in rows], dtype=np.int64)
    stops = np.array([int(r["global_stop"]) for r in rows], dtype=np.int64)
    if not np.all(starts[1:] == stops[:-1]):
        raise SystemExit("manifest shards are not contiguous")
    return rows, starts


def locate(global_rows, manifest, starts):
    manifest_index = np.searchsorted(starts, global_rows, side="right") - 1
    local_row = global_rows - starts[manifest_index]
    n_rows = np.array([int(r["n_rows"]) for r in manifest], dtype=np.int64)
    if np.any(local_row < 0) or np.any(local_row >= n_rows[manifest_index]):
        raise SystemExit("global row fell outside its manifest shard")
    return manifest_index, local_row


_WORKER = {}


def _init_worker(root, population):
    _WORKER["root"] = Path(root)
    _WORKER["population"] = population
    _WORKER["manifest"], _WORKER["starts"] = read_manifest(Path(root))


def _omat_shard_task(payload):
    shard_idx, global_rows = payload
    root, manifest = _WORKER["root"], _WORKER["manifest"]
    entry = manifest[shard_idx]
    local = np.asarray(global_rows, dtype=np.int64) - int(entry["global_start"])
    out = ([], [], [], [])
    with connect(root / entry["raw_path"], readonly=True, use_lock_file=False) as db:
        ids = db.ids
        if len(ids) != int(entry["n_rows"]):
            raise SystemExit(
                f"shard {entry['raw_path']} has {len(ids)} rows, "
                f"manifest says {entry['n_rows']}"
            )
        for lr, gr in zip(local, global_rows):
            atoms = db.get(id=ids[int(lr)]).toatoms()
            for bucket, value in zip(out, structure_descriptors(
                atoms, _WORKER["population"], str(int(gr)),
                {
                    "global_row": int(gr),
                    "subdataset": entry["subdataset"],
                    "shard": entry["shard"],
                    "local_row": int(lr),
                },
            )):
                bucket.append(value)
    return out


def _traj_chunk_task(payload):
    from ase.io import Trajectory

    traj_path, start, stop, meta_rows, row_offset = payload
    out = ([], [], [], [])
    traj = Trajectory(traj_path)
    try:
        for i in range(start, stop):
            meta = meta_rows[i - start]
            for bucket, value in zip(out, structure_descriptors(
                traj[i], _WORKER["population"], meta["key"],
                {
                    "global_row": row_offset + i,
                    "subdataset": meta["subdataset"],
                    "shard": meta["shard"],
                    "local_row": meta["local_row"],
                },
            )):
                bucket.append(value)
    finally:
        traj.close()
    return out


def run_tasks(tasks, root, population, workers, label):
    out = ([], [], [], [])
    total = len(tasks)
    task_fn = _omat_shard_task if label != "traj" else _traj_chunk_task
    with Pool(workers, initializer=_init_worker, initargs=(str(root), population)) as pool:
        for n, chunk in enumerate(pool.imap(task_fn, tasks, chunksize=1), 1):
            for bucket, values in zip(out, chunk):
                bucket.extend(values)
            if n % 25 == 0 or n == total:
                log(f"  {population}: chunk {n:,}/{total:,} ({len(out[0]):,} structures)")
    return out


def write_population(out_dir, population, extracted, sort_key=None):
    records, zcounts, cnhists, cnsums = extracted
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort([sort_key(r) for r in records], kind="stable") if sort_key \
        else np.arange(len(records))

    csv_path = out_dir / f"{population}.csv"
    tmp = csv_path.with_name(csv_path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DESCRIPTOR_FIELDS)
        writer.writeheader()
        writer.writerows(records[i] for i in order)
    os.replace(tmp, csv_path)
    for suffix, values in (("zcounts", zcounts), ("cnhist", cnhists), ("cnsum", cnsums)):
        atomic_write_npy(
            out_dir / f"{population}_{suffix}.npy",
            np.asarray(values, dtype=np.int32)[order],
        )
    log(f"wrote {population}: {len(records):,} structures -> {csv_path}")


def read_query_metadata(root, query):
    """Per-frame metadata rows for a query set, concatenated across splits."""
    spec = QUERY_SETS[query]
    rows = []
    for tsv in spec["meta_tsv"]:
        with (root / tsv).open() as handle:
            rows.append(list(csv.DictReader(handle, delimiter="\t")))
    if query == "mof_off":
        meta = [
            {
                "key": r["record_id"],
                "subdataset": r["cif_name"].split("_")[0],
                "shard": r["temperature_K"],
                "local_row": r["frame"],
            }
            for r in rows[0]
        ]
    else:
        meta = [
            {
                "key": f"{r['split']}:{r['frame']}",
                "subdataset": r["subset"],
                "shard": r["split"],
                "local_row": r["frame"],
            }
            for split in rows for r in split
        ]
    return rows, meta


def command_extract(args):
    root = args.root.resolve()
    out_dir = (args.out or root / DEFAULT_OUT).resolve() / "structures"
    manifest, starts = read_manifest(root)
    spec = QUERY_SETS[args.query]

    idx = np.load(root / spec["indices"] / "indices.npy")[:, :K_NEIGHBOURS]
    if idx.shape[0] != spec["n_expected"]:
        raise SystemExit(f"{args.query}: {idx.shape[0]} queries, expected {spec['n_expected']}")
    union = np.unique(idx)
    log(f"{args.query}: {idx.shape[0]:,} queries x {K_NEIGHBOURS} pc25 neighbours, "
        f"{union.size:,} unique OMAT rows")

    if not args.skip_neighbors:
        population = f"omat_nb_{args.query}"
        manifest_index, _ = locate(union, manifest, starts)
        tasks = [
            (int(s), union[manifest_index == s])
            for s in np.unique(manifest_index)
        ]
        log(f"{population}: {len(tasks):,} shards")
        write_population(
            out_dir, population,
            run_tasks(tasks, root, population, args.workers, "omat"),
            sort_key=lambda r: r["global_row"],
        )

    if not args.skip_queries:
        population = f"query_{args.query}"
        split_rows, meta = read_query_metadata(root, args.query)
        if len(meta) != spec["n_expected"]:
            raise SystemExit(f"{args.query}: metadata has {len(meta)} rows")
        tasks, offset = [], 0
        for traj_path, rows in zip(spec["traj"], split_rows):
            n = len(rows)
            for start in range(0, n, args.chunk):
                stop = min(start + args.chunk, n)
                tasks.append(
                    (str(root / traj_path), start, stop,
                     meta[offset + start:offset + stop], offset)
                )
            offset += n
        log(f"{population}: {len(tasks):,} chunks over {len(spec['traj'])} trajectory file(s)")
        write_population(
            out_dir, population,
            run_tasks(tasks, root, population, args.workers, "traj"),
            sort_key=lambda r: r["global_row"],
        )

    if args.background:
        rng = np.random.default_rng(args.seed)
        total_rows = int(starts[-1]) + int(manifest[-1]["n_rows"])
        plans = {
            "omat_bg_global": np.sort(rng.choice(total_rows, args.background, replace=False)),
        }
        nvt = [i for i, r in enumerate(manifest) if r["subdataset"] == "aimd-from-PBE-3000-nvt"]
        nvt_stops = np.cumsum([int(manifest[i]["n_rows"]) for i in nvt])
        offsets = np.sort(rng.choice(int(nvt_stops[-1]), args.background, replace=False))
        pos = np.searchsorted(nvt_stops, offsets, side="right")
        prev = np.concatenate([[0], nvt_stops[:-1]])
        plans["omat_bg_nvt3000"] = np.sort(np.array(
            [int(manifest[nvt[int(s)]]["global_start"]) + int(o - prev[int(s)])
             for o, s in zip(offsets, pos)], dtype=np.int64))

        for population, picks in plans.items():
            if (out_dir / f"{population}.csv").exists() and not args.force:
                log(f"{population}: exists, skipping (use --force to redo)")
                continue
            manifest_index, _ = locate(picks, manifest, starts)
            tasks = [(int(s), picks[manifest_index == s]) for s in np.unique(manifest_index)]
            log(f"{population}: {len(tasks):,} shards")
            write_population(
                out_dir, population,
                run_tasks(tasks, root, population, args.workers, "omat"),
                sort_key=lambda r: r["global_row"],
            )

    atomic_write_json(
        (args.out or root / DEFAULT_OUT).resolve() / f"extract_metadata_{args.query}.json",
        {
            "created_utc": utc_now(),
            "query": args.query,
            "geometry": "pc25",
            "k_neighbours": K_NEIGHBOURS,
            "indices": spec["indices"],
            "n_queries": int(idx.shape[0]),
            "n_unique_neighbours": int(union.size),
            "bond_scale": BOND_SCALE,
            "bond_scale_wide": BOND_SCALE_WIDE,
            "background_rows": args.background,
            "seed": args.seed,
            "workers": args.workers,
        },
    )
    log("extract complete")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="read structures and write descriptor tables")
    extract.add_argument("--query", choices=sorted(QUERY_SETS), required=True)
    extract.add_argument("--workers", type=int, default=32)
    extract.add_argument("--chunk", type=int, default=2000,
                         help="trajectory frames per worker task")
    extract.add_argument("--background", type=int, default=0,
                         help="rows per OMAT24 baseline (0 disables)")
    extract.add_argument("--seed", type=int, default=20260804)
    extract.add_argument("--skip-neighbors", action="store_true")
    extract.add_argument("--skip-queries", action="store_true")
    extract.add_argument("--force", action="store_true",
                         help="re-extract baselines even if present")
    extract.set_defaults(func=command_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
