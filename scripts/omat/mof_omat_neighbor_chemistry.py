#!/usr/bin/env python
"""Chemistry of the OMAT24 structures nearest to MOF-off R2SCAN-D4 train rows.

The kNN runs under ``runs/omat_knn_probe`` answer *how far* each MOF-off row is
from OMAT24 in UMA latent space.  They do not say *what* the retrieved OMAT24
structures are.  This script reads the actual retrieved structures out of the
OMAT24 ``.aselmdb`` shards and characterises their chemistry: element content,
chemical systems, reduced formulas, atom counts, cell/density, and energetics.

Two neighbor geometries are analysed side by side, because they retrieve almost
disjoint neighbor sets (see ``runs/omat_knn_probe/omat_knn_raw128_mof_off``):

  raw128  Euclidean distance on the raw 128-d UMA embeddings.
  pc25    Euclidean distance after OMAT-fitted standardisation + 25-PC PCA.

Two OMAT24 baselines are extracted so that "what is special about the neighbor
set" can be separated from "what OMAT24 looks like anyway":

  bg_global    uniform random rows over all 100,824,585 manifest rows.
  bg_nvt3000   uniform random rows over ``aimd-from-PBE-3000-nvt`` only, which
               is the single subdataset that supplies 100% of the neighbors.

Subcommands:
  extract   read structures for the neighbor union, both baselines, and the
            MOF-off queries; write per-structure tables + per-Z count matrices.
  validate  check the atoms<->embedding-row mapping via the extensivity of the
            UMA embedding (||x|| against total Z), plus schema/consistency gates.
  analyze   aggregate everything into CSV tables and a markdown report.

Row-mapping contract (verified by ``validate``): global row ``g`` belongs to the
manifest row with ``global_start <= g < global_stop``; ``local_row = g -
global_start``; within a shard the ASE db ids are contiguous and ordered, so the
structure is ``db.get(id=db.ids[local_row])``.
"""

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, chemical_symbols
from ase.db import connect

MAX_Z = 118
DEFAULT_ROOT = Path(
    os.environ.get("PRETRAIN_ANALYSIS_ROOT", Path(__file__).resolve().parents[2])
)
GEOMETRIES = {
    "raw128": "runs/omat_knn_probe/omat_knn_raw128_mof_off/MOF_off_R2SCAN",
    "pc25": "runs/omat_knn_probe/omat_knn_pc25_mof_off/MOF_off_R2SCAN",
}
MANIFEST = "data/processed/omat24/uma_latents_copy/all_uma_embeddings_manifest.csv"
EMBEDDINGS = "data/processed/omat24/uma_latents_copy/all_uma_embeddings.npy"
MOF_TRAJ = "data/processed/mof_off/R2SCAN/R2SCAN_train.traj"
MOF_IDS = "data/processed/mof_off/R2SCAN/R2SCAN_train_sample_ids.tsv"
MOF_LATENTS = "data/processed/mof_off/uma_latents/R2SCAN/train.npy"
DEFAULT_OUT = "runs/omat_knn_probe/mof_omat_neighbor_chemistry"

STRUCTURE_FIELDS = [
    "population", "key", "global_row", "subdataset", "shard", "local_row",
    "formula_hill", "chemical_system", "n_atoms", "n_elements", "total_z",
    "volume_ang3", "volume_per_atom", "density_g_cm3", "energy_ev",
    "energy_per_atom", "force_max", "force_rms", "pressure_gpa",
]

# Element-class taxonomy used for the "what type of atoms" breakdown.  Sets are
# by atomic number so membership tests stay vectorisable.
ELEMENT_CLASSES = {
    "H": [1],
    "organic_CHNO": [1, 6, 7, 8],
    "alkali": [3, 11, 19, 37, 55, 87],
    "alkaline_earth": [4, 12, 20, 38, 56, 88],
    "transition_metal_3d": list(range(21, 31)),
    "transition_metal_4d": list(range(39, 49)),
    "transition_metal_5d": [57] + list(range(72, 81)),
    "lanthanide": list(range(58, 72)),
    "actinide": list(range(89, 104)),
    "metalloid": [5, 14, 32, 33, 51, 52, 84],
    "p_block_metal": [13, 31, 49, 50, 81, 82, 83],
    "halogen": [9, 17, 35, 53, 85],
    "chalcogen_SSeTe": [16, 34, 52],
    "pnictogen_NP": [7, 15],
    "noble_gas": [2, 10, 18, 36, 54, 86],
}
# Anything not H/C/N/O/F/P/S/Cl/Se/Br/I/noble is treated as a metal-ish centre
# for the MOF-motif test; this is the usual inorganic/organic split.
NONMETAL_Z = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 33, 34, 35, 36, 52, 53, 54, 85, 86}


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


def read_manifest(root):
    with (root / MANIFEST).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty manifest: {root / MANIFEST}")
    starts = np.array([int(r["global_start"]) for r in rows], dtype=np.int64)
    stops = np.array([int(r["global_stop"]) for r in rows], dtype=np.int64)
    if not np.all(starts[1:] == stops[:-1]):
        raise SystemExit("manifest shards are not contiguous")
    return rows, starts, stops


def locate(global_rows, manifest, starts):
    """Map global rows to (manifest_index, local_row)."""
    manifest_index = np.searchsorted(starts, global_rows, side="right") - 1
    local_row = global_rows - starts[manifest_index]
    n_rows = np.array([int(r["n_rows"]) for r in manifest], dtype=np.int64)
    if np.any(local_row < 0) or np.any(local_row >= n_rows[manifest_index]):
        raise SystemExit("global row fell outside its manifest shard")
    return manifest_index, local_row


def structure_record(atoms, population, key, extra):
    numbers = atoms.get_atomic_numbers()
    counts = np.bincount(numbers, minlength=MAX_Z + 1)[: MAX_Z + 1]
    present = np.nonzero(counts)[0]
    volume = float(atoms.get_volume())
    mass = float(atomic_masses[numbers].sum())
    n_atoms = int(len(atoms))

    energy = force_max = force_rms = pressure = None
    calc = atoms.calc
    if calc is not None:
        results = calc.results
        if "energy" in results:
            energy = float(results["energy"])
        if "forces" in results:
            forces = np.asarray(results["forces"], dtype=np.float64)
            magnitudes = np.linalg.norm(forces, axis=1)
            force_max = float(magnitudes.max()) if magnitudes.size else None
            force_rms = float(np.sqrt((forces**2).sum() / max(forces.size, 1)))
        if "stress" in results:
            stress = np.asarray(results["stress"], dtype=np.float64).ravel()
            if stress.size >= 3:
                # ASE stress is eV/A^3, Voigt order; positive pressure = compression.
                pressure = float(-stress[:3].mean() * 160.21766208)

    record = {
        "population": population,
        "key": key,
        "global_row": extra.get("global_row", ""),
        "subdataset": extra.get("subdataset", ""),
        "shard": extra.get("shard", ""),
        "local_row": extra.get("local_row", ""),
        "formula_hill": atoms.get_chemical_formula(mode="hill"),
        "chemical_system": "-".join(chemical_symbols[z] for z in present),
        "n_atoms": n_atoms,
        "n_elements": int(present.size),
        "total_z": int(numbers.sum()),
        "volume_ang3": round(volume, 6),
        "volume_per_atom": round(volume / n_atoms, 6),
        "density_g_cm3": round(mass / volume * 1.6605390666, 6),
        "energy_ev": energy,
        "energy_per_atom": (energy / n_atoms) if energy is not None else None,
        "force_max": force_max,
        "force_rms": force_rms,
        "pressure_gpa": pressure,
    }
    return record, counts.astype(np.int32)


def write_population(out_dir, population, records, counts):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{population}.csv"
    tmp = csv_path.with_name(csv_path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRUCTURE_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, csv_path)
    atomic_write_npy(out_dir / f"{population}_zcounts.npy", np.asarray(counts, dtype=np.int32))
    log(f"wrote {population}: {len(records):,} structures -> {csv_path}")


def read_omat_rows(root, global_rows, manifest, starts, population, label):
    """Read OMAT24 structures for the given global rows, grouped by shard."""
    order = np.argsort(global_rows, kind="stable")
    sorted_rows = global_rows[order]
    manifest_index, local_row = locate(sorted_rows, manifest, starts)

    records, counts = [], []
    total = sorted_rows.size
    done = 0
    for shard_idx in np.unique(manifest_index):
        entry = manifest[int(shard_idx)]
        mask = manifest_index == shard_idx
        wanted_local = local_row[mask]
        wanted_global = sorted_rows[mask]
        path = root / entry["raw_path"]
        with connect(path, readonly=True, use_lock_file=False) as db:
            ids = db.ids
            if len(ids) != int(entry["n_rows"]):
                raise SystemExit(
                    f"shard {entry['raw_path']} has {len(ids)} rows, manifest says {entry['n_rows']}"
                )
            for lr, gr in zip(wanted_local, wanted_global):
                atoms = db.get(id=ids[int(lr)]).toatoms()
                record, zcount = structure_record(
                    atoms, population, str(int(gr)),
                    {
                        "global_row": int(gr),
                        "subdataset": entry["subdataset"],
                        "shard": entry["shard"],
                        "local_row": int(lr),
                    },
                )
                records.append(record)
                counts.append(zcount)
        done += int(mask.sum())
        if done % 5000 < int(mask.sum()) or done == total:
            log(f"  {label}: {done:,}/{total:,} structures")
    return records, counts


def load_indices(root, geometry):
    path = root / GEOMETRIES[geometry] / "indices.npy"
    distances = root / GEOMETRIES[geometry] / "distances.npy"
    idx = np.load(path)
    dist = np.load(distances)
    if idx.shape != dist.shape or idx.ndim != 2:
        raise SystemExit(f"{geometry}: indices/distances shape mismatch {idx.shape} {dist.shape}")
    return idx, dist


def command_extract(args):
    root = args.root
    out_dir = args.out / "structures"
    manifest, starts, _ = read_manifest(root)

    neighbor_rows = {}
    for geometry in GEOMETRIES:
        idx, _ = load_indices(root, geometry)
        neighbor_rows[geometry] = idx
        log(f"{geometry}: {idx.shape[0]:,} queries x {idx.shape[1]} neighbors, "
            f"{np.unique(idx).size:,} unique OMAT rows")

    union = np.unique(np.concatenate([v.ravel() for v in neighbor_rows.values()]))
    log(f"union of neighbor rows across geometries: {union.size:,}")

    if not args.skip_neighbors:
        records, counts = read_omat_rows(
            root, union, manifest, starts, "omat_neighbors", "neighbors"
        )
        write_population(out_dir, "omat_neighbors", records, counts)

    rng = np.random.default_rng(args.seed)
    total_rows = int(starts[-1]) + int(manifest[-1]["n_rows"])

    if not args.skip_background:
        picks = np.sort(rng.choice(total_rows, args.background, replace=False))
        records, counts = read_omat_rows(
            root, picks, manifest, starts, "omat_bg_global", "bg_global"
        )
        write_population(out_dir, "omat_bg_global", records, counts)

        nvt_mask = [i for i, r in enumerate(manifest)
                    if r["subdataset"] == "aimd-from-PBE-3000-nvt"]
        nvt_sizes = np.array([int(manifest[i]["n_rows"]) for i in nvt_mask], dtype=np.int64)
        nvt_stops = np.cumsum(nvt_sizes)
        offsets = np.sort(rng.choice(int(nvt_stops[-1]), args.background, replace=False))
        shard_pos = np.searchsorted(nvt_stops, offsets, side="right")
        prev = np.concatenate([[0], nvt_stops[:-1]])
        picks_nvt = np.array(
            [int(manifest[nvt_mask[int(s)]]["global_start"]) + int(o - prev[int(s)])
             for o, s in zip(offsets, shard_pos)],
            dtype=np.int64,
        )
        picks_nvt = np.sort(picks_nvt)
        records, counts = read_omat_rows(
            root, picks_nvt, manifest, starts, "omat_bg_nvt3000", "bg_nvt3000"
        )
        write_population(out_dir, "omat_bg_nvt3000", records, counts)

    if not args.skip_mof:
        from ase.io import Trajectory

        with (root / MOF_IDS).open() as handle:
            sample_ids = list(csv.DictReader(handle, delimiter="\t"))
        traj = Trajectory(str(root / MOF_TRAJ))
        if len(traj) != len(sample_ids):
            raise SystemExit(
                f"MOF traj has {len(traj)} frames but sample_ids has {len(sample_ids)}"
            )
        records, counts = [], []
        for i, atoms in enumerate(traj):
            meta = sample_ids[i]
            if int(meta["split_row"]) != i:
                raise SystemExit(f"sample_ids split_row {meta['split_row']} != frame {i}")
            record, zcount = structure_record(
                atoms, "mof_off_r2scan", meta["record_id"], {"global_row": i}
            )
            record["subdataset"] = meta["cif_name"].split("_")[0]
            record["shard"] = meta["temperature_K"]
            record["local_row"] = meta["frame"]
            records.append(record)
            counts.append(zcount)
            if (i + 1) % 10000 == 0 or i + 1 == len(traj):
                log(f"  mof_off: {i + 1:,}/{len(traj):,} frames")
        traj.close()
        write_population(out_dir, "mof_off_r2scan", records, counts)

    atomic_write_json(
        args.out / "extract_metadata.json",
        {
            "created_utc": utc_now(),
            "root": str(root),
            "query_set": "MOF_off_R2SCAN",
            "geometries": {g: str(GEOMETRIES[g]) for g in GEOMETRIES},
            "neighbor_union_rows": int(union.size),
            "unique_rows_per_geometry": {
                g: int(np.unique(v).size) for g, v in neighbor_rows.items()
            },
            "background_rows_per_baseline": args.background,
            "background_seed": args.seed,
            "row_mapping": "local_row = global_row - global_start; atoms = db.get(id=db.ids[local_row])",
        },
    )
    log("extract complete")


def command_validate(args):
    """Confirm the atoms<->embedding-row mapping and basic table integrity.

    Two independent end-to-end tests, neither of which merely restates the
    manifest arithmetic:

    1. Extensivity.  The UMA embedding aggregates over atoms, so ||x|| tracks
       total Z.  This is measured on the *global* background, which spans the
       full OMAT24 size range; measuring it on the neighbor set instead would
       be range-restricted (all one subdataset, narrow size spread) and would
       understate the correlation.  A shuffled pairing is the control.
    2. Adjacent-row alignment.  Consecutive rows inside one AIMD trajectory
       share a formula and sit very close in embedding space, while a
       trajectory boundary jumps by orders of magnitude.  Because the test is
       evaluated at exact row resolution, any off-by-one in the mapping
       collapses the separation - which extensivity alone would not catch.
    """
    root, out_dir = args.root, args.out / "structures"
    rng = np.random.default_rng(args.seed)
    embeddings = np.load(root / EMBEDDINGS, mmap_mode="r")

    table = load_table(out_dir, "omat_bg_global")
    n = min(args.sample, len(table["global_row"]))
    picks = np.sort(rng.choice(len(table["global_row"]), n, replace=False))
    rows = table["global_row"][picks].astype(np.int64)
    norms = np.array([float(np.linalg.norm(embeddings[int(g)])) for g in rows])

    total_z = table["total_z"][picks].astype(np.float64)
    n_atoms = table["n_atoms"][picks].astype(np.float64)
    corr_z = float(np.corrcoef(norms, total_z)[0, 1])
    corr_n = float(np.corrcoef(norms, n_atoms)[0, 1])
    shuffled = total_z.copy()
    rng.shuffle(shuffled)
    corr_shuffled = float(np.corrcoef(norms, shuffled)[0, 1])

    manifest, starts, _ = read_manifest(root)
    nvt = [r for r in manifest if r["subdataset"] == "aimd-from-PBE-3000-nvt"]
    same, different = [], []
    for _ in range(args.alignment_shards):
        entry = nvt[int(rng.integers(0, len(nvt)))]
        with connect(root / entry["raw_path"], readonly=True, use_lock_file=False) as db:
            ids = db.ids
            for local in rng.integers(0, len(ids) - 1, args.alignment_pairs):
                local = int(local)
                first = db.get(id=ids[local]).toatoms().get_chemical_formula()
                second = db.get(id=ids[local + 1]).toatoms().get_chemical_formula()
                g = int(entry["global_start"]) + local
                delta = float(np.linalg.norm(
                    np.asarray(embeddings[g], dtype=np.float64)
                    - np.asarray(embeddings[g + 1], dtype=np.float64)
                ))
                (same if first == second else different).append(delta)
    del embeddings

    median_same = float(np.median(same)) if same else float("nan")
    median_different = float(np.median(different)) if different else float("nan")
    separation = median_different / median_same if same and different else float("nan")

    checks = {
        "extensivity_population": "omat_bg_global",
        "extensivity_sampled_rows": n,
        "pearson_norm_vs_total_z": corr_z,
        "pearson_norm_vs_n_atoms": corr_n,
        "pearson_norm_vs_shuffled_total_z": corr_shuffled,
        "alignment_same_formula_pairs": len(same),
        "alignment_diff_formula_pairs": len(different),
        "alignment_median_delta_same_formula": median_same,
        "alignment_median_delta_diff_formula": median_different,
        "alignment_separation_ratio": separation,
        "extensivity_gate_passed": bool(corr_z > 0.8 and abs(corr_shuffled) < 0.1),
        "alignment_gate_passed": bool(separation > 20),
    }
    checks["mapping_gate_passed"] = bool(
        checks["extensivity_gate_passed"] and checks["alignment_gate_passed"]
    )

    for population in ("omat_neighbors", "omat_bg_global", "omat_bg_nvt3000", "mof_off_r2scan"):
        path = out_dir / f"{population}.csv"
        if not path.exists():
            continue
        tab = load_table(out_dir, population)
        counts = np.load(out_dir / f"{population}_zcounts.npy")
        checks[f"{population}_rows"] = int(len(tab["n_atoms"]))
        checks[f"{population}_zcounts_shape"] = list(counts.shape)
        checks[f"{population}_zcounts_match_natoms"] = bool(
            np.array_equal(counts.sum(axis=1), tab["n_atoms"].astype(np.int64))
        )
        checks[f"{population}_positive_volume"] = bool(np.all(tab["volume_ang3"] > 0))

    atomic_write_json(args.out / "validation.json", checks)
    log(json.dumps(checks, indent=2))
    if not checks["mapping_gate_passed"]:
        raise SystemExit("row-mapping validation gate FAILED")
    log("validation passed")


def load_table(out_dir, population):
    path = out_dir / f"{population}.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    table = {}
    numeric = {
        "global_row": np.int64, "local_row": np.int64, "n_atoms": np.int64,
        "n_elements": np.int64, "total_z": np.int64, "volume_ang3": np.float64,
        "volume_per_atom": np.float64, "density_g_cm3": np.float64,
        "energy_ev": np.float64, "energy_per_atom": np.float64,
        "force_max": np.float64, "force_rms": np.float64, "pressure_gpa": np.float64,
    }
    for field in STRUCTURE_FIELDS:
        values = [r[field] for r in rows]
        if field in numeric:
            table[field] = np.array(
                [np.nan if v in ("", "None") else float(v) for v in values], dtype=np.float64
            )
            if numeric[field] is np.int64 and not np.any(np.isnan(table[field])):
                table[field] = table[field].astype(np.int64)
        else:
            table[field] = np.array(values, dtype=object)
    return table


def hill_sorted_symbols(present):
    """Element symbols in Hill order for the given atomic numbers."""
    symbols = [chemical_symbols[z] for z in present]
    if "C" in symbols:
        head = [s for s in ("C", "H") if s in symbols]
        tail = sorted(s for s in symbols if s not in ("C", "H"))
        return head + tail
    return sorted(symbols)


def reduced_formula(counts):
    """Formula reduced by the gcd of its element counts, in Hill order."""
    present = np.nonzero(counts)[0]
    if present.size == 0:
        return ""
    divisor = int(np.gcd.reduce(counts[present]))
    lookup = {chemical_symbols[z]: int(counts[z]) // divisor for z in present}
    parts = []
    for symbol in hill_sorted_symbols(present):
        number = lookup[symbol]
        parts.append(symbol if number == 1 else f"{symbol}{number}")
    return "".join(parts)


def make_weights(n_rows, positions):
    return np.bincount(positions, minlength=n_rows).astype(np.float64)


def weighted_quantiles(values, weights, quantiles):
    """Quantiles of `values` under integer-ish `weights`, ignoring NaN."""
    finite = np.isfinite(values)
    v, w = values[finite], weights[finite]
    if v.size == 0 or w.sum() == 0:
        return {q: float("nan") for q in quantiles}
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cumulative = np.cumsum(w) - 0.5 * w
    cumulative /= w.sum()
    return {q: float(np.interp(q, cumulative, v)) for q in quantiles}


def weighted_mean(values, weights):
    finite = np.isfinite(values)
    v, w = values[finite], weights[finite]
    return float((v * w).sum() / w.sum()) if w.sum() else float("nan")


class Profile:
    """One weighted population of structures drawn from a single table."""

    def __init__(self, name, table, zcounts, positions, description):
        self.name = name
        self.table = table
        self.zcounts = zcounts
        self.positions = np.asarray(positions, dtype=np.int64)
        self.description = description
        self.weights = make_weights(zcounts.shape[0], self.positions)
        self.slots = int(self.positions.size)
        self.distinct = int(np.count_nonzero(self.weights))

    def scalar(self, field):
        return self.table[field].astype(np.float64)[self.positions]

    def presence_rate(self):
        return (self.weights[:, None] * (self.zcounts > 0)).sum(0) / self.weights.sum()

    def atom_fraction(self):
        totals = (self.weights[:, None] * self.zcounts).sum(0)
        return totals / totals.sum()

    def weighted_counter(self, values):
        counter = Counter()
        active = np.nonzero(self.weights)[0]
        for i in active:
            counter[values[i]] += self.weights[i]
        return counter


def build_profiles(root, out_dir):
    tables, zcounts = {}, {}
    for population in ("omat_neighbors", "omat_bg_global", "omat_bg_nvt3000", "mof_off_r2scan"):
        tables[population] = load_table(out_dir, population)
        zcounts[population] = np.load(out_dir / f"{population}_zcounts.npy")

    union_rows = tables["omat_neighbors"]["global_row"].astype(np.int64)
    if not np.all(np.diff(union_rows) > 0):
        raise SystemExit("omat_neighbors.csv is not sorted by global_row")

    profiles, neighbor_indices = [], {}
    for geometry in GEOMETRIES:
        idx, dist = load_indices(root, geometry)
        positions = np.searchsorted(union_rows, idx.ravel())
        if not np.array_equal(union_rows[positions], idx.ravel()):
            raise SystemExit(f"{geometry}: neighbor rows missing from extracted table")
        positions = positions.reshape(idx.shape)
        neighbor_indices[geometry] = (idx, dist, positions)

        profiles.append(Profile(
            f"{geometry}_rank1", tables["omat_neighbors"], zcounts["omat_neighbors"],
            positions[:, 0], f"{geometry}: nearest OMAT24 row of each MOF-off query",
        ))
        profiles.append(Profile(
            f"{geometry}_top10", tables["omat_neighbors"], zcounts["omat_neighbors"],
            positions.ravel(), f"{geometry}: all ten neighbors of each MOF-off query",
        ))
        unique_positions = np.unique(positions.ravel())
        profiles.append(Profile(
            f"{geometry}_unique", tables["omat_neighbors"], zcounts["omat_neighbors"],
            unique_positions, f"{geometry}: distinct retrieved OMAT24 structures, unweighted",
        ))
        # The genuinely-close tail: rank-1 neighbors of the decile of queries with
        # the smallest d1.  "Most similar" is otherwise diluted by queries whose
        # nearest OMAT24 row is still far away.
        cutoff = np.quantile(dist[:, 0], 0.10)
        closest = np.nonzero(dist[:, 0] <= cutoff)[0]
        profiles.append(Profile(
            f"{geometry}_rank1_closest10pct", tables["omat_neighbors"],
            zcounts["omat_neighbors"], positions[closest, 0],
            f"{geometry}: nearest row of the 10% of queries with the smallest d1 "
            f"(d1 <= {cutoff:.4f})",
        ))

    for population, description in (
        ("omat_bg_nvt3000", "uniform random rows from aimd-from-PBE-3000-nvt"),
        ("omat_bg_global", "uniform random rows from all of OMAT24"),
        ("mof_off_r2scan", "the MOF-off R2SCAN-D4 train queries themselves"),
    ):
        n = len(tables[population]["n_atoms"])
        profiles.append(Profile(
            population, tables[population], zcounts[population], np.arange(n), description,
        ))

    return profiles, tables, zcounts, neighbor_indices


SCALAR_FIELDS = [
    "n_atoms", "n_elements", "total_z", "volume_ang3", "volume_per_atom",
    "density_g_cm3", "energy_per_atom", "force_max", "force_rms", "pressure_gpa",
]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def analyze_elements(profiles, out):
    names = [p.name for p in profiles]
    presence = {p.name: p.presence_rate() for p in profiles}
    fraction = {p.name: p.atom_fraction() for p in profiles}

    active = sorted({
        z for p in profiles for z in np.nonzero(presence[p.name])[0] if 1 <= z <= MAX_Z
    })
    rows = []
    for z in active:
        row = {"atomic_number": z, "symbol": chemical_symbols[z]}
        for name in names:
            row[f"presence_{name}"] = round(float(presence[name][z]), 8)
            row[f"atomfrac_{name}"] = round(float(fraction[name][z]), 8)
        rows.append(row)
    fields = ["atomic_number", "symbol"] + [
        f"{kind}_{name}" for name in names for kind in ("presence", "atomfrac")
    ]
    write_csv(out / "element_presence_and_abundance.csv", fields, rows)

    # Enrichment of the neighbor set against both OMAT24 baselines and against
    # the MOF-off queries.  log2 ratios with a small floor so absent elements do
    # not produce infinities.
    floor = 1e-6
    enrich_rows = []
    for z in active:
        row = {"atomic_number": z, "symbol": chemical_symbols[z]}
        for geometry in GEOMETRIES:
            target = presence[f"{geometry}_top10"][z]
            for baseline in ("omat_bg_nvt3000", "omat_bg_global", "mof_off_r2scan"):
                ratio = (target + floor) / (presence[baseline][z] + floor)
                row[f"log2_{geometry}_vs_{baseline}"] = round(float(np.log2(ratio)), 6)
        enrich_rows.append(row)
    fields = ["atomic_number", "symbol"] + [
        f"log2_{g}_vs_{b}" for g in GEOMETRIES
        for b in ("omat_bg_nvt3000", "omat_bg_global", "mof_off_r2scan")
    ]
    write_csv(out / "element_enrichment.csv", fields, enrich_rows)
    return presence, fraction, active


def analyze_scalars(profiles, out):
    rows = []
    for p in profiles:
        for field in SCALAR_FIELDS:
            values = p.scalar(field)
            weights = np.ones_like(values)
            quantiles = weighted_quantiles(values, weights, QUANTILES)
            rows.append({
                "population": p.name, "description": p.description, "field": field,
                "slots": p.slots, "distinct_structures": p.distinct,
                "mean": round(weighted_mean(values, weights), 6),
                **{f"p{int(q * 100):02d}": round(quantiles[q], 6) for q in QUANTILES},
            })
    fields = ["population", "description", "field", "slots", "distinct_structures",
              "mean"] + [f"p{int(q * 100):02d}" for q in QUANTILES]
    write_csv(out / "size_and_energy_stats.csv", fields, rows)
    return rows


def analyze_classes(profiles, out):
    rows = []
    for p in profiles:
        presence = p.zcounts > 0
        total = p.weights.sum()
        row = {"population": p.name, "slots": p.slots, "distinct_structures": p.distinct}
        for label, members in ELEMENT_CLASSES.items():
            member = np.zeros(MAX_Z + 1, dtype=bool)
            member[[z for z in members if z <= MAX_Z]] = True
            has = presence[:, member].any(axis=1)
            row[f"has_{label}"] = round(float((p.weights * has).sum() / total), 6)

        metal = np.ones(MAX_Z + 1, dtype=bool)
        metal[list(NONMETAL_Z)] = False
        metal[0] = False
        has_metal = presence[:, metal].any(axis=1)
        has_c = presence[:, 6]
        has_h = presence[:, 1]
        has_n = presence[:, 7]
        has_o = presence[:, 8]

        motifs = {
            "has_metal": has_metal,
            "has_C_and_H": has_c & has_h,
            "has_C_H_O": has_c & has_h & has_o,
            "has_C_H_N_O": has_c & has_h & has_n & has_o,
            "mof_motif_metal_C_H_O": has_metal & has_c & has_h & has_o,
            "purely_inorganic_no_C_no_H": ~(has_c | has_h),
        }
        for label, mask in motifs.items():
            row[label] = round(float((p.weights * mask).sum() / total), 6)
        rows.append(row)

    fields = (["population", "slots", "distinct_structures"]
              + [f"has_{label}" for label in ELEMENT_CLASSES]
              + ["has_metal", "has_C_and_H", "has_C_H_O", "has_C_H_N_O",
                 "mof_motif_metal_C_H_O", "purely_inorganic_no_C_no_H"])
    write_csv(out / "element_classes_and_motifs.csv", fields, rows)
    return rows


def analyze_formulas(profiles, out, top_n):
    system_rows, formula_rows = [], []
    reduced_cache = {}
    for p in profiles:
        systems = p.table["chemical_system"]
        key = id(p.zcounts)
        if key not in reduced_cache:
            reduced_cache[key] = np.array(
                [reduced_formula(p.zcounts[i]) for i in range(p.zcounts.shape[0])],
                dtype=object,
            )
        reduced = reduced_cache[key]
        total = p.weights.sum()
        for label, values, sink in (("chemical_system", systems, system_rows),
                                    ("reduced_formula", reduced, formula_rows)):
            counter = p.weighted_counter(values)
            for rank, (value, weight) in enumerate(counter.most_common(top_n), 1):
                sink.append({
                    "population": p.name, "rank": rank, label: value,
                    "weight": int(weight), "share": round(float(weight / total), 8),
                })
        p.reduced = reduced

    write_csv(out / "top_chemical_systems.csv",
              ["population", "rank", "chemical_system", "weight", "share"], system_rows)
    write_csv(out / "top_reduced_formulas.csv",
              ["population", "rank", "reduced_formula", "weight", "share"], formula_rows)
    return system_rows, formula_rows


def analyze_top_structures(profiles, out, top_n):
    rows = []
    for p in profiles:
        if not p.name.endswith("_rank1"):
            continue
        order = np.argsort(-p.weights)[:top_n]
        total = p.weights.sum()
        for rank, i in enumerate(order, 1):
            if p.weights[i] == 0:
                break
            rows.append({
                "population": p.name, "rank": rank,
                "global_row": int(p.table["global_row"][i]),
                "shard": str(p.table["shard"][i]),
                "formula_hill": str(p.table["formula_hill"][i]),
                "chemical_system": str(p.table["chemical_system"][i]),
                "n_atoms": int(p.table["n_atoms"][i]),
                "times_nearest": int(p.weights[i]),
                "share_of_queries": round(float(p.weights[i] / total), 8),
            })
    write_csv(out / "most_retrieved_structures.csv",
              ["population", "rank", "global_row", "shard", "formula_hill",
               "chemical_system", "n_atoms", "times_nearest", "share_of_queries"], rows)
    return rows


def analyze_pairing(tables, zcounts, neighbor_indices, out):
    """Compare each MOF-off query's composition with its retrieved neighbors."""
    mof_presence = zcounts["mof_off_r2scan"] > 0
    neighbor_presence = zcounts["omat_neighbors"] > 0
    mof_atoms = tables["mof_off_r2scan"]["n_atoms"].astype(np.float64)
    neighbor_atoms = tables["omat_neighbors"]["n_atoms"].astype(np.float64)

    metal = np.ones(MAX_Z + 1, dtype=bool)
    metal[list(NONMETAL_Z)] = False
    metal[0] = False

    rows = []
    per_query = {}
    for geometry, (idx, dist, positions) in neighbor_indices.items():
        n_queries, n_ranks = positions.shape
        for rank in range(n_ranks):
            sel = neighbor_presence[positions[:, rank]]
            intersection = (mof_presence & sel).sum(axis=1)
            union = (mof_presence | sel).sum(axis=1)
            jaccard = intersection / np.maximum(union, 1)

            mof_metals = mof_presence & metal
            sel_metals = sel & metal
            shares_metal = (mof_metals & sel_metals).any(axis=1)

            row = {
                "geometry": geometry, "rank": rank + 1, "queries": int(n_queries),
                "mean_jaccard_elements": round(float(jaccard.mean()), 6),
                "median_jaccard_elements": round(float(np.median(jaccard)), 6),
                "frac_share_any_element": round(float((intersection > 0).mean()), 6),
                "frac_share_any_metal": round(float(shares_metal.mean()), 6),
                "frac_identical_element_set": round(
                    float((intersection == union).mean()), 6),
                "frac_neighbor_has_H": round(float(sel[:, 1].mean()), 6),
                "frac_neighbor_has_C": round(float(sel[:, 6].mean()), 6),
                "frac_neighbor_has_N": round(float(sel[:, 7].mean()), 6),
                "frac_neighbor_has_O": round(float(sel[:, 8].mean()), 6),
                "mean_delta_n_atoms": round(
                    float((neighbor_atoms[positions[:, rank]] - mof_atoms).mean()), 6),
                "corr_n_atoms": round(float(np.corrcoef(
                    mof_atoms, neighbor_atoms[positions[:, rank]])[0, 1]), 6),
                "median_distance": round(float(np.median(dist[:, rank])), 6),
            }
            rows.append(row)
            if rank == 0:
                per_query[geometry] = {
                    "jaccard": jaccard, "intersection": intersection,
                    "shares_metal": shares_metal, "distance": dist[:, 0],
                    "neighbor_position": positions[:, 0],
                }

    write_csv(out / "query_neighbor_pairing.csv", list(rows[0].keys()), rows)

    # Per-MOF-element coverage: of the queries containing element E, how often
    # does the nearest OMAT24 neighbor contain E as well?
    coverage_rows = []
    for geometry, (idx, dist, positions) in neighbor_indices.items():
        sel = neighbor_presence[positions[:, 0]]
        for z in range(1, MAX_Z + 1):
            holders = mof_presence[:, z]
            n_holders = int(holders.sum())
            if n_holders == 0:
                continue
            coverage_rows.append({
                "geometry": geometry, "atomic_number": z, "symbol": chemical_symbols[z],
                "mof_structures_with_element": n_holders,
                "share_of_mof_structures": round(float(n_holders / mof_presence.shape[0]), 6),
                "frac_nearest_neighbor_also_has": round(float(sel[holders, z].mean()), 6),
            })
    write_csv(out / "mof_element_coverage_by_nearest.csv",
              ["geometry", "atomic_number", "symbol", "mof_structures_with_element",
               "share_of_mof_structures", "frac_nearest_neighbor_also_has"], coverage_rows)

    # Per-query drill-down: one row per MOF-off structure and its nearest OMAT24
    # match, so individual cases can be inspected without rerunning anything.
    mof_table = tables["mof_off_r2scan"]
    neighbor_table = tables["omat_neighbors"]
    for geometry, payload in per_query.items():
        position = payload["neighbor_position"]
        drill = [{
            "mof_row": int(i),
            "record_id": str(mof_table["key"][i]),
            "mof_formula": str(mof_table["formula_hill"][i]),
            "mof_chemical_system": str(mof_table["chemical_system"][i]),
            "mof_n_atoms": int(mof_table["n_atoms"][i]),
            "temperature_K": str(mof_table["shard"][i]),
            "distance": round(float(payload["distance"][i]), 6),
            "neighbor_global_row": int(neighbor_table["global_row"][position[i]]),
            "neighbor_formula": str(neighbor_table["formula_hill"][position[i]]),
            "neighbor_chemical_system": str(neighbor_table["chemical_system"][position[i]]),
            "neighbor_n_atoms": int(neighbor_table["n_atoms"][position[i]]),
            "shared_elements": int(payload["intersection"][i]),
            "jaccard_elements": round(float(payload["jaccard"][i]), 6),
            "shares_a_metal": int(payload["shares_metal"][i]),
        } for i in range(len(mof_table["n_atoms"]))]
        write_csv(out / f"nearest_neighbor_per_query_{geometry}.csv",
                  list(drill[0].keys()), drill)

    return rows, coverage_rows, per_query


def analyze_geometry_overlap(neighbor_indices, out):
    (idx_a, _, _), (idx_b, _, _) = (neighbor_indices["raw128"], neighbor_indices["pc25"])
    shared = np.array([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(idx_a, idx_b)])
    payload = {
        "mean_shared_neighbors_of_10": float(shared.mean()),
        "median_shared_neighbors_of_10": float(np.median(shared)),
        "frac_queries_sharing_any_neighbor": float((shared > 0).mean()),
        "frac_queries_same_rank1": float((idx_a[:, 0] == idx_b[:, 0]).mean()),
        "unique_rows_raw128": int(np.unique(idx_a).size),
        "unique_rows_pc25": int(np.unique(idx_b).size),
        "unique_rows_shared": int(len(set(np.unique(idx_a).tolist())
                                      & set(np.unique(idx_b).tolist()))),
    }
    atomic_write_json(out / "geometry_overlap.json", payload)
    return payload


def command_analyze(args):
    out = args.out / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    structures = args.out / "structures"
    log("loading extracted structure tables")
    profiles, tables, zcounts, neighbor_indices = build_profiles(args.root, structures)

    log("element presence / abundance / enrichment")
    presence, fraction, active = analyze_elements(profiles, out)
    log("size, cell and energy statistics")
    analyze_scalars(profiles, out)
    log("element classes and MOF motifs")
    class_rows = analyze_classes(profiles, out)
    log("chemical systems and reduced formulas")
    analyze_formulas(profiles, out, args.top_n)
    log("most frequently retrieved structures")
    analyze_top_structures(profiles, out, args.top_n)
    log("query/neighbor composition pairing")
    pairing_rows, coverage_rows, _ = analyze_pairing(tables, zcounts, neighbor_indices, out)
    log("geometry overlap")
    overlap = analyze_geometry_overlap(neighbor_indices, out)

    atomic_write_json(out / "analysis_metadata.json", {
        "created_utc": utc_now(),
        "query_set": "MOF_off_R2SCAN",
        "profiles": {p.name: {"slots": p.slots, "distinct_structures": p.distinct,
                              "description": p.description} for p in profiles},
        "geometry_overlap": overlap,
        "active_elements": [chemical_symbols[z] for z in active],
    })
    log(f"analysis complete -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="repository root that holds data/ and runs/")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <root>/" + DEFAULT_OUT + ")")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="read structures out of the aselmdb shards")
    extract.add_argument("--background", type=int, default=50000)
    extract.add_argument("--seed", type=int, default=20260731)
    extract.add_argument("--skip-neighbors", action="store_true")
    extract.add_argument("--skip-background", action="store_true")
    extract.add_argument("--skip-mof", action="store_true")
    extract.set_defaults(func=command_extract)

    validate = sub.add_parser("validate", help="check the atoms<->row mapping")
    validate.add_argument("--sample", type=int, default=6000)
    validate.add_argument("--alignment-shards", type=int, default=6)
    validate.add_argument("--alignment-pairs", type=int, default=120)
    validate.add_argument("--seed", type=int, default=20260731)
    validate.set_defaults(func=command_validate)

    analyze = sub.add_parser("analyze", help="aggregate the extracted structures")
    analyze.add_argument("--top-n", type=int, default=30)
    analyze.set_defaults(func=command_analyze)

    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.out is None:
        args.out = args.root / DEFAULT_OUT
    args.out = args.out.resolve()
    args.func(args)


if __name__ == "__main__":
    main()
