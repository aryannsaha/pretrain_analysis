#!/usr/bin/env python
"""SOAP kernel similarity between each MOF and its pc25 top-5 OMAT24 neighbours.

The RDF check in ``mof_omat_coordination_chemistry.py`` is a two-body,
element-blind, density-normalised descriptor: it compares the *shape* of g(r)
and nothing else.  SOAP (Smooth Overlap of Atomic Positions) removes two of
those three limitations -- it is many-body (angular information is retained via
the power spectrum) and species-resolved (a Zn-N contact and a Zn-O contact at
the same distance land in different channels).

Two variants are computed for every pair, because they answer different questions:

  chem   species-resolved SOAP.  "Is the retrieved structure the same kind of
         material?"  Chemistry and geometry together.
  geom   every atom mapped to a single species.  "Is it the same *shape*,
         regardless of what it is made of?"  A strict upgrade over the RDF
         check: same element-blindness, but with angular information added.

If `geom` shows a signal and `chem` does not, the embedding is retrieving
structures that are geometrically similar but chemically unrelated -- which is
exactly the hypothesis the rest of this analysis raises.

COMMON FEATURE SPACE.  A SOAP vector is indexed by species pairs, so two
structures can only be compared inside a fixed species list.  Rather than build
one global list over all ~80 elements present (which would be enormous and
almost entirely zeros), the species list is rebuilt per MOF as the union over
that MOF, its five neighbours and its five random controls.  All eleven
structures for one MOF therefore live in one space, which is what makes the
matched-vs-random contrast a valid paired comparison.  Elements present in one
structure and absent from the other simply give zero blocks, correctly lowering
the similarity.

Kernel: normalised average SOAP (``average="inner"``), cosine, reported at
zeta = 1 for direct comparability with the RDF cosine and at zeta = 4, the
usual sharpening exponent in the SOAP-similarity literature.
"""

import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pc25_top5_structure_descriptors import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_ROOT,
    K_NEIGHBOURS,
    QUERY_SETS,
    atomic_write_json,
    locate,
    log,
    read_manifest,
    utc_now,
)

R_CUT = 5.0
N_MAX = 4
L_MAX = 3
SIGMA = 0.5
ZETAS = (1, 4)


def soap_vectors(structures, species, blind):
    """Unit-normalised average SOAP descriptor for each structure."""
    from ase import Atoms
    from dscribe.descriptors import SOAP

    if blind:
        structures = [
            Atoms(numbers=np.ones(len(a), dtype=int), positions=a.get_positions(),
                  cell=a.cell, pbc=a.pbc)
            for a in structures
        ]
        species = [1]

    descriptor = SOAP(species=sorted(species), r_cut=R_CUT, n_max=N_MAX, l_max=L_MAX,
                      sigma=SIGMA, periodic=True, average="inner", sparse=False)
    matrix = np.asarray(descriptor.create(structures), dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _task(payload):
    """One MOF: cosine to its 5 neighbours and to 5 random controls, both variants."""
    mof, neighbours, controls, row = payload
    structures = [mof] + list(neighbours) + list(controls)
    species = sorted({int(z) for a in structures for z in a.get_atomic_numbers()})
    out = {"mof_row": row, "n_species": len(species), "n_atoms": len(mof)}
    for label, blind in (("chem", False), ("geom", True)):
        try:
            vectors = soap_vectors(structures, species, blind)
        except Exception as error:  # noqa: BLE001 - one bad structure must not stop the sweep
            log(f"  skipped row {row} ({label}): {type(error).__name__} {error}")
            return None
        similarity = vectors[0] @ vectors[1:].T
        matched = similarity[: len(neighbours)]
        random = similarity[len(neighbours):]
        for zeta in ZETAS:
            out[f"{label}_matched_z{zeta}"] = float(np.mean(matched**zeta))
            out[f"{label}_random_z{zeta}"] = float(np.mean(random**zeta))
    return out


def read_structures(root, rows, manifest, starts, label):
    """OMAT24 structures for the given global rows, one LMDB open per shard."""
    from ase.db import connect

    rows = np.asarray(sorted(rows), dtype=np.int64)
    manifest_index, local = locate(rows, manifest, starts)
    structures = {}
    for shard in np.unique(manifest_index):
        entry = manifest[int(shard)]
        mask = manifest_index == shard
        with connect(root / entry["raw_path"], readonly=True, use_lock_file=False) as db:
            ids = db.ids
            for lr, gr in zip(local[mask], rows[mask]):
                structures[int(gr)] = db.get(id=ids[int(lr)]).toatoms()
    log(f"  read {len(structures):,} {label} structures")
    return structures


def command_run(args):
    from ase.io import Trajectory

    root = args.root.resolve()
    out_root = (args.out or root / DEFAULT_OUT).resolve()
    analysis = out_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    spec = QUERY_SETS["mof_off"]
    with (root / spec["meta_tsv"][0]).open() as handle:
        meta = list(csv.DictReader(handle, delimiter="\t"))
    idx = np.load(root / spec["indices"] / "indices.npy")[:, :K_NEIGHBOURS]

    # Same selection rule as the coordination sweep, so the SOAP and RDF numbers
    # describe the same MOFs.
    chosen = {}
    for row, entry in enumerate(meta):
        mof = entry["cif_name"].split("_")[0]
        temperature = int(entry["temperature_K"])
        if mof not in chosen or temperature < chosen[mof][1]:
            chosen[mof] = (row, temperature)
    selected = np.array(sorted(v[0] for v in chosen.values()), dtype=np.int64)
    if args.limit_mofs:
        selected = selected[: args.limit_mofs]
    log(f"{selected.size:,} MOFs (one frame each, lowest sampled temperature)")

    manifest, starts = read_manifest(root)
    pool_rows = np.unique(idx[selected])
    log(f"{pool_rows.size:,} unique pc25 neighbours to read")
    omat = read_structures(root, pool_rows, manifest, starts, "OMAT24")

    traj = Trajectory(str(root / spec["traj"][0]))
    try:
        mofs = [traj[int(r)] for r in selected]
    finally:
        traj.close()
    log(f"  read {len(mofs):,} MOF frames")

    # Controls are drawn from the same pool the neighbours come from, so the
    # contrast isolates "chosen by the embedding" from "an OMAT24 structure".
    rng = np.random.default_rng(args.seed)
    tasks = []
    for position, row in enumerate(selected):
        neighbours = [omat[int(g)] for g in idx[row]]
        draw = rng.choice(pool_rows, size=K_NEIGHBOURS, replace=False)
        tasks.append((mofs[position], neighbours, [omat[int(g)] for g in draw], int(row)))

    results = []
    with Pool(args.workers) as pool:
        for n, record in enumerate(pool.imap_unordered(_task, tasks, chunksize=4), 1):
            if record is not None:
                results.append(record)
            if n % 100 == 0 or n == len(tasks):
                log(f"  SOAP: {n:,}/{len(tasks):,} MOFs")

    fields = sorted(results[0])
    path = analysis / "soap_similarity_per_mof.csv"
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    os.replace(tmp, path)
    log(f"wrote {path.name} ({len(results):,} rows)")

    summary = []
    for label, name in (("chem", "species-resolved SOAP"), ("geom", "element-blind SOAP")):
        for zeta in ZETAS:
            matched = np.array([r[f"{label}_matched_z{zeta}"] for r in results])
            random = np.array([r[f"{label}_random_z{zeta}"] for r in results])
            delta = matched - random
            # Paired: each MOF is its own control, so the test statistic is the
            # mean of the per-MOF differences, not a difference of means.
            error = float(delta.std(ddof=1) / np.sqrt(delta.size))
            summary.append({
                "variant": name,
                "zeta": zeta,
                "n": int(delta.size),
                "matched_mean": round(float(matched.mean()), 5),
                "random_mean": round(float(random.mean()), 5),
                "paired_delta_mean": round(float(delta.mean()), 5),
                "paired_delta_median": round(float(np.median(delta)), 5),
                "paired_delta_stderr": round(error, 5),
                "t_statistic": round(float(delta.mean() / error), 3) if error > 0 else "",
                "pct_mofs_matched_closer": round(100.0 * float((delta > 0).mean()), 2),
            })
    path = analysis / "soap_similarity.csv"
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    os.replace(tmp, path)
    log(f"wrote {path.name}")

    atomic_write_json(analysis / "soap_metadata.json", {
        "created_utc": utc_now(),
        "descriptor": "dscribe SOAP, average='inner'",
        "r_cut": R_CUT, "n_max": N_MAX, "l_max": L_MAX, "sigma": SIGMA,
        "zetas": list(ZETAS),
        "species_space": "per-MOF union over the MOF, its 5 neighbours and 5 controls",
        "mofs": len(results),
        "neighbour_pool": int(pool_rows.size),
        "seed": args.seed,
        "median_species_per_comparison": int(np.median([r["n_species"] for r in results])),
    })
    for row in summary:
        log(f"  {row['variant']} zeta={row['zeta']}: matched {row['matched_mean']:.4f} "
            f"random {row['random_mean']:.4f} delta {row['paired_delta_mean']:+.5f} "
            f"(t={row['t_statistic']}, {row['pct_mofs_matched_closer']}% closer)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-mofs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    command_run(args)


if __name__ == "__main__":
    sys.exit(main())
