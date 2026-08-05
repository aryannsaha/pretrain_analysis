#!/usr/bin/env python
"""Export example query structures and their pc25 top-5 OMAT24 neighbours as CIFs.

The aggregate tables say what the retrieved structures are on average.  This
writes a handful of concrete cases to disk so they can be opened in a viewer:
one directory per example holding the query structure and its five nearest
OMAT24 neighbours, as ``.cif`` (geometry, for visualisation) and ``.extxyz``
(retains the energies/forces/stress that CIF cannot carry).

Examples are chosen to span the retrieval regime rather than to flatter it:

  closest        the query with the smallest pc25 d1 -- best case retrieval.
  best_chemistry the query whose rank-1 neighbour shares the most of its element
                 set (highest Jaccard), i.e. the closest thing to a chemical hit.
  typical        the query nearest the median d1 within the largest query family.
  framework      a 3D-connected query at median distance -- the case where the
                 query is unambiguously a framework, so the neighbour's own
                 dimensionality is a fair test.

Each directory carries a README.md stating what the distance does and does not
mean, because "nearest in UMA latent space" is routinely misread as "similar
material".
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
    QUERY_SETS,
    atomic_write_json,
    locate,
    log,
    read_manifest,
    utc_now,
)

from ase.data import chemical_symbols  # noqa: E402
from ase.db import connect  # noqa: E402
from ase.io import Trajectory, write  # noqa: E402

README = """# {title}

Query: **{query_label}** (`{query_formula}`, {query_atoms} atoms, {query_extra})
Retrieval: pc25 geometry, top-{k} OMAT24 neighbours out of {reference:,} rows.
Selected because: {reason}

| rank | file | formula | atoms | pc25 distance | shared elements | Jaccard |
|---|---|---|---:|---:|---|---:|
{rows}

## What the distance means

Distance is Euclidean in the first 25 standardised principal components of the
OMAT-fitted UMA embedding space. It is **not** coordinate RMSD, composition
matching, chemical identity, or energy/force error. Two structures can sit close
here and share no elements at all -- which is the typical case in this dataset.

The `.cif` files are geometry only. The `.extxyz` files carry the DFT energy,
forces and stress where the source provides them ({calc_note}).
"""


def element_set(zcounts_row):
    return {chemical_symbols[z] for z in np.nonzero(zcounts_row)[0]}


def load_zcounts(structures, population):
    return np.load(structures / f"{population}_zcounts.npy")


def read_table(path, fields):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {f: np.array([r[f] for r in rows], dtype=object) for f in fields}


def command_export(args):
    root = args.root.resolve()
    out_root = (args.out or root / DEFAULT_OUT).resolve()
    structures = out_root / "structures"
    export = out_root / "examples"
    export.mkdir(parents=True, exist_ok=True)

    query = args.query
    spec = QUERY_SETS[query]
    manifest, starts = read_manifest(root)
    reference_rows = int(starts[-1]) + int(manifest[-1]["n_rows"])

    idx = np.load(root / spec["indices"] / "indices.npy")[:, :K_NEIGHBOURS]
    dist = np.load(root / spec["indices"] / "distances.npy")[:, :K_NEIGHBOURS]

    qpop, npop = f"query_{query}", f"omat_nb_{query}"
    qzc, nzc = load_zcounts(structures, qpop), load_zcounts(structures, npop)
    qtab = read_table(structures / f"{qpop}.csv",
                      ["key", "formula_hill", "n_atoms", "subdataset", "shard",
                       "dimensionality", "local_row"])
    ntab = read_table(structures / f"{npop}.csv",
                      ["global_row", "formula_hill", "n_atoms", "subdataset", "shard"])
    nb_rows = ntab["global_row"].astype(np.int64)
    pos = np.searchsorted(nb_rows, idx)

    # Jaccard of the rank-1 neighbour's element set with the query's.
    q_present, n_present = qzc > 0, nzc > 0
    r1 = n_present[pos[:, 0]]
    inter = (q_present & r1).sum(axis=1)
    union = (q_present | r1).sum(axis=1)
    jaccard = inter / np.maximum(union, 1)
    qdim = qtab["dimensionality"].astype(int)

    # One frame per distinct base material, so the picks cannot all be the same
    # MOF sampled at different timesteps.  For mof_off this column holds the base
    # MOF id (cif_name prefix); for MAD it holds the subset.
    family = qtab["subdataset"]
    d1 = dist[:, 0]

    def pick(mask, key, reason, name):
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return None
        best = candidates[np.argmin(key[candidates])]
        return {"row": int(best), "reason": reason, "name": name}

    median_d1 = float(np.median(d1))
    largest_family = max(set(family.tolist()), key=lambda f: int((family == f).sum()))
    # A one- or two-atom query wins "closest" trivially and illustrates nothing,
    # so examples are drawn from structures with real chemistry in them.
    big = qtab["n_atoms"].astype(int) >= args.min_atoms
    log(f"{int(big.sum()):,}/{big.size:,} queries have >= {args.min_atoms} atoms")
    selections = [
        pick(big, d1, f"smallest pc25 d1 among queries with >= {args.min_atoms} atoms",
             "closest"),
        pick(big, -jaccard,
             "rank-1 neighbour shares the largest fraction of the query element set",
             "best_chemistry"),
        pick(big & (family == largest_family), np.abs(d1 - median_d1),
             f"median-distance frame of the largest query family ({largest_family})",
             "typical"),
        pick(big & (qdim == 3), np.abs(d1 - median_d1),
             "3D-connected query at median distance", "framework"),
    ]

    chosen, seen = [], set()
    for selection in selections:
        if selection is None:
            continue
        base = family[selection["row"]]
        if base in seen:
            # Fall back to the next-best row outside the families already taken.
            continue
        seen.add(base)
        chosen.append(selection)

    log(f"{query}: exporting {len(chosen)} examples")

    traj_handles, offsets, offset = [], [], 0
    for path in spec["traj"]:
        traj_handles.append(Trajectory(str(root / path)))
        offsets.append(offset)
        offset += len(traj_handles[-1])

    def query_atoms(row):
        for handle, start in zip(reversed(traj_handles), reversed(offsets)):
            if row >= start:
                return handle[row - start]
        raise SystemExit(f"row {row} outside the trajectory set")

    summary = []
    for n, selection in enumerate(chosen, 1):
        row = selection["row"]
        directory = export / f"{query}_{n:02d}_{selection['name']}_row_{row:06d}"
        directory.mkdir(parents=True, exist_ok=True)

        atoms = query_atoms(row)
        write(directory / f"query_{query}.cif", atoms)
        write(directory / f"query_{query}.extxyz", atoms)
        bundle = [atoms]

        table_rows, neighbour_meta = [], []
        qset = element_set(qzc[row])
        for rank in range(K_NEIGHBOURS):
            global_row = int(idx[row, rank])
            shard_idx, local = locate(np.array([global_row]), manifest, starts)
            entry = manifest[int(shard_idx[0])]
            with connect(root / entry["raw_path"], readonly=True, use_lock_file=False) as db:
                nb = db.get(id=db.ids[int(local[0])]).toatoms()
            stem = f"omat24_rank{rank + 1:02d}_global_{global_row}"
            write(directory / f"{stem}.cif", nb)
            write(directory / f"{stem}.extxyz", nb)
            bundle.append(nb)

            nset = element_set(nzc[pos[row, rank]])
            shared = sorted(qset & nset)
            table_rows.append(
                f"| {rank + 1} | `{stem}.cif` | {nb.get_chemical_formula('hill')} | "
                f"{len(nb)} | {dist[row, rank]:.4f} | "
                f"{', '.join(shared) if shared else '(none)'} | "
                f"{len(qset & nset) / max(len(qset | nset), 1):.3f} |"
            )
            neighbour_meta.append({
                "rank": rank + 1, "global_row": global_row,
                "formula": nb.get_chemical_formula("hill"), "n_atoms": len(nb),
                "pc25_distance": float(dist[row, rank]),
                "subdataset": entry["subdataset"], "shard": entry["shard"],
                "shared_elements": shared,
            })

        write(directory / "comparison_all.extxyz", bundle)
        extra = (f"{qtab['subdataset'][row]}, {qtab['shard'][row]}"
                 if query == "mof_off" else f"{qtab['subdataset'][row]} subset")
        (directory / "README.md").write_text(README.format(
            title=f"{query} example {n}: {selection['name']}",
            query_label=qtab["key"][row],
            query_formula=atoms.get_chemical_formula("hill"),
            query_atoms=len(atoms),
            query_extra=extra,
            k=K_NEIGHBOURS,
            reference=reference_rows,
            reason=selection["reason"],
            rows="\n".join(table_rows),
            calc_note="MOF-off and OMAT24 carry energy/forces; MAD does not",
        ))
        atomic_write_json(directory / "metadata.json", {
            "created_utc": utc_now(),
            "query_set": query,
            "query_row": row,
            "query_key": str(qtab["key"][row]),
            "query_formula": atoms.get_chemical_formula("hill"),
            "query_dimensionality": int(qdim[row]),
            "selection_reason": selection["reason"],
            "geometry": "pc25",
            "neighbours": neighbour_meta,
        })
        summary.append({
            "example": n, "name": selection["name"], "directory": directory.name,
            "query_row": row, "query_key": str(qtab["key"][row]),
            "query_formula": atoms.get_chemical_formula("hill"),
            "query_atoms": len(atoms), "query_dimensionality": int(qdim[row]),
            "d1": round(float(dist[row, 0]), 5),
            "d5": round(float(dist[row, K_NEIGHBOURS - 1]), 5),
            "rank1_formula": neighbour_meta[0]["formula"],
            "rank1_shared_elements": "|".join(neighbour_meta[0]["shared_elements"]),
            "rank1_jaccard": round(float(jaccard[row]), 4),
            "reason": selection["reason"],
        })
        log(f"  {directory.name}: {atoms.get_chemical_formula('hill')} -> "
            f"{neighbour_meta[0]['formula']} (d1={dist[row, 0]:.4f})")

    for handle in traj_handles:
        handle.close()

    path = export / f"selection_summary_{query}.csv"
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    os.replace(tmp, path)
    log(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--query", choices=sorted(QUERY_SETS), default="mof_off")
    parser.add_argument("--min-atoms", type=int, default=24,
                        help="ignore queries smaller than this when picking examples")
    args = parser.parse_args()
    command_export(args)


if __name__ == "__main__":
    sys.exit(main())
