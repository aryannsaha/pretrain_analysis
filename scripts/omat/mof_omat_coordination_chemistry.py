#!/usr/bin/env python
"""Metal-node, linker and SBU chemistry of MOF-off against its pc25 top-5 OMAT24 neighbours.

Composition tables cannot tell a porous Zn-imidazolate framework from a dense
Zn nitride with the same element list.  This module reads coordinates and asks
the coordination-chemistry questions directly:

  metal node   CrystalNN coordination number, ChemEnv coordination environment
               with its continuous symmetry measure, bond-valence sum, and the
               identity/distance of every first-shell donor.
  linker       the binding functional group of each metal-bound donor, derived
               from the covalent bond graph (carboxylate, azolate, phosphonate,
               phenolate, aqua/hydroxo, ...), and the analogous condensed-phase
               motif on the OMAT24 side (carbonate, oxalate, carbide, nitride,
               oxide, hydride).
  SBU          secondary building units detected structurally rather than by
               name -- Zn-N4 (ZIF-like), Zn4O (MOF-5-like), Zr6 oxo cluster
               (UiO-like), Cu2 paddlewheel (HKUST-like), M-O chain (MOF-74-like)
               -- since MOF-off identifies materials only by CSD refcode number.

Two bond definitions are used deliberately and are not interchangeable:
CrystalNN supplies the metal coordination number and donor set (it is built for
crystal environments and handles ionic bonds), while the covalent-radius graph
from ``pc25_top5_structure_descriptors`` supplies the organic connectivity used
for functional-group perception (it reproduces C:4 / O:2 / H:1 valences).

SAMPLING.  Coordination analysis on all 80,643 frames would mostly re-measure
the same 3,269 materials at different timesteps.  One frame per distinct MOF is
analysed, taken at the lowest available temperature, because ChemEnv symmetry
measures are meaningless on a thermally shredded snapshot.  The OMAT24 side has
no such choice available: every retrieved neighbour is a 3000 K AIMD frame, and
that asymmetry is itself a result -- it is reported, not hidden.
"""

import argparse
import csv
import logging
import os
import sys
import warnings
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pc25_top5_structure_descriptors import (  # noqa: E402
    BOND_SCALE,
    BOND_SCALE_WIDE,
    DEFAULT_OUT,
    DEFAULT_ROOT,
    K_NEIGHBOURS,
    MAX_Z,
    METAL_MASK,
    QUERY_SETS,
    atomic_write_json,
    atomic_write_npy,
    locate,
    log,
    read_manifest,
    utc_now,
)

from ase.data import chemical_symbols, covalent_radii  # noqa: E402
from ase.neighborlist import natural_cutoffs, neighbor_list  # noqa: E402

RDF_MAX = 8.0
RDF_BINS = 80

SITE_FIELDS = [
    "population", "key", "global_row", "site_index", "element", "Z",
    "cn_crystalnn", "ce_symbol", "csm", "bvs",
    "donor_formula", "n_donor_O", "n_donor_N", "n_donor_S", "n_donor_C",
    "n_donor_halide", "n_donor_metal", "mean_donor_distance", "min_donor_distance",
    "functional_groups",
]
STRUCTURE_FIELDS = [
    "population", "key", "global_row", "formula_hill", "n_atoms", "n_metals",
    "spacegroup_loose", "spacegroup_tight", "crystal_system",
    "motif_carboxylate", "motif_azolate", "motif_pyridyl", "motif_phenolate",
    "motif_phosphonate", "motif_sulfonate", "motif_aqua_hydroxo",
    "phase_carbonate", "phase_oxalate", "phase_formate", "phase_carbide",
    "phase_nitride", "phase_pure_oxide", "phase_hydride", "phase_intermetallic",
    "sbu_zn_n4", "sbu_zn4o", "sbu_zr6", "sbu_cu_paddlewheel", "sbu_m_o_chain",
]

HALIDES = {9, 17, 35, 53}


def bond_graph(atoms):
    """Covalent-radius bond graph as adjacency lists over (neighbour, shift)."""
    numbers = atoms.get_atomic_numbers()
    i, j, d, shifts = neighbor_list(
        "ijdS", atoms, natural_cutoffs(atoms, mult=BOND_SCALE_WIDE))
    radii = covalent_radii[numbers[i]] + covalent_radii[numbers[j]]
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = np.where(radii > 0, d / radii, np.inf) < BOND_SCALE
    adjacency = [[] for _ in range(len(atoms))]
    for a, b, dist, shift in zip(i[keep], j[keep], d[keep], shifts[keep]):
        adjacency[a].append((int(b), float(dist), tuple(int(s) for s in shift)))
    return adjacency, numbers


def ring_size_through(adjacency, start, skeleton=None, max_size=8):
    """Smallest ring through `start` in the unrolled (PBC-aware) bond graph.

    `skeleton` restricts which atoms a ring may pass through.  Linker rings are
    built from C/N/O/S/P, so metals and hydrogen are excluded by the callers:
    without that, a single close metal-H contact in a 3000 K AIMD snapshot
    closes a spurious three-membered ring and the donor is misclassified.
    """
    if skeleton is not None and not skeleton[start]:
        return 0, []
    best, best_members = 0, []
    neighbours = [(b, s) for b, _, s in adjacency[start]
                  if skeleton is None or skeleton[b]]
    for a_index in range(len(neighbours)):
        for b_index in range(a_index + 1, len(neighbours)):
            source = (neighbours[a_index][0], neighbours[a_index][1])
            target = (neighbours[b_index][0], neighbours[b_index][1])
            parent = {source: None}
            seen = {source, (start, (0, 0, 0))}
            queue = deque([(source, 1)])
            while queue:
                node, depth = queue.popleft()
                if node == target:
                    # depth counts atoms along the path from `source`, so the path
                    # holds depth-1 edges; the ring closes through `start` with two
                    # more edges, giving (depth-1)+2 edges = depth+1 atoms.
                    size = depth + 1
                    if best == 0 or size < best:
                        members, cursor = [start], node
                        while cursor is not None:
                            members.append(cursor[0])
                            cursor = parent[cursor]
                        best, best_members = size, members
                    break
                if depth >= max_size - 1:
                    continue
                shift = node[1]
                for nxt, _, step in adjacency[node[0]]:
                    if skeleton is not None and not skeleton[nxt]:
                        continue
                    moved = (shift[0] + step[0], shift[1] + step[1], shift[2] + step[2])
                    if (nxt, moved) not in seen:
                        seen.add((nxt, moved))
                        parent[(nxt, moved)] = node
                        queue.append(((nxt, moved), depth + 1))
    return best, best_members


def ring_skeleton(numbers):
    """Atoms a linker ring may pass through: non-metal heavy atoms only."""
    return ~METAL_MASK[numbers] & (numbers != 1)


def classify_donor(adjacency, numbers, donor):
    """Binding functional group of a metal-bound donor atom."""
    z = numbers[donor]
    partners = [b for b, _, _ in adjacency[donor]]
    partner_z = [numbers[p] for p in partners]
    heavy = [p for p, pz in zip(partners, partner_z) if not METAL_MASK[pz] and pz != 1]

    if z == 8:
        for p in heavy:
            pz = numbers[p]
            neighbour_z = [numbers[q] for q, _, _ in adjacency[p]]
            if pz == 6:
                n_oxygen = sum(1 for q in neighbour_z if q == 8)
                if n_oxygen >= 3:
                    return "carbonate"
                if n_oxygen == 2:
                    return "carboxylate"
                return "phenolate/alkoxide"
            if pz == 15:
                return "phosphonate"
            if pz == 16:
                return "sulfonate"
            if pz == 7:
                return "nitrate/nitro"
            if pz == 5:
                return "borate"
        if any(pz == 1 for pz in partner_z):
            return "aqua/hydroxo"
        if all(METAL_MASK[pz] for pz in partner_z) and partner_z:
            return "oxo (mu-O)"
        return "other O"
    if z == 7:
        ring, members = ring_size_through(adjacency, donor,
                                          skeleton=ring_skeleton(numbers))
        if ring == 5:
            # Imidazolate/triazolate coordinate through one ring N while the other
            # N sits two bonds away, so the test has to count N in the ring itself.
            n_in_ring = sum(1 for m in members if numbers[m] == 7)
            return "azolate (5-ring)" if n_in_ring >= 2 else "5-ring N-heterocycle"
        if ring == 6:
            return "pyridyl (6-ring)"
        if len(heavy) >= 3:
            return "amine/amide"
        if all(METAL_MASK[pz] for pz in partner_z) and partner_z:
            return "nitride N"
        return "other N"
    if z == 16:
        return "thiolate/sulfide"
    if z == 6:
        return "carbide/organometallic C"
    if z in HALIDES:
        return f"{chemical_symbols[z]} halide"
    if z == 1:
        return "hydride"
    return f"other {chemical_symbols[z]}"


def phase_motifs(adjacency, numbers):
    """Condensed-phase anion motifs, the OMAT24-side analogue of linker groups."""
    flags = dict.fromkeys(
        ["carbonate", "oxalate", "formate", "carbide", "nitride", "pure_oxide",
         "hydride", "intermetallic"], 0)
    carbon = np.flatnonzero(numbers == 6)
    for c in carbon:
        partners = [b for b, _, _ in adjacency[c]]
        pz = [numbers[p] for p in partners]
        n_o = sum(1 for v in pz if v == 8)
        n_c = sum(1 for v in pz if v == 6)
        n_h = sum(1 for v in pz if v == 1)
        if n_o >= 3 and n_c == 0:
            flags["carbonate"] = 1
        if n_o == 2 and n_h == 1:
            flags["formate"] = 1
        if n_o == 2 and n_c == 1:
            partner_c = next(p for p, v in zip(partners, pz) if v == 6)
            if sum(1 for q, _, _ in adjacency[partner_c] if numbers[q] == 8) == 2:
                flags["oxalate"] = 1
        if partners and all(METAL_MASK[v] for v in pz):
            flags["carbide"] = 1
    for element, key in ((7, "nitride"), (8, "pure_oxide"), (1, "hydride")):
        sites = np.flatnonzero(numbers == element)
        if sites.size:
            bound = [s for s in sites if adjacency[s]]
            if bound and all(all(METAL_MASK[numbers[b]] for b, _, _ in adjacency[s])
                             for s in bound):
                flags[key] = 1
    if numbers.size and METAL_MASK[numbers].all():
        flags["intermetallic"] = 1
    return flags


def detect_sbus(numbers, site_records, adjacency, positions, cell, pbc):
    """Structural SBU fingerprints, since MOF-off carries no material names."""
    flags = {"zn_n4": 0, "zn4o": 0, "zr6": 0, "cu_paddlewheel": 0, "m_o_chain": 0}
    for record in site_records:
        z, cn = record["Z"], record["cn_crystalnn"]
        if z == 30 and cn == 4 and record["n_donor_N"] == 4:
            flags["zn_n4"] = 1
        if record["n_donor_O"] >= 4 and cn in (5, 6) and record["n_donor_O"] == cn:
            flags["m_o_chain"] = 1
    # Zn4O: one oxygen bridging exactly four Zn.  Zr6: six Zr mutually bridged.
    for site in np.flatnonzero(numbers == 8):
        zinc = sum(1 for b, _, _ in adjacency[site] if numbers[b] == 30)
        if zinc == 4:
            flags["zn4o"] = 1
    zirconium = np.flatnonzero(numbers == 40)
    if zirconium.size >= 6:
        bridged = sum(
            1 for site in np.flatnonzero(numbers == 8)
            if sum(1 for b, _, _ in adjacency[site] if numbers[b] == 40) >= 2
        )
        if bridged >= 4:
            flags["zr6"] = 1
    copper = [r for r in site_records if r["Z"] == 29 and r["n_donor_O"] >= 4]
    if len(copper) >= 2:
        indices = [r["site_index"] for r in copper]
        from ase.geometry import get_distances

        vectors, distances = get_distances(
            positions[indices], positions[indices], cell=cell, pbc=pbc)
        np.fill_diagonal(distances, np.inf)
        if (distances < 3.0).any():
            flags["cu_paddlewheel"] = 1
    return flags


def radial_distribution(atoms):
    _, _, distances = neighbor_list("ijd", atoms, RDF_MAX)
    if distances.size == 0:
        return np.zeros(RDF_BINS, dtype=np.float32)
    hist, edges = np.histogram(distances, bins=RDF_BINS, range=(0.0, RDF_MAX))
    shells = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    density = hist / (len(atoms) * shells)
    total = density.sum()
    return (density / total).astype(np.float32) if total > 0 else density.astype(np.float32)


_WORKER = {}


def _init_worker(root):
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
        SimplestChemenvStrategy,
    )
    from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
        LocalGeometryFinder,
    )
    from pymatgen.analysis.local_env import CrystalNN

    finder = LocalGeometryFinder()
    finder.setup_parameters(centering_type="centroid",
                            include_central_site_in_centroid=True,
                            structure_refinement="none")
    _WORKER.update(root=Path(root), crystal_nn=CrystalNN(),
                   geometry_finder=finder,
                   strategy=SimplestChemenvStrategy(distance_cutoff=1.4,
                                                    angle_cutoff=0.3))
    manifest, starts = read_manifest(Path(root))
    _WORKER["manifest"], _WORKER["starts"] = manifest, starts


def select_sites(metals, cap):
    """At most `cap` metal sites, evenly spaced so the choice is deterministic.

    A MOF frame carries ~4 metal sites; a retrieved 3000 K OMAT24 neighbour carries
    ~24, and ChemEnv's cost is per site and rises steeply with coordination number,
    so the neighbour side would otherwise dominate the runtime by two orders of
    magnitude.  Site statistics are pooled over thousands of structures, so a
    per-structure cap costs precision that is far below any effect reported here.
    """
    if cap <= 0 or metals.size <= cap:
        return metals
    return metals[np.linspace(0, metals.size - 1, cap).round().astype(int)]


def analyse_structure(atoms, population, key, global_row, chemenv=True, site_cap=0):
    from pymatgen.analysis.bond_valence import calculate_bv_sum
    from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
        LightStructureEnvironments,
    )
    from pymatgen.io.ase import AseAtomsAdaptor

    adjacency, numbers = bond_graph(atoms)
    all_metals = np.flatnonzero(METAL_MASK[numbers])
    metals = select_sites(all_metals, site_cap)
    structure = AseAtomsAdaptor.get_structure(atoms)

    environments = {}
    if chemenv and metals.size:
        try:
            finder = _WORKER["geometry_finder"]
            finder.setup_structure(structure=structure)
            se = finder.compute_structure_environments(
                only_indices=metals.tolist(), maximum_distance_factor=1.45,
                minimum_angle_factor=0.3)
            lse = LightStructureEnvironments.from_structure_environments(
                strategy=_WORKER["strategy"], structure_environments=se)
            for site in metals:
                entry = lse.coordination_environments[int(site)]
                if entry:
                    environments[int(site)] = (entry[0]["ce_symbol"], entry[0]["csm"])
        except Exception:
            environments = {}

    site_records = []
    for site in metals:
        site = int(site)
        record = {field: "" for field in SITE_FIELDS}
        # Donor counts must stay numeric even when CrystalNN returns no neighbours,
        # otherwise the SBU tests compare a string against an int.
        record.update({field: 0 for field in SITE_FIELDS if field.startswith("n_donor_")})
        record.update(population=population, key=key, global_row=global_row,
                      site_index=site, element=chemical_symbols[numbers[site]],
                      Z=int(numbers[site]), cn_crystalnn=0)
        try:
            info = _WORKER["crystal_nn"].get_nn_info(structure, site)
        except Exception:
            info = []
        record["cn_crystalnn"] = len(info)
        if info:
            donor_z = [structure[x["site_index"]].specie.Z for x in info]
            distances = [structure[site].distance(x["site"]) for x in info]
            counts = Counter(donor_z)
            record.update(
                donor_formula="".join(
                    f"{chemical_symbols[z]}{counts[z]}" for z in sorted(counts)),
                n_donor_O=counts.get(8, 0), n_donor_N=counts.get(7, 0),
                n_donor_S=counts.get(16, 0), n_donor_C=counts.get(6, 0),
                n_donor_halide=sum(counts.get(h, 0) for h in HALIDES),
                n_donor_metal=sum(v for k, v in counts.items() if METAL_MASK[k]),
                mean_donor_distance=round(float(np.mean(distances)), 4),
                min_donor_distance=round(float(np.min(distances)), 4),
            )
            try:
                record["bvs"] = round(float(calculate_bv_sum(
                    structure[site], [x["site"] for x in info])), 4)
            except Exception:
                record["bvs"] = ""
            groups = Counter(
                classify_donor(adjacency, numbers, b)
                for b, _, _ in adjacency[site]
                if not METAL_MASK[numbers[b]]
            )
            record["functional_groups"] = "|".join(
                f"{name}:{count}" for name, count in groups.most_common())
        if site in environments:
            # ChemEnv reports a symbol with csm=None when the strategy accepts an
            # environment but cannot score it; keep the label, leave csm blank.
            record["ce_symbol"], csm = environments[site]
            record["csm"] = round(float(csm), 3) if csm is not None else ""
        site_records.append(record)

    structure_record = {field: "" for field in STRUCTURE_FIELDS}
    structure_record.update(
        population=population, key=key, global_row=global_row,
        formula_hill=atoms.get_chemical_formula("hill"), n_atoms=len(atoms),
        n_metals=int(all_metals.size),
    )
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        for label, symprec in (("spacegroup_loose", 0.5), ("spacegroup_tight", 0.1)):
            try:
                analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
                structure_record[label] = analyzer.get_space_group_number()
                if label == "spacegroup_loose":
                    structure_record["crystal_system"] = analyzer.get_crystal_system()
            except Exception:
                structure_record[label] = ""
    except Exception:
        pass

    donor_groups = Counter()
    for record in site_records:
        for item in str(record["functional_groups"]).split("|"):
            if ":" in item:
                name, count = item.rsplit(":", 1)
                donor_groups[name] += int(count)
    for key_name, label in (
        ("carboxylate", "motif_carboxylate"), ("azolate (5-ring)", "motif_azolate"),
        ("pyridyl (6-ring)", "motif_pyridyl"),
        ("phenolate/alkoxide", "motif_phenolate"),
        ("phosphonate", "motif_phosphonate"), ("sulfonate", "motif_sulfonate"),
        ("aqua/hydroxo", "motif_aqua_hydroxo"),
    ):
        structure_record[label] = int(donor_groups.get(key_name, 0) > 0)
    for name, value in phase_motifs(adjacency, numbers).items():
        structure_record[f"phase_{name}"] = value
    for name, value in detect_sbus(numbers, site_records, adjacency,
                                   atoms.get_positions(), atoms.cell.array,
                                   atoms.pbc).items():
        structure_record[f"sbu_{name}"] = value

    return site_records, structure_record, radial_distribution(atoms)


def safe_analyse(atoms, population, key, global_row, chemenv, site_cap):
    """analyse_structure, but a single pathological structure cannot kill the run.

    These are 3000 K AIMD snapshots; pymatgen occasionally raises deep inside
    Voronoi or ChemEnv on a degenerate configuration.  Losing one structure out
    of thousands is acceptable, losing the whole sweep is not, so failures are
    counted and reported rather than propagated.
    """
    try:
        return analyse_structure(atoms, population, key, global_row,
                                 chemenv=chemenv, site_cap=site_cap)
    except Exception as error:  # noqa: BLE001 - deliberate catch-all, see docstring
        log(f"  skipped {population} {global_row}: {type(error).__name__} {error}")
        return [], None, None


def _mof_task(payload):
    from ase.io import Trajectory

    traj_path, rows, keys, chemenv, site_cap = payload
    traj = Trajectory(traj_path)
    sites, structures, rdfs = [], [], []
    try:
        for row, key in zip(rows, keys):
            s, st, rdf = safe_analyse(traj[int(row)], "mof_off", key, int(row),
                                      chemenv, site_cap)
            if st is None:
                continue
            sites.extend(s)
            structures.append(st)
            rdfs.append(rdf)
    finally:
        traj.close()
    return sites, structures, rdfs


def _omat_task(payload):
    from ase.db import connect

    shard_index, rows, chemenv, site_cap = payload
    manifest = _WORKER["manifest"]
    entry = manifest[shard_index]
    local = np.asarray(rows, dtype=np.int64) - int(entry["global_start"])
    sites, structures, rdfs = [], [], []
    with connect(_WORKER["root"] / entry["raw_path"], readonly=True,
                 use_lock_file=False) as db:
        ids = db.ids
        for lr, gr in zip(local, rows):
            atoms = db.get(id=ids[int(lr)]).toatoms()
            s, st, rdf = safe_analyse(atoms, "omat_nb", str(int(gr)), int(gr),
                                      chemenv, site_cap)
            if st is None:
                continue
            sites.extend(s)
            structures.append(st)
            rdfs.append(rdf)
    return sites, structures, rdfs


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    log(f"wrote {path.name} ({len(rows):,} rows)")


def command_analyse(args):
    from multiprocessing import Pool

    root = args.root.resolve()
    out_root = (args.out or root / DEFAULT_OUT).resolve()
    out = out_root / "coordination"
    out.mkdir(parents=True, exist_ok=True)

    spec = QUERY_SETS["mof_off"]
    with (root / spec["meta_tsv"][0]).open() as handle:
        meta = list(csv.DictReader(handle, delimiter="\t"))
    idx = np.load(root / spec["indices"] / "indices.npy")[:, :K_NEIGHBOURS]

    # One frame per distinct MOF, at the lowest temperature sampled for it.
    chosen = {}
    for row, entry in enumerate(meta):
        mof = entry["cif_name"].split("_")[0]
        temperature = int(entry["temperature_K"])
        if mof not in chosen or temperature < chosen[mof][1]:
            chosen[mof] = (row, temperature)
    selected = sorted(v[0] for v in chosen.values())
    if args.limit_mofs:
        selected = selected[: args.limit_mofs]
    selected = np.array(selected, dtype=np.int64)
    log(f"selected {selected.size:,} MOF frames (one per distinct MOF, lowest T)")

    neighbour_rows = np.unique(idx[selected])
    log(f"their pc25 top-5 neighbours: {neighbour_rows.size:,} unique OMAT rows")

    manifest, starts = read_manifest(root)
    chunk = args.chunk
    mof_tasks = [
        (str(root / spec["traj"][0]), selected[start:start + chunk],
         [meta[int(r)]["record_id"] for r in selected[start:start + chunk]],
         not args.no_chemenv, args.site_cap)
        for start in range(0, selected.size, chunk)
    ]
    manifest_index, _ = locate(neighbour_rows, manifest, starts)
    omat_tasks = [
        (int(s), neighbour_rows[manifest_index == s], not args.no_chemenv,
         args.site_cap)
        for s in np.unique(manifest_index)
    ]

    all_sites, all_structures, all_rdfs, rdf_keys = [], [], [], []
    with Pool(args.workers, initializer=_init_worker, initargs=(str(root),)) as pool:
        for label, tasks, fn in (("mof_off", mof_tasks, _mof_task),
                                 ("omat_nb", omat_tasks, _omat_task)):
            done = 0
            for sites, structures, rdfs in pool.imap_unordered(fn, tasks, chunksize=1):
                all_sites.extend(sites)
                all_structures.extend(structures)
                all_rdfs.extend(rdfs)
                rdf_keys.extend((s["population"], s["global_row"]) for s in structures)
                done += 1
                if done % 20 == 0 or done == len(tasks):
                    log(f"  {label}: {done:,}/{len(tasks):,} tasks "
                        f"({len(all_structures):,} structures)")

    write_rows(out / "metal_sites.csv", SITE_FIELDS, all_sites)
    write_rows(out / "structures.csv", STRUCTURE_FIELDS, all_structures)
    atomic_write_npy(out / "rdf.npy", np.asarray(all_rdfs, dtype=np.float32))
    write_rows(out / "rdf_index.csv", ["population", "global_row"],
               [{"population": p, "global_row": g} for p, g in rdf_keys])
    atomic_write_json(out / "coordination_metadata.json", {
        "created_utc": utc_now(),
        "geometry": "pc25",
        "k_neighbours": K_NEIGHBOURS,
        "mof_frames_analysed": int(selected.size),
        "omat_neighbours_analysed": int(neighbour_rows.size),
        "mof_frame_rule": "one frame per distinct MOF at the lowest sampled temperature",
        "bond_definitions": {
            "metal_coordination": "pymatgen CrystalNN",
            "coordination_environment": "pymatgen ChemEnv SimplestChemenvStrategy "
                                        "(distance_cutoff=1.4, angle_cutoff=0.3)",
            "organic_connectivity": f"covalent radii, scale {BOND_SCALE}",
        },
        "rdf": {"max_r": RDF_MAX, "bins": RDF_BINS, "normalised": True},
        "chemenv_enabled": not args.no_chemenv,
        "metal_sites_per_structure_cap": args.site_cap,
        "workers": args.workers,
    })
    log("coordination analysis complete")


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def command_aggregate(args):
    root = args.root.resolve()
    out_root = (args.out or root / DEFAULT_OUT).resolve()
    out = out_root / "coordination"
    analysis = out_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    sites = read_rows(out / "metal_sites.csv")
    structures = read_rows(out / "structures.csv")
    spec = QUERY_SETS["mof_off"]
    with (root / spec["meta_tsv"][0]).open() as handle:
        meta = list(csv.DictReader(handle, delimiter="\t"))
    idx = np.load(root / spec["indices"] / "indices.npy")[:, :K_NEIGHBOURS]

    mof_sites, omat_sites = {}, {}
    for record in sites:
        target = mof_sites if record["population"] == "mof_off" else omat_sites
        target.setdefault(as_int(record["global_row"]), []).append(record)
    mof_struct = {as_int(r["global_row"]): r for r in structures
                  if r["population"] == "mof_off"}
    omat_struct = {as_int(r["global_row"]): r for r in structures
                   if r["population"] == "omat_nb"}
    analysed_mofs = sorted(mof_struct)
    log(f"aggregating {len(analysed_mofs):,} MOF frames and "
        f"{len(omat_struct):,} OMAT neighbours")

    def primary_metal(records):
        counts = Counter(r["element"] for r in records)
        return counts.most_common(1)[0][0] if counts else ""

    def donor_signature(record):
        for label, field in (("O", "n_donor_O"), ("N", "n_donor_N"),
                             ("S", "n_donor_S"), ("C", "n_donor_C"),
                             ("halide", "n_donor_halide")):
            if as_int(record[field]) > 0:
                yield label

    # --- section 2: metal node comparison ------------------------------------
    per_mof, cn_deltas = [], []
    for row in analysed_mofs:
        records = mof_sites.get(row, [])
        if not records:
            continue
        metal = primary_metal(records)
        metal_records = [r for r in records if r["element"] == metal]
        mof_cn = float(np.mean([as_int(r["cn_crystalnn"]) for r in metal_records]))
        mof_donors = set()
        for record in metal_records:
            mof_donors.update(donor_signature(record))
        mof_bvs = np.nanmean([as_float(r["bvs"]) for r in metal_records])
        mof_csm = np.nanmean([as_float(r["csm"]) for r in metal_records])

        shares_metal = same_cn = same_donor = 0
        neighbour_cn = []
        for neighbour in idx[row]:
            neighbour_records = omat_sites.get(int(neighbour), [])
            matched = [r for r in neighbour_records if r["element"] == metal]
            if matched:
                shares_metal = 1
                cn = float(np.mean([as_int(r["cn_crystalnn"]) for r in matched]))
                neighbour_cn.append(cn)
                if abs(cn - mof_cn) < 0.5:
                    same_cn = 1
                donors = set()
                for record in matched:
                    donors.update(donor_signature(record))
                if donors & mof_donors:
                    same_donor = 1
        if neighbour_cn:
            cn_deltas.append(float(np.mean(neighbour_cn)) - mof_cn)

        entry = mof_struct[row]
        per_mof.append({
            "mof_row": row,
            "record_id": meta[row]["record_id"],
            "mof_id": meta[row]["cif_name"].split("_")[0],
            "temperature_K": meta[row]["temperature_K"],
            "formula": entry["formula_hill"],
            "primary_metal": metal,
            "n_metal_sites": len(metal_records),
            "mof_cn": round(mof_cn, 3),
            "mof_bvs": round(float(mof_bvs), 3) if np.isfinite(mof_bvs) else "",
            "mof_csm": round(float(mof_csm), 3) if np.isfinite(mof_csm) else "",
            "mof_donors": "+".join(sorted(mof_donors)),
            "neighbour_shares_metal": shares_metal,
            "neighbour_same_cn": same_cn,
            "neighbour_shares_donor_type": same_donor,
            "neighbour_mean_cn": round(float(np.mean(neighbour_cn)), 3)
            if neighbour_cn else "",
            "cn_delta": round(float(np.mean(neighbour_cn)) - mof_cn, 3)
            if neighbour_cn else "",
            "sbu": "|".join(k[4:] for k in entry if k.startswith("sbu_")
                            and as_int(entry[k])),
            "linker_groups": "|".join(k[6:] for k in entry if k.startswith("motif_")
                                      and as_int(entry[k])),
        })
    write_rows(analysis / "per_mof_summary.csv", list(per_mof[0]), per_mof)

    metal_rows = []
    for element in sorted({r["primary_metal"] for r in per_mof if r["primary_metal"]}):
        subset = [r for r in per_mof if r["primary_metal"] == element]
        if len(subset) < args.min_mofs_per_metal:
            continue
        deltas = [as_float(r["cn_delta"]) for r in subset
                  if r["cn_delta"] != "" and np.isfinite(as_float(r["cn_delta"]))]
        metal_rows.append({
            "metal": element,
            "n_mofs": len(subset),
            "mof_mean_cn": round(float(np.mean([r["mof_cn"] for r in subset])), 3),
            "mof_mean_bvs": round(float(np.nanmean(
                [as_float(r["mof_bvs"]) for r in subset])), 3),
            "pct_neighbour_shares_metal": round(
                100.0 * np.mean([r["neighbour_shares_metal"] for r in subset]), 2),
            "pct_neighbour_same_cn": round(
                100.0 * np.mean([r["neighbour_same_cn"] for r in subset]), 2),
            "pct_neighbour_shares_donor": round(
                100.0 * np.mean([r["neighbour_shares_donor_type"] for r in subset]), 2),
            "median_cn_delta": round(float(np.median(deltas)), 3) if deltas else "",
        })
    write_rows(analysis / "metal_node_agreement.csv", list(metal_rows[0]), metal_rows)

    # ChemEnv coverage and symmetry-measure contrast.
    chemenv_rows = []
    for label, collection in (("MOF-off (lowest T)", mof_sites),
                              ("OMAT24 pc25 top-5 (3000 K AIMD)", omat_sites)):
        records = [r for group in collection.values() for r in group]
        csm = np.array([as_float(r["csm"]) for r in records])
        resolved = np.isfinite(csm)
        environments = Counter(r["ce_symbol"] for r in records if r["ce_symbol"])
        chemenv_rows.append({
            "population": label,
            "metal_sites": len(records),
            "chemenv_resolved_pct": round(100.0 * resolved.mean(), 2),
            "median_csm": round(float(np.median(csm[resolved])), 3) if resolved.any() else "",
            "pct_csm_below_2": round(100.0 * float((csm[resolved] < 2).mean()), 2)
            if resolved.any() else "",
            "top_environments": "; ".join(
                f"{k} {100.0 * v / max(sum(environments.values()), 1):.1f}%"
                for k, v in environments.most_common(5)),
            "mean_cn": round(float(np.mean([as_int(r["cn_crystalnn"]) for r in records])), 3),
        })
    write_rows(analysis / "chemenv_summary.csv", list(chemenv_rows[0]), chemenv_rows)

    # --- section 3: linker groups vs condensed-phase motifs -------------------
    linker_rows = []
    motif_keys = [k for k in structures[0] if k.startswith("motif_")]
    phase_keys = [k for k in structures[0] if k.startswith("phase_")]
    for key in motif_keys + phase_keys:
        entry = {"flag": key}
        for label, collection in (("MOF-off", mof_struct), ("OMAT24 neighbours", omat_struct)):
            values = [as_int(r[key]) for r in collection.values()]
            entry[label] = round(100.0 * float(np.mean(values)), 2) if values else ""
        linker_rows.append(entry)
    write_rows(analysis / "linker_and_phase_motifs.csv", list(linker_rows[0]), linker_rows)

    # Does an organic-rich MOF simply retrieve carbon-rich phases?
    organic_rows = []
    zc_mof = np.load(out_root / "structures" / "query_mof_off_zcounts.npy")
    zc_nb = np.load(out_root / "structures" / "omat_nb_mof_off_zcounts.npy")
    with (out_root / "structures" / "omat_nb_mof_off.csv").open(newline="") as handle:
        nb_rows_all = np.array([int(r["global_row"]) for r in csv.DictReader(handle)])
    organic_z = [1, 6, 7, 8]
    frac_organic = zc_mof[:, organic_z].sum(axis=1) / np.maximum(zc_mof.sum(axis=1), 1)
    nb_frac_c = zc_nb[:, 6] / np.maximum(zc_nb.sum(axis=1), 1)
    nb_frac_metal = (zc_nb * METAL_MASK[None, : MAX_Z + 1]).sum(axis=1) / \
        np.maximum(zc_nb.sum(axis=1), 1)
    edges = np.array([0.0, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01])
    bucket = np.digitize(frac_organic, edges) - 1
    for b in range(len(edges) - 1):
        rows = np.flatnonzero(bucket == b)
        if rows.size < 50:
            continue
        positions = np.searchsorted(nb_rows_all, idx[rows].ravel())
        organic_rows.append({
            "mof_organic_fraction": f"{edges[b]:.2f}-{edges[b + 1]:.2f}",
            "n_frames": int(rows.size),
            "neighbour_mean_C_fraction": round(float(nb_frac_c[positions].mean()), 4),
            "neighbour_mean_metal_fraction": round(float(nb_frac_metal[positions].mean()), 4),
            "neighbour_pct_containing_C": round(
                100.0 * float((zc_nb[positions, 6] > 0).mean()), 2),
            "neighbour_pct_containing_H": round(
                100.0 * float((zc_nb[positions, 1] > 0).mean()), 2),
        })
    write_rows(analysis / "organic_fraction_vs_neighbours.csv",
               list(organic_rows[0]), organic_rows)

    # --- section 7: SBU control set ------------------------------------------
    sbu_targets = {
        "zn_n4": ("ZIF-like Zn-N4", "Zn", "N"),
        "zn4o": ("MOF-5-like Zn4O", "Zn", "O"),
        "zr6": ("UiO-like Zr6 oxo cluster", "Zr", "O"),
        "cu_paddlewheel": ("HKUST-like Cu paddlewheel", "Cu", "O"),
        "m_o_chain": ("MOF-74-like M-O chain", None, "O"),
    }
    sbu_rows = []
    for key, (label, metal, donor) in sbu_targets.items():
        members = [r for r in per_mof if key in str(r["sbu"]).split("|")]
        if not members:
            sbu_rows.append({"sbu": label, "n_mofs": 0, "target_metal": metal or "same as MOF",
                             "target_donor": donor, "pct_neighbour_has_metal": "",
                             "pct_neighbour_has_metal_with_donor": "", "example_mofs": ""})
            continue
        hit_metal = hit_env = 0
        for record in members:
            want = metal or record["primary_metal"]
            found_metal = found_env = False
            for neighbour in idx[record["mof_row"]]:
                matched = [r for r in omat_sites.get(int(neighbour), [])
                           if r["element"] == want]
                if matched:
                    found_metal = True
                    if any(as_int(r[f"n_donor_{donor}"]) > 0 for r in matched):
                        found_env = True
            hit_metal += found_metal
            hit_env += found_env
        sbu_rows.append({
            "sbu": label,
            "n_mofs": len(members),
            "target_metal": metal or "same as MOF",
            "target_donor": donor,
            "pct_neighbour_has_metal": round(100.0 * hit_metal / len(members), 2),
            "pct_neighbour_has_metal_with_donor": round(100.0 * hit_env / len(members), 2),
            "example_mofs": ", ".join(r["mof_id"] for r in members[:3]),
        })
    write_rows(analysis / "sbu_control_set.csv", list(sbu_rows[0]), sbu_rows)

    # --- section 5: independent similarity check via the RDF ------------------
    rdf = np.load(out / "rdf.npy")
    rdf_index = read_rows(out / "rdf_index.csv")
    position = {(r["population"], as_int(r["global_row"])): i
                for i, r in enumerate(rdf_index)}
    normed = rdf / np.maximum(np.linalg.norm(rdf, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(args.seed)
    omat_positions = np.array([position[("omat_nb", r)] for r in sorted(omat_struct)])
    matched_similarity, random_similarity = [], []
    for row in analysed_mofs:
        anchor = position.get(("mof_off", row))
        if anchor is None:
            continue
        neighbours = [position[("omat_nb", int(n))] for n in idx[row]
                      if ("omat_nb", int(n)) in position]
        if not neighbours:
            continue
        matched_similarity.append(float(np.mean(normed[anchor] @ normed[neighbours].T)))
        draw = rng.choice(omat_positions, size=len(neighbours), replace=False)
        random_similarity.append(float(np.mean(normed[anchor] @ normed[draw].T)))
    matched_similarity = np.array(matched_similarity)
    random_similarity = np.array(random_similarity)
    rdf_rows = [{
        "comparison": label,
        "n": int(values.size),
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
    } for label, values in (
        ("MOF vs its pc25 top-5 neighbours", matched_similarity),
        ("MOF vs random analysed OMAT24 structures", random_similarity),
    )]
    rdf_rows.append({
        "comparison": "paired difference (matched - random)",
        "n": int(matched_similarity.size),
        "mean": round(float((matched_similarity - random_similarity).mean()), 4),
        "median": round(float(np.median(matched_similarity - random_similarity)), 4),
        "p10": round(float(np.percentile(matched_similarity - random_similarity, 10)), 4),
        "p90": round(float(np.percentile(matched_similarity - random_similarity, 90)), 4),
    })
    write_rows(analysis / "rdf_similarity.csv", list(rdf_rows[0]), rdf_rows)

    # --- space groups ---------------------------------------------------------
    spacegroup_rows = []
    for label, collection in (("MOF-off (lowest T)", mof_struct),
                              ("OMAT24 pc25 top-5", omat_struct)):
        loose = [as_int(r["spacegroup_loose"], 0) for r in collection.values()]
        systems = Counter(r["crystal_system"] for r in collection.values()
                          if r["crystal_system"])
        spacegroup_rows.append({
            "population": label,
            "n": len(loose),
            "pct_P1_symprec_0.5": round(100.0 * float(np.mean(np.array(loose) == 1)), 2),
            "pct_symmetric": round(100.0 * float(np.mean(np.array(loose) > 1)), 2),
            "top_crystal_systems": "; ".join(
                f"{k} {100.0 * v / max(sum(systems.values()), 1):.1f}%"
                for k, v in systems.most_common(4)),
        })
    write_rows(analysis / "spacegroup_summary.csv",
               list(spacegroup_rows[0]), spacegroup_rows)

    atomic_write_json(analysis / "coordination_aggregate_metadata.json", {
        "created_utc": utc_now(),
        "mofs_analysed": len(analysed_mofs),
        "omat_neighbours_analysed": len(omat_struct),
        "cn_match_tolerance": 0.5,
        "pct_neighbour_shares_metal": round(
            100.0 * float(np.mean([r["neighbour_shares_metal"] for r in per_mof])), 3),
        "pct_neighbour_same_cn": round(
            100.0 * float(np.mean([r["neighbour_same_cn"] for r in per_mof])), 3),
        "pct_neighbour_shares_donor_type": round(
            100.0 * float(np.mean([r["neighbour_shares_donor_type"] for r in per_mof])), 3),
        "median_cn_delta": round(float(np.median(cn_deltas)), 3) if cn_deltas else None,
        "seed": args.seed,
    })
    log("aggregation complete")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    analyse = sub.add_parser("analyse")
    analyse.add_argument("--workers", type=int, default=16)
    analyse.add_argument("--chunk", type=int, default=40)
    analyse.add_argument("--limit-mofs", type=int, default=0)
    analyse.add_argument("--no-chemenv", action="store_true")
    analyse.add_argument("--site-cap", type=int, default=6,
                         help="max metal sites analysed per structure (0 = all)")
    analyse.set_defaults(func=command_analyse)

    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--min-mofs-per-metal", type=int, default=20)
    aggregate.add_argument("--seed", type=int, default=20260804)
    aggregate.set_defaults(func=command_aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
