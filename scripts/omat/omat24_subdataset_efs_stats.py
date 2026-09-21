#!/usr/bin/env python
"""Full-population energy / force / stress statistics per OMAT24 train subdataset.

Two stages, same shape as scripts/analysis/compute_energy_kl.py:

  map     scan a stride of the raw .aselmdb shards and accumulate, per subdataset,
          exact running moments plus a fine histogram for every channel
  reduce  merge the parts, check frame counts against the official totals, and
          write the statistics (JSON / CSV / Markdown) and the figures

Nothing is sampled: every frame, every atom's force and every stress tensor is
counted. Moments (mean, std, RMS, min, max) are exact; quantiles come from the
histograms, whose bins are 1 meV/atom for energy and <0.5% relative for the
symlog force/stress channels.

Stress is stored as a 6-vector in eV/A^3 (Voigt xx yy zz yz xz xy) and reported
here in GPa with the sign left exactly as stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
DEFAULT_OMAT_ROOT = ROOT / "data/raw/omat24/train"
DEFAULT_OMAT_RECOVERED_ROOT = ROOT / "data/raw/omat24/missing_train"
DEFAULT_OUTPUT_DIR = ROOT / "runs/omat24_subdataset_efs_stats"

EV_A3_TO_GPA = 160.21766208

# Presentation order: the 3000 K sets first, then 1000 K, then the rattled family.
ORDER = [
    "aimd-from-PBE-3000-nvt", "aimd-from-PBE-3000-npt",
    "aimd-from-PBE-1000-npt", "aimd-from-PBE-1000-nvt",
    "rattled-relax", "rattled-300", "rattled-300-subsampled",
    "rattled-500", "rattled-500-subsampled",
    "rattled-1000", "rattled-1000-subsampled",
]
ALL = "ALL"
OFFICIAL = {  # frames per subdataset in the 100,824,585-row embedding manifest
    "aimd-from-PBE-1000-npt": 21269486, "aimd-from-PBE-1000-nvt": 20256650,
    "aimd-from-PBE-3000-npt": 6076290, "aimd-from-PBE-3000-nvt": 7839846,
    "rattled-1000": 11388510, "rattled-1000-subsampled": 3879741, "rattled-300": 6319139,
    "rattled-300-subsampled": 3464007, "rattled-500": 6922197, "rattled-500-subsampled": 3975416,
    "rattled-relax": 9433303,
}
# Same entity colors as scripts/omat/plot_omat24_subdataset_natoms.py.
COLORS = {
    "aimd-from-PBE-3000-nvt": "#1baf7a", "aimd-from-PBE-3000-npt": "#eb6834",
    "aimd-from-PBE-1000-nvt": "#2a78d6", "aimd-from-PBE-1000-npt": "#4a3aa7",
    "rattled-1000": "#eda100", "rattled-1000-subsampled": "#e87ba4", "rattled-300": "#008300",
    "rattled-300-subsampled": "#e34948", "rattled-500": "#898781",
    "rattled-500-subsampled": "#0d366b", "rattled-relax": "#6e280c", ALL: "#52514e",
}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

QUANTILES = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
MOMENT_FIELDS = ["count", "nonfinite", "sum", "sumsq", "sumabs", "min", "max", "under", "over"]
COUNTER_FIELDS = ["frames", "atoms", "shards", "empty_shards", "missing_forces",
                  "missing_stress", "zero_stress"]
NATOMS_BINS = 2048


def linear_edges(lo: float, hi: float, width: float) -> np.ndarray:
    return np.linspace(lo, hi, int(round((hi - lo) / width)) + 1)


def symlog_edges(x0: float, tmax: float, dt: float, signed: bool) -> np.ndarray:
    """Uniform bins in t = sign(x) log10(1 + |x|/x0): linear below x0, log above."""
    t = np.arange(0.0, tmax + dt / 2, dt)
    pos = x0 * (10.0 ** t - 1.0)
    return np.concatenate([-pos[:0:-1], pos]) if signed else pos


# channel -> (unit, edges). 1 meV/atom energy bins; symlog bins are 0.46% wide in
# the tails and 4.6e-6 wide at the origin, reaching 1e5 eV/A and 1e5 GPa.
CHANNELS = {
    "e_per_atom": ("eV/atom", linear_edges(-100.0, 100.0, 0.001)),
    "e_total": ("eV", linear_edges(-6000.0, 6000.0, 0.5)),
    "f_comp": ("eV/A", symlog_edges(1e-3, 8.0, 0.002, True)),
    "f_norm": ("eV/A", symlog_edges(1e-3, 8.0, 0.002, False)),
    "f_rms_frame": ("eV/A", symlog_edges(1e-3, 8.0, 0.002, False)),
    "f_max_frame": ("eV/A", symlog_edges(1e-3, 8.0, 0.002, False)),
    "s_hydro": ("GPa", symlog_edges(1e-3, 8.0, 0.002, True)),
    "s_normal": ("GPa", symlog_edges(1e-3, 8.0, 0.002, True)),
    "s_shear": ("GPa", symlog_edges(1e-3, 8.0, 0.002, True)),
    "s_vonmises": ("GPa", symlog_edges(1e-3, 8.0, 0.002, False)),
    "vol_per_atom": ("A^3/atom", symlog_edges(1.0, 5.0, 0.0005, False)),
}
CHANNEL_NOTES = {
    "e_per_atom": "total DFT energy / N atoms, one value per frame",
    "e_total": "total DFT energy, one value per frame",
    "f_comp": "signed Cartesian force components, three values per atom",
    "f_norm": "|F| per atom",
    "f_rms_frame": "sqrt(mean |F|^2) over the atoms of a frame",
    "f_max_frame": "max |F| over the atoms of a frame",
    "s_hydro": "tr(sigma)/3, one value per frame, sign as stored",
    "s_normal": "sigma_xx, sigma_yy, sigma_zz, three values per frame",
    "s_shear": "sigma_yz, sigma_xz, sigma_xy, three values per frame",
    "s_vonmises": "von Mises equivalent stress, one value per frame",
    "vol_per_atom": "|det(cell)| / N atoms, one value per frame",
}


class Channel:
    def __init__(self, edges: np.ndarray):
        self.edges = edges
        self.hist = np.zeros(edges.size - 1, dtype=np.int64)
        self.mom = np.zeros(len(MOMENT_FIELDS), dtype=np.float64)
        self.mom[5], self.mom[6] = math.inf, -math.inf

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64).ravel()
        finite = np.isfinite(values)
        if not finite.all():
            self.mom[1] += values.size - int(finite.sum())
            values = values[finite]
        if values.size == 0:
            return
        self.hist += np.histogram(values, bins=self.edges)[0]
        m = self.mom
        m[0] += values.size
        m[2] += values.sum()
        m[3] += np.square(values).sum()
        m[4] += np.abs(values).sum()
        m[5] = min(m[5], values.min())
        m[6] = max(m[6], values.max())
        m[7] += np.count_nonzero(values < self.edges[0])
        m[8] += np.count_nonzero(values > self.edges[-1])

    def merge(self, hist: np.ndarray, mom: np.ndarray) -> None:
        self.hist += hist
        for i in (0, 1, 2, 3, 4, 7, 8):
            self.mom[i] += mom[i]
        self.mom[5] = min(self.mom[5], mom[5])
        self.mom[6] = max(self.mom[6], mom[6])


class Accumulator:
    """Everything tracked for one subdataset."""

    def __init__(self):
        self.channels = {name: Channel(edges) for name, (_, edges) in CHANNELS.items()}
        self.natoms = np.zeros(NATOMS_BINS, dtype=np.int64)
        self.counters = dict.fromkeys(COUNTER_FIELDS, 0)
        self.extremes: dict[str, dict] = {}

    def offer_extreme(self, name: str, value: float, where: dict, largest: bool) -> None:
        best = self.extremes.get(name)
        if best is None or (value > best["value"] if largest else value < best["value"]):
            self.extremes[name] = {"value": float(value), **where}


def shard_sort_key(relative_path: str) -> tuple[str, int, str]:
    subdataset, filename = relative_path.split("/", 1)
    try:
        shard_index = int(filename.removeprefix("db_").removesuffix(".aselmdb"))
    except ValueError:
        shard_index = -1
    return subdataset, shard_index, filename


def omat_shards(root: Path, recovered_root: Path) -> list[tuple[str, Path]]:
    """Return unique OMAT shards, preferring recovered copies on overlap."""
    shards: dict[str, Path] = {}
    for source_root in (root, recovered_root):
        if not source_root.exists():
            continue
        for path in source_root.glob("*/*.aselmdb"):
            shards[str(path.relative_to(source_root))] = path
    return sorted(shards.items(), key=lambda item: shard_sort_key(item[0]))


def manifest_sha256(shards: list[tuple[str, Path]]) -> str:
    return hashlib.sha256("\n".join(rel for rel, _ in shards).encode()).hexdigest()


def scan_shard(relative: str, path: Path, acc: Accumulator) -> int:
    import lmdb
    import orjson

    keys, sids, energies, natoms, cells, forces, stresses, stress_rows = [], [], [], [], [], [], [], []
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False,
                    readahead=False, max_readers=1)
    try:
        with env.begin(buffers=True) as txn:
            for key, compressed in txn.cursor():
                if not bytes(key).isdigit():
                    continue
                record = orjson.loads(zlib.decompress(compressed))
                n = len(record["numbers"])
                row = len(energies)
                keys.append(int(bytes(key)))
                sids.append((record.get("data") or {}).get("sid", ""))
                energies.append(float(record["energy"]))
                natoms.append(n)
                cells.append(record["cell"])
                f = record.get("forces")
                if f is None:
                    acc.counters["missing_forces"] += 1
                    f = np.full((n, 3), np.nan)
                forces.append(np.asarray(f, dtype=np.float64).reshape(n, 3))
                s = record.get("stress")
                if s is None:
                    acc.counters["missing_stress"] += 1
                else:
                    s = np.asarray(s, dtype=np.float64).ravel()
                    if s.size == 9:  # full 3x3 -> Voigt
                        s = s[[0, 4, 8, 5, 2, 1]]
                    stresses.append(s)
                    stress_rows.append(row)
    finally:
        env.close()

    acc.counters["shards"] += 1
    if not energies:
        acc.counters["empty_shards"] += 1
        return 0

    energy = np.asarray(energies)
    nat = np.asarray(natoms, dtype=np.int64)
    acc.counters["frames"] += energy.size
    acc.counters["atoms"] += int(nat.sum())
    acc.natoms += np.bincount(np.minimum(nat, NATOMS_BINS - 1), minlength=NATOMS_BINS)

    e_per_atom = energy / nat
    volume = np.abs(np.linalg.det(np.asarray(cells, dtype=np.float64).reshape(-1, 3, 3)))
    force = np.concatenate(forces)
    norm2 = np.square(force).sum(axis=1)
    offsets = np.concatenate([[0], np.cumsum(nat)[:-1]])
    f_rms = np.sqrt(np.add.reduceat(norm2, offsets) / nat)
    f_max = np.sqrt(np.maximum.reduceat(norm2, offsets))

    ch = acc.channels
    ch["e_per_atom"].update(e_per_atom)
    ch["e_total"].update(energy)
    ch["vol_per_atom"].update(volume / nat)
    ch["f_comp"].update(force)
    ch["f_norm"].update(np.sqrt(norm2))
    ch["f_rms_frame"].update(f_rms)
    ch["f_max_frame"].update(f_max)

    def where(row: int) -> dict:
        return {"shard": relative, "row_key": keys[row], "sid": sids[row], "natoms": int(nat[row])}

    lo, hi = int(np.nanargmin(e_per_atom)), int(np.nanargmax(e_per_atom))
    acc.offer_extreme("e_per_atom_min", e_per_atom[lo], where(lo), largest=False)
    acc.offer_extreme("e_per_atom_max", e_per_atom[hi], where(hi), largest=True)
    if np.isfinite(f_max).any():
        top = int(np.nanargmax(f_max))
        acc.offer_extreme("f_max_frame_max", f_max[top], where(top), largest=True)

    if stresses:
        sigma = np.asarray(stresses) * EV_A3_TO_GPA
        acc.counters["zero_stress"] += int(np.count_nonzero(~sigma.any(axis=1)))
        hydro = sigma[:, :3].mean(axis=1)
        xx, yy, zz, yz, xz, xy = sigma.T
        von_mises = np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                            + 3.0 * (yz ** 2 + xz ** 2 + xy ** 2))
        ch["s_hydro"].update(hydro)
        ch["s_normal"].update(sigma[:, :3])
        ch["s_shear"].update(sigma[:, 3:])
        ch["s_vonmises"].update(von_mises)
        if np.isfinite(hydro).any():
            a, b = int(np.nanargmin(hydro)), int(np.nanargmax(hydro))
            acc.offer_extreme("s_hydro_min", hydro[a], where(stress_rows[a]), largest=False)
            acc.offer_extreme("s_hydro_max", hydro[b], where(stress_rows[b]), largest=True)
    return int(energy.size)


def part_path(output_dir: Path, task_index: int, num_tasks: int) -> Path:
    return output_dir / "parts" / f"part_{task_index:03d}_of_{num_tasks:03d}.npz"


def map_partition(args: argparse.Namespace) -> None:
    if not 0 <= args.task_index < args.num_tasks:
        raise ValueError("--task-index must be in [0, num-tasks)")
    shards = omat_shards(args.omat_root, args.omat_recovered_root)
    assigned = shards[args.task_index::args.num_tasks]
    if args.max_shards:
        assigned = assigned[:args.max_shards]
    if not assigned:
        raise ValueError(f"Task {args.task_index} has no shards (num_tasks={args.num_tasks})")

    accs: dict[str, Accumulator] = {}
    rows = 0
    for local_index, (relative, path) in enumerate(assigned, start=1):
        sub = relative.split("/", 1)[0]
        rows += scan_shard(relative, path, accs.setdefault(sub, Accumulator()))
        if local_index == 1 or local_index % 10 == 0 or local_index == len(assigned):
            print(f"task={args.task_index}/{args.num_tasks} shards={local_index}/{len(assigned)} "
                  f"rows={rows} latest={relative}", flush=True)

    payload = {
        "task_index": np.int64(args.task_index), "num_tasks": np.int64(args.num_tasks),
        "total_shards": np.int64(len(shards)), "max_shards": np.int64(args.max_shards),
        "manifest_sha256": np.asarray(manifest_sha256(shards)),
        "assigned_shards": np.asarray([rel for rel, _ in assigned]),
        "subdatasets": np.asarray(sorted(accs)),
    }
    for sub, acc in accs.items():
        payload[f"{sub}|counters"] = np.asarray([acc.counters[k] for k in COUNTER_FIELDS], dtype=np.int64)
        payload[f"{sub}|natoms"] = acc.natoms
        payload[f"{sub}|extremes"] = np.asarray(json.dumps(acc.extremes, sort_keys=True))
        for name, channel in acc.channels.items():
            payload[f"{sub}|{name}|hist"] = channel.hist
            payload[f"{sub}|{name}|mom"] = channel.mom

    output = part_path(args.output_dir, args.task_index, args.num_tasks)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(output)
    print(f"wrote {output}", flush=True)


# ----------------------------------------------------------------------------- reduce


def load_parts(args: argparse.Namespace) -> dict[str, Accumulator]:
    accs: dict[str, Accumulator] = {}
    manifest, seen = None, set()
    for task_index in range(args.num_tasks):
        path = part_path(args.output_dir, task_index, args.num_tasks)
        if not path.exists():
            raise FileNotFoundError(f"missing map output {path}")
        with np.load(path) as part:
            digest = str(part["manifest_sha256"])
            if manifest not in (None, digest):
                raise ValueError(f"{path} was built from a different shard manifest")
            manifest = digest
            if int(part["max_shards"]) and not args.allow_partial:
                raise ValueError(f"{path} is a --max-shards smoke part; pass --allow-partial")
            seen.update(str(s) for s in part["assigned_shards"])
            for sub in (str(s) for s in part["subdatasets"]):
                acc = accs.setdefault(sub, Accumulator())
                for k, v in zip(COUNTER_FIELDS, part[f"{sub}|counters"]):
                    acc.counters[k] += int(v)
                acc.natoms += part[f"{sub}|natoms"]
                for name, extreme in json.loads(str(part[f"{sub}|extremes"])).items():
                    where = {k: v for k, v in extreme.items() if k != "value"}
                    acc.offer_extreme(name, extreme["value"], where, largest=name.endswith("max"))
                for name, channel in acc.channels.items():
                    channel.merge(part[f"{sub}|{name}|hist"], part[f"{sub}|{name}|mom"])
    print(f"merged {args.num_tasks} parts covering {len(seen)} shards", flush=True)
    return accs


def pooled(accs: dict[str, Accumulator]) -> Accumulator:
    total = Accumulator()
    for acc in accs.values():
        for k in COUNTER_FIELDS:
            total.counters[k] += acc.counters[k]
        total.natoms += acc.natoms
        for name, extreme in acc.extremes.items():
            where = {k: v for k, v in extreme.items() if k != "value"}
            total.offer_extreme(name, extreme["value"], where, largest=name.endswith("max"))
        for name, channel in acc.channels.items():
            total.channels[name].merge(channel.hist, channel.mom)
    return total


def cdf(channel: Channel) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative counts at the histogram edges, out-of-range mass included."""
    m = dict(zip(MOMENT_FIELDS, channel.mom))
    return channel.edges, m["under"] + np.concatenate([[0], np.cumsum(channel.hist)])


def quantile(channel: Channel, q: float) -> float:
    m = dict(zip(MOMENT_FIELDS, channel.mom))
    if m["count"] == 0:
        return math.nan
    edges, cum = cdf(channel)
    target = q * m["count"]
    if target <= cum[0]:
        return m["min"]
    if target >= cum[-1]:
        return m["max"]
    return float(np.clip(np.interp(target, cum, edges), m["min"], m["max"]))


def channel_stats(channel: Channel) -> dict:
    m = dict(zip(MOMENT_FIELDS, channel.mom))
    n = m["count"]
    if n == 0:
        return {"count": 0}
    mean = m["sum"] / n
    out = {
        "count": int(n), "mean": mean,
        "std": math.sqrt(max(m["sumsq"] / n - mean * mean, 0.0)),
        "rms": math.sqrt(m["sumsq"] / n), "mean_abs": m["sumabs"] / n,
        "min": m["min"], "max": m["max"],
    }
    out.update({f"p{100 * q:g}": quantile(channel, q) for q in QUANTILES})
    out.update({"nonfinite": int(m["nonfinite"]),
                "outside_histogram": int(m["under"] + m["over"])})
    return out


def summarize(acc: Accumulator) -> dict:
    frames = acc.counters["frames"]
    sizes = np.arange(NATOMS_BINS)
    natoms_cum = np.cumsum(acc.natoms)
    return {
        **acc.counters,
        "atoms_per_frame_mean": acc.counters["atoms"] / frames if frames else math.nan,
        "atoms_per_frame_median": int(np.searchsorted(natoms_cum, 0.5 * frames)) if frames else None,
        "atoms_per_frame_min": int(sizes[acc.natoms > 0].min()) if frames else None,
        "atoms_per_frame_max": int(sizes[acc.natoms > 0].max()) if frames else None,
        "channels": {name: channel_stats(channel) for name, channel in acc.channels.items()},
        "extremes": acc.extremes,
    }


def rebin_fraction(channel: Channel, new_edges: np.ndarray) -> np.ndarray:
    """Fraction of the channel's values in each new bin, by interpolating the CDF."""
    edges, cum = cdf(channel)
    return np.diff(np.interp(new_edges, edges, cum)) / max(channel.mom[0], 1.0)


def rebin_density(channel: Channel, new_edges: np.ndarray) -> np.ndarray:
    return rebin_fraction(channel, new_edges) / np.diff(new_edges)


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}g}"


def write_tables(summary: dict, output_dir: Path) -> None:
    columns = ["count", "mean", "std", "rms", "mean_abs", "min"] + \
              [f"p{100 * q:g}" for q in QUANTILES] + ["max", "nonfinite", "outside_histogram"]
    lines = ["subdataset,channel,unit," + ",".join(columns)]
    for sub, entry in summary.items():
        for name, stats in entry["channels"].items():
            lines.append(",".join([sub, name, CHANNELS[name][0]] +
                                  [repr(stats.get(c, "")) for c in columns]))
    (output_dir / "omat24_subdataset_efs_stats.csv").write_text("\n".join(lines) + "\n")

    md = ["# OMAT24 train subdatasets: energy / force / stress statistics", "",
          "Full population, nothing sampled. Moments are exact; quantiles are read from",
          "fine histograms (1 meV/atom energy bins, <0.5% relative force/stress bins).",
          "Stress is in GPa with the sign exactly as stored in the dataset.", "",
          "## Frames and atoms", "",
          "| subdataset | frames | official | atoms | atoms/frame | median | min | max | zero-stress frames |",
          "|---|---|---|---|---|---|---|---|---|"]
    for sub, e in summary.items():
        official = OFFICIAL.get(sub, sum(OFFICIAL.values()))
        md.append(f"| {sub} | {e['frames']:,} | {official:,} | {e['atoms']:,} | "
                  f"{e['atoms_per_frame_mean']:.2f} | {e['atoms_per_frame_median']} | "
                  f"{e['atoms_per_frame_min']} | {e['atoms_per_frame_max']} | {e['zero_stress']:,} |")
    shown = ["mean", "std", "rms", "min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9", "max"]
    for name, (unit, _) in CHANNELS.items():
        md += ["", f"## {name} ({unit})", "", CHANNEL_NOTES[name] + ".", "",
               "| subdataset | " + " | ".join(shown) + " |", "|---|" + "---|" * len(shown)]
        for sub, e in summary.items():
            stats = e["channels"][name]
            md.append(f"| {sub} | " + " | ".join(fmt(stats.get(c)) for c in shown) + " |")
    md += ["", "## Most extreme frames", "",
           "| subdataset | which | value | shard | row key | atoms | sid |", "|---|---|---|---|---|---|---|"]
    for sub, e in summary.items():
        if sub == ALL:
            continue
        for name, x in sorted(e["extremes"].items()):
            md.append(f"| {sub} | {name} | {fmt(x['value'], 6)} | {x['shard']} | {x['row_key']} | "
                      f"{x['natoms']} | {x['sid']} |")
    (output_dir / "omat24_subdataset_efs_stats.md").write_text("\n".join(md) + "\n")


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7, length=2.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def small_multiples(accs, total, summary, output_dir, *, channel, stem, title, xlabel, ylabel,
                    lo, hi, log_y=False, note_stats=("mean", "std"), note_side="right"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    edges = np.linspace(lo, hi, 241)
    centers = 0.5 * (edges[1:] + edges[:-1])
    reference = rebin_density(total.channels[channel], edges)
    panels = [s for s in ORDER if s in accs] + [ALL]

    fig, axes = plt.subplots(3, 4, figsize=(15, 9.2), sharex=True, sharey=True, facecolor=SURFACE)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, sub in zip(axes.ravel(), panels):
        style_axes(ax)
        acc = total if sub == ALL else accs[sub]
        density = rebin_density(acc.channels[channel], edges)
        if log_y:
            ax.plot(centers, np.where(density > 0, density, np.nan), color=COLORS[sub], linewidth=1.6)
        else:
            ax.fill_between(centers, density, step="mid", color=COLORS[sub], alpha=0.30, linewidth=0)
            ax.plot(centers, density, color=COLORS[sub], linewidth=1.4, drawstyle="steps-mid")
        if sub != ALL:
            ax.plot(centers, np.where(reference > 0, reference, np.nan) if log_y else reference,
                    color=INK_2, linewidth=0.9, drawstyle="default" if log_y else "steps-mid")
        stats = summary[sub]["channels"][channel]
        note = f"{summary[sub]['frames']:,} frames\n" + "\n".join(
            f"{k} {fmt(stats.get(k))}" for k in note_stats)
        ax.text(0.97 if note_side == "right" else 0.03, 0.95, note, transform=ax.transAxes,
                ha=note_side, va="top", fontsize=6.8, color=INK_2, linespacing=1.35)
        ax.set_title("all OMAT24 train, pooled" if sub == ALL else sub,
                     fontsize=9, color=INK, loc="left", pad=4)
        if log_y:
            ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, fontsize=8, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8, color=INK_2)
    axes[0, 0].set_xlim(lo, hi)
    handles = [Line2D([], [], color=INK_2, linewidth=0.9, label="all OMAT24 train, pooled")]
    handles.insert(0, Line2D([], [], color=INK, linewidth=1.6, label="this subdataset (panel title)")
                   if log_y else Patch(facecolor=MUTED, alpha=0.45, label="this subdataset (panel title)"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=2,
               frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle(title, x=0.015, y=0.985, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output_dir / f"{stem}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {output_dir / (stem + '.png')}", flush=True)


def short_name(sub: str) -> str:
    return "all OMAT24\npooled" if sub == ALL else sub.replace("aimd-from-PBE-", "aimd-")


def value_violins(accs, total, summary, output_dir, *, channel, stem, title, ylabel):
    """One violin per subdataset with the quantity itself on a log y axis.

    Violin width is the share of values at that height, each violin scaled to its
    own widest point; the markers are p1-p99 (thin), p25-p75 (bar) and the median.
    """
    import matplotlib.pyplot as plt

    columns = [s for s in ORDER if s in accs] + [ALL]
    stats = {sub: summary[sub]["channels"][channel] for sub in columns}
    lo = 10 ** math.floor(math.log10(min(s["p0.1"] for s in stats.values())))
    top = max(s["max"] for s in stats.values())
    edges = np.logspace(math.log10(lo), math.log10(top), 221)
    centers = np.sqrt(edges[1:] * edges[:-1])

    fig, ax = plt.subplots(figsize=(15, 6.8), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    for x, sub in enumerate(columns):
        acc = total if sub == ALL else accs[sub]
        share = rebin_fraction(acc.channels[channel], edges)
        filled = np.flatnonzero(share > 0)  # draw only where this subdataset has values
        span = slice(filled[0], filled[-1] + 1)
        half, height = 0.42 * share[span] / share.max(), centers[span]
        ax.fill_betweenx(height, x - half, x + half, color=COLORS[sub], alpha=0.30, linewidth=0)
        ax.plot(np.concatenate([x - half, (x + half)[::-1]]), np.concatenate([height, height[::-1]]),
                color=COLORS[sub], linewidth=1.0)
        s = stats[sub]
        ax.plot([x, x], [s["p1"], s["p99"]], color=INK, linewidth=1.0)
        ax.plot([x, x], [s["p25"], s["p75"]], color=INK, linewidth=4.5, solid_capstyle="round")
        ax.plot(x, s["p50"], "o", markersize=6, color=SURFACE, markeredgecolor=INK, markeredgewidth=1.4)
        ax.plot([x - 0.16, x + 0.16], [s["max"], s["max"]], color=MUTED, linewidth=1.2)
        ax.text(x, 1.035, fmt(s["p50"], 3), transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8, color=INK)
        ax.text(x, 1.0, f"p99 {fmt(s['p99'], 3)}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=7, color=INK_2)
    ax.text(-0.62, 1.035, "median", transform=ax.get_xaxis_transform(), ha="right", va="bottom",
            fontsize=8, color=INK_2)
    ax.set_yscale("log")
    ax.set_ylim(lo, top * 1.6)
    ax.set_xlim(-0.6, len(columns) - 0.4)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([short_name(s) for s in columns], fontsize=8, color=INK, rotation=25, ha="right",
                       rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2)
    fig.suptitle(title, x=0.015, y=0.985, ha="left", fontsize=12, color=INK)
    fig.text(0.015, 0.925, "violin width: share of values at that height (each violin scaled to its own widest "
             "point)   |   thin line p1-p99, bar p25-p75, open dot median, grey dash maximum",
             fontsize=8, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / f"{stem}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {output_dir / (stem + '.png')}", flush=True)


def normal_vs_shear(accs, total, summary, output_dir, lo, hi):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    edges = np.linspace(lo, hi, 241)
    centers = 0.5 * (edges[1:] + edges[:-1])
    series = [("s_normal", "#2a78d6", "normal  xx, yy, zz"), ("s_shear", "#eb6834", "shear  yz, xz, xy")]
    fig, axes = plt.subplots(3, 4, figsize=(15, 9.2), sharex=True, sharey=True, facecolor=SURFACE)
    panels = [s for s in ORDER if s in accs] + [ALL]
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for ax, sub in zip(axes.ravel(), panels):
        style_axes(ax)
        acc = total if sub == ALL else accs[sub]
        for name, color, _ in series:
            density = rebin_density(acc.channels[name], edges)
            ax.plot(centers, np.where(density > 0, density, np.nan), color=color, linewidth=1.6)
        ch = summary[sub]["channels"]
        ax.text(0.97, 0.95, f"normal std {fmt(ch['s_normal'].get('std'))}\n"
                            f"shear std {fmt(ch['s_shear'].get('std'))}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=INK_2, linespacing=1.35)
        ax.set_title("all OMAT24 train, pooled" if sub == ALL else sub, fontsize=9, color=INK, loc="left", pad=4)
        ax.set_yscale("log")
    for ax in axes[-1]:
        ax.set_xlabel("stress component (GPa)", fontsize=8, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("density (1/GPa)", fontsize=8, color=INK_2)
    axes[0, 0].set_xlim(lo, hi)
    fig.legend(handles=[Line2D([], [], color=c, linewidth=1.6, label=l) for _, c, l in series],
               loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=2, frameon=False,
               fontsize=8, labelcolor=INK_2)
    fig.suptitle("OMAT24 stress components by subdataset: normal vs shear", x=0.015, y=0.985,
                 ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output_dir / "stress_components.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {output_dir / 'stress_components.png'}", flush=True)


def percentile_ranges(summary, output_dir):
    """One row per subdataset: p1-p99 whisker, p25-p75 bar, median dot."""
    import matplotlib.pyplot as plt

    specs = [("e_per_atom", "energy per atom (eV/atom)", False),
             ("f_norm", "per-atom |F| (eV/A)", True),
             ("s_hydro", "mean normal stress tr(sigma)/3 (GPa)", False)]
    rows = [s for s in ORDER if s in summary] + [ALL]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True, facecolor=SURFACE)
    for ax, (name, label, log_x) in zip(axes, specs):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        for y, sub in enumerate(rows):
            s = summary[sub]["channels"][name]
            if not s.get("count"):
                continue
            ax.plot([s["p1"], s["p99"]], [y, y], color=COLORS[sub], linewidth=1.2, solid_capstyle="round")
            ax.plot([s["p25"], s["p75"]], [y, y], color=COLORS[sub], linewidth=6, solid_capstyle="round")
            ax.plot(s["p50"], y, "o", markersize=6.5, color=INK, markeredgecolor=SURFACE, markeredgewidth=1.4)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(label, fontsize=8.5, color=INK_2)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(["all OMAT24, pooled" if r == ALL else r for r in rows], fontsize=8, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("OMAT24 subdatasets at a glance: thin line p1-p99, bar p25-p75, dot median",
                 x=0.015, y=0.975, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "summary_percentile_ranges.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {output_dir / 'summary_percentile_ranges.png'}", flush=True)


def nice(value: float, step: float, up: bool) -> float:
    return (math.ceil if up else math.floor)(value / step) * step


def reduce_partitions(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")

    accs = load_parts(args)
    total = pooled(accs)
    ordered = [s for s in ORDER if s in accs] + sorted(set(accs) - set(ORDER))
    summary = {sub: summarize(accs[sub]) for sub in ordered}
    summary[ALL] = summarize(total)

    found = {s: summary.get(s, {}).get("frames", 0) for s in OFFICIAL}
    mismatched = {s: (found[s], OFFICIAL[s]) for s in OFFICIAL if found[s] != OFFICIAL[s]}
    for sub, (got, want) in mismatched.items():
        print(f"!! {sub}: {got:,} frames, official {want:,}", flush=True)
    if mismatched and not args.allow_partial:
        raise ValueError("frame counts differ from the official totals; pass --allow-partial to proceed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "omat24_subdataset_efs_stats.json").write_text(json.dumps({
        "units": {name: unit for name, (unit, _) in CHANNELS.items()},
        "channel_notes": CHANNEL_NOTES, "quantiles": QUANTILES,
        "frame_counts_match_official": not mismatched, "subdatasets": summary,
    }, indent=2) + "\n")
    write_tables(summary, args.output_dir)

    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)
    pool = summary[ALL]["channels"]
    common = dict(accs=accs, total=total, summary=summary, output_dir=figures)
    small_multiples(**common, channel="e_per_atom", stem="energy_per_atom",
                    title="OMAT24 energy per atom by subdataset", xlabel="energy per atom (eV/atom)",
                    ylabel="density (atom/eV)", lo=nice(pool["e_per_atom"]["p0.1"], 0.5, False),
                    hi=nice(pool["e_per_atom"]["p99.9"], 0.5, True), note_stats=("mean", "std", "min", "max"))
    value_violins(**common, channel="f_norm", stem="force_norm_per_atom",
                  title="OMAT24 per-atom force magnitude by subdataset", ylabel="|F| per atom (eV/A)")
    span = nice(max(abs(pool["f_comp"]["p0.1"]), pool["f_comp"]["p99.9"]), 1.0, True)
    small_multiples(**common, channel="f_comp", stem="force_components",
                    title="OMAT24 force components (x, y, z pooled) by subdataset",
                    xlabel="force component (eV/A)", ylabel="density (A/eV)", lo=-span, hi=span,
                    log_y=True, note_stats=("std", "mean_abs", "min", "max"))
    value_violins(**common, channel="f_max_frame", stem="force_max_per_frame",
                  title="OMAT24 largest |F| in each frame by subdataset", ylabel="max |F| in frame (eV/A)")
    s_lo, s_hi = nice(pool["s_hydro"]["p0.1"], 5.0, False), nice(pool["s_hydro"]["p99.9"], 5.0, True)
    small_multiples(**common, channel="s_hydro", stem="stress_hydrostatic",
                    title="OMAT24 mean normal stress tr(sigma)/3 by subdataset (sign as stored)",
                    xlabel="tr(sigma)/3 (GPa)", ylabel="density (1/GPa)", lo=s_lo, hi=s_hi, log_y=True,
                    note_stats=("mean", "std", "min", "max"), note_side="left")
    value_violins(**common, channel="s_vonmises", stem="stress_von_mises",
                  title="OMAT24 von Mises stress by subdataset", ylabel="von Mises stress (GPa)")
    c_span = nice(max(abs(pool["s_normal"]["p0.1"]), pool["s_normal"]["p99.9"]), 5.0, True)
    normal_vs_shear(accs, total, summary, figures, -c_span, c_span)
    percentile_ranges(summary, figures)
    print(f"wrote {args.output_dir / 'omat24_subdataset_efs_stats.md'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["map", "reduce"])
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0, help="smoke test: cap shards per task")
    parser.add_argument("--allow-partial", action="store_true", help="reduce smoke-test parts")
    parser.add_argument("--omat-root", type=Path, default=DEFAULT_OMAT_ROOT)
    parser.add_argument("--omat-recovered-root", type=Path, default=DEFAULT_OMAT_RECOVERED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (map_partition if args.stage == "map" else reduce_partitions)(args)


if __name__ == "__main__":
    main()
