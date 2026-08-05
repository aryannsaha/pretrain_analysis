#!/usr/bin/env python
"""Driver: compute LogME and every baseline for a list of frozen MACE checkpoints.

For each (checkpoint x downstream dataset x layer x feature variant) this
produces one row containing ``logme_energy``, ``logme_forcemag``, all
baselines, and the shapes and diagnostics needed to judge them.  The
post-fine-tune MAEs are joined in later by ``build_results_table.py`` once the
fine-tunes finish -- this script never needs them, which is the entire point of
a training-free estimator.

Example
-------
::

    python scripts/transfer/run_logme_analysis.py \\
        --checkpoint-dir models/logme_frozen_foundations \\
        --target-lmdb data/processed/mof_off/R2SCAN/R2SCAN_val.lmdb \\
        --target-name mof_off \\
        --source-lmdb data/processed/omat24/stratified_random_nested/sr_100k/train.lmdb \\
        --out outputs/logme
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Data, checkpoints and outputs live in the main checkout even when this code
# is executed from a git worktree, so they are addressed absolutely rather than
# relative to this file.
DATA_ROOT = Path(
    os.environ.get("PRETRAIN_ANALYSIS_ROOT", "/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
)

from pretrain_analysis.transfer import baselines as bl  # noqa: E402
from pretrain_analysis.transfer.extract import (  # noqa: E402
    ExtractionConfig,
    MaceActivationExtractor,
    load_mace_model,
    pool_features,
    sha256_file,
)
from pretrain_analysis.transfer.logme import LogMEConvergenceWarning, logme  # noqa: E402

LOGGER = logging.getLogger("logme_driver")

# The scale ladder actually on disk. Note there is NO 10k model in this family;
# see the results write-up.
SCALES = {
    "omat_100k": 100_000,
    "omat_500k": 500_000,
    "omat_1m": 1_000_000,
    "omat_2m": 2_000_000,
    "omat_5m": 5_000_000,
    "omat_10m": 10_000_000,
}


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def extract_dataset(
    model,
    extractor: MaceActivationExtractor,
    lmdb_path: str,
    *,
    n_structures: int,
    max_atoms: int,
    batch_size: int,
    device: str,
    seed: int,
    need_labels: bool = True,
):
    """Run the frozen model over a subsample and return invariant features.

    Returns a dict with, per feature-set name, the per-structure pooled
    features (sum and mean) and a per-atom subsample, plus the aligned targets.
    """
    import torch
    from mace.data import LMDBDataset
    from mace.tools import AtomicNumberTable, torch_geometric

    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = list(getattr(model, "heads", ["Default"]))
    dataset = LMDBDataset(
        lmdb_path, r_max=float(model.r_max), z_table=z_table, heads=heads, head=heads[0]
    )

    rng = np.random.default_rng(seed)
    n_avail = len(dataset)
    take = min(n_structures, n_avail)
    # A fixed random subset, identical across checkpoints for a given seed, so
    # differences between checkpoints are differences between models and not
    # between samples.
    picked = np.sort(rng.choice(n_avail, size=take, replace=False))

    subset = torch.utils.data.Subset(dataset, picked.tolist())
    loader = torch_geometric.dataloader.DataLoader(
        subset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    fwd = dict(
        training=False, compute_force=False, compute_virials=False, compute_stress=False
    )

    pooled_sum: dict[str, list] = {}
    pooled_mean: dict[str, list] = {}
    atom_feats: dict[str, list] = {}
    energies, natoms, force_mags, atomic_numbers, struct_ids = [], [], [], [], []

    t0 = time.time()
    n_done = 0
    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad(), extractor.hooks():
            model(batch.to_dict(), **fwd)
        feats = extractor.feature_sets(extractor.pop())

        ptr = batch.ptr.cpu().numpy()
        counts = np.diff(ptr)
        for name, arr in feats.items():
            pooled_sum.setdefault(name, []).append(pool_features(arr, ptr, how="sum"))
            pooled_mean.setdefault(name, []).append(pool_features(arr, ptr, how="mean"))
            atom_feats.setdefault(name, []).append(arr.astype(np.float16))

        natoms.append(counts)
        atomic_numbers.append(batch.node_attrs.argmax(dim=1).cpu().numpy())
        struct_ids.append(np.repeat(picked[n_done : n_done + counts.size], counts))
        if need_labels:
            energies.append(batch.energy.detach().cpu().numpy().reshape(-1))
            f = batch.forces.detach().cpu().numpy()
            force_mags.append(np.linalg.norm(f, axis=1))
        n_done += counts.size

        if bi % 25 == 0:
            LOGGER.info(
                "    batch %4d  structures %6d/%6d  (%.1fs)",
                bi, n_done, take, time.time() - t0,
            )

    out = {
        "pooled_sum": {k: np.concatenate(v, axis=0) for k, v in pooled_sum.items()},
        "pooled_mean": {k: np.concatenate(v, axis=0) for k, v in pooled_mean.items()},
        "n_atoms": np.concatenate(natoms),
        "structure_ids": picked,
    }
    if need_labels:
        out["energy_total"] = np.concatenate(energies)
        out["energy_per_atom"] = out["energy_total"] / out["n_atoms"]

    # Subsample atoms once, and use the SAME rows for every feature set so the
    # per-atom scores are computed on identical atoms and stay comparable.
    all_atoms = np.concatenate(atomic_numbers)
    n_atoms_total = all_atoms.size
    if n_atoms_total > max_atoms:
        sel = np.sort(rng.choice(n_atoms_total, size=max_atoms, replace=False))
    else:
        sel = np.arange(n_atoms_total)

    out["atom_features"] = {
        k: np.concatenate(v, axis=0)[sel] for k, v in atom_feats.items()
    }
    out["atom_z_index"] = all_atoms[sel]
    out["atom_structure_id"] = np.concatenate(struct_ids)[sel]
    out["n_atoms_total"] = n_atoms_total
    if need_labels:
        out["force_magnitude"] = np.concatenate(force_mags)[sel]

    LOGGER.info(
        "    done: %d structures, %d atoms (kept %d) in %.1fs",
        take, n_atoms_total, sel.size, time.time() - t0,
    )
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def logme_with_subsample_variance(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    n_subsamples: int,
    subsample_fraction: float,
    seed: int,
) -> dict:
    """LogME on the full sample, plus its spread over random subsamples.

    If the score is not stable across subsamples then every rank correlation
    downstream is measuring sampling noise, so this is reported for every row
    rather than spot-checked.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LogMEConvergenceWarning)
        full = logme(features, targets)

    rng = np.random.default_rng(seed)
    n = features.shape[0]
    k = max(int(round(subsample_fraction * n)), features.shape[1] + 2, 10)
    scores = []
    if k < n:
        for _ in range(n_subsamples):
            idx = rng.choice(n, size=k, replace=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LogMEConvergenceWarning)
                try:
                    scores.append(logme(features[idx], targets[idx]).score)
                except ValueError:
                    pass

    return {
        "logme": full.score,
        "logme_alpha": full.alpha,
        "logme_beta": full.beta,
        "logme_resid_std": float(1.0 / np.sqrt(full.beta)) if full.beta > 0 else np.nan,
        "logme_gamma": full.gamma,
        "logme_rank": full.rank,
        "logme_converged": full.converged,
        "logme_n": full.n,
        "logme_d": full.d,
        "logme_subsample_mean": float(np.mean(scores)) if scores else np.nan,
        "logme_subsample_std": float(np.std(scores)) if scores else np.nan,
        "logme_n_subsamples": len(scores),
    }


def score_one(
    name: str,
    target_data: dict,
    source_data: dict,
    *,
    pooling: str,
    n_subsamples: int,
    subsample_fraction: float,
    seed: int,
    gmm_components: int,
) -> dict:
    """Every predictor for one feature set."""
    row: dict = {"feature_set": name, "pooling": pooling}

    # ---- energy target: per-structure -------------------------------------
    pooled_key = "pooled_mean" if pooling == "mean" else "pooled_sum"
    f_struct = np.asarray(target_data[pooled_key][name], dtype=np.float64)
    # mean-pool pairs with per-atom energy; sum-pool pairs with total energy.
    y_energy = (
        target_data["energy_per_atom"]
        if pooling == "mean"
        else target_data["energy_total"]
    )
    row.update(
        {
            f"energy_{k}": v
            for k, v in logme_with_subsample_variance(
                f_struct, y_energy,
                n_subsamples=n_subsamples,
                subsample_fraction=subsample_fraction,
                seed=seed,
            ).items()
        }
    )

    ridge_e = bl.ridge_probe(f_struct, y_energy, seed=seed)
    row.update({f"energy_{k}": v for k, v in ridge_e.as_dict().items()})
    row["energy_h_score"] = bl.h_score(f_struct, y_energy)

    # ---- force-magnitude target: per-atom ---------------------------------
    f_atom = np.asarray(target_data["atom_features"][name], dtype=np.float64)
    y_force = target_data["force_magnitude"]
    row.update(
        {
            f"forcemag_{k}": v
            for k, v in logme_with_subsample_variance(
                f_atom, y_force,
                n_subsamples=n_subsamples,
                subsample_fraction=subsample_fraction,
                seed=seed,
            ).items()
        }
    )
    ridge_f = bl.ridge_probe(f_atom, y_force, seed=seed)
    row.update({f"forcemag_{k}": v for k, v in ridge_f.as_dict().items()})
    row["forcemag_h_score"] = bl.h_score(f_atom, y_force)

    # ---- target-free: representational richness ---------------------------
    row["participation_ratio"] = bl.participation_ratio(f_atom)
    row["effective_rank"] = bl.effective_rank(f_atom)

    # ---- distributional distance to the pretraining cloud -----------------
    if source_data is not None:
        src = np.asarray(source_data["atom_features"][name], dtype=np.float64)
        # Cap the source sample: the GMM is O(n k d^2) and 20k atoms is plenty
        # to characterize the cloud.
        if src.shape[0] > 20_000:
            src = src[
                np.random.default_rng(seed).choice(src.shape[0], 20_000, replace=False)
            ]
        tgt = f_atom
        if tgt.shape[0] > 20_000:
            tgt = tgt[
                np.random.default_rng(seed + 1).choice(tgt.shape[0], 20_000, replace=False)
            ]
        row.update(bl.gaussian_distance(src, tgt))
        try:
            row.update(
                bl.gmm_log_likelihood(
                    src, tgt, n_components=gmm_components, seed=seed
                ).as_dict()
            )
        except np.linalg.LinAlgError as exc:
            LOGGER.warning("    GMM failed for %s: %s", name, exc)
            row["gmm_target_loglik"] = np.nan

    return row


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", default=str(DATA_ROOT / "models/logme_frozen_foundations"))
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="explicit checkpoint paths; overrides --checkpoint-dir")
    ap.add_argument("--target-lmdb", required=True)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--source-lmdb", default=None,
                    help="OMat24 LMDB for the distributional-distance baseline")
    ap.add_argument("--out", default=str(DATA_ROOT / "outputs/logme"))
    ap.add_argument("--n-structures", type=int, default=3000)
    ap.add_argument("--n-source-structures", type=int, default=1500)
    ap.add_argument("--max-atoms", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-subsamples", type=int, default=8)
    ap.add_argument("--subsample-fraction", type=float, default=0.5)
    ap.add_argument("--gmm-components", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoints:
        ckpts = [Path(p) for p in args.checkpoints]
    else:
        d = Path(args.checkpoint_dir)
        ckpts = [d / f"{k}.model" for k in SCALES if (d / f"{k}.model").exists()]
    if not ckpts:
        LOGGER.error("no checkpoints found")
        return 1

    out_dir = Path(args.out) / args.target_name
    out_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("device=%s  checkpoints=%d  out=%s", device, len(ckpts), out_dir)

    rows = []
    for ckpt in ckpts:
        scale_name = ckpt.stem
        LOGGER.info("=" * 70)
        LOGGER.info("checkpoint %s", ckpt)

        model = load_mace_model(ckpt, device=device)
        extractor = MaceActivationExtractor(model)
        desc = extractor.describe()
        LOGGER.info("  layout: %s", desc)

        LOGGER.info("  extracting target (%s)", args.target_name)
        target_data = extract_dataset(
            model, extractor, args.target_lmdb,
            n_structures=args.n_structures, max_atoms=args.max_atoms,
            batch_size=args.batch_size, device=device, seed=args.seed,
        )

        source_data = None
        if args.source_lmdb:
            LOGGER.info("  extracting source (OMat24)")
            source_data = extract_dataset(
                model, extractor, args.source_lmdb,
                n_structures=args.n_source_structures, max_atoms=args.max_atoms,
                batch_size=args.batch_size, device=device, seed=args.seed + 100,
                need_labels=False,
            )

        # Provenance, written once per checkpoint.
        import mace

        cfg = ExtractionConfig(
            checkpoint=str(ckpt),
            checkpoint_sha256=sha256_file(ckpt),
            dataset=args.target_lmdb,
            n_structures=int(target_data["structure_ids"].size),
            max_atoms=args.max_atoms,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            mace_version=getattr(mace, "__version__", "unknown"),
            model_class=type(model).__name__,
            r_max=float(model.r_max),
            n_layers=desc["n_layers"],
            layer_irreps=desc["layer_irreps"],
            readout_hidden_dim=desc["readout_hidden_dim"],
            dtype=str(next(p.dtype for p in model.parameters() if p.is_floating_point())),
            atomic_numbers=[int(z) for z in model.atomic_numbers],
        )
        cfg.to_json(out_dir / f"{scale_name}_provenance.json")

        for name in sorted(target_data["atom_features"]):
            for pooling in ("mean", "sum"):
                LOGGER.info("  scoring %-22s pooling=%s", name, pooling)
                row = score_one(
                    name, target_data, source_data,
                    pooling=pooling,
                    n_subsamples=args.n_subsamples,
                    subsample_fraction=args.subsample_fraction,
                    seed=args.seed,
                    gmm_components=args.gmm_components,
                )
                row.update(
                    {
                        "checkpoint": scale_name,
                        "n_pretrain": SCALES.get(scale_name, np.nan),
                        "log_n_pretrain": np.log(SCALES[scale_name])
                        if scale_name in SCALES else np.nan,
                        "target": args.target_name,
                        "checkpoint_sha256": cfg.checkpoint_sha256[:16],
                        "n_structures": cfg.n_structures,
                        "n_atoms_sampled": int(target_data["atom_z_index"].size),
                    }
                )
                rows.append(row)

        del model, extractor, target_data, source_data
        if device == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ save
    import pandas as pd

    df = pd.DataFrame(rows)
    csv = out_dir / f"logme_scores_{args.target_name}.csv"
    df.to_csv(csv, index=False)
    LOGGER.info("wrote %s  (%d rows, %d cols)", csv, len(df), df.shape[1])

    (out_dir / "run_args.json").write_text(json.dumps(vars(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
