#!/usr/bin/env python
"""Zero-shot error of each frozen checkpoint on a downstream set.

The post-fine-tune MAEs are the study's real ground truth, but they take days.
Zero-shot error is available immediately and is a legitimate -- if different --
downstream performance measure, so it gives the rank-correlation machinery
something to chew on now and answers a necessary-condition question: if LogME
cannot even rank the frozen models by how well they already do on the target,
it is unlikely to rank them by how well they will fine-tune.

**This is not a substitute for the fine-tune result.** Zero-shot and
post-fine-tune performance can disagree: a model can start poorly and adapt
well. Treat the correlation reported here as a sanity check on the pipeline and
a weak prior, not as the answer.

Metrics
-------
``force_mae``
    Mean absolute force error, eV/A. **The trustworthy one** -- forces are a
    derivative, so the arbitrary energy reference cancels.
``energy_mae_per_atom_raw``
    Straight |E_pred - E_true| / N. Dominated by the reference mismatch: the
    models carry OMat24's E0 table while MOF-OFF is r2SCAN-D4 and AM is raw
    VASP, so this is large and mostly uninformative.
``energy_mae_per_atom_shifted``
    Same, after removing the single best constant offset per checkpoint. This
    removes the reference mismatch but not its composition dependence, so it is
    a lower bound on the reference-corrected error rather than the truth.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DATA_ROOT = Path(
    os.environ.get("PRETRAIN_ANALYSIS_ROOT", "/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
)

from pretrain_analysis.transfer.extract import load_mace_model  # noqa: E402

LOGGER = logging.getLogger("zeroshot")

SCALES = ["omat_100k", "omat_500k", "omat_1m", "omat_2m", "omat_5m", "omat_10m"]


def evaluate(model, lmdb_path, *, n_structures, batch_size, device, seed):
    import torch
    from mace.data import LMDBDataset
    from mace.tools import AtomicNumberTable, torch_geometric

    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = list(getattr(model, "heads", ["Default"]))
    dataset = LMDBDataset(
        lmdb_path, r_max=float(model.r_max), z_table=z_table, heads=heads, head=heads[0]
    )
    rng = np.random.default_rng(seed)
    take = min(n_structures, len(dataset))
    picked = np.sort(rng.choice(len(dataset), size=take, replace=False))
    loader = torch_geometric.dataloader.DataLoader(
        torch.utils.data.Subset(dataset, picked.tolist()),
        batch_size=batch_size, shuffle=False, drop_last=False,
    )

    e_pred, e_true, n_at, f_err = [], [], [], []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        batch = batch.to(device)
        # Forces need autograd, so this cannot run under no_grad.
        with torch.enable_grad():
            out = model(
                batch.to_dict(), training=False, compute_force=True,
                compute_virials=False, compute_stress=False,
            )
        e_pred.append(out["energy"].detach().cpu().numpy().reshape(-1))
        e_true.append(batch.energy.detach().cpu().numpy().reshape(-1))
        n_at.append(np.diff(batch.ptr.cpu().numpy()))
        f_err.append(
            np.abs(
                out["forces"].detach().cpu().numpy()
                - batch.forces.detach().cpu().numpy()
            ).mean(axis=1)
        )
        if bi % 25 == 0:
            LOGGER.info("    batch %4d  (%.1fs)", bi, time.time() - t0)

    e_pred = np.concatenate(e_pred)
    e_true = np.concatenate(e_true)
    n_at = np.concatenate(n_at)
    f_err = np.concatenate(f_err)

    per_atom_resid = (e_pred - e_true) / n_at
    # One constant offset per checkpoint, chosen as the median because MAE is
    # minimized by the median, not the mean.
    shift = float(np.median(per_atom_resid))

    return {
        "n_structures": int(take),
        "n_atoms": int(n_at.sum()),
        "force_mae": float(f_err.mean()),
        "force_rmse": float(np.sqrt((f_err**2).mean())),
        "energy_mae_per_atom_raw": float(np.abs(per_atom_resid).mean()),
        "energy_mae_per_atom_shifted": float(np.abs(per_atom_resid - shift).mean()),
        "energy_offset_per_atom": shift,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default=str(DATA_ROOT / "models/logme_frozen_foundations"))
    ap.add_argument("--target-lmdb", required=True)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--out", default=str(DATA_ROOT / "outputs/logme"))
    ap.add_argument("--n-structures", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for scale in SCALES:
        ckpt = Path(args.checkpoint_dir) / f"{scale}.model"
        if not ckpt.exists():
            LOGGER.warning("missing %s", ckpt)
            continue
        LOGGER.info("=== %s ===", scale)
        model = load_mace_model(ckpt, device=device)
        rec = evaluate(
            model, args.target_lmdb,
            n_structures=args.n_structures, batch_size=args.batch_size,
            device=device, seed=args.seed,
        )
        rec.update({"checkpoint": scale, "target": args.target_name})
        LOGGER.info(
            "  force_mae=%.4f eV/A  E_mae_shifted=%.4f eV/atom  offset=%.3f",
            rec["force_mae"], rec["energy_mae_per_atom_shifted"],
            rec["energy_offset_per_atom"],
        )
        rows.append(rec)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    import pandas as pd

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"zeroshot_{args.target_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    LOGGER.info("wrote %s", path)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
