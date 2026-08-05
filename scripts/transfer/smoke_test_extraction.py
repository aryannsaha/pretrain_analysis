#!/usr/bin/env python
"""Smoke-test MACE activation extraction against a real checkpoint.

Checks, in order:

1. The checkpoint loads and the hooks fire on every product block.
2. The captured widths match the irreps the checkpoint advertises.
3. **Rotation invariance on a real structure.**  The whole extraction rests on
   the assumption that e3nn lays an irreps block out as ``(mul, 2l+1)`` with
   multiplicity outermost.  If that is wrong, ``normed`` silently mixes
   channels and every score built on it is meaningless.  This test rotates a
   real periodic MOF structure and asserts the reductions do not move while
   the raw features do.

Run with the MACE environment::

    /scratch/gpfs/ROSENGROUP/aryan/software/conda_envs/pretrain_analysis_env_mace/bin/python \
        scripts/transfer/smoke_test_extraction.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from pretrain_analysis.transfer.extract import (  # noqa: E402
    MaceActivationExtractor,
    load_mace_model,
    pool_features,
)

DEFAULT_CKPT = (
    "/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/models/"
    "logme_frozen_foundations/omat_1m.model"
)
DEFAULT_LMDB = (
    "/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis/data/processed/"
    "mof_off/R2SCAN/R2SCAN_val.lmdb"
)


def random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:  # keep it a proper rotation
        q[:, 0] *= -1
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--lmdb", default=DEFAULT_LMDB)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-structures", type=int, default=4)
    args = ap.parse_args()

    print(f"device            : {args.device}")
    print(f"checkpoint        : {args.checkpoint}")

    model = load_mace_model(args.checkpoint, device=args.device)
    print(f"model class       : {type(model).__name__}")
    print(f"r_max             : {float(model.r_max)}")
    print(f"n elements        : {len(model.atomic_numbers)}")
    print(f"heads             : {list(getattr(model, 'heads', ['Default']))}")
    print(f"scale_shift       : scale={model.scale_shift.scale.tolist()} "
          f"shift={model.scale_shift.shift.tolist()}")

    ex = MaceActivationExtractor(model)
    desc = ex.describe()
    print("\n-- extractor layout --")
    for k, v in desc.items():
        print(f"  {k}: {v}")

    # ---------------------------------------------------------------- data
    from mace.data import LMDBDataset
    from mace.tools import AtomicNumberTable, torch_geometric

    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = list(getattr(model, "heads", ["Default"]))
    dataset = LMDBDataset(
        args.lmdb,
        r_max=float(model.r_max),
        z_table=z_table,
        heads=heads,
        head=heads[0],
    )
    print(f"\ndataset size      : {len(dataset)}")

    loader = torch_geometric.dataloader.DataLoader(
        dataset,
        batch_size=args.n_structures,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader)).to(args.device)
    ptr = batch.ptr.cpu().numpy()
    n_atoms = int(ptr[-1])
    print(f"batch             : {len(ptr) - 1} structures, {n_atoms} atoms")
    print(f"energy (eV)       : {batch.energy.cpu().numpy()}")
    print(f"forces shape      : {tuple(batch.forces.shape)}")

    fwd = dict(
        training=False,
        compute_force=False,
        compute_virials=False,
        compute_stress=False,
    )

    # ------------------------------------------------- unrotated features
    with torch.no_grad(), ex.hooks():
        model(batch.to_dict(), **fwd)
    captured = ex.pop()

    print("\n-- captured activations --")
    ok = True
    for name, arr in sorted(captured.items()):
        print(f"  {name:16s} {arr.shape}")
        if arr.shape[0] != n_atoms:
            print(f"    !! expected {n_atoms} rows")
            ok = False

    feats = ex.feature_sets(captured)
    print("\n-- invariant feature sets --")
    for name, arr in sorted(feats.items()):
        finite = np.isfinite(arr).all()
        print(f"  {name:22s} {arr.shape!s:14s} finite={finite} "
              f"std={arr.std():.4g}")
        if not finite:
            ok = False

    for i, layout in enumerate(ex.layer_irreps):
        exp_inv, exp_norm = layout.n_invariant, layout.n_normed
        got_inv = feats[f"layer{i}/inv"].shape[1]
        got_norm = feats[f"layer{i}/normed"].shape[1]
        if (got_inv, got_norm) != (exp_inv, exp_norm):
            print(f"  !! layer{i} width mismatch: got ({got_inv},{got_norm}) "
                  f"expected ({exp_inv},{exp_norm})")
            ok = False

    # --------------------------------------------------- rotation invariance
    print("\n-- rotation invariance (the load-bearing check) --")
    rot = random_rotation(0)
    rot_t = torch.tensor(rot, dtype=batch.positions.dtype, device=args.device)

    rotated = batch.clone()
    rotated.positions = batch.positions @ rot_t.T
    rotated.shifts = batch.shifts @ rot_t.T
    rotated.cell = batch.cell @ rot_t.T

    with torch.no_grad(), ex.hooks():
        model(rotated.to_dict(), **fwd)
    captured_rot = ex.pop()
    feats_rot = ex.feature_sets(captured_rot)

    for i in range(len(ex.layer_irreps)):
        raw, raw_rot = captured[f"layer{i}"], captured_rot[f"layer{i}"]
        moved = np.abs(raw - raw_rot).max()
        print(f"  layer{i} raw irreps   max|delta| = {moved:.3e}   "
              f"(expected NONZERO for l>0 layers)")

    for name in sorted(feats):
        a, b = feats[name].astype(np.float64), feats_rot[name].astype(np.float64)
        delta = np.abs(a - b).max()
        scale = max(np.abs(a).max(), 1e-30)
        rel = delta / scale
        verdict = "OK" if rel < 1e-5 else "FAIL"
        if rel >= 1e-5:
            ok = False
        print(f"  {name:22s} max|delta| = {delta:.3e}  rel = {rel:.3e}  {verdict}")

    # ------------------------------------------------------------- pooling
    print("\n-- pooling --")
    inv0 = feats["layer0/inv"]
    summed = pool_features(inv0, ptr, how="sum")
    meaned = pool_features(inv0, ptr, how="mean")
    counts = np.diff(ptr)
    print(f"  atoms per structure : {counts.tolist()}")
    print(f"  sum-pooled  : {summed.shape}")
    print(f"  mean-pooled : {meaned.shape}")
    if not np.allclose(summed, meaned * counts[:, None], rtol=1e-5):
        print("  !! sum != mean * n_atoms")
        ok = False
    else:
        print("  sum == mean * n_atoms  OK")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
