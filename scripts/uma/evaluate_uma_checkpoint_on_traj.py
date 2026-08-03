#!/usr/bin/env python
"""Evaluate one UMA checkpoint on one ASE trajectory (or ASE-LMDB).

UMA counterpart of ``scripts/mace/evaluate_mace_checkpoint_on_traj.py``: it runs
the checkpoint over every frame, reports energy/force/stress MAEs, and writes a
parity figure.

  python scripts/uma/evaluate_uma_checkpoint_on_traj.py \
      --checkpoint runs/uma/.../checkpoints/final/inference_ckpt.pt \
      --data data/processed/matpes/r2scan_flatiron/r2scan_3/lte_val.traj \
      --output-dir runs/matpes_parity/uma_adhoc/lte3_val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_uma_parity_batched import (  # noqa: E402
    MAX_DFT_FORCE_EV_PER_A,
    make_parity_plot,
)
from scripts.internal.uma_parity_inference import (  # noqa: E402
    prepare_predictions,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="UMA inference checkpoint (inference_ckpt.pt).")
    parser.add_argument("--data", type=Path, required=True,
                        help="ASE trajectory or ASE-LMDB of reference frames.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-name", help="UMA dataset/task name, e.g. omat.")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-atoms-per-batch", type=int, default=4096)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Evaluate only the first N frames. Useful for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--plot", type=Path, default=None,
                        help="Parity figure path. Defaults to <output-dir>/parity.png.")
    parser.add_argument("--plot-style", default="hexbin", choices=["hexbin", "scatter"],
                        help="Parity rendering; both draw every point.")
    parser.add_argument("--gridsize", type=int, default=160,
                        help="Hexbin resolution along x.")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_args = SimpleNamespace(
        model=args.checkpoint,
        data=args.data,
        output_dir=args.output_dir,
        task_name=args.task_name,
        manifest=None,
        task_index=None,
        device=args.device,
        batch_size=args.batch_size,
        max_atoms_per_batch=args.max_atoms_per_batch,
        max_frames=args.max_frames,
        progress_every=args.progress_every,
    )
    csv_path, metadata_path, metadata, model_path, data_path = prepare_predictions(
        inference_args
    )

    plot_path = args.plot or (Path(args.output_dir).expanduser().resolve() / "parity.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or f"{model_path.parent.parent.parent.name} on {data_path.name}"
    plot = make_parity_plot(
        csv_path, plot_path, title, metadata["counts"],
        style=args.plot_style, gridsize=args.gridsize,
    )
    metadata["plot"] = {
        "complete": True,
        "path": str(plot_path),
        "size_bytes": plot_path.stat().st_size,
        **plot,
    }
    write_json(metadata_path, metadata)

    print(f"Checkpoint: {model_path}")
    print(f"Data: {data_path}")
    print(f"Energy MAE: {1000 * plot['energy_mae_eV_per_atom']:.6g} meV/atom")
    print(f"Force MAE: {1000 * plot['force_mae_eV_per_A']:.6g} meV/A")
    stress = plot["stress_mae_eV_per_A3"]
    if stress is None:
        print("Stress MAE: unavailable")
    else:
        print(f"Stress MAE: {1000 * stress:.6g} meV/A^3")
    print(f"Frames used: {plot['included_frames']:,}")
    print(f"Excluded frames above {MAX_DFT_FORCE_EV_PER_A} eV/A: {plot['excluded_frames']:,}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
