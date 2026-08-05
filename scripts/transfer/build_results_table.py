#!/usr/bin/env python
"""Join LogME scores to post-fine-tune MAEs and report rank agreement.

Reads the per-checkpoint scores written by ``run_logme_analysis.py``, scrapes
the fine-tune MAEs out of MACE's JSONL result logs, and emits the correlation
table plus an honest assessment of whether anything was demonstrated.

Runs that have not finished are reported as such rather than silently dropped:
a correlation over four of six checkpoints is a different claim from one over
six, and the difference must be visible.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Runs and outputs live in the main checkout even when this executes from a
# worktree.
DATA_ROOT = Path(
    os.environ.get("PRETRAIN_ANALYSIS_ROOT", "/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
)

from pretrain_analysis.transfer.evaluate import (  # noqa: E402
    compare_predictors,
    minimum_achievable_p,
)

LOGGER = logging.getLogger("results")

SCALES = ["omat_100k", "omat_500k", "omat_1m", "omat_2m", "omat_5m", "omat_10m"]

# Fine-tune run directories, keyed by downstream target.
FINETUNE_DIRS = {
    "mof_off": DATA_ROOT / "runs/mace/mof_off_ft_rs",
    "am": DATA_ROOT / "runs/mace/am_ft_rs",
}
RUN_PREFIX = {"mof_off": "mace_mof_off_rs_small_ft_", "am": "mace_am_rs_small_ft_"}

# Predictors where a LARGER value means a WORSE expected transfer.
LOWER_IS_BETTER = (
    "mahalanobis_mean",
    "mahalanobis_median",
    "energy_ridge_mae",
    "forcemag_ridge_mae",
)


def scrape_finetune_metrics(results_file: Path) -> dict:
    """Best and final validation MAEs from a MACE JSONL training log.

    MACE writes many rows per epoch and a pre-training row with
    ``epoch: null``.  Only ``mode == "eval"`` rows with a real epoch number are
    training progress; the null-epoch rows are the *initial* evaluation of the
    frozen foundation model, which is interesting in its own right (it is a
    zero-shot baseline) so it is reported separately.
    """
    rows = []
    with open(results_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    evals = [r for r in rows if r.get("mode") == "eval"]
    trained = [r for r in evals if r.get("epoch") is not None]
    initial = [r for r in evals if r.get("epoch") is None]

    out: dict = {
        "n_eval_rows": len(evals),
        "n_epochs_logged": len({r["epoch"] for r in trained}) if trained else 0,
        "max_epoch": max((r["epoch"] for r in trained), default=np.nan),
    }
    if initial:
        out["zeroshot_mae_e_per_atom"] = float(initial[0].get("mae_e_per_atom", np.nan))
        out["zeroshot_mae_f"] = float(initial[0].get("mae_f", np.nan))
    if trained:
        e = np.array([r.get("mae_e_per_atom", np.nan) for r in trained], dtype=float)
        f = np.array([r.get("mae_f", np.nan) for r in trained], dtype=float)
        last = max(trained, key=lambda r: r["epoch"])
        out.update(
            {
                "best_mae_e_per_atom": float(np.nanmin(e)),
                "best_mae_f": float(np.nanmin(f)),
                "final_mae_e_per_atom": float(last.get("mae_e_per_atom", np.nan)),
                "final_mae_f": float(last.get("mae_f", np.nan)),
            }
        )
    return out


def collect_finetune_table(targets) -> pd.DataFrame:
    rows = []
    for target in targets:
        base = FINETUNE_DIRS.get(target)
        if base is None or not base.exists():
            LOGGER.warning("no fine-tune directory for target %s (%s)", target, base)
            continue
        for scale in SCALES:
            short = scale.replace("omat_", "")
            run_dir = base / f"{RUN_PREFIX[target]}{short}"
            rec = {"target": target, "checkpoint": scale, "run_dir": str(run_dir)}
            results = sorted(run_dir.glob("results/*_train.txt"))
            if not results:
                rec["status"] = "no results yet"
                rows.append(rec)
                continue
            rec.update(scrape_finetune_metrics(results[0]))
            rec["status"] = (
                "finished" if rec.get("n_epochs_logged", 0) > 0 else "started, no epochs"
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logme-dir", default=str(DATA_ROOT / "outputs/logme"))
    ap.add_argument("--targets", nargs="*", default=["mof_off", "am"])
    ap.add_argument("--out", default=str(DATA_ROOT / "outputs/logme"))
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- fine-tune ground truth ------------------------------------------
    ft = collect_finetune_table(args.targets)
    ft_path = out_dir / "finetune_metrics.csv"
    ft.to_csv(ft_path, index=False)
    LOGGER.info("fine-tune status written to %s", ft_path)
    if not ft.empty:
        print("\n=== fine-tune run status ===")
        cols = [c for c in ("target", "checkpoint", "status", "max_epoch",
                            "best_mae_e_per_atom", "best_mae_f") if c in ft.columns]
        print(ft[cols].to_string(index=False))

    # ---- LogME scores -----------------------------------------------------
    frames = []
    for target in args.targets:
        p = Path(args.logme_dir) / target / f"logme_scores_{target}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            LOGGER.warning("missing LogME scores: %s", p)
    if not frames:
        LOGGER.error("no LogME score files found; run run_logme_analysis.py first")
        return 1
    scores = pd.concat(frames, ignore_index=True)
    LOGGER.info("loaded %d LogME rows", len(scores))

    usable = ft[ft.get("best_mae_e_per_atom").notna()] if "best_mae_e_per_atom" in ft else pd.DataFrame()
    if usable.empty:
        print(
            "\n*** No fine-tune has logged a completed epoch yet, so no rank "
            "correlation can be computed. The LogME and baseline scores are "
            "final and written; re-run this script once the fine-tunes finish. ***"
        )
        scores.to_csv(out_dir / "logme_scores_all.csv", index=False)
        return 0

    merged = scores.merge(
        usable[["target", "checkpoint", "best_mae_e_per_atom", "best_mae_f"]],
        on=["target", "checkpoint"],
        how="left",
    )
    merged.to_csv(out_dir / "logme_scores_with_mae.csv", index=False)

    # ---- rank agreement, per (target, feature_set, pooling) ---------------
    predictor_cols = [
        c
        for c in (
            "energy_logme", "forcemag_logme", "log_n_pretrain",
            "energy_ridge_r2", "forcemag_ridge_r2",
            "energy_h_score", "forcemag_h_score",
            "gaussian_loglik_mean", "gmm_target_loglik",
            "mahalanobis_mean", "participation_ratio", "effective_rank",
        )
        if c in merged.columns
    ]

    results = []
    for (target, fset, pooling), grp in merged.groupby(
        ["target", "feature_set", "pooling"]
    ):
        grp = grp.dropna(subset=["best_mae_e_per_atom"])
        if len(grp) < 3:
            continue
        for r in compare_predictors(
            grp,
            predictor_cols,
            ["best_mae_e_per_atom", "best_mae_f"],
            lower_is_better_predictors=LOWER_IS_BETTER,
            n_bootstrap=args.n_bootstrap,
        ):
            d = r.as_dict()
            d.update({"downstream": target, "feature_set": fset, "pooling": pooling})
            results.append(d)

    corr = pd.DataFrame(results)
    corr_path = out_dir / "rank_correlations.csv"
    corr.to_csv(corr_path, index=False)
    LOGGER.info("wrote %s (%d rows)", corr_path, len(corr))

    n = int(corr["n_models"].max()) if not corr.empty else 0
    print(f"\n=== power at n = {n} ===")
    if n >= 3:
        print(f"  best attainable one-sided exact p = 1/{n}! = {minimum_achievable_p(n):.5f}")
    print(
        "  At n = 6 a Spearman rho below 0.83 does not clear p < 0.05. "
        "Any correlation reported here should be read against that bar."
    )

    if not corr.empty:
        print("\n=== headline: weighted Kendall tau_w vs energy MAE ===")
        head = (
            corr[corr["target"] == "best_mae_e_per_atom"]
            .sort_values("weighted_kendall", ascending=False)
            .head(25)
        )
        print(
            head[
                ["downstream", "feature_set", "pooling", "predictor",
                 "weighted_kendall", "spearman", "perm_p_spearman", "n_models"]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
