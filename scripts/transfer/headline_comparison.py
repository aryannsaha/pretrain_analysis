#!/usr/bin/env python
"""The pre-specified headline comparison, without cherry-picking.

``build_results_table.py`` reports the best feature set per predictor. That is
useful for exploration and **misleading as a headline**: it maximizes over 10
(feature set x pooling) combinations per predictor, and at n = 6 the maximum of
10 rank correlations is biased sharply upward. A predictor with no signal at all
clears rho = 1.0 in one of ten tries far more often than 0.1% of the time.

This script instead fixes the feature set *before* looking at the answer.
``layer0/normed`` with mean-pooling is chosen on principle, not on performance:

* ``layer0`` because it is the only interaction block with ``l > 0`` content
  (the second and last block emits scalars only);
* ``normed`` because it is a strict superset of ``inv``;
* mean-pooling because it is the pairing that matches a per-atom energy target
  exactly (see the caveats).

It also prints the zero-shot ground truth so the monotonicity that drives every
correlation to +1 is visible rather than inferred.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DATA_ROOT = Path(
    os.environ.get("PRETRAIN_ANALYSIS_ROOT", "/scratch/gpfs/ROSENGROUP/aryan/pretrain_analysis")
)

from pretrain_analysis.transfer.evaluate import rank_agreement  # noqa: E402

SCALES = ["omat_100k", "omat_500k", "omat_1m", "omat_2m", "omat_5m", "omat_10m"]
FEATURE_SET = "layer0/normed"
POOLING = "mean"

PREDICTORS = [
    ("energy_logme", True, "LogME (energy)"),
    ("forcemag_logme", True, "LogME (force magnitude)"),
    ("log_n_pretrain", True, "log N_pretrain  [trivial baseline]"),
    ("energy_ridge_r2", True, "Ridge probe R^2 (energy)"),
    ("forcemag_ridge_r2", True, "Ridge probe R^2 (force mag)"),
    ("energy_h_score", True, "H-score (energy)"),
    ("forcemag_h_score", True, "H-score (force mag)"),
    ("gmm_target_loglik", True, "GMM log-likelihood"),
    ("gaussian_loglik_mean", True, "Gaussian log-likelihood"),
    ("mahalanobis_mean", False, "Mahalanobis distance"),
    ("participation_ratio", True, "Participation ratio"),
    ("effective_rank", True, "Effective rank"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logme-dir", default=str(DATA_ROOT / "outputs/logme"))
    ap.add_argument("--targets", nargs="*", default=["mof_off", "am"])
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    pd.set_option("display.width", 200)

    for target in args.targets:
        sp = Path(args.logme_dir) / target / f"logme_scores_{target}.csv"
        zp = Path(args.logme_dir) / f"zeroshot_{target}.csv"
        if not sp.exists() or not zp.exists():
            print(f"\n### {target}: missing inputs, skipping")
            continue

        scores = pd.read_csv(sp)
        zs = pd.read_csv(zp)
        df = (
            scores[(scores.feature_set == FEATURE_SET) & (scores.pooling == POOLING)]
            .merge(zs, on=["target", "checkpoint"])
            .set_index("checkpoint")
            .reindex(SCALES)
            .reset_index()
        )

        print(f"\n{'=' * 78}")
        print(f"### {target}   feature set = {FEATURE_SET}, pooling = {POOLING}")
        print(f"{'=' * 78}")

        print("\n-- zero-shot ground truth (frozen model, no fine-tuning) --")
        gt = df[["checkpoint", "force_mae", "energy_mae_per_atom_shifted"]].copy()
        gt.columns = ["checkpoint", "force MAE (eV/A)", "E MAE shifted (eV/atom)"]
        print(gt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        fm = df["force_mae"].to_numpy(dtype=float)
        print(
            f"   force MAE strictly decreasing with scale: "
            f"{bool(np.all(np.diff(fm) < 0))}"
        )

        print("\n-- predictors --")
        vals = df[["checkpoint"] + [c for c, _, _ in PREDICTORS if c in df.columns]]
        print(vals.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        for truth, truth_label in (
            ("force_mae", "zero-shot force MAE"),
            ("energy_mae_per_atom_shifted", "zero-shot energy MAE (offset-corrected)"),
        ):
            print(f"\n-- rank agreement vs {truth_label} --")
            rows = []
            for col, higher_better, label in PREDICTORS:
                if col not in df.columns or df[col].isna().all():
                    continue
                try:
                    r = rank_agreement(
                        df[col].to_numpy(dtype=float),
                        df[truth].to_numpy(dtype=float),
                        predictor=label,
                        higher_predictor_is_better=higher_better,
                        n_bootstrap=args.n_bootstrap,
                    )
                except ValueError:
                    continue
                rows.append(
                    {
                        "predictor": label,
                        "tau_w": r.weighted_kendall,
                        "rho": r.spearman,
                        "exact p": r.perm_p_spearman,
                        "rho 95% CI": f"[{r.spearman_ci[0]:+.2f}, {r.spearman_ci[1]:+.2f}]",
                        "sig?": "yes" if r.perm_p_spearman < 0.05 else "no",
                    }
                )
            out = pd.DataFrame(rows).sort_values("tau_w", ascending=False)
            print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

            # The verdict that matters: did anything beat the trivial baseline?
            perfect = out[out["rho"] == 1.0]["predictor"].tolist()
            trivial = "log N_pretrain  [trivial baseline]"
            trivial_row = out[out["predictor"] == trivial]
            trivial_rho = float(trivial_row["rho"].iloc[0]) if not trivial_row.empty else np.nan
            trivial_tau = float(trivial_row["tau_w"].iloc[0]) if not trivial_row.empty else np.nan

            print(f"\n   log(N_pretrain) baseline: rho = {trivial_rho:+.4f}, "
                  f"tau_w = {trivial_tau:+.4f}")

            if trivial in perfect:
                print(
                    f"   *** {len(perfect)} predictors tied at rho = 1.000, INCLUDING the "
                    "trivial baseline.\n"
                    "       At n = 6 a perfect ranking is the only outcome that clears "
                    "p < 0.01, so a tie\n"
                    "       here is not evidence for any of them -- it is evidence that the "
                    "ground truth is\n"
                    "       monotonic in pretraining scale and the design cannot discriminate."
                )
            else:
                beat = out[out["tau_w"] > trivial_tau + 1e-9]["predictor"].tolist()
                if beat:
                    print(
                        f"   *** The ground truth is NOT monotonic in scale, so the trivial "
                        "baseline is beatable\n"
                        f"       here -- and {len(beat)} predictor(s) beat it on tau_w: "
                        f"{', '.join(beat)}.\n"
                        "       This is the only configuration in the study where LogME can "
                        "demonstrate value,\n"
                        "       and at n = 6 it rests on a single pair of models swapping."
                    )
                else:
                    print("   *** Nothing beat the trivial baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
