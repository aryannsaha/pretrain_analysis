"""Training-free transferability estimation for pretrained MACE potentials.

Modules
-------
logme
    From-scratch LogME (You et al., ICML 2021) via evidence maximization.
extract
    Hook-based extraction of O(3)-invariant activations from MACE models.
baselines
    Competing predictors: ridge probe, H-score, distributional distance,
    effective dimensionality.
evaluate
    Rank-correlation machinery and bootstrap confidence intervals.
"""

from pretrain_analysis.transfer.logme import LogMEResult, logme, logme_multi

__all__ = ["LogMEResult", "logme", "logme_multi"]
