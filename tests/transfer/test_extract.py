"""Tests for the parts of the extractor that do not need a MACE checkpoint.

The model-dependent behaviour -- hooks firing, irreps layout matching the real
network, rotation invariance on a real periodic structure -- is covered by
``scripts/transfer/smoke_test_extraction.py``, which needs a GPU-loadable
checkpoint and so cannot live in the unit suite.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pretrain_analysis.transfer.extract import cache_paths, pool_features, sha256_file

# --------------------------------------------------------------------------
# pooling -- where a sum/mean mix-up would silently break the energy target
# --------------------------------------------------------------------------


def test_sum_and_mean_pooling_agree_up_to_atom_count():
    rng = np.random.default_rng(0)
    ptr = np.array([0, 3, 7, 8])  # 3 structures of 3, 4 and 1 atoms
    x = rng.standard_normal((8, 5))

    summed = pool_features(x, ptr, how="sum")
    meaned = pool_features(x, ptr, how="mean")
    counts = np.diff(ptr)[:, None]

    assert summed.shape == meaned.shape == (3, 5)
    np.testing.assert_allclose(summed, meaned * counts)


def test_pooling_respects_structure_boundaries():
    ptr = np.array([0, 2, 5])
    x = np.array(
        [[1.0], [2.0], [10.0], [20.0], [30.0]]
    )
    np.testing.assert_allclose(pool_features(x, ptr, how="sum"), [[3.0], [60.0]])
    np.testing.assert_allclose(pool_features(x, ptr, how="mean"), [[1.5], [20.0]])


def test_single_atom_structure_pools_to_itself():
    ptr = np.array([0, 1])
    x = np.array([[3.0, 4.0]])
    np.testing.assert_allclose(pool_features(x, ptr, how="sum"), x)
    np.testing.assert_allclose(pool_features(x, ptr, how="mean"), x)


def test_pooling_promotes_to_float64():
    """Features are cached as float16; pooling must not accumulate in float16.

    Two separate error sources, and only one of them is fixable here:

    * **Accumulation** in float16 over 1000 terms would lose several digits.
      Pooling promotes to float64, so this is eliminated -- the sum is exact
      given its inputs.
    * **Quantization** of the stored values is not recoverable. float16 carries
      ~3 decimal digits, so 0.01 is stored as 0.00995, and the exact sum of
      1000 of them is 9.953125, not 10.0. That 0.47% offset is a property of
      the cache format, not of the arithmetic.

    The test asserts both: exactness against the quantized inputs, and the
    magnitude of the quantization itself, so that the float16 caching decision
    stays visible rather than being discovered later as a mystery discrepancy.
    """
    ptr = np.array([0, 1000])
    x = np.full((1000, 2), 0.01, dtype=np.float16)
    out = pool_features(x, ptr, how="sum")

    assert out.dtype == np.float64
    # Exact given the stored (quantized) values -- no accumulation error.
    np.testing.assert_allclose(out, np.full((1, 2), x[0, 0].astype(np.float64) * 1000))
    # And the quantization itself is a ~0.5% effect at this magnitude.
    assert abs(out[0, 0] - 10.0) / 10.0 < 0.01


def test_float16_caching_perturbs_logme_only_marginally():
    """Bound the cost of the float16 cache on the score it feeds.

    Per-atom features are stored as float16, so LogME sees quantized inputs.
    This checks the resulting shift is small relative to the between-model
    differences the study needs to resolve, rather than assuming it.
    """
    from pretrain_analysis.transfer.logme import logme

    rng = np.random.default_rng(0)
    n, d = 5_000, 128
    f = rng.standard_normal((n, d)) * 0.1  # MACE-like magnitudes
    y = f @ rng.standard_normal(d) + 0.05 * rng.standard_normal(n)

    exact = logme(f, y).score
    quantized = logme(f.astype(np.float16).astype(np.float64), y).score

    assert abs(exact - quantized) < 0.02 * abs(exact), (
        f"float16 shifted LogME from {exact:.4f} to {quantized:.4f}"
    )


def test_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="unknown pooling"):
        pool_features(np.zeros((3, 2)), np.array([0, 3]), how="max")


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "ckpt.model"
    payload = b"not really a checkpoint" * 5000
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_sha256_is_chunk_size_independent(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(bytes(range(256)) * 4000)
    assert sha256_file(p, chunk=97) == sha256_file(p, chunk=1 << 20)


def test_cache_paths_are_filesystem_safe(tmp_path):
    paths = cache_paths(tmp_path, "omat_1m", "layer0/inv")
    assert "/" not in paths["features"].name
    assert paths["features"].name == "layer0__inv.f16.npy"
    assert paths["dir"] == tmp_path / "omat_1m"
    assert paths["provenance"].name == "provenance.json"
