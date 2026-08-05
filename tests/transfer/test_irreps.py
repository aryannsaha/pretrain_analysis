"""Tests for the irreps -> invariant-feature reductions.

The load-bearing test here is
:func:`test_reductions_are_invariant_under_a_simulated_rotation`, which
simulates how a rotation acts on an irreps tensor (an orthogonal Wigner-D
matrix applied within each ``2l+1`` block, shared across the channels of that
block) and checks that the ``inv`` and ``normed`` reductions do not move while
the raw flattened features do.  That contrast is the entire reason this module
exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from pretrain_analysis.transfer.irreps import (
    block_norms,
    feature_names,
    invariant_features,
    normed_features,
    parse_irreps,
    reduce_features,
)


def _random_orthogonal(dim, rng):
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))  # fix the sign convention so q is deterministic


def _apply_block_rotation(x, layout, rng):
    """Act on an irreps tensor the way a rotation of the structure would.

    Scalars are untouched; each ``l > 0`` block is hit with a single orthogonal
    ``(2l+1) x (2l+1)`` matrix shared by all ``mul`` channels of that block --
    which is exactly the structure of a Wigner-D action.
    """
    out = x.copy()
    for b in layout.higher_blocks:
        d = _random_orthogonal(b.dim, rng)
        block = out[:, b.start : b.stop].reshape(-1, b.mul, b.dim)
        out[:, b.start : b.stop] = (block @ d.T).reshape(-1, b.mul * b.dim)
    return out


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_parses_the_mace_hidden_irreps():
    """The layout these six checkpoints actually use."""
    layout = parse_irreps("128x0e + 128x1o")
    assert layout.dim == 128 * 1 + 128 * 3 == 512
    assert layout.n_invariant == 128
    assert layout.n_normed == 256
    assert [str(b) for b in layout.blocks] == ["128x0e", "128x1o"]
    assert layout.blocks[0].start == 0 and layout.blocks[0].stop == 128
    assert layout.blocks[1].start == 128 and layout.blocks[1].stop == 512


def test_parses_last_layer_scalars_only():
    """MACE's final product block emits scalars only, so normed == inv there."""
    layout = parse_irreps("128x0e")
    assert layout.dim == 128
    assert layout.n_invariant == layout.n_normed == 128
    assert layout.higher_blocks == ()


@pytest.mark.parametrize(
    "spec,dim,n_inv,n_normed",
    [
        ("0e", 1, 1, 1),
        ("16x0e", 16, 16, 16),
        ("32x0e+16x1o+8x2e", 32 + 48 + 40, 32, 32 + 16 + 8),
        ("1x0e+1x1o+1x2e+1x3o", 1 + 3 + 5 + 7, 1, 4),
    ],
)
def test_parse_dimensions(spec, dim, n_inv, n_normed):
    layout = parse_irreps(spec)
    assert layout.dim == dim
    assert layout.n_invariant == n_inv
    assert layout.n_normed == n_normed


def test_parse_tolerates_whitespace_and_zero_multiplicity():
    assert parse_irreps("  128x0e  +  128x1o ").dim == 512
    # e3nn drops 0x terms; so do we.
    assert parse_irreps("16x0e+0x1o").dim == 16


def test_parse_rejects_garbage():
    with pytest.raises(ValueError, match="Cannot parse"):
        parse_irreps("128y0e")
    with pytest.raises(ValueError, match="Cannot parse"):
        parse_irreps("128x0")


def test_parse_accepts_an_object_with_an_irreps_str():
    class FakeIrreps:
        def __str__(self):
            return "128x0e+128x1o"

    assert parse_irreps(FakeIrreps()).dim == 512


# --------------------------------------------------------------------------
# the invariance property -- the reason this module exists
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec", ["128x0e+128x1o", "32x0e+16x1o+8x2e", "1x0e+1x1o+1x2e+1x3o"]
)
def test_reductions_are_invariant_under_a_simulated_rotation(spec):
    rng = np.random.default_rng(0)
    layout = parse_irreps(spec)
    x = rng.standard_normal((200, layout.dim))
    x_rot = _apply_block_rotation(x, layout, rng)

    # The reductions must not move ...
    np.testing.assert_allclose(
        invariant_features(x, layout), invariant_features(x_rot, layout), atol=1e-12
    )
    np.testing.assert_allclose(
        block_norms(x, layout), block_norms(x_rot, layout), atol=1e-10
    )
    np.testing.assert_allclose(
        normed_features(x, layout), normed_features(x_rot, layout), atol=1e-10
    )

    # ... while the raw features do, which is why they must never be used as F.
    assert not np.allclose(x, x_rot, atol=1e-6)


def test_rotation_only_touches_the_higher_blocks():
    """Sanity check on the test's own rotation helper."""
    rng = np.random.default_rng(1)
    layout = parse_irreps("32x0e+16x1o")
    x = rng.standard_normal((50, layout.dim))
    x_rot = _apply_block_rotation(x, layout, rng)
    np.testing.assert_allclose(x[:, :32], x_rot[:, :32], atol=1e-15)
    assert not np.allclose(x[:, 32:], x_rot[:, 32:])


def test_norms_use_the_channel_outer_layout():
    """Guard the mul-outer convention, which is where a layout bug would hide.

    Built so the two conventions give different answers: channel ``i`` of the
    ``1o`` block is the vector ``(i, 0, 0)``, so the correct norms are
    ``0, 1, 2, 3`` and any transposed reading gives something else.
    """
    layout = parse_irreps("2x0e+4x1o")
    x = np.zeros((1, layout.dim))
    x[0, :2] = [7.0, 8.0]
    for i in range(4):
        x[0, 2 + 3 * i] = float(i)  # first component of channel i

    np.testing.assert_allclose(invariant_features(x, layout), [[7.0, 8.0]])
    np.testing.assert_allclose(block_norms(x, layout), [[0.0, 1.0, 2.0, 3.0]])


def test_normed_is_inv_then_norms():
    rng = np.random.default_rng(2)
    layout = parse_irreps("8x0e+4x1o+2x2e")
    x = rng.standard_normal((10, layout.dim))
    expected = np.concatenate(
        [invariant_features(x, layout), block_norms(x, layout)], axis=1
    )
    np.testing.assert_allclose(normed_features(x, layout), expected)
    assert normed_features(x, layout).shape[1] == layout.n_normed


# --------------------------------------------------------------------------
# API surface
# --------------------------------------------------------------------------


def test_feature_names_match_widths():
    layout = parse_irreps("128x0e+128x1o")
    assert len(feature_names(layout, "inv")) == layout.n_invariant
    assert len(feature_names(layout, "normed")) == layout.n_normed
    assert feature_names(layout, "normed")[128] == "|l1o|[0]"


def test_reduce_features_dispatches():
    rng = np.random.default_rng(3)
    layout = parse_irreps("16x0e+8x1o")
    x = rng.standard_normal((20, layout.dim))
    np.testing.assert_allclose(
        reduce_features(x, layout, "inv"), invariant_features(x, layout)
    )
    np.testing.assert_allclose(
        reduce_features(x, layout, "normed"), normed_features(x, layout)
    )


def test_reduce_features_refuses_raw_components():
    """Asking for the equivariant components must fail loudly, not silently work."""
    layout = parse_irreps("16x0e+8x1o")
    x = np.zeros((5, layout.dim))
    with pytest.raises(ValueError, match="not rotation"):
        reduce_features(x, layout, "raw")


def test_width_mismatch_is_caught():
    layout = parse_irreps("128x0e+128x1o")
    with pytest.raises(ValueError, match="does not match irreps"):
        invariant_features(np.zeros((4, 640)), layout)
