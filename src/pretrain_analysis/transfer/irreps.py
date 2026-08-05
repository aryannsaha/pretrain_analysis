"""Irreps bookkeeping for turning MACE node features into invariant vectors.

MACE node features are O(3)-equivariant irreps, not plain vectors.  Flattening
them into a design matrix would be wrong: under a rotation of the input
structure the ``l > 0`` components mix among themselves, so a linear model
fitted on them is fitting one arbitrary choice of frame, and both the fit and
any transferability score derived from it are meaningless.

This module provides the two invariant reductions used throughout:

``inv``
    The ``l = 0`` channels alone.  Exactly invariant, and exactly the input
    that MACE's own readout consumes, which is what makes the LogME linear
    model match the network's real functional form.

``normed``
    ``inv`` concatenated with the per-``l``, per-channel L2 norms of the
    ``l > 0`` blocks.  Each norm is invariant because rotations act on a
    ``(2l+1)``-dimensional block by an orthogonal Wigner-D matrix, which
    preserves the block's Euclidean length.  This recovers geometric
    information that ``inv`` discards.

Layout convention
-----------------
e3nn stores an irreps tensor as concatenated ``mul x ir`` blocks in the order
the irreps are declared.  Within a block of ``mul`` copies of an irrep of
dimension ``2l+1``, the multiplicity index is the *outer* one, so the block
reshapes as ``(..., mul, 2l+1)``.  MACE's own ``reshape_irreps`` in
``mace/modules/irreps_tools.py`` relies on the same convention.

The irreps string is parsed here rather than handed to ``e3nn.o3.Irreps`` so
that this logic is testable without e3nn and unaffected by the e3nn version
split between the project's two conda environments (0.4.4 in the MACE env,
0.6.0 in the analysis env).  Parsing accepts either a string or anything whose
``str()`` is an irreps expression, so an ``o3.Irreps`` object works directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

__all__ = [
    "IrrepBlock",
    "IrrepsLayout",
    "parse_irreps",
]

_TERM = re.compile(r"^\s*(?:(\d+)\s*x\s*)?(\d+)([eo])\s*$")


@dataclass(frozen=True)
class IrrepBlock:
    """One ``mul x l{p}`` block and where it sits in the flat feature vector."""

    mul: int
    l: int  # noqa: E741 - `l` is the standard name for the degree
    parity: str  # "e" or "o"
    start: int  # flat offset of the block
    stop: int  # exclusive end; stop - start == mul * (2l + 1)

    @property
    def dim(self) -> int:
        return 2 * self.l + 1

    def __str__(self) -> str:
        return f"{self.mul}x{self.l}{self.parity}"


@dataclass(frozen=True)
class IrrepsLayout:
    """Parsed irreps expression with flat-offset bookkeeping."""

    blocks: tuple[IrrepBlock, ...]

    @property
    def dim(self) -> int:
        """Total flat dimension."""
        return self.blocks[-1].stop if self.blocks else 0

    @property
    def scalar_blocks(self) -> tuple[IrrepBlock, ...]:
        return tuple(b for b in self.blocks if b.l == 0)

    @property
    def higher_blocks(self) -> tuple[IrrepBlock, ...]:
        return tuple(b for b in self.blocks if b.l > 0)

    @property
    def n_invariant(self) -> int:
        """Width of the ``inv`` feature vector."""
        return sum(b.mul for b in self.scalar_blocks)

    @property
    def n_normed(self) -> int:
        """Width of the ``normed`` feature vector (``inv`` + one norm per channel)."""
        return self.n_invariant + sum(b.mul for b in self.higher_blocks)

    def __str__(self) -> str:
        return " + ".join(str(b) for b in self.blocks)


def parse_irreps(spec) -> IrrepsLayout:
    """Parse an irreps expression such as ``"128x0e + 128x1o"``.

    Accepts a string or any object whose ``str()`` is one (notably
    ``e3nn.o3.Irreps``).  A bare ``"0e"`` is read as multiplicity 1, matching
    e3nn.  Zero-multiplicity terms are dropped, as e3nn also drops them.
    """
    text = str(spec).strip()
    if not text:
        return IrrepsLayout(blocks=())

    blocks: list[IrrepBlock] = []
    offset = 0
    for term in text.split("+"):
        m = _TERM.match(term)
        if not m:
            raise ValueError(f"Cannot parse irreps term {term!r} in {text!r}")
        mul = int(m.group(1)) if m.group(1) is not None else 1
        ell = int(m.group(2))
        parity = m.group(3)
        if mul == 0:
            continue
        width = mul * (2 * ell + 1)
        blocks.append(
            IrrepBlock(mul=mul, l=ell, parity=parity, start=offset, stop=offset + width)
        )
        offset += width
    return IrrepsLayout(blocks=tuple(blocks))


def _as_2d(x: np.ndarray, layout: IrrepsLayout) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D (n, dim) array, got shape {x.shape}")
    if x.shape[1] != layout.dim:
        raise ValueError(
            f"feature width {x.shape[1]} does not match irreps '{layout}' "
            f"(expected {layout.dim})"
        )
    return x


def invariant_features(x: np.ndarray, layout: IrrepsLayout) -> np.ndarray:
    """Extract the ``l = 0`` channels: the ``inv`` feature set.

    Returns an ``(n, n_invariant)`` array.
    """
    x = _as_2d(x, layout)
    if not layout.scalar_blocks:
        return np.empty((x.shape[0], 0), dtype=x.dtype)
    return np.concatenate([x[:, b.start : b.stop] for b in layout.scalar_blocks], axis=1)


def block_norms(x: np.ndarray, layout: IrrepsLayout) -> np.ndarray:
    """Per-``l``, per-channel L2 norms of the ``l > 0`` blocks.

    Returns an ``(n, sum_of_higher_multiplicities)`` array.  Each column is the
    length of one channel's ``(2l+1)``-vector, which a rotation leaves fixed.
    """
    x = _as_2d(x, layout)
    out = []
    for b in layout.higher_blocks:
        block = x[:, b.start : b.stop].reshape(x.shape[0], b.mul, b.dim)
        out.append(np.linalg.norm(block, axis=-1))
    if not out:
        return np.empty((x.shape[0], 0), dtype=x.dtype)
    return np.concatenate(out, axis=1)


def normed_features(x: np.ndarray, layout: IrrepsLayout) -> np.ndarray:
    """``inv`` concatenated with :func:`block_norms`: the ``normed`` feature set."""
    return np.concatenate(
        [invariant_features(x, layout), block_norms(x, layout)], axis=1
    )


def feature_names(layout: IrrepsLayout, variant: str) -> list[str]:
    """Human-readable column labels, so a cached matrix stays interpretable."""
    if variant not in ("inv", "normed"):
        raise ValueError(f"unknown variant {variant!r}")
    names = []
    for b in layout.scalar_blocks:
        names += [f"l0{b.parity}[{i}]" for i in range(b.mul)]
    if variant == "normed":
        for b in layout.higher_blocks:
            names += [f"|l{b.l}{b.parity}|[{i}]" for i in range(b.mul)]
    return names


def reduce_features(x: np.ndarray, layout: IrrepsLayout, variant: str) -> np.ndarray:
    """Dispatch to the requested invariant reduction."""
    if variant == "inv":
        return invariant_features(x, layout)
    if variant == "normed":
        return normed_features(x, layout)
    raise ValueError(
        f"unknown variant {variant!r}; expected 'inv' or 'normed'. Raw l>0 "
        "components are deliberately not offered: they are not rotation "
        "invariant and a linear model on them is frame-dependent."
    )
