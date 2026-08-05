"""Hook-based extraction of O(3)-invariant activations from frozen MACE models.

Registers forward hooks on the interaction/product blocks and the readout
rather than modifying MACE, so the vendored fork at ``mace-src/`` stays
untouched and the extraction follows whatever the checkpoint actually does.

What is extracted, per layer
----------------------------
``inv``
    The ``l = 0`` channels of that layer's ``EquivariantProductBasisBlock``
    output.  Primary feature set, and the exact input MACE's own readout sees.
``normed``
    ``inv`` plus the per-``l``, per-channel norms of the ``l > 0`` blocks --
    invariant, and carrying geometric information ``inv`` throws away.
``readout_hidden``
    Post-activation hidden layer of the final ``NonLinearReadoutBlock``.
    Already scalars (``16x0e`` in these checkpoints), so it is invariant by
    construction.  Included because the readout is what a fine-tune head most
    directly reuses.

Raw ``l > 0`` components are never written to a feature matrix.  See
:mod:`pretrain_analysis.transfer.irreps` for why.

Pooling and the energy target
-----------------------------
MACE computes ``E_inter = sum_i readout(h_i)``, so the *total* interaction
energy is exactly linear in the **sum**-pooled features, and the *per-atom*
energy is exactly linear in the **mean**-pooled features.  Those are the two
self-consistent pairings; mixing them (sum-pool against per-atom energy)
introduces a system-size dependence that is not in the model's functional
form.  :func:`pool_features` produces both, and the driver defaults to the
consistent pairing while reporting the alternative.

A second wrinkle: MACE's total energy is ``E0 + E_inter``, where ``E0`` is a
fixed per-element sum the readout never sees.  A linear model on node features
can only explain ``E_inter``.  ``subtract_e0`` uses the checkpoint's own
``atomic_energies`` table to remove that offset; both variants are reported,
because the model's OMat24-fitted E0s are not the downstream dataset's E0s and
neither choice is obviously right.

Caching
-------
Features are written as float16 memmaps (cast to float64 only inside LogME)
with a sidecar index mapping row -> (structure_id, atom_index, atomic_number)
and a provenance JSON recording the checkpoint SHA-256, the MACE version, the
irreps layout and the extraction settings.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pretrain_analysis.transfer.irreps import (
    IrrepsLayout,
    parse_irreps,
    reduce_features,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ExtractionConfig",
    "MaceActivationExtractor",
    "cache_paths",
    "load_cached",
    "load_mace_model",
    "pool_features",
    "sha256_file",
]

VARIANTS = ("inv", "normed")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a checkpoint, recorded alongside every cache."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def _jit_load_on_cpu():
    """Let ``torch.load`` of a MACE ``.model`` succeed on a CPU-only node.

    e3nn's codegen mixin calls ``torch.jit.load(buffer)`` with no
    ``map_location`` while unpickling, which raises "No CUDA GPUs are
    available" on a login node even though the outer ``torch.load`` was given
    ``map_location="cpu"``.  Patching the default for the duration of the load
    is the least invasive fix and leaves ``mace-src`` alone.
    """
    import torch

    original = torch.jit.load
    torch.jit.load = lambda *a, **k: original(*a, **{**k, "map_location": "cpu"})
    try:
        yield
    finally:
        torch.jit.load = original


def load_mace_model(path: str | Path, device: str = "cpu"):
    """Load a pickled ``ScaleShiftMACE`` and put it in frozen eval mode.

    Also sets the global default dtype to the checkpoint's, which must happen
    *before* any ``AtomicData`` is built: ``AtomicData.from_config`` uses
    ``torch.get_default_dtype()`` for every tensor and the model does no
    casting, so a mismatch raises inside e3nn.
    """
    import torch
    from mace.tools import torch_tools

    if device == "cpu":
        with _jit_load_on_cpu():
            model = torch.load(str(path), map_location="cpu", weights_only=False)
    else:
        model = torch.load(str(path), map_location=device, weights_only=False)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
    torch_tools.set_default_dtype(str(dtype).removeprefix("torch."))
    LOGGER.info("Loaded %s (%s, dtype=%s)", Path(path).name, type(model).__name__, dtype)
    return model


@dataclass
class ExtractionConfig:
    """Everything that affects the numbers, recorded next to them."""

    checkpoint: str
    checkpoint_sha256: str
    dataset: str
    n_structures: int
    max_atoms: int
    batch_size: int
    seed: int
    device: str
    mace_version: str
    model_class: str
    r_max: float
    n_layers: int
    layer_irreps: list[str]
    readout_hidden_dim: int
    dtype: str
    atomic_numbers: list[int] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))


class MaceActivationExtractor:
    """Forward hooks on a frozen MACE model, yielding invariant per-atom features.

    Usage::

        ex = MaceActivationExtractor(model)
        with ex.hooks():
            model(batch.to_dict(), training=False, compute_force=False, ...)
        per_layer = ex.pop()   # {layer_name: (n_atoms, dim) float32 array}
    """

    def __init__(self, model):
        self.model = model
        self._captured: dict[str, list] = {}
        self._handles: list = []

        self.layer_irreps: list[IrrepsLayout] = []
        for product in model.products:
            self.layer_irreps.append(parse_irreps(product.linear.irreps_out))

        # The final readout is a NonLinearReadoutBlock in these checkpoints;
        # earlier ones are plain Linear and have no hidden layer to hook.
        self.readout_hidden_dim = 0
        self._readout_module = None
        last = model.readouts[-1]
        if hasattr(last, "non_linearity") and hasattr(last, "linear_1"):
            self._readout_module = last.non_linearity
            self.readout_hidden_dim = parse_irreps(last.linear_1.irreps_out).dim

    # -- hook plumbing ----------------------------------------------------

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self._captured.setdefault(name, []).append(
                tensor.detach().to("cpu", dtype=None).float().numpy()
            )

        return hook

    @contextlib.contextmanager
    def hooks(self):
        """Register hooks for the duration of a forward pass."""
        try:
            for i, product in enumerate(self.model.products):
                self._handles.append(
                    product.register_forward_hook(self._make_hook(f"layer{i}"))
                )
            if self._readout_module is not None:
                self._handles.append(
                    self._readout_module.register_forward_hook(
                        self._make_hook("readout_hidden")
                    )
                )
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()

    def pop(self) -> dict[str, np.ndarray]:
        """Concatenate and clear everything captured since the last call."""
        out = {k: np.concatenate(v, axis=0) for k, v in self._captured.items()}
        self._captured.clear()
        return out

    # -- reductions -------------------------------------------------------

    def feature_sets(self, captured: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Turn raw hook output into named invariant feature matrices.

        Keys are ``"layer{i}/{variant}"`` plus ``"readout_hidden"``.  A layer
        whose irreps are scalars only (MACE's last product block is ``128x0e``)
        has ``normed == inv``; it is still emitted under both names so the
        downstream grid stays rectangular, and the report notes the duplication
        rather than pretending they are independent measurements.
        """
        out: dict[str, np.ndarray] = {}
        for i, layout in enumerate(self.layer_irreps):
            raw = captured.get(f"layer{i}")
            if raw is None:
                continue
            for variant in VARIANTS:
                out[f"layer{i}/{variant}"] = reduce_features(raw, layout, variant)
        if "readout_hidden" in captured:
            out["readout_hidden"] = captured["readout_hidden"]
        return out

    def describe(self) -> dict:
        return {
            "n_layers": len(self.layer_irreps),
            "layer_irreps": [str(x) for x in self.layer_irreps],
            "layer_dims": {
                f"layer{i}/{v}": (
                    layout.n_invariant if v == "inv" else layout.n_normed
                )
                for i, layout in enumerate(self.layer_irreps)
                for v in VARIANTS
            },
            "readout_hidden_dim": self.readout_hidden_dim,
        }


def pool_features(
    per_atom: np.ndarray,
    ptr: np.ndarray,
    *,
    how: str = "mean",
) -> np.ndarray:
    """Pool per-atom features into per-structure features.

    ``ptr`` is the batch pointer from ``torch_geometric`` (length
    ``n_structures + 1``), so structure ``g`` owns rows ``ptr[g]:ptr[g+1]``.

    ``how="sum"`` reproduces MACE's own aggregation and pairs with a *total*
    energy target; ``how="mean"`` pairs with a *per-atom* energy target.  Use
    the pairing that matches the target, or the linear model no longer matches
    the network's functional form.
    """
    if how not in ("sum", "mean"):
        raise ValueError(f"unknown pooling {how!r}; expected 'sum' or 'mean'")

    # Promote BEFORE reducing, not after. numpy accumulates in the input dtype,
    # so summing a float16 block and assigning the result into a float64 array
    # keeps the float16 accumulation error -- over a 224-atom MOF that is a
    # ~0.5% error on the pooled features that carry the energy target.
    per_atom = np.asarray(per_atom, dtype=np.float64)

    ptr = np.asarray(ptr, dtype=np.int64)
    n_graphs = ptr.size - 1
    out = np.empty((n_graphs, per_atom.shape[1]), dtype=np.float64)
    for g in range(n_graphs):
        block = per_atom[ptr[g] : ptr[g + 1]]
        out[g] = block.sum(axis=0) if how == "sum" else block.mean(axis=0)
    return out


def cache_paths(root: Path, tag: str, name: str) -> dict[str, Path]:
    """Where a given (checkpoint, dataset, feature-set) cache lives."""
    base = Path(root) / tag
    safe = name.replace("/", "__")
    return {
        "dir": base,
        "features": base / f"{safe}.f16.npy",
        "index": base / "index.npz",
        "provenance": base / "provenance.json",
    }


def load_cached(root: Path, tag: str, name: str) -> tuple[np.ndarray, dict]:
    """Memory-map a cached feature matrix and load its index.

    Features come back as float16 memmaps.  Cast to float64 before any linear
    algebra -- LogME does this internally, but anything else touching them must
    too.
    """
    paths = cache_paths(root, tag, name)
    features = np.load(paths["features"], mmap_mode="r")
    index = dict(np.load(paths["index"], allow_pickle=False))
    return features, index
