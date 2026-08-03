#!/usr/bin/env python
"""Rebuild the ``.model`` artefacts of a MACE run from a training checkpoint.

``mace_run_train`` only writes ``<checkpoints_dir>/<tag>.model`` and
``<model_dir>/<name>{,_compiled}.model`` after the training loop returns.  A run
that dies on the SLURM wall clock therefore leaves a perfectly good
``<tag>_epoch-<N>.pt`` checkpoint behind but no ``.model`` file, which is what
downstream fine-tuning scripts consume as ``foundation_model``.

Checkpoints only hold ``state_dict``s, so the model *object* has to come from
somewhere.  This script takes it from a reference ``.model`` of a sibling run
that was trained with the same architecture (same ``statistics_file``, same
hyperparameters), loads the checkpoint weights into it with ``strict=True`` and
re-saves it exactly the way ``run_train.py`` does.

Everything that distinguishes two runs of the same architecture -- weights,
``atomic_energies``, the ``scale``/``shift`` of the ``ScaleShiftBlock``,
``r_max``, ``atomic_numbers``, ``num_interactions`` -- lives in the state dict
and is overwritten by the load.  What does *not* live in the state dict is
``avg_num_neighbors`` (a plain float on each interaction block) and ``heads``,
so those are checked explicitly against the run's config before saving.

Example
-------
    python scripts/mace/checkpoint_to_model.py \
        --run-dir runs/mace/omat24/stratified_random_nested/omat_100k \
        --reference-model runs/mace/omat24/stratified_random_nested/omat_500k/models/omat_500k.model
"""

import argparse
import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path

# e3nn unpickles its Wigner constants at import time, which torch>=2.6 refuses
# under the default weights_only=True.  mace/__init__.py sets this too, but only
# once mace is imported -- and e3nn gets there first in a standalone script.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch  # noqa: E402
import yaml  # noqa: E402
from e3nn import o3  # noqa: E402
from e3nn.util import jit  # noqa: E402

from mace.tools.scripts_utils import (  # noqa: E402
    convert_to_json_format,
    extract_config_mace_model,
    print_git_commit,
)

EPOCH_RE = re.compile(r"^(?P<tag>.+)_epoch-(?P<epochs>\d+)(?P<swa>_swa)?\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="MACE run directory (holds checkpoints/, the run yaml, and models/)",
    )
    parser.add_argument(
        "--reference-model",
        type=Path,
        required=True,
        help="A .model from a run with an identical architecture, used as the "
        "model object into which the checkpoint weights are loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to convert (default: highest-epoch .pt in <run-dir>/checkpoints)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run yaml (default: the single *.yaml in <run-dir>)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device the saved model lives on; run_train saves on the training "
        "device unless --save_cpu was passed",
    )
    parser.add_argument(
        "--skip-compiled",
        action="store_true",
        help="Do not write the TorchScript-compiled <name>_compiled.model",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite .model files that already exist",
    )
    return parser.parse_args()


def find_checkpoint(checkpoints_dir: Path) -> Path:
    """Return the highest-epoch checkpoint, mirroring CheckpointIO's naming."""
    candidates = []
    for path in sorted(checkpoints_dir.glob("*.pt")):
        match = EPOCH_RE.match(path.name)
        if match:
            candidates.append((int(match.group("epochs")), path))
    if not candidates:
        raise SystemExit(f"No <tag>_epoch-<N>.pt checkpoints found in {checkpoints_dir}")
    return max(candidates)[1]


def find_config(run_dir: Path) -> Path:
    configs = sorted(run_dir.glob("*.yaml")) + sorted(run_dir.glob("*.yml"))
    if len(configs) != 1:
        raise SystemExit(
            f"Expected exactly one yaml in {run_dir}, found {len(configs)}; pass --config"
        )
    return configs[0]


def avg_num_neighbors(model: torch.nn.Module):
    values = {
        float(interaction.avg_num_neighbors)
        for interaction in model.interactions
        if hasattr(interaction, "avg_num_neighbors")
    }
    return values


def check_architecture(model: torch.nn.Module, config: dict) -> None:
    """Fail loudly if the reference model does not match the run's config.

    ``load_state_dict(strict=True)`` already catches every parameter- and
    buffer-shaped difference.  This covers the handful of architectural knobs
    that are stored as plain Python attributes instead.
    """
    extracted = extract_config_mace_model(model)
    if "error" in extracted:
        raise SystemExit(f"Reference model is not a ScaleShiftMACE: {extracted['error']}")

    problems = []

    def compare(name, expected, actual):
        if expected is None:
            return
        if expected != actual:
            problems.append(f"  {name}: config={expected!r} reference model={actual!r}")

    compare("r_max", float(config["r_max"]), float(extracted["r_max"]))
    compare("max_ell", int(config["max_ell"]), int(extracted["max_ell"]))
    compare(
        "num_interactions",
        int(config["num_interactions"]),
        int(extracted["num_interactions"]),
    )
    compare("correlation", int(config["correlation"]), int(extracted["correlation"]))
    compare(
        "hidden_irreps",
        o3.Irreps(config["hidden_irreps"]),
        o3.Irreps(str(extracted["hidden_irreps"])),
    )
    compare(
        "interaction",
        config.get("interaction"),
        extracted["interaction_cls"].__name__,
    )
    compare(
        "interaction_first",
        config.get("interaction_first"),
        extracted["interaction_cls_first"].__name__,
    )
    if "atomic_numbers" in config:
        expected_numbers = config["atomic_numbers"]
        if isinstance(expected_numbers, str):
            expected_numbers = json.loads(expected_numbers)
        compare("num_elements", len(expected_numbers), int(extracted["num_elements"]))
    compare(
        "distance_transform",
        config.get("distance_transform"),
        extracted.get("distance_transform"),
    )

    # avg_num_neighbors is a float attribute on the interaction blocks, not a
    # buffer, so a mismatched reference model would silently rescale messages.
    statistics_file = config.get("statistics_file")
    neighbors = avg_num_neighbors(model)
    if len(neighbors) != 1:
        problems.append(f"  avg_num_neighbors differs between blocks: {neighbors}")
    elif statistics_file is not None:
        with open(statistics_file, "r", encoding="utf-8") as handle:
            statistics = json.load(handle)
        expected = float(statistics["avg_num_neighbors"])
        actual = next(iter(neighbors))
        if abs(expected - actual) > 1e-9:
            problems.append(
                f"  avg_num_neighbors: statistics_file={expected!r} "
                f"reference model={actual!r}"
            )

    if problems:
        raise SystemExit(
            "Reference model architecture does not match the run config:\n"
            + "\n".join(problems)
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    run_dir = args.run_dir.resolve()
    checkpoints_dir = run_dir / "checkpoints"
    model_dir = run_dir / "models"

    checkpoint_path = (
        args.checkpoint.resolve() if args.checkpoint else find_checkpoint(checkpoints_dir)
    )
    config_path = args.config.resolve() if args.config else find_config(run_dir)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    match = EPOCH_RE.match(checkpoint_path.name)
    if not match:
        raise SystemExit(f"Cannot parse tag/epoch out of {checkpoint_path.name}")
    tag = match.group("tag")
    epoch = int(match.group("epochs"))
    swa = bool(match.group("swa"))
    name = config["name"]
    suffix = "_stagetwo" if swa else ""

    logging.info("Run directory: %s", run_dir)
    logging.info("Config: %s", config_path)
    logging.info("Checkpoint: %s (tag=%s, epoch=%d, swa=%s)", checkpoint_path, tag, epoch, swa)
    logging.info("Reference model: %s", args.reference_model)

    device = torch.device(args.device)
    torch.set_default_dtype(
        torch.float64 if config.get("default_dtype", "float64") == "float64" else torch.float32
    )

    model = torch.load(args.reference_model, map_location=device, weights_only=False)
    check_architecture(model, config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    # strict=True is the real architecture check: every parameter and buffer of
    # the reference object must be present in the checkpoint with a matching
    # shape, and vice versa.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # load_state_dict casts into the destination dtype, so verify the weights
    # really are bit-identical to what the checkpoint held.
    loaded = model.state_dict()
    mismatched = []
    for key, expected in state_dict.items():
        actual = loaded[key]
        if actual.dtype != expected.dtype:
            mismatched.append(f"  {key}: dtype {expected.dtype} -> {actual.dtype}")
        elif not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
            mismatched.append(f"  {key}: values differ after load")
    if mismatched:
        raise SystemExit("Checkpoint weights were not loaded faithfully:\n" + "\n".join(mismatched))
    logging.info("Verified %d tensors match the checkpoint exactly", len(state_dict))

    outputs = [checkpoints_dir / f"{tag}{suffix}.model", model_dir / f"{name}{suffix}.model"]
    if not args.skip_compiled:
        outputs.append(model_dir / f"{name}{suffix}_compiled.model")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite existing files (pass --overwrite):\n"
            + "\n".join(f"  {path}" for path in existing)
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    commit = print_git_commit()
    extra_files = {
        "commit.txt": commit.encode("utf-8") if commit is not None else b"",
        "config.yaml": json.dumps(convert_to_json_format(extract_config_mace_model(model))),
    }

    for path in (checkpoints_dir / f"{tag}{suffix}.model", model_dir / f"{name}{suffix}.model"):
        logging.info("Saving model to %s", path)
        torch.save(model, path)

    if not args.skip_compiled:
        path_compiled = model_dir / f"{name}{suffix}_compiled.model"
        logging.info("Compiling model, saving metadata to %s", path_compiled)
        model_compiled = jit.compile(deepcopy(model))
        torch.jit.save(model_compiled, path_compiled, _extra_files=extra_files)

    logging.info("Done: %s is the epoch-%d checkpoint of run %s", f"{name}{suffix}.model", epoch, name)


if __name__ == "__main__":
    main()
