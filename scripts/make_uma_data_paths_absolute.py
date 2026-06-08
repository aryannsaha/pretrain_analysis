#!/usr/bin/env python
"""Set UMA data-task train/validation src fields to absolute paths."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _set_nested(config: dict[str, Any], keys: list[str], value: str) -> None:
    cursor: dict[str, Any] = config
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text()) or {}
    _set_nested(
        config,
        ["train_dataset", "splits", "train", "src"],
        str(args.train.resolve()),
    )
    _set_nested(
        config,
        ["val_dataset", "splits", "val", "src"],
        str(args.val.resolve()),
    )
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
