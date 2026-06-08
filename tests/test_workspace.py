from __future__ import annotations

from pathlib import Path


def test_workspace_scaffold_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_paths = [
        root / "Fine-tuning Procedures.md",
        root / "configs" / "uma",
        root / "configs" / "upet",
        root / "configs" / "mace",
        root / "data" / "raw",
        root / "data" / "processed",
        root / "models" / "pretrained",
        root / "models" / "checkpoints",
        root / "runs" / "uma",
        root / "runs" / "upet",
        root / "runs" / "mace",
        root / "logs" / "slurm",
        root / "scripts",
    ]

    missing = [path for path in expected_paths if not path.exists()]
    assert missing == []
