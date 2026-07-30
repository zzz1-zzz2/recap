"""Checkpoint loader — load frozen C₁ from disk."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("recap.checkpoint")


class Checkpoint:
    """Frozen first-failure checkpoint C₁ = (x₁, p₁, τ₁, w₁)."""

    def __init__(self, checkpoint_dir: str | Path):
        self.dir = Path(checkpoint_dir)
        self.r1_dir = self.dir / "round1"
        self._validate()

    def _validate(self) -> None:
        required = [
            self.r1_dir / "trajectory.json",
            self.r1_dir / "failure_witness.json",
            self.r1_dir / "workspace.patch",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Checkpoint missing: {missing}")

    @property
    def trajectory(self) -> list[dict]:
        data = json.loads((self.r1_dir / "trajectory.json").read_text())
        if isinstance(data, list):
            return data
        return data.get("messages", data.get("trajectory", []))

    @property
    def failure_witness(self) -> dict:
        return json.loads((self.r1_dir / "failure_witness.json").read_text())

    @property
    def workspace_patch(self) -> str:
        return (self.r1_dir / "workspace.patch").read_text()

    @property
    def workspace_snapshot(self) -> dict:
        p = self.r1_dir / "workspace_snapshot.json"
        if p.exists():
            return json.loads(p.read_text())
        return {}
