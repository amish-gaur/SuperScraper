"""Persist browser agent step artifacts for debugging and observability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StepArtifactLogger:
    """Write per-step browser artifacts to disk."""

    def __init__(self, *, root_dir: str | Path, run_id: str) -> None:
        self.root_dir = Path(root_dir) / run_id
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def log_step(
        self,
        *,
        step: int,
        snapshot: str,
        page_state: dict[str, Any],
        decision: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        step_dir = self.root_dir / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "snapshot.txt").write_text(snapshot, encoding="utf-8")
        (step_dir / "page_state.json").write_text(
            json.dumps(page_state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if decision is not None:
            (step_dir / "decision.json").write_text(
                json.dumps(decision, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if metadata is not None:
            (step_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
