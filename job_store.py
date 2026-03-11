"""File-backed job manifests for API polling and artifact lookup."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from settings import AppSettings


class JobStore:
    """Persist small JSON manifests for long-running queue jobs."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.root = settings.jobs_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def create_job(self, *, goal: str, max_agents: int) -> dict[str, Any]:
        job_id = str(uuid4())
        payload = {
            "job_id": job_id,
            "goal": goal,
            "max_agents": max_agents,
            "status": "queued",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "artifacts": {},
            "error": None,
            "progress": {"stage": "queued", "message": "Job accepted", "detail": None},
        }
        self._write(job_id, payload)
        return payload

    def mark_started(self, job_id: str) -> None:
        payload = self.read(job_id)
        payload["status"] = "running"
        payload["updated_at"] = _utc_now()
        payload["started_at"] = _utc_now()
        self._write(job_id, payload)

    def mark_progress(
        self,
        job_id: str,
        *,
        stage: str,
        message: str,
        detail: dict[str, Any] | None,
    ) -> None:
        payload = self.read(job_id)
        payload["status"] = "running"
        payload["updated_at"] = _utc_now()
        payload["progress"] = {"stage": stage, "message": message, "detail": detail}
        self._write(job_id, payload)

    def mark_success(self, job_id: str, *, artifacts: dict[str, Any]) -> None:
        payload = self.read(job_id)
        payload["status"] = "completed"
        payload["updated_at"] = _utc_now()
        payload["completed_at"] = _utc_now()
        payload["artifacts"] = artifacts
        payload["error"] = None
        payload["progress"] = {
            "stage": "completed",
            "message": "Artifacts ready",
            "detail": {"dataset_name": artifacts.get("dataset_name")},
        }
        self._write(job_id, payload)

    def mark_failure(self, job_id: str, *, error: str) -> None:
        payload = self.read(job_id)
        payload["status"] = "failed"
        payload["updated_at"] = _utc_now()
        payload["error"] = error
        payload["progress"] = {
            "stage": "failed",
            "message": error,
            "detail": None,
        }
        self._write(job_id, payload)

    def read(self, job_id: str) -> dict[str, Any]:
        path = self._job_path(job_id)
        if not path.exists():
            raise FileNotFoundError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def job_dir(self, job_id: str) -> Path:
        path = self.root / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifact_path(self, job_id: str, artifact_name: str) -> Path:
        payload = self.read(job_id)
        artifact_value = payload.get("artifacts", {}).get(artifact_name)
        if not artifact_value:
            raise FileNotFoundError(f"{job_id}:{artifact_name}")
        return Path(artifact_value)

    def _write(self, job_id: str, payload: dict[str, Any]) -> None:
        job_dir = self.job_dir(job_id)
        temp_path = job_dir / "status.tmp"
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temp_path.replace(self._job_path(job_id))

    def _job_path(self, job_id: str) -> Path:
        return self.root / job_id / "status.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
