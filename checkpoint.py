"""Checkpoint persistence for resumable swarm runs."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, ValidationError

try:
    import fcntl
except ImportError as exc:  # pragma: no cover - this pipeline targets POSIX environments.
    raise RuntimeError("checkpoint.py requires POSIX file locking support") from exc


LOGGER = logging.getLogger(__name__)


class CheckpointManager:
    """Persist raw extracted rows with file-lock protection."""

    def __init__(self, cache_path: str | Path = "pipeline_cache.json") -> None:
        self.cache_path = Path(cache_path)
        self.lock_path = self.cache_path.with_suffix(f"{self.cache_path.suffix}.lock")

    def exists(self) -> bool:
        """Return True when a cache file is present on disk."""
        return self.cache_path.exists()

    def load_records(
        self,
        *,
        goal: str,
        dataset_name: str,
        row_model: type[BaseModel],
    ) -> list[BaseModel]:
        """Load cached records when the cache metadata matches the active run."""
        if not self.cache_path.exists():
            return []

        with self._locked(lock_type=fcntl.LOCK_SH):
            payload = self._read_payload()

        if not payload:
            return []
        if not self._payload_matches(payload, goal=goal, dataset_name=dataset_name, row_model=row_model):
            LOGGER.info("Ignoring cache at %s because it does not match this run", self.cache_path)
            return []

        records: list[BaseModel] = []
        for index, record_payload in enumerate(payload.get("records", []), start=1):
            try:
                records.append(row_model.model_validate(record_payload))
            except ValidationError as exc:
                LOGGER.warning("Skipping invalid checkpoint row %d: %s", index, exc)

        LOGGER.info("Loaded %d cached raw records from %s", len(records), self.cache_path)
        return records

    def append_record(
        self,
        *,
        goal: str,
        dataset_name: str,
        row_model: type[BaseModel],
        record: BaseModel,
    ) -> None:
        """Append one validated record to the checkpoint, deduplicating by fingerprint."""
        serialized = record.model_dump(mode="json")

        with self._locked(lock_type=fcntl.LOCK_EX):
            payload = self._read_payload()
            if not self._payload_matches(payload, goal=goal, dataset_name=dataset_name, row_model=row_model):
                payload = self._build_payload(goal=goal, dataset_name=dataset_name, row_model=row_model)

            records = payload.setdefault("records", [])
            fingerprint = self._fingerprint(serialized)
            fingerprints = {self._fingerprint(existing) for existing in records if isinstance(existing, dict)}
            if fingerprint in fingerprints:
                return

            records.append(serialized)
            self._write_payload(payload)

    def _build_payload(
        self,
        *,
        goal: str,
        dataset_name: str,
        row_model: type[BaseModel],
    ) -> dict[str, Any]:
        return {
            "goal": goal,
            "dataset_name": dataset_name,
            "row_schema": row_model.model_json_schema(),
            "records": [],
        }

    def _payload_matches(
        self,
        payload: dict[str, Any] | None,
        *,
        goal: str,
        dataset_name: str,
        row_model: type[BaseModel],
    ) -> bool:
        if not payload:
            return False
        return (
            payload.get("goal") == goal
            and payload.get("dataset_name") == dataset_name
            and payload.get("row_schema") == row_model.model_json_schema()
        )

    def _read_payload(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}

        try:
            raw = self.cache_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            LOGGER.warning("Failed to read checkpoint %s: %s", self.cache_path, exc)
            return {}

        if not raw:
            return {}

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Checkpoint file %s is not valid JSON: %s", self.cache_path, exc)
            return {}

        if not isinstance(payload, dict):
            LOGGER.warning("Checkpoint file %s has invalid top-level structure", self.cache_path)
            return {}
        return payload

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        temp_path.write_text(serialized, encoding="utf-8")
        os.replace(temp_path, self.cache_path)

    def _fingerprint(self, record_payload: dict[str, Any]) -> str:
        return json.dumps(record_payload, sort_keys=True, separators=(",", ":"))

    @contextmanager
    def _locked(self, *, lock_type: int) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), lock_type)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
