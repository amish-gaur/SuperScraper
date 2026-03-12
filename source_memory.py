"""Persistent memory of successful source patterns for related goals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from goal_intent import decompose_goal


DEFAULT_SOURCE_MEMORY_PATH = Path(__file__).parent / "storage" / "source_memory.json"


@dataclass(frozen=True, slots=True)
class SourceMemoryEntry:
    """Compact persisted view of a successful source set."""

    goal: str
    domain_intent: str
    entity_intent: str
    target_hint: str | None
    feature_hints: tuple[str, ...]
    urls: tuple[str, ...]


class SourceMemory:
    """Persist and retrieve successful sources for similar future goals."""

    def __init__(self, path: str | Path = DEFAULT_SOURCE_MEMORY_PATH) -> None:
        self.path = Path(path)

    def record_success(self, goal: str, urls: list[str]) -> None:
        cleaned_urls = tuple(dict.fromkeys(url.strip() for url in urls if url.strip()))
        if not cleaned_urls:
            return
        decomposition = decompose_goal(goal)
        entry = SourceMemoryEntry(
            goal=goal.strip(),
            domain_intent=decomposition.domain_intent,
            entity_intent=decomposition.entity_intent,
            target_hint=decomposition.target_hint,
            feature_hints=decomposition.feature_hints,
            urls=cleaned_urls,
        )
        entries = self._load_entries()

        deduped: list[SourceMemoryEntry] = []
        replaced = False
        for existing in entries:
            if existing.goal == entry.goal:
                deduped.append(entry)
                replaced = True
            else:
                deduped.append(existing)
        if not replaced:
            deduped.append(entry)
        self._save_entries(deduped[-50:])

    def similar_urls(self, goal: str, *, limit: int = 6) -> list[str]:
        decomposition = decompose_goal(goal)
        scored: list[tuple[int, SourceMemoryEntry]] = []
        for entry in self._load_entries():
            score = 0
            if entry.domain_intent == decomposition.domain_intent:
                score += 5
            if entry.entity_intent == decomposition.entity_intent:
                score += 4
            if entry.target_hint and entry.target_hint == decomposition.target_hint:
                score += 5
            score += len(set(entry.feature_hints).intersection(decomposition.feature_hints)) * 2
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].goal))

        urls: list[str] = []
        for _, entry in scored:
            for url in entry.urls:
                if url not in urls:
                    urls.append(url)
                if len(urls) >= limit:
                    return urls
        return urls

    def _load_entries(self) -> list[SourceMemoryEntry]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []

        entries: list[SourceMemoryEntry] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            urls = item.get("urls")
            if not isinstance(urls, list):
                continue
            entries.append(
                SourceMemoryEntry(
                    goal=str(item.get("goal") or "").strip(),
                    domain_intent=str(item.get("domain_intent") or "generic"),
                    entity_intent=str(item.get("entity_intent") or "entity"),
                    target_hint=str(item["target_hint"]) if item.get("target_hint") is not None else None,
                    feature_hints=tuple(str(value) for value in item.get("feature_hints") or [] if value),
                    urls=tuple(str(url).strip() for url in urls if str(url).strip()),
                )
            )
        return [entry for entry in entries if entry.goal and entry.urls]

    def _save_entries(self, entries: list[SourceMemoryEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: list[dict[str, Any]] = []
        for entry in entries:
            row = asdict(entry)
            row["feature_hints"] = list(entry.feature_hints)
            row["urls"] = list(entry.urls)
            payload.append(row)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))
