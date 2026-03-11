"""Entity normalization and fuzzy-resolution helpers for dataset assembly."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re

import pandas as pd


COMMON_ENTITY_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "ltd",
    "llc",
    "plc",
    "group",
    "holdings",
}


@dataclass(slots=True)
class EntityResolver:
    """Normalize entity labels and collapse near-duplicate keys."""

    similarity_threshold: float = 0.94

    def canonical_key(self, value: str | object) -> str:
        if value is None or pd.isna(value):
            return ""
        value = str(value).strip()
        if not value or value.lower() in {"nan", "none"}:
            return ""
        compact_ticker_pattern = bool(
            re.search(r"[a-z][A-Z]{2,}$", value)
            or re.search(r"\d{3,5}\.[A-Z]{1,3}$", value)
            or re.search(r"[A-Za-z]+\d{2,5}$", value)
        )
        value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
        value = re.sub(r"([A-Za-z])(\d)", r"\1 \2", value)
        value = re.sub(r"(\d)([A-Za-z])", r"\1 \2", value)
        lowered = value.lower()
        lowered = re.sub(r"\[[^\]]+\]", "", lowered)
        lowered = lowered.replace("&", " and ")
        lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
        tokens = [token for token in lowered.split() if token and token not in COMMON_ENTITY_SUFFIXES]
        if len(tokens) >= 2 and tokens[-1].isalpha() and len(tokens[-1]) <= 2:
            tokens = tokens[:-1]
        if compact_ticker_pattern:
            if len(tokens) >= 2 and tokens[-1].isalpha() and len(tokens[-1]) <= 5:
                tokens = tokens[:-1]
            if len(tokens) >= 2 and tokens[-1].isdigit() and len(tokens[-1]) <= 5:
                tokens = tokens[:-1]
            if len(tokens) >= 2 and tokens[-1].isalpha() and len(tokens[-1]) <= 3:
                tokens = tokens[:-1]
        return " ".join(tokens)

    def resolve_frame(self, frame: pd.DataFrame, *, entity_column: str) -> pd.DataFrame:
        """Rewrite an entity column so near duplicates collapse to one canonical key."""
        resolved = frame.copy()
        canonical_map: dict[str, str] = {}
        seen_keys: list[str] = []

        for raw_value in resolved[entity_column].astype(str):
            key = self.canonical_key(raw_value)
            matched = self._best_match(key, seen_keys)
            canonical = matched or key
            canonical_map[raw_value] = canonical
            if not matched:
                seen_keys.append(canonical)

        resolved[entity_column] = resolved[entity_column].astype(str).map(canonical_map)
        return resolved

    def _best_match(self, key: str, candidates: list[str]) -> str | None:
        for candidate in candidates:
            if key == candidate:
                return candidate
            if SequenceMatcher(a=key, b=candidate).ratio() >= self.similarity_threshold:
                return candidate
        return None
