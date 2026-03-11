"""Heuristic extraction for visible list, table, and directory pages."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError


LINE_REF_RE = re.compile(r"@?(e\d+)\b", flags=re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", flags=re.IGNORECASE)
UPPER_TOKEN_RE = re.compile(r"[A-Z][A-Za-z&.'-]+")

BLOCKED_LINE_PHRASES = (
    "jump to content",
    "main menu",
    "search wikipedia",
    "create account",
    "log in",
    "donate",
    "help",
    "navigation",
    "button",
    "searchbox",
    "combobox",
    "radio",
    "appearance",
    "page tools",
    "personal tools",
    "site",
    "views",
    "tools",
    "automatic",
    "dark",
    "light",
    "enabled",
    "disabled",
    "thumbnail for",
    "request that a redirect be created",
    "create a draft and submit it for review",
)

BLOCKED_PAGE_PHRASES = (
    "search results",
    "search wikipedia",
    "create account",
    "request that a redirect be created",
    "results for",
)

POSITIVE_PAGE_PHRASES = (
    "list of",
    "teams",
    "schools",
    "companies",
    "directory",
    "standings",
    "rankings",
    "roster",
)


@dataclass(slots=True)
class ListPageExtractor:
    """Extract simple flat rows from dense list or directory pages."""

    row_model: type[BaseModel]

    def should_extract(self, snapshot: str) -> bool:
        """Return True when the snapshot looks like a useful directory page."""
        lowered = snapshot.lower()
        if any(phrase in lowered for phrase in BLOCKED_PAGE_PHRASES):
            return False

        candidate_lines = self._candidate_lines(snapshot)
        if len(candidate_lines) < 5:
            return False

        if any(phrase in lowered for phrase in POSITIVE_PAGE_PHRASES):
            return True

        return len(candidate_lines) >= 12

    def candidate_count(self, snapshot: str) -> int:
        """Return the number of visible row-like lines in the snapshot."""
        return len(self._candidate_lines(snapshot))

    def extract(self, snapshot: str, *, source_url: str) -> list[BaseModel]:
        """Convert repeated visible entity-like lines into row candidates."""
        records: list[BaseModel] = []
        seen: set[str] = set()
        for line in self._candidate_lines(snapshot):
            payload = self._line_to_payload(line, source_url=source_url)
            if not payload:
                continue
            fingerprint = str(sorted(payload.items()))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            try:
                records.append(self.row_model.model_validate(payload))
            except ValidationError:
                continue
        return records

    def _candidate_lines(self, snapshot: str) -> list[str]:
        lines: list[str] = []
        for raw_line in snapshot.splitlines():
            line = raw_line.strip()
            if not self._line_is_candidate(line):
                continue
            lines.append(line)
        return lines

    def _line_is_candidate(self, line: str) -> bool:
        if not line:
            return False
        if len(line) < 8 or len(line) > 220:
            return False
        if not LINE_REF_RE.search(line):
            return False

        lowered = line.lower()
        if any(phrase in lowered for phrase in BLOCKED_LINE_PHRASES):
            return False
        if lowered.startswith(("button", "menu", "sign in", "subscribe", "navigation", "searchbox")):
            return False
        if lowered.count('"') < 2:
            return False

        text = self._clean_line_text(line)
        if len(text) < 5:
            return False
        if text.count(" ") > 10:
            return False
        if len(UPPER_TOKEN_RE.findall(text)) == 0:
            return False
        return True

    def _line_to_payload(self, line: str, *, source_url: str) -> dict[str, object]:
        text = self._clean_line_text(line)
        if not text or not self._looks_like_entity_name(text):
            return {}

        payload: dict[str, object] = {}
        primary_name = self._primary_name_field()
        if primary_name:
            payload[primary_name] = text

        if "school" in self.row_model.model_fields and "school" not in payload:
            payload["school"] = self._derive_school_name(text)
        if "category" in self.row_model.model_fields:
            payload["category"] = self._infer_category(source_url)
        if "sport" in self.row_model.model_fields:
            payload["sport"] = self._infer_sport(source_url)
        if "team_profile_url" in self.row_model.model_fields:
            payload["team_profile_url"] = self._extract_url(line)
        if "reference_url" in self.row_model.model_fields:
            payload["reference_url"] = self._extract_url(line)
        if "source_url" in self.row_model.model_fields:
            payload["source_url"] = source_url
        for optional in ("conference", "division", "head_coach", "location"):
            if optional in self.row_model.model_fields:
                payload.setdefault(optional, None)

        return payload

    def _clean_line_text(self, line: str) -> str:
        line = LINE_REF_RE.sub("", line)
        line = URL_RE.sub("", line)
        line = re.sub(r"\[[^\]]*\]", "", line)
        line = re.sub(r'^(?:link|heading|listitem|row|cell)\s+', "", line, flags=re.IGNORECASE)
        line = re.sub(r'\s+', " ", line).strip(" -|:\t\"'")
        return line

    def _looks_like_entity_name(self, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in BLOCKED_LINE_PHRASES):
            return False
        if "wikipedia" in lowered or "search results" in lowered:
            return False
        tokens = text.split()
        if not (1 <= len(tokens) <= 8):
            return False
        uppercase_tokens = UPPER_TOKEN_RE.findall(text)
        return len(uppercase_tokens) >= 1

    def _primary_name_field(self) -> str | None:
        for candidate in (
            "team_name",
            "entity_name",
            "name",
            "school",
            "company_name",
            "organization_name",
            "title",
        ):
            if candidate in self.row_model.model_fields:
                return candidate
        return None

    def _derive_school_name(self, text: str) -> str:
        tokens = text.split()
        if len(tokens) <= 2:
            return text
        multiword_mascot_suffixes = {
            ("blue", "devils"),
            ("tar", "heels"),
            ("wolf", "pack"),
            ("golden", "eagles"),
            ("crimson", "tide"),
            ("fighting", "irish"),
            ("red", "raiders"),
            ("hill", "toppers"),
            ("scarlet", "knights"),
        }
        lowered_tokens = tuple(token.lower() for token in tokens)
        for suffix in multiword_mascot_suffixes:
            if lowered_tokens[-len(suffix):] == suffix and len(tokens) > len(suffix):
                return " ".join(tokens[:-len(suffix)])
        mascot_suffixes = {
            "tigers", "eagles", "bulldogs", "wildcats", "hawks", "bears", "lions",
            "bruins", "spartans", "crimson", "gators", "heels", "devils", "seminoles",
            "cardinals", "owls", "wolves", "pirates", "terriers", "chargers",
        }
        if tokens[-1].lower() in mascot_suffixes:
            return " ".join(tokens[:-1])
        return text

    def _extract_url(self, line: str) -> str | None:
        match = URL_RE.search(line)
        if match:
            return match.group(0)
        return None

    def _infer_category(self, source_url: str) -> str | None:
        if "wikipedia" in source_url:
            return "list_entry"
        return "directory_entry"

    def _infer_sport(self, source_url: str) -> str | None:
        lowered = source_url.lower()
        if "basketball" in lowered:
            return "basketball"
        if "football" in lowered:
            return "football"
        if "soccer" in lowered:
            return "soccer"
        return None
