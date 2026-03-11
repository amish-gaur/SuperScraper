"""Deterministic HTML table extraction for public list and directory pages."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import logging
import re
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from pydantic import BaseModel, ValidationError
import urllib3

from source_health import FailureReason, fetch_url, REGISTRY


LOGGER = logging.getLogger(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass(slots=True)
class HtmlTableExtractor:
    """Extract records from HTML tables without requiring browser automation."""

    row_model: type[BaseModel]
    timeout_seconds: float = 20.0
    domain_blacklist: set[str] | None = None

    def extract(self, url: str) -> list[BaseModel]:
        """Fetch a page and parse any tables that map cleanly to the row schema."""
        outcome = self._fetch_html(url)
        if not outcome.ok or not outcome.text:
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=outcome.reason)
            return []
        return self.extract_from_html(url, outcome.text)

    def extract_from_html(self, url: str, html_text: str) -> list[BaseModel]:
        """Parse pre-fetched HTML tables and map them into the row schema."""
        if not html_text.strip():
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.EMPTY_CONTENT)
            return []

        try:
            tables = pd.read_html(StringIO(html_text))
        except ValueError:
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.NO_TABLES)
            return []
        except Exception as exc:
            LOGGER.warning("Failed to parse HTML tables from %s: %s", url, exc)
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.NO_TABLES)
            return []

        records: list[BaseModel] = []
        seen: set[str] = set()
        for table in tables:
            flattened = self._flatten_table(table)
            mapping = self._map_columns(list(flattened.columns))
            if not mapping:
                continue

            for row in flattened.to_dict(orient="records"):
                payload = self._row_to_payload(row, mapping=mapping, source_url=url)
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

        if records:
            REGISTRY.record_extraction(url, records_extracted=len(records), success=True, reason=FailureReason.SUCCESS)
        else:
            REGISTRY.record_extraction(url, records_extracted=0, success=False, reason=FailureReason.SCHEMA_MISMATCH)
        return records

    def _fetch_html(self, url: str):
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; WebScraperPrototype/1.0; "
                "+https://example.com/web-scraper)"
            )
        }
        outcome = fetch_url(
            url,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
            verify=False,
        )
        if outcome.reason in {FailureReason.HTTP_403, FailureReason.HTTP_429, FailureReason.ANTI_BOT}:
            self._blacklist_domain(url)
        if not outcome.ok:
            LOGGER.warning(
                "Failed to fetch %s for deterministic table extraction: %s",
                url,
                outcome.detail or outcome.reason.value,
            )
        return outcome

    def _blacklist_domain(self, url: str) -> None:
        if self.domain_blacklist is None:
            return
        domain = _root_domain(url)
        if domain:
            self.domain_blacklist.add(domain)

    def _flatten_table(self, table: pd.DataFrame) -> pd.DataFrame:
        frame = table.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                " ".join(str(part) for part in column if str(part) not in {"nan", ""}).strip()
                for column in frame.columns
            ]
        frame.columns = [str(column).strip() for column in frame.columns]
        frame = self._promote_header_row(frame)
        frame = self._dedupe_columns(frame)
        return frame

    def _promote_header_row(self, frame: pd.DataFrame) -> pd.DataFrame:
        current_columns = [str(column).strip() for column in frame.columns]
        if not all(
            column.isdigit() or column.lower().startswith("unnamed")
            for column in current_columns
        ):
            return frame
        if frame.empty:
            return frame

        header_candidates = [str(value).strip() for value in frame.iloc[0].tolist()]
        informative = [
            value for value in header_candidates if value and value.lower() not in {"nan", "none"}
        ]
        if len(informative) < max(2, len(header_candidates) // 2):
            return frame
        if not any(
            any(token in value.lower() for token in {"name", "team", "school", "player", "company", "conference"})
            for value in informative
        ):
            return frame

        promoted = frame.iloc[1:].reset_index(drop=True).copy()
        promoted.columns = header_candidates
        return promoted

    def _dedupe_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized_counts: dict[str, int] = {}
        deduped_columns: list[str] = []
        for column in frame.columns:
            base = str(column).strip() or "column"
            count = normalized_counts.get(base, 0)
            deduped_columns.append(base if count == 0 else f"{base}_{count + 1}")
            normalized_counts[base] = count + 1
        frame.columns = deduped_columns
        return frame

    def _map_columns(self, columns: list[str]) -> dict[str, str]:
        model_fields = set(self.row_model.model_fields.keys())
        normalized_columns = {
            column: _normalize_label(column)
            for column in columns
        }
        mapping: dict[str, str] = {}
        for column in columns:
            normalized = normalized_columns[column]
            if normalized in {"school", "college", "university"} and "school" in model_fields:
                mapping["school"] = column
            elif normalized in {"nickname", "teamname", "mascot"}:
                if "nickname" in model_fields:
                    mapping["nickname"] = column
                elif "team_name" in model_fields:
                    mapping["team_name"] = column
                elif "name" in model_fields:
                    mapping["name"] = column
            elif normalized in {"conference", "league"} and "conference" in model_fields:
                mapping["conference"] = column
            elif normalized in {"homearena", "arena"} and "home_arena" in model_fields:
                mapping["home_arena"] = column
            elif normalized in {"homestadium", "stadium"} and "stadium" in model_fields:
                mapping["stadium"] = column
            elif normalized in {"tournamentappearances", "appearances"} and "tournament_appearances" in model_fields:
                mapping["tournament_appearances"] = column
            elif normalized in {"finalfourappearances", "finalfour"} and "final_four_appearances" in model_fields:
                mapping["final_four_appearances"] = column
            elif normalized in {"championshipwins", "championships"} and "championship_wins" in model_fields:
                mapping["championship_wins"] = column
            elif normalized in {"team", "name", "program"}:
                for candidate in ("team_name", "name", "entity_name", "organization_name"):
                    if candidate in model_fields:
                        mapping.setdefault(candidate, column)
                        break

        for field_name in model_fields:
            if field_name in mapping:
                continue
            field_normalized = _normalize_label(field_name)
            matched_column = self._match_generic_column(
                field_name=field_name,
                field_normalized=field_normalized,
                normalized_columns=normalized_columns,
            )
            if matched_column is not None:
                mapping[field_name] = matched_column
        return mapping

    def _match_generic_column(
        self,
        *,
        field_name: str,
        field_normalized: str,
        normalized_columns: dict[str, str],
    ) -> str | None:
        best_column: str | None = None
        best_score = 0
        for column, normalized in normalized_columns.items():
            score = self._score_generic_column(
                field_name=field_name,
                field_normalized=field_normalized,
                column_name=column,
                column_normalized=normalized,
            )
            if score > best_score:
                best_column = column
                best_score = score
        if best_score < 4:
            return None
        return best_column

    def _score_generic_column(
        self,
        *,
        field_name: str,
        field_normalized: str,
        column_name: str,
        column_normalized: str,
    ) -> int:
        if not column_normalized or column_normalized in {"rank", "geo.sort", "geosort", "#", "#.1", "#.2", "#.3"}:
            return 0

        aliases = {
            "state": (
                {"state"},
                {"territory"},
                {"district"},
                {"federal", "district"},
            ),
            "population": (
                {"population"},
                {"pop"},
                {"census", "population"},
            ),
            "population_growth_rate": (
                {"population", "growth"},
                {"population", "change"},
                {"pop", "change"},
                {"growth", "rate"},
                {"percent", "change"},
                {"change"},
            ),
            "gdp": (
                {"gdp"},
                {"gross", "domestic", "product"},
                {"nominal", "gdp"},
            ),
            "gdp_growth_rate": (
                {"gdp", "growth"},
                {"gdp", "percent", "change"},
                {"economic", "growth", "rate"},
                {"real", "gdp", "growth", "rate"},
            ),
            "name": (
                {"name"},
                {"team"},
                {"organization"},
                {"company"},
            ),
        }
        text = f"{column_name} {column_normalized}"
        tokens = _tokenize_label(text)
        accepted_groups = aliases.get(field_name, ({field_normalized},))
        score = 0
        for group in accepted_groups:
            if all(token in tokens for token in group):
                score = max(score, len(group) * 3)
        if column_normalized and field_normalized and column_normalized == field_normalized:
            score = max(score, 6)
        elif column_normalized and field_normalized and (
            column_normalized.startswith(field_normalized)
            or field_normalized.startswith(column_normalized)
        ):
            score = max(score, 4)

        if field_name == "population_growth_rate":
            if "gdp" in tokens:
                score = 0
            elif "population" in tokens and any(token in tokens for token in {"growth", "change"}):
                score = max(score, 9)
            elif "change" in tokens and "population" not in tokens and "gdp" not in tokens:
                score = max(score, 5)
        elif field_name == "population":
            if "population" in tokens or "pop" in tokens:
                score = max(score, 8)
            if score and any(year in text for year in ("2025", "2024", "2023", "2022", "2021", "2020")):
                score += 1
        elif field_name == "gdp":
            if any(token in tokens for token in {"growth", "change", "percent", "rate"}):
                return 0
            if "gdp" in tokens and "growth" not in tokens and "change" not in tokens:
                score = max(score, 8)
            if score and any(year in text for year in ("2025", "2024", "2023", "2022", "2021", "2020")):
                score += 1
        elif field_name == "gdp_growth_rate":
            if "gdp" in tokens and any(token in tokens for token in {"growth", "change"}):
                score = max(score, 9)
        elif field_name == "state":
            if "state" in tokens:
                score = max(score, 8)
            if "territory" in tokens or "district" in tokens:
                score = max(score, 6)
        return score

    def _row_to_payload(
        self,
        row: dict[str, Any],
        *,
        mapping: dict[str, str],
        source_url: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field_name, column_name in mapping.items():
            cleaned = self._clean_cell(row.get(column_name), field_name=field_name)
            if cleaned is None:
                continue
            payload[field_name] = cleaned

        if "source_url" in self.row_model.model_fields:
            payload["source_url"] = source_url
        if "reference_url" in self.row_model.model_fields and "reference_url" not in payload:
            payload["reference_url"] = source_url

        self._fill_derived_fields(payload)
        if not self._has_primary_identity(payload):
            return {}
        if self._is_aggregate_row(payload):
            return {}
        return payload

    def _fill_derived_fields(self, payload: dict[str, Any]) -> None:
        if "team_name" in self.row_model.model_fields and not payload.get("team_name"):
            if payload.get("nickname"):
                payload["team_name"] = payload["nickname"]
            elif payload.get("school"):
                payload["team_name"] = payload["school"]
        if "name" in self.row_model.model_fields and not payload.get("name"):
            for candidate in ("team_name", "school", "nickname"):
                value = payload.get(candidate)
                if value:
                    payload["name"] = value
                    break

    def _has_primary_identity(self, payload: dict[str, Any]) -> bool:
        for candidate in ("name", "team_name", "school", "entity_name", "organization_name", "state"):
            value = payload.get(candidate)
            if isinstance(value, str) and value.strip():
                return True
        return False

    def _is_aggregate_row(self, payload: dict[str, Any]) -> bool:
        state_value = payload.get("state")
        if not isinstance(state_value, str):
            return False
        normalized = state_value.strip().casefold()
        blocked = {
            "united states",
            "district of columbia",
            "puerto rico",
            "guam",
            "american samoa",
            "northern mariana islands",
            "u.s. virgin islands",
            "virgin islands",
            "50 states and district of columbia",
            "island areas (territories)",
            "new england",
            "mid-atlantic",
            "east north central",
            "west north central",
            "south atlantic",
            "east south central",
            "west south central",
            "mountain",
            "pacific",
            "northeast",
            "midwest",
            "south",
            "west",
        }
        if normalized in blocked:
            return True
        return any(token in normalized for token in ("division", "region", "total"))

    def _clean_cell(self, value: Any, *, field_name: str) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None

        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "n/a"}:
            return None

        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None
        if self._field_expects_numeric(field_name):
            return _parse_numeric_value(text)
        return text

    def _field_expects_numeric(self, field_name: str) -> bool:
        schema = self.row_model.model_json_schema()
        property_schema = schema.get("properties", {}).get(field_name, {})
        return _schema_allows_numeric(property_schema)


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _tokenize_label(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", value.lower())
        if token
    }


def _schema_allows_numeric(schema: dict[str, Any]) -> bool:
    direct_type = schema.get("type")
    if direct_type in {"number", "integer"}:
        return True
    for branch in schema.get("anyOf", []):
        if branch.get("type") in {"number", "integer"}:
            return True
    return False


def _parse_numeric_value(text: str) -> float | int | None:
    normalized = text.strip()
    if not normalized:
        return None
    normalized = normalized.replace("\u2212", "-")
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("€", "")
    normalized = normalized.replace("£", "")
    normalized = normalized.replace("US$", "")
    multiplier = 1.0
    lowered = normalized.casefold()
    if lowered.endswith("%"):
        normalized = normalized[:-1]
    suffix_multipliers = {
        "k": 1_000.0,
        "m": 1_000_000.0,
        "b": 1_000_000_000.0,
        "t": 1_000_000_000_000.0,
    }
    suffix = normalized[-1:].casefold()
    if suffix in suffix_multipliers:
        multiplier = suffix_multipliers[suffix]
        normalized = normalized[:-1]
    normalized = normalized.strip()
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    value = float(match.group(0)) * multiplier
    if value.is_integer():
        return int(value)
    return value


def _root_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    if not hostname:
        return ""
    parts = hostname.split(".")
    if len(parts) <= 2:
        return hostname
    if parts[-2] in {"co", "com", "org", "gov", "ac"} and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])
