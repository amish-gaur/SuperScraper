"""Semantic validation and source-agreement checks for exported datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Literal
from urllib.parse import urlparse

import pandas as pd
from pydantic import BaseModel


IDENTITY_FIELDS = {
    "source_url",
    "reference_url",
    "entity_name",
    "raw_entity_name",
    "name",
    "team_name",
    "school",
    "state",
}
NON_NEGATIVE_TOKENS = {
    "age",
    "asset",
    "attendance",
    "block",
    "count",
    "display",
    "fee",
    "funding",
    "gdp",
    "height",
    "income",
    "loss",
    "market_cap",
    "payroll",
    "points",
    "population",
    "price",
    "profit",
    "ram",
    "rank",
    "rating",
    "rebound",
    "revenue",
    "salary",
    "score",
    "size",
    "steal",
    "turnover",
    "valuation",
    "weight",
    "win",
}
PERCENT_TOKENS = {"pct", "percent", "percentage", "share"}
RATE_TOKENS = {"rate"}
GROWTH_TOKENS = {"growth", "change", "delta"}
LOWER_BOUND_SUFFIXES = ("_min", "_minimum", "_low")
UPPER_BOUND_SUFFIXES = ("_max", "_maximum", "_high")


class DataValidationError(RuntimeError):
    """Raised when semantic validation leaves too little trustworthy data."""


@dataclass(slots=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    category: str
    message: str
    field: str | None = None
    row_index: int | None = None
    entity: str | None = None


@dataclass(slots=True)
class ValidationReport:
    dataset_name: str
    original_row_count: int
    retained_row_count: int
    dropped_row_count: int
    error_count: int
    warning_count: int
    cross_source_entity_count: int
    dropped_entities: list[str]
    issues: list[ValidationIssue]

    def write(self, output_dir: str | Path = ".") -> Path:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        slug = _slugify(self.dataset_name)
        path = directory / f"{slug}_validation.json"
        payload = {
            "dataset_name": self.dataset_name,
            "original_row_count": self.original_row_count,
            "retained_row_count": self.retained_row_count,
            "dropped_row_count": self.dropped_row_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "cross_source_entity_count": self.cross_source_entity_count,
            "dropped_entities": self.dropped_entities,
            "issues": [asdict(issue) for issue in self.issues],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


@dataclass(slots=True)
class ValidationResult:
    dataframe: pd.DataFrame
    report: ValidationReport


class SemanticDataValidator:
    """Reject implausible rows before dataset export."""

    def __init__(self, *, row_model: type[BaseModel]) -> None:
        self.row_model = row_model
        self.target_field = self._identify_target_field()

    def validate(
        self,
        dataframe: pd.DataFrame,
        *,
        dataset_name: str,
        raw_records: list[BaseModel] | None = None,
    ) -> ValidationResult:
        working = dataframe.copy()
        issues: list[ValidationIssue] = []
        dropped_entities: list[str] = []
        rows_to_drop: set[int] = set()
        entity_column = self._entity_column(working)

        rows_to_drop.update(
            self._apply_row_rules(working, issues=issues, entity_column=entity_column)
        )
        rows_to_drop.update(
            self._apply_outlier_rules(working, issues=issues, entity_column=entity_column)
        )

        conflicting_entities = self._detect_cross_source_conflicts(raw_records or [], issues=issues)
        if entity_column is not None and conflicting_entities:
            for index, row in working.iterrows():
                entity = self._normalize_entity(row.get(entity_column))
                if entity in conflicting_entities:
                    rows_to_drop.add(index)
                    dropped_entities.append(str(row.get(entity_column)).strip())

        if rows_to_drop:
            working = working.drop(index=sorted(rows_to_drop)).reset_index(drop=True)

        original_count = len(dataframe)
        retained_count = len(working)
        dropped_count = original_count - retained_count
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        unique_dropped_entities = sorted({entity for entity in dropped_entities if entity})

        if retained_count == 0:
            raise DataValidationError("Semantic validation rejected all rows; refusing to export an empty dataset.")
        if original_count >= 10 and dropped_count / max(original_count, 1) > 0.35:
            raise DataValidationError(
                "Semantic validation rejected more than 35% of rows; source data is too inconsistent to export safely."
            )

        report = ValidationReport(
            dataset_name=dataset_name,
            original_row_count=original_count,
            retained_row_count=retained_count,
            dropped_row_count=dropped_count,
            error_count=error_count,
            warning_count=warning_count,
            cross_source_entity_count=len(conflicting_entities),
            dropped_entities=unique_dropped_entities,
            issues=issues,
        )
        return ValidationResult(dataframe=working, report=report)

    def _apply_row_rules(
        self,
        dataframe: pd.DataFrame,
        *,
        issues: list[ValidationIssue],
        entity_column: str | None,
    ) -> set[int]:
        rows_to_drop: set[int] = set()
        enums = self._enum_values()

        for index, row in dataframe.iterrows():
            entity = self._row_entity(row, entity_column)
            for column in dataframe.columns:
                value = row.get(column)
                if self._is_missing(value):
                    continue
                if column in enums and str(value) not in enums[column]:
                    rows_to_drop.add(index)
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="enum_mismatch",
                            field=str(column),
                            row_index=int(index),
                            entity=entity,
                            message=f"Value '{value}' is outside the allowed set for '{column}'.",
                        )
                    )
                if not pd.api.types.is_numeric_dtype(dataframe[column]):
                    continue
                numeric = _to_float(value)
                if numeric is None:
                    continue
                if self._violates_hard_bounds(str(column), numeric):
                    rows_to_drop.add(index)
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="hard_bounds",
                            field=str(column),
                            row_index=int(index),
                            entity=entity,
                            message=f"Value {numeric:g} for '{column}' is outside the plausible range.",
                        )
                    )

            for lower_column, upper_column in self._bound_column_pairs(dataframe.columns):
                lower_value = _to_float(row.get(lower_column))
                upper_value = _to_float(row.get(upper_column))
                if lower_value is None or upper_value is None:
                    continue
                if lower_value > upper_value:
                    rows_to_drop.add(index)
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            category="cross_field_consistency",
                            field=f"{lower_column},{upper_column}",
                            row_index=int(index),
                            entity=entity,
                            message=f"'{lower_column}' exceeds '{upper_column}' for the same row.",
                        )
                    )
        return rows_to_drop

    def _apply_outlier_rules(
        self,
        dataframe: pd.DataFrame,
        *,
        issues: list[ValidationIssue],
        entity_column: str | None,
    ) -> set[int]:
        if len(dataframe) < 12:
            return set()

        flagged_columns_by_row: dict[int, list[str]] = {}
        target_hits: set[int] = set()
        for column in dataframe.select_dtypes(include=["number", "bool"]).columns:
            if column in IDENTITY_FIELDS:
                continue
            series = pd.to_numeric(dataframe[column], errors="coerce")
            valid = series.dropna()
            if len(valid) < 8:
                continue

            median = float(valid.median())
            mad = float((valid - median).abs().median())
            q1 = float(valid.quantile(0.25))
            q3 = float(valid.quantile(0.75))
            iqr = q3 - q1
            for index, value in series.items():
                if pd.isna(value):
                    continue
                robust_score = abs(0.6745 * (float(value) - median) / mad) if mad > 0 else 0.0
                beyond_iqr = (
                    iqr > 0
                    and (
                        float(value) < q1 - 5.0 * iqr
                        or float(value) > q3 + 5.0 * iqr
                    )
                )
                if robust_score <= 8.0 and not beyond_iqr:
                    continue
                flagged_columns_by_row.setdefault(int(index), []).append(str(column))
                if column == self.target_field and robust_score > 12.0:
                    target_hits.add(int(index))
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        category="outlier",
                        field=str(column),
                        row_index=int(index),
                        entity=self._row_entity(dataframe.loc[index], entity_column),
                        message=f"'{column}' contains an extreme outlier value ({float(value):g}).",
                    )
                )

        rows_to_drop = {
            index
            for index, columns in flagged_columns_by_row.items()
            if len(columns) >= 2 or index in target_hits
        }
        for index in rows_to_drop:
            issues.append(
                ValidationIssue(
                    severity="error",
                    category="outlier_rejection",
                    row_index=index,
                    entity=self._row_entity(dataframe.loc[index], entity_column),
                    message="Row was rejected because it contains multiple extreme numeric outliers.",
                )
            )
        return rows_to_drop

    def _detect_cross_source_conflicts(
        self,
        raw_records: list[BaseModel],
        *,
        issues: list[ValidationIssue],
    ) -> set[str]:
        if len(raw_records) < 2:
            return set()

        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in raw_records:
            payload = record.model_dump(mode="json")
            entity_key = self._payload_identity_key(payload)
            if entity_key is None:
                continue
            grouped.setdefault(entity_key, []).append(payload)

        rejected_entities: set[str] = set()
        for entity_key, payloads in grouped.items():
            entity_value = entity_key.split(":", 1)[1]
            source_count = len(
                {
                    _root_domain(payload.get("source_url") or payload.get("reference_url"))
                    for payload in payloads
                    if _root_domain(payload.get("source_url") or payload.get("reference_url"))
                }
            )
            if source_count < 2:
                continue

            conflict_fields: list[str] = []
            for field_name in self.row_model.model_fields:
                if field_name in {"source_url", "reference_url"}:
                    continue
                values = [
                    payload.get(field_name)
                    for payload in payloads
                    if not self._is_missing(payload.get(field_name))
                ]
                if len(values) < 2:
                    continue
                if self._field_values_conflict(field_name, values):
                    conflict_fields.append(field_name)
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            category="cross_source_conflict",
                            field=field_name,
                            entity=entity_value,
                            message=f"Cross-source disagreement detected for '{field_name}' across {source_count} sources.",
                        )
                    )
            if self.target_field in conflict_fields or len(conflict_fields) >= 2:
                rejected_entities.add(entity_value)
                issues.append(
                    ValidationIssue(
                        severity="error",
                        category="cross_source_rejection",
                        entity=entity_value,
                        message="Entity was rejected because multiple sources disagree on key values.",
                    )
                )
        return rejected_entities

    def _field_values_conflict(self, field_name: str, values: list[Any]) -> bool:
        numeric_values = [_to_float(value) for value in values]
        if all(value is not None for value in numeric_values):
            ordered = sorted(float(value) for value in numeric_values if value is not None)
            if len(ordered) < 2:
                return False
            median = ordered[len(ordered) // 2]
            spread = ordered[-1] - ordered[0]
            denominator = max(abs(median), 1.0)
            threshold = 0.2 if self._looks_percentage_like(field_name) else 0.35
            return spread / denominator > threshold

        normalized = {
            _normalize_text(value)
            for value in values
            if not self._is_missing(value)
        }
        return len({value for value in normalized if value}) > 1

    def _violates_hard_bounds(self, field_name: str, value: float) -> bool:
        lowered = field_name.lower()
        tokens = set(re.findall(r"[a-z0-9]+", lowered))

        if "year" in tokens:
            return value < 1800 or value > 2100
        if tokens & NON_NEGATIVE_TOKENS and value < 0:
            return True
        if self._looks_percentage_like(field_name):
            if tokens & GROWTH_TOKENS:
                return value < -100 or value > 500
            return value < 0 or value > 100
        if tokens & RATE_TOKENS and not tokens & GROWTH_TOKENS:
            return value < 0 or value > 100
        return False

    def _bound_column_pairs(self, columns: Any) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        column_names = [str(column) for column in columns]
        for column in column_names:
            for suffix in LOWER_BOUND_SUFFIXES:
                if not column.endswith(suffix):
                    continue
                prefix = column[: -len(suffix)]
                for upper_suffix in UPPER_BOUND_SUFFIXES:
                    candidate = f"{prefix}{upper_suffix}"
                    if candidate in column_names:
                        pairs.append((column, candidate))
        return pairs

    def _enum_values(self) -> dict[str, set[str]]:
        schema = self.row_model.model_json_schema()
        result: dict[str, set[str]] = {}
        for field_name, property_schema in schema.get("properties", {}).items():
            enum = property_schema.get("enum")
            if isinstance(enum, list) and enum:
                result[str(field_name)] = {str(value) for value in enum}
        return result

    def _entity_column(self, dataframe: pd.DataFrame) -> str | None:
        for field_name in (
            "entity_name",
            "name",
            "team_name",
            "school",
            "state",
            "organization_name",
            "player_name",
            "company_name",
        ):
            if field_name in dataframe.columns:
                return field_name
        return None

    def _payload_identity_key(self, payload: dict[str, Any]) -> str | None:
        for field_name in (
            "state",
            "school",
            "team_name",
            "name",
            "entity_name",
            "organization_name",
            "player_name",
            "company_name",
        ):
            value = payload.get(field_name)
            if isinstance(value, str) and value.strip():
                return f"{field_name}:{value.strip().casefold()}"
        return None

    def _row_entity(self, row: pd.Series, entity_column: str | None) -> str | None:
        if entity_column is None:
            return None
        value = row.get(entity_column)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def _normalize_entity(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip().casefold()

    def _identify_target_field(self) -> str | None:
        for field_name, model_field in self.row_model.model_fields.items():
            extra = model_field.json_schema_extra or {}
            if extra.get("x-ml-role") == "target":
                return field_name

        schema = self.row_model.model_json_schema()
        for field_name, property_schema in schema.get("properties", {}).items():
            if property_schema.get("x-ml-role") == "target":
                return str(field_name)
        target_keywords = (
            "target",
            "label",
            "outcome",
            "result",
            "winner",
            "salary",
            "price",
            "valuation",
            "revenue",
            "income",
            "profit",
            "score",
        )
        for field_name in self.row_model.model_fields:
            if any(keyword in field_name.lower() for keyword in target_keywords):
                return field_name
        return None

    def _looks_percentage_like(self, field_name: str) -> bool:
        tokens = set(re.findall(r"[a-z0-9]+", field_name.lower()))
        return bool(tokens & PERCENT_TOKENS)

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, str) and value.strip().casefold() in {"", "na", "n/a", "none", "null", "unknown"}:
            return True
        return False


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().casefold()


def _root_domain(url: Any) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    hostname = (urlparse(text).hostname or "").lower().strip(".")
    if not hostname:
        return ""
    parts = hostname.split(".")
    if len(parts) <= 2:
        return hostname
    if parts[-2] in {"co", "com", "org", "gov", "ac"} and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "dataset"
