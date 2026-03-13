"""Shared post-extraction dataset pruning for predictive and fallback flows."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from typing import Any

import pandas as pd

from llm import LLMError, LLMGateway
from settings import get_settings


LOGGER = logging.getLogger(__name__)

PROTECTED_METADATA_COLUMNS = {"name", "entity_name", "raw_entity_name", "source", "source_url"}


@dataclass(slots=True)
class PruningResult:
    """Pruned dataframe plus audit metadata."""

    dataframe: pd.DataFrame
    provenance_map: dict[str, str]
    pruning_audit: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PostExtractionPruner:
    """Apply shared statistical and optional semantic pruning after extraction."""

    goal: str
    target_field: str | None = None
    core_feature_fields: list[str] = field(default_factory=list)
    required_feature_fields: list[str] = field(default_factory=list)
    llm_gateway: LLMGateway | None = None

    def process(
        self,
        dataframe: pd.DataFrame,
        *,
        provenance_map: dict[str, str] | None = None,
    ) -> PruningResult:
        pruned = dataframe.copy()
        if pruned.empty:
            return PruningResult(
                dataframe=pruned,
                provenance_map=self._prune_provenance_map(provenance_map or {}, pruned.columns),
                pruning_audit={},
            )

        target_column = self._resolve_quality_column(pruned, [self.target_field] if self.target_field else [])
        if target_column is None:
            LOGGER.info("Skipping dataset pruning because target column '%s' could not be resolved", self.target_field)
            return PruningResult(
                dataframe=pruned.reset_index(drop=True),
                provenance_map=self._prune_provenance_map(provenance_map or {}, pruned.columns),
                pruning_audit={},
            )

        original_row_count = len(pruned)
        original_columns = list(map(str, pruned.columns))
        protected_columns = self._protected_schema_columns()

        missing_target_mask = self._series_missing_mask(pruned[target_column])
        rows_missing_target = int(missing_target_mask.sum())
        if rows_missing_target:
            pruned = pruned.loc[~missing_target_mask].copy()

        feature_columns = [
            column
            for column in pruned.columns
            if column != target_column and str(column) not in PROTECTED_METADATA_COLUMNS
        ]
        rows_sparse_features = 0
        if feature_columns:
            feature_missing = pd.DataFrame(
                {str(column): self._series_missing_mask(pruned[column]) for column in feature_columns},
                index=pruned.index,
            )
            rows_sparse_mask = feature_missing.mean(axis=1) > 0.50
            rows_sparse_features = int(rows_sparse_mask.sum())
            if rows_sparse_features:
                pruned = pruned.loc[~rows_sparse_mask].copy()

        sparse_columns = [
            str(column)
            for column in pruned.columns
            if (
                column != target_column
                and str(column) not in PROTECTED_METADATA_COLUMNS
                and str(column) not in protected_columns
                and self._missing_rate(pruned[column]) > 0.40
            )
        ]
        if sparse_columns:
            pruned = pruned.drop(columns=sparse_columns, errors="ignore")

        zero_variance_columns = [
            str(column)
            for column in pruned.columns
            if (
                column != target_column
                and str(column) not in PROTECTED_METADATA_COLUMNS
                and str(column) not in protected_columns
                and self._has_zero_variance(pruned[column])
            )
        ]
        if zero_variance_columns:
            pruned = pruned.drop(columns=zero_variance_columns, errors="ignore")

        semantic_columns = self._semantic_prune_columns(pruned, target_column=target_column)
        if semantic_columns:
            pruned = pruned.drop(columns=semantic_columns, errors="ignore")

        remaining_columns = [column for column in original_columns if column in pruned.columns]
        pruned = pruned.loc[:, remaining_columns].reset_index(drop=True)
        pruning_audit = {
            "target_column": target_column,
            "rows_before": original_row_count,
            "rows_after": int(len(pruned)),
            "columns_before": list(original_columns),
            "columns_after": list(map(str, pruned.columns)),
            "dropped_rows_missing_target": rows_missing_target,
            "dropped_rows_sparse_features": rows_sparse_features,
            "dropped_sparse_columns": sorted(sparse_columns),
            "dropped_zero_variance_columns": sorted(zero_variance_columns),
            "dropped_semantic_columns": sorted(semantic_columns),
            "protected_schema_columns": sorted(protected_columns),
            "semantic_pruning_enabled": self._semantic_pruning_enabled(),
        }
        pruned.attrs["pruning_audit"] = pruning_audit

        LOGGER.info(
            "Pruned dataset rows/columns: dropped %d rows missing target, %d rows with >50%% missing features, %d sparse columns, %d zero-variance columns, %d semantically irrelevant columns",
            rows_missing_target,
            rows_sparse_features,
            len(sparse_columns),
            len(zero_variance_columns),
            len(semantic_columns),
        )
        if sparse_columns:
            LOGGER.info("Dropped sparse columns: %s", ", ".join(sorted(sparse_columns)))
        if zero_variance_columns:
            LOGGER.info("Dropped zero-variance columns: %s", ", ".join(sorted(zero_variance_columns)))
        if semantic_columns:
            LOGGER.info(
                "Dropped semantically irrelevant columns for target '%s': %s",
                target_column,
                ", ".join(sorted(semantic_columns)),
            )

        return PruningResult(
            dataframe=pruned,
            provenance_map=self._prune_provenance_map(provenance_map or {}, pruned.columns),
            pruning_audit=pruning_audit,
        )

    def _semantic_prune_columns(self, frame: pd.DataFrame, *, target_column: str) -> list[str]:
        if not self._semantic_pruning_enabled():
            return []
        candidate_columns = [
            str(column)
            for column in frame.columns
            if (
                column != target_column
                and str(column) not in PROTECTED_METADATA_COLUMNS
                and str(column) not in self._expected_schema_fields()
            )
        ]
        if not candidate_columns:
            return []

        gateway = self._get_semantic_pruning_gateway()
        if gateway is None:
            return []

        system_prompt = (
            "You are a machine learning feature-pruning assistant. "
            "Review candidate dataset columns and identify columns that are clearly UI artifacts, random noise, page chrome, rankings-only clutter, or otherwise irrelevant to predicting the target variable. "
            "Be conservative: keep plausible business, statistical, demographic, operational, or domain features. "
            "Return JSON only."
        )
        user_prompt = (
            f"Identify any columns in this list that are clearly UI artifacts, random noise, or completely irrelevant to predicting the target variable '{target_column}'. "
            "Return a JSON list of column names to drop.\n\n"
            f"Columns:\n{json.dumps(candidate_columns, indent=2)}"
        )
        try:
            response = gateway.complete_text(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=300,
            )
            parsed = self._parse_json_list(response)
        except LLMError as exc:
            LOGGER.warning("Semantic pruning skipped because the LLM call failed: %s", exc)
            return []
        except ValueError as exc:
            LOGGER.warning("Semantic pruning skipped because the LLM returned invalid JSON: %s", exc)
            return []

        valid_columns = set(candidate_columns)
        drops: list[str] = []
        seen: set[str] = set()
        for column in parsed:
            normalized = str(column).strip()
            if normalized in valid_columns and normalized not in seen:
                seen.add(normalized)
                drops.append(normalized)
        return drops

    def _get_semantic_pruning_gateway(self) -> LLMGateway | None:
        if self.llm_gateway is not None:
            return self.llm_gateway
        try:
            self.llm_gateway = LLMGateway(model="gpt-4o-mini", max_tokens=400)
        except LLMError as primary_exc:
            LOGGER.warning("Fast semantic pruning model unavailable: %s", primary_exc)
            try:
                self.llm_gateway = LLMGateway(max_tokens=400)
            except LLMError as fallback_exc:
                LOGGER.warning("Semantic pruning disabled because no LLM gateway is available: %s", fallback_exc)
                return None
        return self.llm_gateway

    def _semantic_pruning_enabled(self) -> bool:
        return bool(get_settings().enable_semantic_feature_pruning)

    def _protected_schema_columns(self) -> set[str]:
        protected = {field_name for field_name in self.required_feature_fields if field_name}
        if self.target_field:
            protected.add(self.target_field)
        return protected

    def _expected_schema_fields(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for field_name in [*self.core_feature_fields, self.target_field]:
            if not field_name or field_name in seen:
                continue
            seen.add(field_name)
            ordered.append(field_name)
        return ordered

    def _resolve_quality_column(self, frame: pd.DataFrame, field_names: list[str]) -> str | None:
        if not field_names:
            return None
        normalized_columns = {str(column): self._normalize_column_name(str(column)) for column in frame.columns}
        for field_name in field_names:
            if not field_name:
                continue
            if field_name in frame.columns:
                return field_name
        for field_name in field_names:
            if not field_name:
                continue
            normalized_field = self._normalize_column_name(field_name)
            for column, normalized_column in normalized_columns.items():
                if normalized_column == normalized_field:
                    return column
            for column, normalized_column in normalized_columns.items():
                if normalized_column.startswith(normalized_field) or normalized_field in normalized_column:
                    return column
        return None

    def _series_missing_mask(self, series: pd.Series) -> pd.Series:
        missing = series.isna()
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized = series.fillna("").astype(str).str.strip().str.casefold()
            missing = missing | normalized.isin({"", "unknown", "n/a", "na", "none", "null"})
        return missing

    def _missing_rate(self, series: pd.Series) -> float:
        if len(series) == 0:
            return 1.0
        return float(self._series_missing_mask(series).mean())

    def _has_zero_variance(self, series: pd.Series) -> bool:
        if len(series) == 0:
            return True
        non_missing = series.loc[~self._series_missing_mask(series)]
        if non_missing.empty:
            return True
        normalized = non_missing.map(lambda value: str(value).strip().casefold() if isinstance(value, str) else value)
        return normalized.nunique(dropna=True) <= 1

    def _prune_provenance_map(self, provenance_map: dict[str, str], columns: Any) -> dict[str, str]:
        column_names = {str(column) for column in columns}
        return {
            column: source
            for column, source in provenance_map.items()
            if column in column_names
        }

    def _parse_json_list(self, value: str) -> list[str]:
        cleaned = value.strip()
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("Expected a JSON list")
        return [str(item) for item in parsed if isinstance(item, str)]

    def _normalize_column_name(self, column: str) -> str:
        normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(column))
        normalized = "_".join(part for part in normalized.split("_") if part)
        return normalized or "column"
