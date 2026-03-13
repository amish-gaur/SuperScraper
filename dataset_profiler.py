"""Profiling utilities for exported datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from llm import LLMError, LLMGateway, StructuredEnvelope


@dataclass(slots=True)
class DatasetProfile:
    dataset_name: str
    row_count: int
    column_count: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    null_fraction_by_column: dict[str, float]
    unique_count_by_column: dict[str, int]
    likely_target_column: str | None
    inferred_target_column: str | None
    ml_task_type: str
    dropped_features: list[str]
    provenance_by_column: dict[str, str]
    leakage_warnings: list[str]
    pruning_audit: dict[str, Any]


class TargetInference(StructuredEnvelope):
    """Strict target-inference contract for the final ML context pass."""

    inferred_target_column: str | None = Field(
        default=None,
        description="Exact column name that is the most likely supervised learning target.",
    )
    ml_task_type: str = Field(
        description='One of "regression", "binary classification", "multiclass classification", "ranking", or "clustering".'
    )
    dropped_features: list[str] = Field(
        default_factory=list,
        description="Columns that should likely be excluded from training because they are IDs, URLs, names, or too sparse.",
    )


class DatasetProfiler:
    """Produce and persist high-signal dataset metadata."""

    def profile(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str,
        *,
        goal: str,
        provenance_map: dict[str, str] | None = None,
        pruning_audit: dict[str, Any] | None = None,
        llm_gateway: LLMGateway | None = None,
    ) -> DatasetProfile:
        numeric_columns = list(dataframe.select_dtypes(include=["number", "bool"]).columns)
        categorical_columns = list(
            dataframe.select_dtypes(include=["object", "string"]).columns
        )
        null_fraction = {
            column: round(float(dataframe[column].isna().mean()), 4)
            for column in dataframe.columns
        }
        unique_counts = {
            column: int(dataframe[column].nunique(dropna=True))
            for column in dataframe.columns
        }
        heuristic_target = self._infer_target_column(dataframe, goal=goal)
        target_inference = self._infer_target_with_llm(
            goal=goal,
            dataframe=dataframe,
            llm_gateway=llm_gateway,
        )
        inferred_target = self._choose_best_target_candidate(
            dataframe,
            goal=goal,
            candidates=[
                target_inference.inferred_target_column,
                heuristic_target,
            ],
        )
        dropped_features = self._normalize_dropped_features(
            target_inference.dropped_features,
            dataframe,
        )
        leakage_warnings = self._detect_leakage_warnings(dataframe, inferred_target)
        return DatasetProfile(
            dataset_name=dataset_name,
            row_count=int(len(dataframe)),
            column_count=int(len(dataframe.columns)),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            null_fraction_by_column=null_fraction,
            unique_count_by_column=unique_counts,
            likely_target_column=heuristic_target,
            inferred_target_column=inferred_target,
            ml_task_type=target_inference.ml_task_type,
            dropped_features=dropped_features,
            provenance_by_column=provenance_map or {
                column: "unknown_source" for column in dataframe.columns
            },
            leakage_warnings=leakage_warnings,
            pruning_audit=pruning_audit or {},
        )

    def write(self, profile: DatasetProfile, output_dir: str | Path = ".") -> Path:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        slug = _slugify(profile.dataset_name)
        path = directory / f"{slug}_profile.json"
        path.write_text(json.dumps(asdict(profile), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _infer_target_column(self, dataframe: pd.DataFrame, *, goal: str = "") -> str | None:
        if dataframe.empty:
            return None
        columns = list(map(str, dataframe.columns))
        scored = sorted(
            columns,
            key=lambda column: self._target_column_score(
                goal=goal,
                column=column,
                series=dataframe[column],
            ),
            reverse=True,
        )
        best = scored[0] if scored else None
        if best is None:
            return None
        best_score = self._target_column_score(goal=goal, column=best, series=dataframe[best])
        if best_score[0] <= 0:
            numeric_columns = list(dataframe.select_dtypes(include=["number", "bool"]).columns)
            return str(numeric_columns[-1]) if numeric_columns else None
        return best

    def _infer_target_with_llm(
        self,
        *,
        goal: str,
        dataframe: pd.DataFrame,
        llm_gateway: LLMGateway | None,
    ) -> TargetInference:
        if llm_gateway is None:
            return self._heuristic_target_inference(dataframe)

        system_prompt = (
            "You are an ML dataset profiler. "
            "Given a supervised-learning goal and a list of dataset columns, identify the most likely target column, "
            "the task type, and low-value columns to drop before training. "
            "Return strict JSON only."
        )
        user_prompt = (
            f"Goal:\n{goal}\n\n"
            f"Columns:\n{json.dumps(list(map(str, dataframe.columns.tolist())), indent=2)}\n\n"
            "Choose the exact existing target column name. "
            "If no obvious supervised target exists, choose the best numeric business outcome column when possible. "
            "Mark IDs, names, URLs, and very likely leakage columns in dropped_features."
        )
        try:
            return llm_gateway.complete_structured(
                response_model=TargetInference,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name="dataset_target_inference",
                max_tokens=500,
            )
        except LLMError:
            return self._heuristic_target_inference(dataframe, goal=goal)

    def _heuristic_target_inference(self, dataframe: pd.DataFrame, *, goal: str = "") -> TargetInference:
        inferred_target = self._infer_target_column(dataframe, goal=goal)
        lowered_columns = [column.lower() for column in dataframe.columns]
        if inferred_target and any(token in inferred_target.lower() for token in ("pct", "ratio", "price", "revenue", "assets", "salary", "margin")):
            task_type = "regression"
        elif inferred_target and dataframe[inferred_target].nunique(dropna=True) <= 2:
            task_type = "binary classification"
        else:
            task_type = "regression"

        dropped_features = [
            column for column in dataframe.columns
            if any(token in column.lower() for token in ("id", "uuid", "url", "source", "name", "entity_name"))
        ]
        return TargetInference(
            inferred_target_column=inferred_target,
            ml_task_type=task_type,
            dropped_features=dropped_features,
        )

    def _choose_best_target_candidate(
        self,
        dataframe: pd.DataFrame,
        *,
        goal: str,
        candidates: list[str | None],
    ) -> str | None:
        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate and candidate in dataframe.columns
        ]
        if not valid_candidates:
            return None
        return max(
            valid_candidates,
            key=lambda column: self._target_column_score(
                goal=goal,
                column=column,
                series=dataframe[column],
            ),
        )

    def _target_column_score(self, *, goal: str, column: str, series: pd.Series) -> tuple[int, int, int, int]:
        lowered = column.lower()
        tokens = {token for token in lowered.replace("/", "_").split("_") if token}
        goal_lower = goal.lower()
        score = 0

        if any(token in lowered for token in ("source", "url", "name", "entity_name", "id", "uuid")):
            score -= 50
        if pd.api.types.is_numeric_dtype(series):
            score += 8
        if "target" in tokens or "label" in tokens or "outcome" in tokens:
            score += 30
        if "salary" in goal_lower and "salary" in tokens:
            score += 40
        if any(token in goal_lower for token in ("valuation", "market value")) and "valuation" in tokens:
            score += 40
        if any(token in goal_lower for token in ("price", "pricing", "cost")) and "price" in tokens:
            score += 40
        if "revenue growth" in goal_lower and {"revenue", "growth"} <= tokens:
            score += 40
        elif "revenue" in goal_lower and "revenue" in tokens:
            score += 35
        if "population" in goal_lower and any(token in goal_lower for token in ("growth", "change")):
            if "population" in tokens and ({"growth", "rate"} <= tokens or "change" in tokens):
                score += 50
            if "gdp" in tokens and any(token in tokens for token in ("growth", "change")):
                score -= 20
        if "gdp" in goal_lower and any(token in goal_lower for token in ("growth", "change")):
            if "gdp" in tokens and any(token in tokens for token in ("growth", "change", "rate")):
                score += 35
        if any(token in goal_lower for token in ("win", "winning")) and any(token in tokens for token in ("winning", "wins", "pct")):
            score += 25
        if "points" in goal_lower and "points" in tokens:
            score += 20

        generic_priority_tokens = ("salary", "valuation", "price", "revenue", "income", "margin", "growth", "pct")
        if any(token in lowered for token in generic_priority_tokens):
            score += 10

        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        compact_name_bonus = max(0, 12 - min(len(tokens), 12))
        return (score, non_null, compact_name_bonus, unique)

    def _normalize_dropped_features(self, dropped_features: list[str], dataframe: pd.DataFrame) -> list[str]:
        valid = []
        seen: set[str] = set()
        columns = set(map(str, dataframe.columns))
        for feature in dropped_features:
            if feature not in columns or feature in seen:
                continue
            seen.add(feature)
            valid.append(feature)
        return valid

    def _detect_leakage_warnings(self, dataframe: pd.DataFrame, target: str | None) -> list[str]:
        if not target or target not in dataframe.columns:
            return []

        warnings: list[str] = []
        target_series = dataframe[target]
        if target_series.nunique(dropna=True) <= 1:
            warnings.append(f"Target column '{target}' is constant or nearly constant.")

        if pd.api.types.is_numeric_dtype(target_series):
            correlations = dataframe.select_dtypes(include=["number", "bool"]).corr(numeric_only=True)
            if target in correlations:
                for column, value in correlations[target].items():
                    if column == target or pd.isna(value):
                        continue
                    if abs(float(value)) >= 0.995:
                        warnings.append(
                            f"Column '{column}' is almost perfectly correlated with target '{target}' ({value:.3f})."
                        )
        return warnings


def _slugify(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")
