"""Profiling utilities for exported datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

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
        heuristic_target = self._infer_target_column(dataframe)
        target_inference = self._infer_target_with_llm(
            goal=goal,
            dataframe=dataframe,
            llm_gateway=llm_gateway,
        )
        inferred_target = target_inference.inferred_target_column or heuristic_target
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
        )

    def write(self, profile: DatasetProfile, output_dir: str | Path = ".") -> Path:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        slug = _slugify(profile.dataset_name)
        path = directory / f"{slug}_profile.json"
        path.write_text(json.dumps(asdict(profile), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _infer_target_column(self, dataframe: pd.DataFrame) -> str | None:
        priority_tokens = ("target", "label", "outcome", "revenue", "price", "salary", "margin", "pct")
        for column in dataframe.columns:
            lowered = column.lower()
            if any(token in lowered for token in priority_tokens):
                return str(column)
        numeric_columns = list(dataframe.select_dtypes(include=["number", "bool"]).columns)
        return numeric_columns[-1] if numeric_columns else None

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
            return self._heuristic_target_inference(dataframe)

    def _heuristic_target_inference(self, dataframe: pd.DataFrame) -> TargetInference:
        inferred_target = self._infer_target_column(dataframe)
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
