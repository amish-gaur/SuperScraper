"""ML-focused dataframe formatting and export."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Sequence

import pandas as pd

from dataset_profiler import DatasetProfiler
from llm import LLMGateway


LOGGER = logging.getLogger(__name__)


class MLFormatter:
    """Convert synthesized records into an ML-ready dataframe."""

    def __init__(self, records: Sequence[dict[str, Any]], dataset_name: str) -> None:
        self.dataset_name = dataset_name
        self.dataframe = pd.DataFrame(records)

    def handle_missing_values(self) -> pd.DataFrame:
        """Impute numeric values with median and categoricals with 'Unknown'."""
        numeric_columns = self.dataframe.select_dtypes(include=["number", "bool"]).columns
        for column in numeric_columns:
            median = self.dataframe[column].median()
            self.dataframe[column] = self.dataframe[column].fillna(median)

        categorical_columns = self.dataframe.select_dtypes(include=["object", "string"]).columns
        for column in categorical_columns:
            self.dataframe[column] = self.dataframe[column].fillna("Unknown")
            self.dataframe[column] = self.dataframe[column].replace("", "Unknown")

        LOGGER.info("Handled missing values for dataframe shape=%s", self.dataframe.shape)
        return self.dataframe

    def encode_categoricals(self) -> pd.DataFrame:
        """One-hot encode categorical text columns."""
        categorical_columns = self.dataframe.select_dtypes(include=["object", "string"]).columns
        if len(categorical_columns) > 0:
            self.dataframe = pd.get_dummies(
                self.dataframe,
                columns=list(categorical_columns),
                dummy_na=False,
            )
        LOGGER.info("Encoded categoricals; dataframe shape=%s", self.dataframe.shape)
        return self.dataframe

    def export(self, output_dir: str | Path = ".") -> tuple[Path, Path]:
        """Export the processed dataframe as CSV and Parquet."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        slug = _slugify(self.dataset_name)
        csv_path = directory / f"{slug}.csv"
        parquet_path = directory / f"{slug}.parquet"

        self.dataframe.to_csv(csv_path, index=False)
        self.dataframe.to_parquet(parquet_path, index=False)
        LOGGER.info("Exported dataframe to %s and %s", csv_path, parquet_path)
        return csv_path, parquet_path

    def export_profile(
        self,
        *,
        goal: str,
        provenance_map: dict[str, str] | None = None,
        pruning_audit: dict[str, Any] | None = None,
        llm_gateway: LLMGateway | None = None,
        output_dir: str | Path = ".",
    ) -> Path:
        """Profile the dataframe and write a JSON report next to the dataset export."""
        profiler = DatasetProfiler()
        profile = profiler.profile(
            self.dataframe,
            self.dataset_name,
            goal=goal,
            provenance_map=provenance_map,
            pruning_audit=pruning_audit,
            llm_gateway=llm_gateway,
        )
        path = profiler.write(profile, output_dir=output_dir)
        LOGGER.info("Exported dataset profile to %s", path)
        return path


def _slugify(value: str) -> str:
    """Convert an arbitrary dataset name into a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "ml_dataset"
