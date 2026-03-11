"""Reusable pipeline runner for CLI, worker, and API flows."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from architect import (
    DatasetArchitect,
    json_schema_to_pydantic_model,
    relax_pydantic_model,
    schema_to_json_schema,
)
from checkpoint import CheckpointManager
from formatter import MLFormatter
from goal_intent import infer_goal_cardinality
from llm import LLMGateway
from predictive_dataset_builder import PredictiveDatasetBuilder
from synthesizer import DataSynthesizer
from swarm import SwarmDispatcher


LOGGER = logging.getLogger(__name__)
ProgressCallback = Callable[[str, str, dict[str, Any] | None], None]


@dataclass(slots=True)
class PipelineArtifacts:
    dataset_name: str
    csv_path: str
    parquet_path: str
    profile_path: str
    rows: int
    columns: int


def run_pipeline(
    *,
    goal: str,
    max_agents: int,
    llm_gateway: LLMGateway | None = None,
    output_dir: str | Path = ".",
    checkpoint_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineArtifacts:
    """Run the four-stage pipeline and return exported artifact metadata."""
    domain_blacklist: set[str] = set()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(
        checkpoint_path or output_dir_path / "pipeline_cache.json"
    )

    _notify(
        progress_callback,
        stage="architect",
        message="Designing dataset schema",
        detail={"goal": goal},
    )
    architect = DatasetArchitect(llm_gateway=llm_gateway)
    blueprint = architect.design(goal, forbidden_domains=domain_blacklist)

    predictive_builder = PredictiveDatasetBuilder(
        goal=goal,
        dataset_name=blueprint.dataset_name,
        starting_urls=blueprint.starting_urls,
        domain_blacklist=domain_blacklist,
    )
    if predictive_builder.is_applicable():
        _notify(
            progress_callback,
            stage="predictive_builder",
            message="Attempting direct predictive dataset assembly",
            detail={"dataset_name": blueprint.dataset_name},
        )
        predictive_result = predictive_builder.build()
        if predictive_result:
            predictive_records = predictive_result.records
            target_field = blueprint.row_schema.target_field
            if predictive_records and _predictive_result_matches_goal(
                predictive_result.dataframe,
                goal=goal,
                target_field=target_field,
            ):
                formatter = MLFormatter(predictive_records, blueprint.dataset_name)
                formatter.handle_missing_values()
                csv_path, parquet_path = formatter.export(output_dir=output_dir_path)
                profile_path = formatter.export_profile(
                    goal=goal,
                    provenance_map=predictive_result.provenance_map,
                    llm_gateway=llm_gateway,
                    output_dir=output_dir_path,
                )
                return PipelineArtifacts(
                    dataset_name=blueprint.dataset_name,
                    csv_path=str(csv_path.resolve()),
                    parquet_path=str(parquet_path.resolve()),
                    profile_path=str(profile_path.resolve()),
                    rows=len(predictive_records),
                    columns=len(predictive_records[0]),
                )

    row_model = json_schema_to_pydantic_model(
        schema_to_json_schema(blueprint.row_schema),
        model_name=blueprint.row_schema.title,
    )
    scrape_row_model = relax_pydantic_model(
        row_model,
        model_name=f"{blueprint.row_schema.title}ScrapePartial",
    )

    _notify(
        progress_callback,
        stage="swarm",
        message="Dispatching scraping swarm",
        detail={"target_records": blueprint.target_record_count},
    )
    if checkpoint_manager.exists():
        LOGGER.info(
            "Detected checkpoint cache at %s; swarm will attempt resume",
            checkpoint_manager.cache_path,
        )
    swarm = SwarmDispatcher(
        goal=goal,
        blueprint=blueprint,
        row_model=scrape_row_model,
        agent_count=max(1, max_agents),
        llm_gateway=llm_gateway,
        architect=architect,
        domain_blacklist=domain_blacklist,
        checkpoint_manager=checkpoint_manager,
    )
    raw_records = asyncio.run(swarm.run())
    LOGGER.info("Swarm returned %d raw records", len(raw_records))
    if not raw_records:
        raise RuntimeError("Swarm returned 0 records. Pipeline failed.")

    _notify(
        progress_callback,
        stage="synthesizer",
        message="Synthesizing clean entities",
        detail={"raw_records": len(raw_records)},
    )
    synthesizer = DataSynthesizer(row_model=row_model, llm_gateway=llm_gateway)
    clean_records = synthesizer.synthesize(goal, raw_records)
    LOGGER.info("Synthesizer returned %d clean records", len(clean_records))
    if not clean_records or _records_look_like_ui_chrome(clean_records):
        raise RuntimeError("Clean records failed quality checks. Pipeline failed.")

    _notify(
        progress_callback,
        stage="formatter",
        message="Formatting ML-ready dataset",
        detail={"clean_records": len(clean_records)},
    )
    formatter = MLFormatter(clean_records, blueprint.dataset_name)
    formatter.handle_missing_values()
    csv_path, parquet_path = formatter.export(output_dir=output_dir_path)
    profile_path = formatter.export_profile(
        goal=goal,
        provenance_map=_fallback_provenance_map(clean_records),
        llm_gateway=llm_gateway,
        output_dir=output_dir_path,
    )
    _log_cardinality_gap(goal=goal, actual_rows=len(clean_records))
    return PipelineArtifacts(
        dataset_name=blueprint.dataset_name,
        csv_path=str(csv_path.resolve()),
        parquet_path=str(parquet_path.resolve()),
        profile_path=str(profile_path.resolve()),
        rows=len(clean_records),
        columns=len(clean_records[0]),
    )


def _notify(
    callback: ProgressCallback | None,
    *,
    stage: str,
    message: str,
    detail: dict[str, Any] | None = None,
) -> None:
    if callback is not None:
        callback(stage, message, detail)


def _records_look_like_ui_chrome(records: list[dict[str, object]]) -> bool:
    blocked_phrases = (
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
        "radio",
        "appearance",
        "page tools",
        "personal tools",
        "views",
        "site",
        "tools",
        "thumbnail for",
    )
    suspicious = 0
    for record in records[:20]:
        for value in record.values():
            if isinstance(value, str) and any(
                phrase in value.lower() for phrase in blocked_phrases
            ):
                suspicious += 1
                break
    return suspicious >= max(1, min(len(records), 5) // 2)


def _predictive_result_matches_goal(dataframe, *, goal: str, target_field: str | None) -> bool:
    import re

    columns = {str(column).lower() for column in dataframe.columns}
    markers = set()
    if target_field:
        markers.update(
            token
            for token in re.findall(r"[a-z0-9]+", target_field.lower())
            if len(token) >= 3
        )
    markers.update(
        token
        for token in re.findall(r"[a-z0-9]+", goal.lower())
        if token
        in {
            "salary",
            "valuation",
            "revenue",
            "income",
            "profit",
            "growth",
            "spend",
            "transfer",
            "playoff",
            "population",
            "gdp",
        }
    )
    if not markers:
        return True
    if any(marker in column for marker in markers for column in columns):
        return True
    if "growth" in markers and any(
        token in column for token in {"change", "growth"} for column in columns
    ):
        return True
    return False


def _fallback_provenance_map(records: list[dict[str, object]]) -> dict[str, str]:
    if not records:
        return {}
    source_ref = "synthesized"
    first = records[0]
    if isinstance(first.get("source_url"), str) and first["source_url"]:
        source_ref = str(first["source_url"])
    return {column: source_ref for column in first.keys()}


def _log_cardinality_gap(*, goal: str, actual_rows: int) -> None:
    cardinality = infer_goal_cardinality(goal)
    if cardinality is None:
        return
    expected_rows = cardinality.count
    if actual_rows >= expected_rows:
        return
    LOGGER.warning(
        "Dataset row count (%d) is below the goal-implied expected count (%d, %s)",
        actual_rows,
        expected_rows,
        cardinality.reason or "goal_cardinality",
    )
