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
from data_validation import DataValidationError, SemanticDataValidator
from demo_datasets import demo_dataset_for_goal
from formatter import MLFormatter
from goal_intent import infer_goal_cardinality
from llm import LLMGateway
from post_extraction_pruner import PostExtractionPruner
from predictive_dataset_builder import DataQualityError, PredictiveDatasetBuilder
from settings import get_settings
from source_memory import SourceMemory
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
    validation_path: str
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
    settings = get_settings()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = CheckpointManager(
        checkpoint_path or output_dir_path / "pipeline_cache.json"
    )
    source_memory = SourceMemory()

    _notify(
        progress_callback,
        stage="architect",
        message="Designing dataset schema",
        detail={"goal": goal},
    )
    architect = DatasetArchitect(llm_gateway=llm_gateway)
    blueprint = architect.design(goal, forbidden_domains=domain_blacklist)
    row_model = json_schema_to_pydantic_model(
        schema_to_json_schema(blueprint.row_schema),
        model_name=blueprint.row_schema.title,
    )
    validator = SemanticDataValidator(row_model=row_model)
    post_extraction_pruner = PostExtractionPruner(
        goal=goal,
        target_field=blueprint.row_schema.target_field,
        core_feature_fields=[
            field.name
            for field in blueprint.row_schema.fields
            if field.ml_role == "feature"
        ],
        required_feature_fields=[
            field.name
            for field in blueprint.row_schema.fields
            if field.ml_role == "feature" and not field.nullable
        ],
        llm_gateway=llm_gateway,
    )

    predictive_builder = PredictiveDatasetBuilder(
        goal=goal,
        dataset_name=blueprint.dataset_name,
        starting_urls=blueprint.starting_urls,
        target_field=blueprint.row_schema.target_field,
        core_feature_fields=[
            field.name
            for field in blueprint.row_schema.fields
            if field.ml_role == "feature"
        ],
        required_feature_fields=[
            field.name
            for field in blueprint.row_schema.fields
            if field.ml_role == "feature" and not field.nullable
        ],
        domain_blacklist=domain_blacklist,
        llm_gateway=llm_gateway,
        post_extraction_pruner=post_extraction_pruner,
        progress_callback=lambda message, detail: _notify(
            progress_callback,
            stage="predictive_builder",
            message=message,
            detail=detail,
        ),
    )
    if predictive_builder.is_applicable():
        _notify(
            progress_callback,
            stage="predictive_builder",
            message="Attempting direct predictive dataset assembly",
            detail={"dataset_name": blueprint.dataset_name},
        )
        try:
            predictive_result = predictive_builder.build()
        except DataQualityError as exc:
            LOGGER.warning("Predictive dataset assembly rejected by quality gate: %s", exc)
            _notify(
                progress_callback,
                stage="predictive_builder",
                message="Direct predictive dataset rejected by quality gate; falling back to swarm",
                detail={"error": str(exc)},
            )
            predictive_result = None
        if predictive_result:
            predictive_records = predictive_result.records
            target_field = blueprint.row_schema.target_field
            if predictive_records and _predictive_result_matches_goal(
                predictive_result.dataframe,
                goal=goal,
                target_field=target_field,
            ):
                _notify(
                    progress_callback,
                    stage="validation",
                    message="Validating dataset semantics",
                    detail={"candidate_rows": len(predictive_records)},
                )
                try:
                    validation_result = validator.validate(
                        predictive_result.dataframe,
                        dataset_name=blueprint.dataset_name,
                    )
                except DataValidationError as exc:
                    raise RuntimeError(
                        f"Predictive dataset rejected by semantic validation: {exc}"
                    ) from exc
                formatter = MLFormatter(
                    validation_result.dataframe.to_dict(orient="records"),
                    blueprint.dataset_name,
                )
                formatter.handle_missing_values()
                csv_path, parquet_path = formatter.export(output_dir=output_dir_path)
                profile_path = formatter.export_profile(
                    goal=goal,
                    provenance_map=predictive_result.provenance_map,
                    pruning_audit=predictive_result.pruning_audit,
                    llm_gateway=llm_gateway,
                    output_dir=output_dir_path,
                )
                validation_path = validation_result.report.write(output_dir=output_dir_path)
                source_memory.record_success(goal, blueprint.starting_urls)
                return PipelineArtifacts(
                    dataset_name=blueprint.dataset_name,
                    csv_path=str(csv_path.resolve()),
                    parquet_path=str(parquet_path.resolve()),
                    profile_path=str(profile_path.resolve()),
                    validation_path=str(validation_path.resolve()),
                    rows=len(validation_result.dataframe),
                    columns=len(validation_result.dataframe.columns),
                )

    demo_dataset = demo_dataset_for_goal(goal)
    if demo_dataset is not None:
        LOGGER.warning(
            "Falling back to deterministic demo dataset for hosted environment: %s",
            demo_dataset.source_label,
        )
        _notify(
            progress_callback,
            stage="formatter",
            message="Using hosted demo fallback dataset",
            detail={"source": demo_dataset.source_label},
        )
        formatter = MLFormatter(demo_dataset.records, blueprint.dataset_name)
        _notify(
            progress_callback,
            stage="validation",
            message="Validating demo fallback dataset",
            detail={"candidate_rows": len(formatter.dataframe)},
        )
        validation_result = validator.validate(
            formatter.dataframe,
            dataset_name=blueprint.dataset_name,
        )
        formatter.dataframe = validation_result.dataframe
        formatter.handle_missing_values()
        csv_path, parquet_path = formatter.export(output_dir=output_dir_path)
        profile_path = formatter.export_profile(
            goal=goal,
            provenance_map=demo_dataset.provenance_map,
            pruning_audit={"mode": "demo_fallback"},
            llm_gateway=llm_gateway,
            output_dir=output_dir_path,
        )
        validation_path = validation_result.report.write(output_dir=output_dir_path)
        return PipelineArtifacts(
            dataset_name=blueprint.dataset_name,
            csv_path=str(csv_path.resolve()),
            parquet_path=str(parquet_path.resolve()),
            profile_path=str(profile_path.resolve()),
            validation_path=str(validation_path.resolve()),
            rows=len(validation_result.dataframe),
            columns=len(validation_result.dataframe.columns),
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
        stage="post_processing",
        message="Pruning synthesized dataset",
        detail={"clean_records": len(clean_records)},
    )
    fallback_provenance = _fallback_provenance_map(clean_records)
    formatter = MLFormatter(clean_records, blueprint.dataset_name)
    pruning_result = post_extraction_pruner.process(
        formatter.dataframe,
        provenance_map=fallback_provenance,
    )
    formatter.dataframe = pruning_result.dataframe
    try:
        predictive_builder._enforce_fill_rate(formatter.dataframe)
    except DataQualityError as exc:
        raise RuntimeError(
            f"Final dataset rejected due to low fill rate; refusing to export artifacts: {exc}"
        ) from exc
    _notify(
        progress_callback,
        stage="validation",
        message="Validating dataset semantics",
        detail={"candidate_rows": len(formatter.dataframe)},
    )
    try:
        validation_result = validator.validate(
            formatter.dataframe,
            dataset_name=blueprint.dataset_name,
            raw_records=raw_records,
        )
    except DataValidationError as exc:
        raise RuntimeError(f"Final dataset rejected by semantic validation: {exc}") from exc
    formatter.dataframe = validation_result.dataframe
    formatter.handle_missing_values()
    csv_path, parquet_path = formatter.export(output_dir=output_dir_path)
    profile_path = formatter.export_profile(
        goal=goal,
        provenance_map=pruning_result.provenance_map,
        pruning_audit=pruning_result.pruning_audit,
        llm_gateway=llm_gateway,
        output_dir=output_dir_path,
    )
    validation_path = validation_result.report.write(output_dir=output_dir_path)
    source_memory.record_success(goal, blueprint.starting_urls)
    _log_cardinality_gap(goal=goal, actual_rows=len(validation_result.dataframe))
    return PipelineArtifacts(
        dataset_name=blueprint.dataset_name,
        csv_path=str(csv_path.resolve()),
        parquet_path=str(parquet_path.resolve()),
        profile_path=str(profile_path.resolve()),
        validation_path=str(validation_path.resolve()),
        rows=len(validation_result.dataframe),
        columns=len(validation_result.dataframe.columns),
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
