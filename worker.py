"""Celery tasks for background dataset generation."""

from __future__ import annotations

import logging

from celery import current_task

from celery_app import celery_app
from env_utils import load_env_into_process
from job_store import JobStore
from llm import LLMError, LLMGateway
from pipeline_service import run_pipeline
from settings import get_settings
from tracing import configure_llm_tracing
from swarm import SwarmAbortError


LOGGER = logging.getLogger(__name__)
load_env_into_process()
settings = get_settings()
settings.validate_for_service()
configure_llm_tracing(settings)


@celery_app.task(name="worker.run_scrape_job", bind=True)
def run_scrape_job(self, *, job_id: str, goal: str, max_agents: int) -> dict[str, object]:
    """Run one dataset generation job in the background."""
    store = JobStore(settings)
    store.mark_started(job_id)
    current_task.update_state(
        state="STARTED",
        meta={"job_id": job_id, "stage": "started", "message": "Worker picked up job"},
    )

    def on_progress(stage: str, message: str, detail: dict[str, object] | None) -> None:
        store.mark_progress(job_id, stage=stage, message=message, detail=detail)
        current_task.update_state(
            state="STARTED",
            meta={"job_id": job_id, "stage": stage, "message": message, "detail": detail},
        )

    try:
        llm_gateway = LLMGateway()
        artifacts = run_pipeline(
            goal=goal,
            max_agents=max_agents,
            llm_gateway=llm_gateway,
            output_dir=store.job_dir(job_id),
            checkpoint_path=store.job_dir(job_id) / "pipeline_cache.json",
            progress_callback=on_progress,
        )
    except SwarmAbortError as exc:
        if exc.partial_records > 0:
            store.mark_partial_success(
                job_id,
                message=str(exc),
                detail=exc.detail,
            )
        else:
            store.mark_failure(job_id, error=str(exc))
        LOGGER.exception("Job %s aborted by fail-fast protection", job_id)
        raise
    except (LLMError, ValueError, RuntimeError) as exc:
        store.mark_failure(job_id, error=str(exc))
        LOGGER.exception("Job %s failed", job_id)
        raise
    except Exception as exc:
        store.mark_failure(job_id, error=f"Unhandled pipeline error: {exc}")
        LOGGER.exception("Job %s failed", job_id)
        raise

    payload = {
        "dataset_name": artifacts.dataset_name,
        "csv_path": artifacts.csv_path,
        "parquet_path": artifacts.parquet_path,
        "profile_path": artifacts.profile_path,
        "validation_path": artifacts.validation_path,
        "rows": artifacts.rows,
        "columns": artifacts.columns,
    }
    store.mark_success(job_id, artifacts=payload)
    return payload
