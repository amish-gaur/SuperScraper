"""FastAPI application for submitting and monitoring scraping jobs."""

from __future__ import annotations

import logging
from pathlib import Path
import socket
import threading
from typing import Literal
from urllib.parse import urlparse

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from celery_app import celery_app
from env_utils import load_env_into_process
from job_store import JobStore
from llm import LLMError, LLMGateway
from pipeline_service import run_pipeline
from settings import get_settings
from tracing import configure_llm_tracing
from worker import run_scrape_job


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

load_env_into_process()
settings = get_settings()
settings.validate_for_service()
configure_llm_tracing(settings)
store = JobStore(settings)
app = FastAPI(title="Web Scraper API", version="1.0.0")


class JobCreateRequest(BaseModel):
    goal: str = Field(min_length=1, description="High-level machine learning data goal.")
    max_agents: int = Field(default_factory=lambda: settings.max_agents_default, ge=1, le=10)

    @field_validator("goal")
    @classmethod
    def _strip_goal(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("goal must not be empty")
        return cleaned


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    goal: str
    max_agents: int
    progress: dict[str, object] | None
    artifacts: dict[str, object]
    error: str | None


@app.get("/healthz")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/jobs", response_model=JobCreateResponse, status_code=status.HTTP_202_ACCEPTED)
def create_job(request: JobCreateRequest) -> JobCreateResponse:
    job = store.create_job(goal=request.goal, max_agents=request.max_agents)
    if _celery_broker_is_reachable(settings.redis_url):
        try:
            run_scrape_job.apply_async(
                kwargs={
                    "job_id": job["job_id"],
                    "goal": job["goal"],
                    "max_agents": job["max_agents"],
                },
                task_id=job["job_id"],
            )
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Falling back to in-process job runner for %s because Celery submission failed: %s",
                job["job_id"],
                exc,
            )
            _launch_in_process_job(
                job_id=job["job_id"],
                goal=job["goal"],
                max_agents=job["max_agents"],
            )
    else:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Falling back to in-process job runner for %s because Redis broker %r is unreachable",
            job["job_id"],
            settings.redis_url,
        )
        _launch_in_process_job(
            job_id=job["job_id"],
            goal=job["goal"],
            max_agents=job["max_agents"],
        )
    return JobCreateResponse(job_id=job["job_id"], status=job["status"])


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    try:
        payload = store.read(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if payload["status"] in {"queued", "running"} and _celery_broker_is_reachable(settings.redis_url):
        try:
            task_result = AsyncResult(job_id, app=celery_app)
            if task_result.failed():
                payload["status"] = "failed"
                payload["error"] = str(task_result.result)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Unable to query Celery task state for %s: %s",
                job_id,
                exc,
            )

    return JobStatusResponse(**payload)


@app.get("/jobs/{job_id}/download/{artifact_name}")
def download_artifact(
    job_id: str,
    artifact_name: Literal["csv", "parquet", "profile"],
) -> FileResponse:
    try:
        payload = store.read(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if payload["status"] != "completed":
        raise HTTPException(status_code=409, detail="Job is not complete yet")

    artifact_key = {
        "csv": "csv_path",
        "parquet": "parquet_path",
        "profile": "profile_path",
    }[artifact_name]
    try:
        artifact_path = Path(store.artifact_path(job_id, artifact_key))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc

    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    media_type = {
        "csv": "text/csv",
        "parquet": "application/octet-stream",
        "profile": "application/json",
    }[artifact_name]
    return FileResponse(artifact_path, media_type=media_type, filename=artifact_path.name)


def _launch_in_process_job(*, job_id: str, goal: str, max_agents: int) -> None:
    thread = threading.Thread(
        target=_run_job_in_process,
        kwargs={
            "job_id": job_id,
            "goal": goal,
            "max_agents": max_agents,
        },
        name=f"job-{job_id[:8]}",
        daemon=True,
    )
    thread.start()


def _run_job_in_process(*, job_id: str, goal: str, max_agents: int) -> None:
    logger = logging.getLogger(__name__)
    store.mark_started(job_id)

    def on_progress(stage: str, message: str, detail: dict[str, object] | None) -> None:
        store.mark_progress(job_id, stage=stage, message=message, detail=detail)

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
    except (LLMError, ValueError, RuntimeError) as exc:
        store.mark_failure(job_id, error=str(exc))
        logger.exception("In-process job %s failed", job_id)
        return
    except Exception as exc:
        store.mark_failure(job_id, error=f"Unhandled pipeline error: {exc}")
        logger.exception("In-process job %s failed", job_id)
        return

    payload = {
        "dataset_name": artifacts.dataset_name,
        "csv_path": artifacts.csv_path,
        "parquet_path": artifacts.parquet_path,
        "profile_path": artifacts.profile_path,
        "rows": artifacts.rows,
        "columns": artifacts.columns,
    }
    store.mark_success(job_id, artifacts=payload)


def _celery_broker_is_reachable(redis_url: str) -> bool:
    parsed = urlparse(redis_url)
    host = parsed.hostname
    port = parsed.port or 6379
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False
