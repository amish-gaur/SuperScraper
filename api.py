"""FastAPI application for submitting and monitoring scraping jobs."""

from __future__ import annotations

import logging
import json
from pathlib import Path
import socket
import threading
from typing import Literal
from urllib.parse import urlparse

import pandas as pd
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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
FRONTEND_DIST_DIR = Path(__file__).parent / "frontend" / "dist"
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class JobPreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, object]]


class JobLogResponse(BaseModel):
    lines: list[str]


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
                error_message = str(task_result.result)
                store.mark_failure(job_id, error=error_message)
                payload = store.read(job_id)
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
    artifact_name: Literal["csv", "parquet", "profile", "validation"],
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
        "validation": "validation_path",
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
        "validation": "application/json",
    }[artifact_name]
    return FileResponse(artifact_path, media_type=media_type, filename=artifact_path.name)


@app.get("/jobs/{job_id}/profile")
def get_profile(job_id: str) -> dict[str, object]:
    _assert_job_completed(job_id)
    profile_path = _artifact_path_for(job_id, "profile_path")
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Profile artifact is invalid JSON") from exc


@app.get("/jobs/{job_id}/preview", response_model=JobPreviewResponse)
def get_preview(job_id: str, limit: int = 10) -> JobPreviewResponse:
    _assert_job_completed(job_id)
    csv_path = _artifact_path_for(job_id, "csv_path")
    preview_limit = min(max(limit, 1), 50)
    dataframe = pd.read_csv(csv_path).head(preview_limit)
    dataframe = dataframe.where(pd.notnull(dataframe), None)
    rows = dataframe.to_dict(orient="records")
    return JobPreviewResponse(columns=[str(column) for column in dataframe.columns], rows=rows)


@app.get("/jobs/{job_id}/logs", response_model=JobLogResponse)
def get_job_logs(job_id: str, limit: int = 120) -> JobLogResponse:
    try:
        store.read(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    log_path = store.job_dir(job_id) / "runtime.log"
    if not log_path.exists():
        return JobLogResponse(lines=[])

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail_limit = min(max(limit, 1), 500)
    return JobLogResponse(lines=lines[-tail_limit:])


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
    log_handler = _build_job_log_handler(job_id)
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

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
    finally:
        root_logger.removeHandler(log_handler)
        log_handler.close()

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


def _assert_job_completed(job_id: str) -> dict[str, object]:
    try:
        payload = store.read(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    if payload["status"] != "completed":
        raise HTTPException(status_code=409, detail="Job is not complete yet")
    return payload


def _artifact_path_for(job_id: str, artifact_key: str) -> Path:
    try:
        artifact_path = Path(store.artifact_path(job_id, artifact_key))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")
    return artifact_path


class _ThreadNameFilter(logging.Filter):
    def __init__(self, thread_name: str) -> None:
        super().__init__()
        self.thread_name = thread_name

    def filter(self, record: logging.LogRecord) -> bool:
        return record.threadName == self.thread_name


def _build_job_log_handler(job_id: str) -> logging.Handler:
    thread_name = f"job-{job_id[:8]}"
    log_path = store.job_dir(job_id) / "runtime.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.addFilter(_ThreadNameFilter(thread_name))
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    return handler


if FRONTEND_DIST_DIR.exists():
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/", response_class=HTMLResponse)
    def serve_frontend_index() -> HTMLResponse:
        return HTMLResponse((FRONTEND_DIST_DIR / "index.html").read_text(encoding="utf-8"))


    @app.get("/{full_path:path}", response_class=HTMLResponse)
    def serve_frontend_routes(full_path: str) -> HTMLResponse:
        if full_path.startswith(("jobs", "docs", "openapi.json", "redoc", "healthz")):
            raise HTTPException(status_code=404, detail="Not found")
        return HTMLResponse((FRONTEND_DIST_DIR / "index.html").read_text(encoding="utf-8"))
