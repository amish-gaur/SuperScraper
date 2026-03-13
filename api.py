"""FastAPI application for submitting and monitoring scraping jobs."""

from __future__ import annotations

import logging
import json
from pathlib import Path
import re
import threading
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from env_utils import load_env_into_process
from job_store import JobStore
from llm import LLMError, LLMGateway
from pipeline_service import run_pipeline
from settings import get_settings
from tracing import configure_llm_tracing
from swarm import SwarmAbortError


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
    allow_origin_regex=settings.cors_origin_regex,
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
    status: Literal["queued", "running", "completed", "failed", "partial_success"]
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
    normalized_goal = _normalize_predictive_goal(request.goal)
    job = store.create_job(
        goal=request.goal,
        max_agents=request.max_agents,
        normalized_goal=normalized_goal,
    )
    if settings.running_on_vercel:
        _run_job_inline(
            job_id=job["job_id"],
            goal=job["normalized_goal"],
            max_agents=job["max_agents"],
        )
        refreshed_job = store.read(job["job_id"])
        return JobCreateResponse(job_id=job["job_id"], status=refreshed_job["status"])

    _launch_in_process_job(
        job_id=job["job_id"],
        goal=job["normalized_goal"],
        max_agents=job["max_agents"],
    )
    return JobCreateResponse(job_id=job["job_id"], status=job["status"])


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    try:
        payload = store.read(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    return JobStatusResponse(**payload)


@app.get("/jobs/{job_id}/download/{artifact_name}")
@app.head("/jobs/{job_id}/download/{artifact_name}", include_in_schema=False)
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
    thread_name = f"job-{job_id[:8]}"
    thread = threading.Thread(
        target=_run_job_with_lifecycle,
        kwargs={
            "job_id": job_id,
            "goal": goal,
            "max_agents": max_agents,
            "log_thread_name": thread_name,
        },
        name=thread_name,
        daemon=True,
    )
    thread.start()


def _run_job_inline(*, job_id: str, goal: str, max_agents: int) -> None:
    _run_job_with_lifecycle(
        job_id=job_id,
        goal=goal,
        max_agents=max_agents,
        log_thread_name=threading.current_thread().name,
    )


def _run_job_with_lifecycle(
    *,
    job_id: str,
    goal: str,
    max_agents: int,
    log_thread_name: str,
) -> None:
    logger = logging.getLogger(__name__)
    store.mark_started(job_id)
    log_handler = _build_job_log_handler(job_id, thread_name=log_thread_name)
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
    except SwarmAbortError as exc:
        if exc.partial_records > 0:
            store.mark_partial_success(
                job_id,
                message=str(exc),
                detail=exc.detail,
            )
        else:
            store.mark_failure(job_id, error=str(exc))
        logger.exception("In-process job %s aborted by fail-fast protection", job_id)
        return
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


def _build_job_log_handler(job_id: str, *, thread_name: str) -> logging.Handler:
    log_path = store.job_dir(job_id) / "runtime.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.addFilter(_ThreadNameFilter(thread_name))
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    return handler


def _normalize_predictive_goal(goal: str) -> str:
    cleaned = re.sub(r"\s+", " ", goal).strip()
    if not cleaned:
        return cleaned

    lowered = cleaned.lower()
    if any(token in lowered for token in ("dataset", "table", "rows", "columns", "features")):
        return cleaned

    target = _extract_prediction_target(cleaned)
    if target is None:
        return cleaned

    return (
        f"Build a predictive dataset to predict {target}. "
        "Infer the most useful feature columns, choose sensible row entities, "
        "find public web sources, validate the scraped data, and return a clean modeling table."
    )


def _extract_prediction_target(goal: str) -> str | None:
    patterns = [
        r"^\s*i want to predict\s+(?P<target>.+?)\s*$",
        r"^\s*i'?d like to predict\s+(?P<target>.+?)\s*$",
        r"^\s*help me predict\s+(?P<target>.+?)\s*$",
        r"^\s*predict\s+(?P<target>.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, goal, flags=re.IGNORECASE)
        if not match:
            continue
        target = match.group("target").strip().rstrip(".!?")
        return target or None
    return None


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
