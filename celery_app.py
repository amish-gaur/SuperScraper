"""Celery application definition."""

from __future__ import annotations

from celery import Celery

from env_utils import load_env_into_process
from settings import get_settings


load_env_into_process()
settings = get_settings()
celery_app = Celery(
    "web_scraper",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86400,
    timezone="UTC",
)
