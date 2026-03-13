"""Application settings and runtime validation."""

from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path
import shutil
from typing import Literal

from env_utils import ENV_FILE_PATH, read_env_file
from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    class BaseSettings(BaseModel):
        """Minimal fallback when pydantic-settings is unavailable."""

        model_config = ConfigDict(extra="ignore")

        def __init__(self, **data: object) -> None:
            merged = read_env_file(ENV_FILE_PATH, lowercase_keys=True)
            for field_name in self.__class__.model_fields:
                env_name = field_name.upper()
                if env_name in os.environ:
                    merged[field_name] = os.environ[env_name]
            merged.update(data)
            super().__init__(**merged)

    def SettingsConfigDict(**kwargs: object) -> ConfigDict:
        return ConfigDict(**kwargs)

class AppSettings(BaseSettings):
    """Shared typed settings for CLI, API, and worker processes."""

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "production", "test"] = "development"
    redis_url: str = "redis://redis:6379/0"
    enable_background_jobs: bool = False
    artifact_root: Path = Field(default=Path("artifacts"))
    max_agents_default: int = 2
    frontend_origin: str | None = None
    allow_vercel_preview_origins: bool = False

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    openai_model: str | None = None
    openai_base_url: str | None = None
    openrouter_site_url: str = "https://localhost"
    openrouter_app_name: str = "Autonomous Data Engineer"
    enable_semantic_feature_pruning: bool = False

    agent_browser_bin: str = "agent-browser"
    agent_browser_native: bool = True
    agent_browser_cdp_url: str | None = None
    browserbase_api_key: str | None = None
    browserbase_project_id: str | None = None
    browserbase_cdp_url: str | None = None

    llm_tracing_backend: Literal["none", "langsmith", "phoenix"] = "none"
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None
    langsmith_project: str = "web-scraper"
    phoenix_collector_endpoint: str | None = None
    phoenix_project_name: str = "web-scraper"

    @model_validator(mode="after")
    def _validate_partial_browserbase_config(self) -> "AppSettings":
        if bool(self.browserbase_api_key) ^ bool(self.browserbase_project_id):
            raise ValueError(
                "BROWSERBASE_API_KEY and BROWSERBASE_PROJECT_ID must be provided together."
            )
        return self

    @property
    def jobs_dir(self) -> Path:
        return self.artifact_root / "jobs"

    @property
    def has_llm_api_key(self) -> bool:
        return any(
            value
            for value in (
                self.openai_api_key,
                self.gemini_api_key,
                self.groq_api_key,
            )
        )

    def validate_for_service(self) -> None:
        if not self.has_llm_api_key:
            logging.getLogger(__name__).warning(
                "No LLM API key configured. Service will still start, but LLM-backed jobs may fail."
            )
        if "/" in self.agent_browser_bin:
            browser_available = Path(self.agent_browser_bin).exists()
        else:
            browser_available = shutil.which(self.agent_browser_bin) is not None
        if not browser_available:
            logging.getLogger(__name__).warning(
                "AGENT_BROWSER_BIN is not installed or not on PATH. Browser fallback may be unavailable."
            )

    @property
    def cors_origins(self) -> list[str]:
        origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
        if self.frontend_origin and self.frontend_origin.strip():
            origins.append(self.frontend_origin.strip().rstrip("/"))
        return list(dict.fromkeys(origins))

    @property
    def cors_origin_regex(self) -> str | None:
        if self.allow_vercel_preview_origins or self.app_env != "production":
            return r"https://.*\.vercel\.app"
        return None

    @property
    def running_on_vercel(self) -> bool:
        return os.environ.get("VERCEL") == "1" or bool(os.environ.get("VERCEL_ENV"))


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
