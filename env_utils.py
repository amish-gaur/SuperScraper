"""Shared environment-file and provider key helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


ENV_FILE_PATH = Path(".env.local")
API_KEY_ENV_VARS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")
PLACEHOLDER_API_KEYS = {"your_api_key_here", "replace_me", "changeme"}


def read_env_file(path: Path = ENV_FILE_PATH, *, lowercase_keys: bool = False) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip().lower() if lowercase_keys else key.strip()
        values[normalized_key] = value.strip().strip("'\"")
    return values


def load_env_into_process(path: Path = ENV_FILE_PATH) -> None:
    """Populate os.environ from the local env file without clobbering usable values."""
    for key, value in read_env_file(path).items():
        if not key or not value:
            continue
        if key in API_KEY_ENV_VARS:
            current = os.environ.get(key)
            if not env_value_is_usable(current, key=key):
                os.environ[key] = value
            continue
        if key not in os.environ:
            os.environ[key] = value


def env_value_is_usable(value: str | None, *, key: str) -> bool:
    if not value:
        return False
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in PLACEHOLDER_API_KEYS:
        return False
    if any(character.isspace() for character in cleaned):
        return False
    if any(ord(character) < 32 for character in cleaned):
        return False
    if key == "GROQ_API_KEY":
        return cleaned.startswith("gsk_")
    if key == "GEMINI_API_KEY":
        return cleaned.startswith("AI")
    return True


def configured_api_key_present(env: dict[str, str] | None = None, *, env_vars: Iterable[str] = API_KEY_ENV_VARS) -> bool:
    source = env if env is not None else {}
    for env_var in env_vars:
        value = source.get(env_var) if env is not None else None
        if value is None and env is None:
            value = os.getenv(env_var, "")
        if value is None:
            continue
        if env_value_is_usable(str(value).strip(), key=env_var):
            return True
    return False
