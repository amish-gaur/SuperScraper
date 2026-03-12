"""Regression tests for shared environment parsing and key detection."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import tempfile

from env_utils import configured_api_key_present, env_value_is_usable, read_env_file
from job_store import JobStore
from main import load_local_env
from settings import AppSettings, get_settings


def test_read_env_file_supports_case_preserving_and_lowercase_modes() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / ".env.local"
        path.write_text("OPENAI_API_KEY='sk-test'\nOPENAI_MODEL='gpt-test'\n", encoding="utf-8")
        assert read_env_file(path) == {"OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "gpt-test"}
        assert read_env_file(path, lowercase_keys=True) == {
            "openai_api_key": "sk-test",
            "openai_model": "gpt-test",
        }


def test_load_local_env_replaces_placeholder_keys_only() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / ".env.local"
        path.write_text("OPENAI_API_KEY='sk-file'\nOPENAI_MODEL='gpt-test'\n", encoding="utf-8")
        original_key = os.environ.get("OPENAI_API_KEY")
        original_model = os.environ.get("OPENAI_MODEL")
        try:
            os.environ["OPENAI_API_KEY"] = "replace_me"
            os.environ.pop("OPENAI_MODEL", None)
            load_local_env(path)
            assert os.environ["OPENAI_API_KEY"] == "sk-file"
            assert os.environ["OPENAI_MODEL"] == "gpt-test"
        finally:
            _restore_env_var("OPENAI_API_KEY", original_key)
            _restore_env_var("OPENAI_MODEL", original_model)


def test_configured_api_key_present_uses_shared_validation_rules() -> None:
    assert configured_api_key_present(
        {
            "OPENAI_API_KEY": "replace_me",
            "GEMINI_API_KEY": "AI_valid_provider_key",
        }
    )
    assert not configured_api_key_present({"OPENAI_API_KEY": "replace_me"})
    assert not configured_api_key_present({})
    assert env_value_is_usable("gsk_valid", key="GROQ_API_KEY")
    assert not env_value_is_usable("bad key", key="OPENAI_API_KEY")


def test_api_get_job_persists_failed_celery_state() -> None:
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_browser_bin = os.environ.get("AGENT_BROWSER_BIN")
    try:
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["AGENT_BROWSER_BIN"] = "sh"
        get_settings.cache_clear()
        api = importlib.import_module("api")

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = AppSettings(
                artifact_root=Path(temp_dir),
                openai_api_key="sk-test",
                agent_browser_bin="sh",
            )
            store = JobStore(settings)
            job = store.create_job(goal="x", max_agents=1)
            store.mark_started(job["job_id"])

            api.store = store
            api.settings = settings
            api._celery_broker_is_reachable = lambda redis_url: True

            class FailedResult:
                result = RuntimeError("boom")

                def failed(self) -> bool:
                    return True

            api.AsyncResult = lambda job_id, app=None: FailedResult()
            payload = api.get_job(job["job_id"])
            persisted = store.read(job["job_id"])

            assert payload.status == "failed"
            assert persisted["status"] == "failed"
            assert persisted["error"] == "boom"
    finally:
        get_settings.cache_clear()
        _restore_env_var("OPENAI_API_KEY", original_openai_key)
        _restore_env_var("AGENT_BROWSER_BIN", original_browser_bin)


def _restore_env_var(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def main(*, verbose: bool = True) -> int:
    tests = [
        test_read_env_file_supports_case_preserving_and_lowercase_modes,
        test_load_local_env_replaces_placeholder_keys_only,
        test_configured_api_key_present_uses_shared_validation_rules,
        test_api_get_job_persists_failed_celery_state,
    ]
    for test in tests:
        test()
        if verbose:
            print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
