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


def test_api_create_job_runs_inline_on_vercel() -> None:
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_browser_bin = os.environ.get("AGENT_BROWSER_BIN")
    original_vercel = os.environ.get("VERCEL")
    try:
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["AGENT_BROWSER_BIN"] = "sh"
        os.environ["VERCEL"] = "1"
        get_settings.cache_clear()
        api = importlib.import_module("api")

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = AppSettings(
                artifact_root=Path(temp_dir),
                openai_api_key="sk-test",
                agent_browser_bin="sh",
            )
            store = JobStore(settings)

            api.store = store
            api.settings = settings

            def fake_run_job_inline(*, job_id: str, goal: str, max_agents: int) -> None:
                store.mark_started(job_id)
                store.mark_success(
                    job_id,
                    artifacts={
                        "dataset_name": "demo",
                        "csv_path": str(Path(temp_dir) / "demo.csv"),
                        "parquet_path": str(Path(temp_dir) / "demo.parquet"),
                        "profile_path": str(Path(temp_dir) / "profile.json"),
                        "validation_path": str(Path(temp_dir) / "validation.json"),
                        "rows": 1,
                        "columns": 1,
                    },
                )

            api._run_job_inline = fake_run_job_inline
            response = api.create_job(api.JobCreateRequest(goal="x", max_agents=1))
            persisted = store.read(response.job_id)

            assert response.status == "completed"
            assert persisted["status"] == "completed"
    finally:
        get_settings.cache_clear()
        _restore_env_var("OPENAI_API_KEY", original_openai_key)
        _restore_env_var("AGENT_BROWSER_BIN", original_browser_bin)
        _restore_env_var("VERCEL", original_vercel)


def test_api_normalizes_short_predictive_goals() -> None:
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

            api.store = store
            api.settings = settings

            captured: dict[str, object] = {}

            def fake_launch(*, job_id: str, goal: str, max_agents: int) -> None:
                captured["job_id"] = job_id
                captured["goal"] = goal
                captured["max_agents"] = max_agents

            api._launch_in_process_job = fake_launch
            response = api.create_job(api.JobCreateRequest(goal="I want to predict NBA player salary", max_agents=2))
            persisted = store.read(response.job_id)

            assert response.status == "queued"
            assert persisted["goal"] == "I want to predict NBA player salary"
            assert persisted["normalized_goal"].startswith("Build a predictive dataset to predict NBA player salary.")
            assert captured["goal"] == persisted["normalized_goal"]
            assert captured["max_agents"] == 2
    finally:
        get_settings.cache_clear()
        _restore_env_var("OPENAI_API_KEY", original_openai_key)
        _restore_env_var("AGENT_BROWSER_BIN", original_browser_bin)


def test_settings_disable_vercel_preview_origins_in_production_by_default() -> None:
    settings = AppSettings(app_env="production")
    assert settings.cors_origin_regex is None


def test_settings_allow_vercel_preview_origins_outside_production() -> None:
    settings = AppSettings(app_env="development")
    assert settings.cors_origin_regex == r"https://.*\.vercel\.app"


def test_worker_persists_validation_artifact_path() -> None:
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    original_browser_bin = os.environ.get("AGENT_BROWSER_BIN")
    try:
        os.environ["OPENAI_API_KEY"] = "sk-test"
        os.environ["AGENT_BROWSER_BIN"] = "sh"
        get_settings.cache_clear()

        import worker

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = AppSettings(
                artifact_root=Path(temp_dir),
                openai_api_key="sk-test",
                agent_browser_bin="sh",
            )
            store = JobStore(settings)
            job = store.create_job(goal="x", max_agents=1)

            worker.settings = settings

            class FakeCurrentTask:
                def update_state(self, **kwargs: object) -> None:
                    return None

            class FakeArtifacts:
                dataset_name = "demo"
                csv_path = str(Path(temp_dir) / "demo.csv")
                parquet_path = str(Path(temp_dir) / "demo.parquet")
                profile_path = str(Path(temp_dir) / "profile.json")
                validation_path = str(Path(temp_dir) / "validation.json")
                rows = 1
                columns = 1

            worker.current_task = FakeCurrentTask()
            worker.run_pipeline = lambda **kwargs: FakeArtifacts()
            worker.run_scrape_job.run(
                job_id=job["job_id"],
                goal=job["goal"],
                max_agents=job["max_agents"],
            )

            persisted = store.read(job["job_id"])
            assert persisted["artifacts"]["validation_path"] == FakeArtifacts.validation_path
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
        test_api_create_job_runs_inline_on_vercel,
        test_api_normalizes_short_predictive_goals,
        test_settings_disable_vercel_preview_origins_in_production_by_default,
        test_settings_allow_vercel_preview_origins_outside_production,
        test_worker_persists_validation_artifact_path,
    ]
    for test in tests:
        test()
        if verbose:
            print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
