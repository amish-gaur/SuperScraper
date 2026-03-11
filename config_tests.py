"""Regression tests for shared environment parsing and key detection."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from env_utils import configured_api_key_present, env_value_is_usable, read_env_file
from main import load_local_env


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
    ]
    for test in tests:
        test()
        if verbose:
            print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
