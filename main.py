"""CLI entrypoint for the autonomous data engineering pipeline."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re
import sys

from env_utils import (
    API_KEY_ENV_VARS,
    ENV_FILE_PATH,
    configured_api_key_present,
    env_value_is_usable,
    load_env_into_process,
    read_env_file,
)
from llm import LLMError, LLMGateway
from pipeline_service import run_pipeline
from settings import get_settings
from tracing import configure_llm_tracing

ESCAPE_SEQUENCE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def configure_logging(*, level: str = "INFO") -> None:
    """Configure logging for pipeline progress."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Autonomous Data Engineer pipeline."
    )
    parser.add_argument(
        "--goal",
        help="High-level machine learning data goal.",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=2,
        help="Maximum number of parallel research agents to run.",
    )
    parser.add_argument(
        "--set-api-key",
        action="store_true",
        help="Prompt for an API key, save it to .env.local, and exit.",
    )
    parser.add_argument(
        "--clear-api-key",
        action="store_true",
        help="Remove stored API keys from .env.local and exit.",
    )
    parser.add_argument(
        "--api-key-status",
        action="store_true",
        help="Print the currently configured API key source and exit.",
    )
    parser.add_argument(
        "--set-model",
        action="store_true",
        help="Prompt for a preferred model, save it to .env.local, and exit.",
    )
    parser.add_argument(
        "--clear-model",
        action="store_true",
        help="Remove OPENAI_MODEL from .env.local and exit.",
    )
    parser.add_argument(
        "--model-status",
        action="store_true",
        help="Print the currently configured model and exit.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Interactive setup for API key and preferred model, then exit.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run local deployment and routing checks, then exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity for the current run.",
    )
    args = parser.parse_args(argv)
    if not (
        args.set_api_key
        or args.clear_api_key
        or args.api_key_status
        or args.set_model
        or args.clear_model
        or args.model_status
        or args.setup
        or args.doctor
    ) and not args.goal:
        parser.error("the following arguments are required: --goal")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the four-stage autonomous data engineering pipeline."""
    args = parse_args(argv or sys.argv[1:])
    load_local_env()
    if args.set_api_key:
        return prompt_and_store_api_key()
    if args.setup:
        return interactive_setup()
    if args.clear_api_key:
        return clear_stored_api_keys()
    if args.api_key_status:
        return print_api_key_status()
    if args.set_model:
        return prompt_and_store_model()
    if args.clear_model:
        return clear_stored_model()
    if args.model_status:
        return print_model_status()
    if args.doctor:
        from deployment_checks import format_results, run_deployment_checks

        load_local_env()
        results, failures = run_deployment_checks()
        print(format_results(results))
        return 0 if failures == 0 else 1
    maybe_prompt_for_api_key()
    configure_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    configure_llm_tracing(get_settings())
    llm_gateway = _optional_llm_gateway()
    try:
        artifacts = run_pipeline(
            goal=args.goal,
            max_agents=max(1, args.max_agents),
            llm_gateway=llm_gateway,
        )
        print(f"Dataset name: {artifacts.dataset_name}")
        print(f"CSV: {artifacts.csv_path}")
        print(f"Parquet: {artifacts.parquet_path}")
        print(f"Profile: {artifacts.profile_path}")
        print(f"Validation: {artifacts.validation_path}")
        print(f"Rows: {artifacts.rows}")
        print(f"Columns: {artifacts.columns}")
        return 0
    except LLMError as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Unhandled pipeline error: %s", exc)
        return 3


def _records_look_like_ui_chrome(records: list[dict[str, object]]) -> bool:
    blocked_phrases = (
        "jump to content",
        "main menu",
        "search wikipedia",
        "create account",
        "log in",
        "donate",
        "help",
        "navigation",
        "button",
        "searchbox",
        "radio",
        "appearance",
        "page tools",
        "personal tools",
        "views",
        "site",
        "tools",
        "thumbnail for",
    )
    suspicious = 0
    for record in records[:20]:
        for value in record.values():
            if isinstance(value, str) and any(phrase in value.lower() for phrase in blocked_phrases):
                suspicious += 1
                break
    return suspicious >= max(1, min(len(records), 5) // 2)


def _predictive_result_matches_goal(dataframe, *, goal: str, target_field: str | None) -> bool:
    columns = {str(column).lower() for column in dataframe.columns}
    markers = set()
    if target_field:
        markers.update(token for token in re.findall(r"[a-z0-9]+", target_field.lower()) if len(token) >= 3)
    markers.update(
        token
        for token in re.findall(r"[a-z0-9]+", goal.lower())
        if token in {"salary", "valuation", "revenue", "income", "profit", "growth", "spend", "transfer", "playoff", "population", "gdp"}
    )
    if not markers:
        return True
    if any(marker in column for marker in markers for column in columns):
        return True
    if "growth" in markers and any(token in column for token in {"change", "growth"} for column in columns):
        return True
    return False


def load_local_env(path: Path = ENV_FILE_PATH) -> None:
    """Load locally stored environment variables without requiring python-dotenv."""
    load_env_into_process(path)


def maybe_prompt_for_api_key(path: Path = ENV_FILE_PATH) -> None:
    """On first interactive boot, offer to save an API key for future runs."""
    if _configured_api_key_present():
        return
    if not sys.stdin.isatty():
        return

    print(
        "No API key found for LLM features.\n"
        "Enter an API key to store in .env.local for future runs,\n"
        "or press Enter to continue without one."
    )
    prompt_and_store_api_key(path)


def interactive_setup(path: Path = ENV_FILE_PATH) -> int:
    """Guide the user through storing an API key and optional model."""
    print("Interactive setup")
    return prompt_and_store_api_key(path)


def prompt_and_store_api_key(path: Path = ENV_FILE_PATH) -> int:
    """Prompt for an API key, persist it locally, and load it into the process."""
    if not sys.stdin.isatty():
        print("Cannot prompt for an API key without an interactive terminal.")
        return 1

    api_key = input("API key (input is visible): ").strip()
    if not api_key:
        print("No API key entered.")
        return 1

    env_var = _infer_api_env_var(api_key)
    _store_api_key(path, env_var, api_key)
    os.environ[env_var] = api_key
    print(f"Stored {env_var} in {path}.")
    suggested_model = _default_model_for_env_var(env_var)
    model = _clean_prompt_input(input(
        f"Preferred model [{suggested_model}] (press Enter to keep default): "
    ))
    if model:
        _store_env_value(path, "OPENAI_MODEL", model)
        os.environ["OPENAI_MODEL"] = model
        print(f"Stored OPENAI_MODEL in {path}.")
    return 0


def clear_stored_api_keys(path: Path = ENV_FILE_PATH) -> int:
    """Remove stored API keys from the local env file."""
    if not path.exists():
        print(f"No {path} file found.")
        return 0

    existing = read_env_file(path)
    for env_var in API_KEY_ENV_VARS:
        existing.pop(env_var, None)

    if existing:
        serialized = "\n".join(f"{key}='{value}'" for key, value in sorted(existing.items()))
        path.write_text(f"{serialized}\n", encoding="utf-8")
    else:
        path.unlink()

    for env_var in API_KEY_ENV_VARS:
        os.environ.pop(env_var, None)
    print("Cleared stored API keys.")
    return 0


def prompt_and_store_model(path: Path = ENV_FILE_PATH) -> int:
    """Prompt for a preferred model and persist it locally."""
    if not sys.stdin.isatty():
        print("Cannot prompt for a model without an interactive terminal.")
        return 1

    suggested = _suggested_model_from_environment()
    model = _clean_prompt_input(input(f"Model [{suggested}]: ")) or suggested
    _store_env_value(path, "OPENAI_MODEL", model)
    os.environ["OPENAI_MODEL"] = model
    print(f"Stored OPENAI_MODEL='{model}' in {path}.")
    return 0


def clear_stored_model(path: Path = ENV_FILE_PATH) -> int:
    """Remove the locally stored preferred model."""
    if not path.exists():
        print(f"No {path} file found.")
        return 0

    existing = read_env_file(path)
    if "OPENAI_MODEL" not in existing:
        print("No stored model found.")
        return 0

    existing.pop("OPENAI_MODEL", None)
    if existing:
        serialized = "\n".join(f"{key}='{value}'" for key, value in sorted(existing.items()))
        path.write_text(f"{serialized}\n", encoding="utf-8")
    else:
        path.unlink()
    os.environ.pop("OPENAI_MODEL", None)
    print("Cleared stored model.")
    return 0


def print_api_key_status(path: Path = ENV_FILE_PATH) -> int:
    """Print which provider key is currently configured."""
    for env_var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        value = os.getenv(env_var, "").strip()
        if env_value_is_usable(value, key=env_var):
            source = "environment"
            if path.exists() and read_env_file(path).get(env_var) == value:
                source = str(path)
            print(f"{env_var} configured via {source}.")
            return 0
    print("No API key configured.")
    return 0


def print_model_status(path: Path = ENV_FILE_PATH) -> int:
    """Print the currently configured model source."""
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not model:
        print("No model configured. Provider default will be used.")
        return 0

    source = "environment"
    if path.exists() and read_env_file(path).get("OPENAI_MODEL") == model:
        source = str(path)
    print(f"OPENAI_MODEL='{model}' configured via {source}.")
    return 0


def _configured_api_key_present() -> bool:
    return configured_api_key_present()


def _infer_api_env_var(api_key: str) -> str:
    if api_key.startswith("gsk_"):
        return "GROQ_API_KEY"
    if api_key.startswith("AI"):
        return "GEMINI_API_KEY"
    return "OPENAI_API_KEY"


def _store_api_key(path: Path, env_var: str, api_key: str) -> None:
    _store_env_value(path, env_var, api_key)


def _store_env_value(path: Path, key: str, value: str) -> None:
    existing = read_env_file(path)
    existing[key] = value
    serialized = "\n".join(f"{item_key}='{item_value}'" for item_key, item_value in sorted(existing.items()))
    path.write_text(f"{serialized}\n", encoding="utf-8")


def _default_model_for_env_var(env_var: str) -> str:
    if env_var == "GROQ_API_KEY":
        return "openai/gpt-oss-20b"
    if env_var == "GEMINI_API_KEY":
        return "gemini-2.0-flash-lite"
    return "gpt-4.1-mini"


def _suggested_model_from_environment() -> str:
    for env_var in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(env_var, "").strip()
        if env_value_is_usable(value, key=env_var):
            return _default_model_for_env_var(env_var)
    return "gpt-4.1-mini"


def _clean_prompt_input(value: str) -> str:
    cleaned = ESCAPE_SEQUENCE_RE.sub("", value).strip()
    if cleaned in {"^[[A", "^[[B", "^[[C", "^[[D"}:
        return ""
    return cleaned


def _optional_llm_gateway() -> LLMGateway | None:
    try:
        return LLMGateway()
    except LLMError:
        return None


def _fallback_provenance_map(records: list[dict[str, object]]) -> dict[str, str]:
    if not records:
        return {}
    source_ref = "synthesized"
    first = records[0]
    if isinstance(first.get("source_url"), str) and first["source_url"]:
        source_ref = str(first["source_url"])
    return {column: source_ref for column in first.keys()}


if __name__ == "__main__":
    raise SystemExit(main())
