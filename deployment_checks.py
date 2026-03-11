"""Deployability checks for local runtime readiness."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
import pathlib
import py_compile
import traceback

from browser import BrowserController
from config_tests import main as config_tests_main
from edge_case_tests import main as edge_case_tests_main
from fixture_integration_tests import main as fixture_integration_main
from env_utils import (
    ENV_FILE_PATH,
    configured_api_key_present,
    read_env_file,
)
from router_fixture_integration_tests import main as router_fixture_integration_main
from routing_smoke_tests import main as routing_smoke_main


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_deployment_checks() -> tuple[list[CheckResult], int]:
    """Run local checks that should be stable in CI and on deployment hosts."""
    results = [
        _check_python_compilation(),
        _check_required_modules(),
        _check_routing_smoke_tests(),
        _check_local_regressions(),
        _check_env_file_parse(),
        _check_llm_configuration(),
        _check_browser_binary(),
        _check_workspace_writable(),
    ]
    failures = sum(1 for result in results if not result.ok)
    return results, failures


def main() -> int:
    """Run deployment checks as a standalone script."""
    results, failures = run_deployment_checks()
    print(format_results(results))
    return 0 if failures == 0 else 1


def format_results(results: list[CheckResult]) -> str:
    """Format deployment checks for CLI output."""
    lines: list[str] = []
    for result in results:
        prefix = "PASS" if result.ok else "FAIL"
        lines.append(f"{prefix} {result.name}: {result.detail}")
    return "\n".join(lines)


def _check_python_compilation() -> CheckResult:
    try:
        count = 0
        for path in sorted(pathlib.Path(".").glob("*.py")):
            py_compile.compile(str(path), doraise=True)
            count += 1
        return CheckResult("python_compile", True, f"compiled {count} Python files")
    except Exception as exc:
        return CheckResult("python_compile", False, f"compilation failed: {exc}")


def _check_required_modules() -> CheckResult:
    missing: list[str] = []
    for module_name in (
        "openai",
        "pydantic",
        "pydantic_settings",
        "pandas",
        "requests",
        "pyarrow",
        "html5lib",
        "fastapi",
        "celery",
        "redis",
    ):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        return CheckResult("dependencies", False, f"missing modules: {', '.join(missing)}")
    return CheckResult("dependencies", True, "required Python modules are importable")


def _check_routing_smoke_tests() -> CheckResult:
    try:
        exit_code = routing_smoke_main(verbose=False)
    except Exception as exc:
        return CheckResult(
            "routing_smoke_tests",
            False,
            f"raised {exc.__class__.__name__}: {exc}\n{traceback.format_exc(limit=2).strip()}",
        )
    if exit_code != 0:
        return CheckResult("routing_smoke_tests", False, f"returned exit code {exit_code}")
    return CheckResult("routing_smoke_tests", True, "routing smoke tests passed")


def _check_local_regressions() -> CheckResult:
    checks = [
        ("config_tests", config_tests_main),
        ("edge_case_tests", edge_case_tests_main),
        ("fixture_integration_tests", fixture_integration_main),
        ("router_fixture_integration_tests", router_fixture_integration_main),
    ]
    failures: list[str] = []
    for name, runner in checks:
        try:
            exit_code = runner(verbose=False)
        except Exception as exc:
            return CheckResult(
                "local_regressions",
                False,
                f"{name} raised {exc.__class__.__name__}: {exc}",
            )
        if exit_code != 0:
            failures.append(f"{name}={exit_code}")
    if failures:
        return CheckResult("local_regressions", False, f"failed suites: {', '.join(failures)}")
    return CheckResult("local_regressions", True, "config, edge-case, and fixture regression suites passed")


def _check_env_file_parse() -> CheckResult:
    try:
        values = read_env_file(ENV_FILE_PATH)
    except Exception as exc:
        return CheckResult("env_file", False, f"failed to parse {ENV_FILE_PATH}: {exc}")
    if not ENV_FILE_PATH.exists():
        return CheckResult("env_file", True, f"{ENV_FILE_PATH} not present; using process environment only")
    return CheckResult("env_file", True, f"loaded {len(values)} values from {ENV_FILE_PATH}")


def _check_llm_configuration() -> CheckResult:
    if configured_api_key_present():
        return CheckResult("llm_config", True, "LLM API key detected")
    return CheckResult("llm_config", True, "no API key configured; deterministic paths will still work")


def _check_browser_binary() -> CheckResult:
    controller = BrowserController()
    return CheckResult("browser_binary", controller.is_available(), controller.availability_detail())


def _check_workspace_writable() -> CheckResult:
    probe = pathlib.Path(".deployability_check.tmp")
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return CheckResult("workspace_write", False, f"workspace is not writable: {exc}")
    return CheckResult("workspace_write", True, "workspace is writable")


if __name__ == "__main__":
    raise SystemExit(main())
