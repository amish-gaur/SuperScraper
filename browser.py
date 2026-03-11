"""Browser controller for driving the agent-browser CLI."""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from urllib.parse import urlencode, urlsplit, urlunsplit
from dataclasses import dataclass, field
from typing import Sequence


LOGGER = logging.getLogger(__name__)
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class BrowserControllerError(RuntimeError):
    """Raised when the browser controller cannot complete an operation."""


@dataclass(slots=True)
class BrowserController:
    """Thin subprocess wrapper around the `agent-browser` CLI."""

    binary: str = field(
        default_factory=lambda: os.getenv("AGENT_BROWSER_BIN")
        or shutil.which("agent-browser")
        or "agent-browser"
    )
    timeout_seconds: float = 30.0
    extra_args: Sequence[str] = field(default_factory=tuple)
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    use_native: bool = field(
        default_factory=lambda: os.getenv("AGENT_BROWSER_NATIVE", "1") != "0"
    )
    browserbase_api_key: str | None = field(
        default_factory=lambda: os.getenv("BROWSERBASE_API_KEY")
    )
    browserbase_project_id: str | None = field(
        default_factory=lambda: os.getenv("BROWSERBASE_PROJECT_ID")
    )
    explicit_cdp_url: str | None = field(
        default_factory=lambda: os.getenv("AGENT_BROWSER_CDP_URL")
        or os.getenv("BROWSERBASE_CDP_URL")
    )
    _is_open: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self._uses_browserbase() and self.retry_delay_seconds < 5.0:
            self.retry_delay_seconds = 5.0
        if self._uses_browserbase() and self.retry_attempts > 2:
            self.retry_attempts = 2

    def is_available(self) -> bool:
        """Return whether the configured browser binary appears runnable."""
        if os.path.sep not in self.binary:
            return shutil.which(self.binary) is not None
        return os.path.exists(self.binary) and os.access(self.binary, os.X_OK)

    def availability_detail(self) -> str:
        """Return a short human-readable browser availability status."""
        if self.is_available():
            return f"browser binary available at '{self.binary}'"
        return (
            f"browser binary '{self.binary}' was not found. "
            "Set AGENT_BROWSER_BIN or install agent-browser before enabling browser fallback."
        )

    def open(self, url: str) -> None:
        """Open a target URL in the browser session."""
        if not url:
            raise ValueError("url must be a non-empty string")

        LOGGER.info("Opening URL: %s", url)
        self._run_cli("open", url)
        self._is_open = True

    def snapshot(self) -> str:
        """Capture a page snapshot of the current page."""
        self._ensure_open()
        result = self._run_cli("snapshot")
        snapshot = self._clean_output(result.stdout)
        if not snapshot:
            raise BrowserControllerError("agent-browser returned an empty snapshot")
        return snapshot

    def execute(self, command: str) -> str:
        """Execute an agent-browser action like `click @e1` or `type @e4 \"text\"`."""
        self._ensure_open()
        if not command or not command.strip():
            raise ValueError("command must be a non-empty string")
        if any(token in command for token in (";", "&&", "||")):
            raise BrowserControllerError(
                "Browser command must contain exactly one agent-browser action"
            )

        argv = shlex.split(command)
        if not argv:
            raise ValueError("command must contain at least one token")
        if argv[0] == self.binary:
            argv = argv[1:]

        LOGGER.info("Executing browser command: %s", command)
        result = self._run_cli(*argv)
        return self._clean_output(result.stdout)

    def close(self) -> None:
        """Close the browser session on a best-effort basis."""
        if not self._is_open:
            return

        try:
            LOGGER.info("Closing browser session")
            self._run_cli("close")
        except BrowserControllerError as exc:
            LOGGER.warning("Failed to close browser session cleanly: %s", exc)
        finally:
            self._is_open = False

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise BrowserControllerError("browser session is not open")

    def _run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        command = [self.binary, *self._remote_args(), *self.extra_args, *args]
        display_command = _redact_command(command)
        last_error: BrowserControllerError | None = None
        for attempt in range(1, self.retry_attempts + 1):
            backoff_seconds: float | None = None
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=self._command_env(),
                )
            except FileNotFoundError as exc:
                raise BrowserControllerError(
                    f"Browser binary '{self.binary}' was not found. "
                    "Set AGENT_BROWSER_BIN to the executable path or add it to PATH."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                last_error = BrowserControllerError(
                    f"Command timed out after {self.timeout_seconds:.1f}s: {display_command}"
                )
            except OSError as exc:
                last_error = BrowserControllerError(
                    f"Failed to execute browser command {display_command}: {exc}"
                )
            else:
                if result.returncode == 0:
                    return result
                stderr = result.stderr.strip() or "<no stderr>"
                backoff_seconds = _parse_retry_after_seconds(stderr) if self._uses_browserbase() else None
                last_error = BrowserControllerError(
                    f"Browser command failed with exit code {result.returncode}: "
                    f"{display_command} | stderr: {stderr}"
                )

            if attempt < self.retry_attempts:
                LOGGER.warning(
                    "Browser command failed, retrying (%d/%d): %s",
                    attempt,
                    self.retry_attempts,
                    display_command,
                )
                time.sleep(backoff_seconds or self.retry_delay_seconds)

        raise last_error or BrowserControllerError(
            f"Browser command failed after {self.retry_attempts} attempts: {display_command}"
        )

    def _clean_output(self, output: str) -> str:
        """Normalize CLI output by removing ANSI noise and blank padding."""
        cleaned = ANSI_ESCAPE_RE.sub("", output)
        lines = [line.rstrip() for line in cleaned.splitlines()]
        return "\n".join(line for line in lines if line.strip()).strip()

    def _remote_args(self) -> list[str]:
        """Build remote CDP CLI arguments when Browserbase or a custom CDP URL is configured."""
        if self.explicit_cdp_url:
            return ["--cdp", self.explicit_cdp_url.strip()]

        api_key = (self.browserbase_api_key or "").strip()
        project_id = (self.browserbase_project_id or "").strip()
        if api_key and project_id:
            return ["-p", "browserbase"]

        return []

    def _resolved_cdp_url(self) -> str | None:
        """Resolve the remote CDP WebSocket URL, if any."""
        if self.explicit_cdp_url:
            return self.explicit_cdp_url.strip()
        api_key = (self.browserbase_api_key or "").strip()
        project_id = (self.browserbase_project_id or "").strip()
        if api_key and project_id:
            return _build_browserbase_cdp_url(
                api_key=api_key,
                project_id=project_id,
            )
        return None

    def _uses_browserbase(self) -> bool:
        return bool((self.browserbase_api_key or "").strip() and (self.browserbase_project_id or "").strip())

    def _command_env(self) -> dict[str, str]:
        """Build the process environment for agent-browser invocations."""
        env = os.environ.copy()
        env["AGENT_BROWSER_NATIVE"] = "1" if self.use_native else "0"
        if self.browserbase_api_key:
            env["BROWSERBASE_API_KEY"] = self.browserbase_api_key
        if self.browserbase_project_id:
            env["BROWSERBASE_PROJECT_ID"] = self.browserbase_project_id
        return env


def _build_browserbase_cdp_url(*, api_key: str, project_id: str) -> str:
    """Construct the Browserbase remote CDP connection URL without exposing raw secrets in logs."""
    query = urlencode(
        {
            "apiKey": api_key,
            "projectId": project_id,
        },
        safe="",
    )
    return f"wss://connect.browserbase.com?{query}"


def _redact_command(command: Sequence[str]) -> str:
    """Render a shell-safe command string without leaking WebSocket credentials."""
    return " ".join(
        shlex.quote(_redact_url(token)) if token.startswith(("ws://", "wss://")) else shlex.quote(token)
        for token in command
    )


def _redact_url(url: str) -> str:
    """Mask sensitive query parameters before logging a remote CDP URL."""
    parts = urlsplit(url)
    if not parts.query:
        return url

    redacted_items: list[str] = []
    for item in parts.query.split("&"):
        key, sep, value = item.partition("=")
        if key in {"apiKey", "projectId"} and value:
            redacted_items.append(f"{key}=***")
            continue
        redacted_items.append(item if not sep else f"{key}={value}")

    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, "&".join(redacted_items), parts.fragment)
    )


def _parse_retry_after_seconds(stderr: str) -> float | None:
    """Extract a provider-provided retry hint from Browserbase rate-limit errors."""
    match = re.search(r"try again in (\d+) seconds", stderr, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))
