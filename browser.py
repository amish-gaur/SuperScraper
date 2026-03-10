"""Browser controller for driving the agent-browser CLI."""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import time
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
    _is_open: bool = field(default=False, init=False)

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
        command = [self.binary, *self.extra_args, *args]
        last_error: BrowserControllerError | None = None
        for attempt in range(1, self.retry_attempts + 1):
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
                    f"Command timed out after {self.timeout_seconds:.1f}s: {' '.join(command)}"
                )
            except OSError as exc:
                last_error = BrowserControllerError(
                    f"Failed to execute browser command {' '.join(command)}: {exc}"
                )
            else:
                if result.returncode == 0:
                    return result
                stderr = result.stderr.strip() or "<no stderr>"
                last_error = BrowserControllerError(
                    f"Browser command failed with exit code {result.returncode}: "
                    f"{' '.join(command)} | stderr: {stderr}"
                )

            if attempt < self.retry_attempts:
                LOGGER.warning(
                    "Browser command failed, retrying (%d/%d): %s",
                    attempt,
                    self.retry_attempts,
                    " ".join(command),
                )
                time.sleep(self.retry_delay_seconds)

        raise last_error or BrowserControllerError(
            f"Browser command failed after {self.retry_attempts} attempts: {' '.join(command)}"
        )

    def _clean_output(self, output: str) -> str:
        """Normalize CLI output by removing ANSI noise and blank padding."""
        cleaned = ANSI_ESCAPE_RE.sub("", output)
        lines = [line.rstrip() for line in cleaned.splitlines()]
        return "\n".join(line for line in lines if line.strip()).strip()

    def _command_env(self) -> dict[str, str]:
        """Build the process environment for agent-browser invocations."""
        env = os.environ.copy()
        env["AGENT_BROWSER_NATIVE"] = "1" if self.use_native else "0"
        return env
