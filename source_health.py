"""Shared runtime source health tracking and structured fetch outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from urllib.parse import urlparse

import requests


LOGGER = logging.getLogger(__name__)


class FailureReason(str, Enum):
    """Canonical failure reasons used across deterministic and browser paths."""

    HTTP_403 = "http_403"
    HTTP_429 = "http_429"
    HTTP_ERROR = "http_error"
    NETWORK_ERROR = "network_error"
    ANTI_BOT = "anti_bot"
    NO_TABLES = "no_tables_found"
    SNAPSHOT_STATIC = "snapshot_static"
    SNAPSHOT_LOW_DENSITY = "snapshot_low_density"
    SCHEMA_MISMATCH = "schema_mismatch"
    EMPTY_CONTENT = "empty_content"
    BROWSER_ERROR = "browser_error"
    SUCCESS = "success"


@dataclass(frozen=True, slots=True)
class FetchOutcome:
    """Result of a fetch attempt with enough metadata for routing decisions."""

    url: str
    ok: bool
    text: str = ""
    status_code: int | None = None
    reason: FailureReason = FailureReason.SUCCESS
    detail: str = ""


@dataclass(slots=True)
class DomainHealth:
    """Aggregated runtime health for a root domain."""

    attempts: int = 0
    successes: int = 0
    anti_bot_failures: int = 0
    hard_failures: int = 0
    soft_failures: int = 0
    empty_results: int = 0
    records_extracted: int = 0
    last_reason: str | None = None


@dataclass(slots=True)
class SourceHealthRegistry:
    """In-memory registry for runtime source outcomes."""

    domains: dict[str, DomainHealth] = field(default_factory=dict)

    def root_domain(self, url: str) -> str:
        hostname = (urlparse(url).hostname or "").lower().strip(".")
        if not hostname:
            return ""
        parts = hostname.split(".")
        if len(parts) <= 2:
            return hostname
        if parts[-2] in {"co", "com", "org", "gov", "ac"} and len(parts) >= 3:
            return ".".join(parts[-3:])
        return ".".join(parts[-2:])

    def record_fetch(self, outcome: FetchOutcome) -> None:
        domain = self.root_domain(outcome.url)
        if not domain:
            return
        stats = self.domains.setdefault(domain, DomainHealth())
        stats.attempts += 1
        stats.last_reason = outcome.reason.value
        if outcome.ok:
            stats.successes += 1
            if not outcome.text.strip():
                stats.empty_results += 1
            return

        if outcome.reason in {FailureReason.HTTP_403, FailureReason.HTTP_429, FailureReason.ANTI_BOT}:
            stats.anti_bot_failures += 1
            stats.hard_failures += 1
        elif outcome.reason in {FailureReason.NETWORK_ERROR, FailureReason.HTTP_ERROR, FailureReason.BROWSER_ERROR}:
            stats.hard_failures += 1
        else:
            stats.soft_failures += 1

    def record_extraction(self, url: str, *, records_extracted: int, success: bool, reason: FailureReason) -> None:
        domain = self.root_domain(url)
        if not domain:
            return
        stats = self.domains.setdefault(domain, DomainHealth())
        stats.last_reason = reason.value
        stats.records_extracted += max(records_extracted, 0)
        if success:
            stats.successes += 1
            return
        if reason in {FailureReason.NO_TABLES, FailureReason.EMPTY_CONTENT, FailureReason.SCHEMA_MISMATCH}:
            stats.soft_failures += 1

    def domain_penalty(self, url: str) -> int:
        stats = self.domains.get(self.root_domain(url))
        if stats is None:
            return 0
        penalty = stats.hard_failures * 6
        penalty += stats.anti_bot_failures * 8
        penalty += stats.soft_failures * 2
        penalty += min(stats.empty_results, 3)
        penalty -= min(stats.successes, 3) * 2
        penalty -= min(stats.records_extracted // 25, 4) * 2
        return penalty

    def should_cooldown(self, url: str) -> bool:
        stats = self.domains.get(self.root_domain(url))
        if stats is None:
            return False
        return stats.anti_bot_failures >= 2 or stats.hard_failures >= 3

    def known_bad_domains(self) -> set[str]:
        return {
            domain
            for domain, stats in self.domains.items()
            if stats.anti_bot_failures >= 1 or stats.hard_failures >= 2
        }


REGISTRY = SourceHealthRegistry()


def fetch_url(
    url: str,
    *,
    headers: dict[str, str],
    timeout_seconds: float,
    verify: bool = False,
) -> FetchOutcome:
    """Fetch a URL and return a structured outcome suitable for source routing."""
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout_seconds,
            verify=verify,
        )
    except requests.RequestException as exc:
        outcome = FetchOutcome(
            url=url,
            ok=False,
            reason=FailureReason.NETWORK_ERROR,
            detail=str(exc),
        )
        REGISTRY.record_fetch(outcome)
        return outcome

    lowered = response.text.lower()
    anti_bot_markers = (
        "verify you are human",
        "enable javascript and cookies",
        "just a moment",
        "access denied",
        "cf-browser-verification",
        "cf-challenge",
        "why do i have to complete a captcha",
    )
    if response.status_code == 403:
        outcome = FetchOutcome(
            url=url,
            ok=False,
            status_code=response.status_code,
            reason=FailureReason.HTTP_403,
            detail="403 response",
        )
    elif response.status_code == 429:
        outcome = FetchOutcome(
            url=url,
            ok=False,
            status_code=response.status_code,
            reason=FailureReason.HTTP_429,
            detail="429 response",
        )
    elif any(token in lowered for token in anti_bot_markers):
        outcome = FetchOutcome(
            url=url,
            ok=False,
            status_code=response.status_code,
            reason=FailureReason.ANTI_BOT,
            detail="anti-bot text detected",
        )
    elif response.status_code >= 400:
        outcome = FetchOutcome(
            url=url,
            ok=False,
            status_code=response.status_code,
            reason=FailureReason.HTTP_ERROR,
            detail=f"http status {response.status_code}",
        )
    else:
        text = response.text or ""
        outcome = FetchOutcome(
            url=url,
            ok=bool(text.strip()),
            text=text,
            status_code=response.status_code,
            reason=FailureReason.SUCCESS if text.strip() else FailureReason.EMPTY_CONTENT,
            detail="",
        )

    REGISTRY.record_fetch(outcome)
    if not outcome.ok:
        LOGGER.warning("Fetch failed for %s: %s", url, outcome.reason.value)
    return outcome
