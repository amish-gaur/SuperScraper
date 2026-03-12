"""Domain-aware extraction adapters for known high-value sites."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from crawlee import Request
import requests

from architect import SourceTarget


LOGGER = logging.getLogger(__name__)


class DomainAdapter:
    """Base interface for domain-specific extraction shortcuts."""

    def matches(self, source_target: SourceTarget) -> bool:
        """Return whether this adapter can handle the given source target."""
        raise NotImplementedError

    def prefers_crawlee(self) -> bool:
        """Return whether this adapter should run through Crawlee routing."""
        return True

    def requires_javascript(self, source_target: SourceTarget) -> bool:
        """Return whether this adapter needs a Playwright-backed crawl."""
        return False

    def build_request(self, source_target: SourceTarget) -> Request:
        """Build the initial Crawlee request enqueued by the extraction router."""
        return Request.from_url(
            source_target.url,
            user_data={
                "source_target_url": source_target.url,
                "expected_source_type": source_target.expected_source_type,
            },
        )

    def fetch_payload(self, source_target: SourceTarget) -> dict[str, Any] | None:
        """Compatibility fallback for adapters that do not need Crawlee context."""
        return None

    async def fetch_payload_with_context(
        self,
        source_target: SourceTarget,
        context: Any,
    ) -> dict[str, Any] | None:
        """Return a raw payload using Crawlee request context."""
        return self.fetch_payload(source_target)


@dataclass(slots=True)
class NBAStatsAdapter(DomainAdapter):
    """Proof-of-concept adapter for nba.com/stats hidden JSON endpoints."""

    retry_attempts: int = 2

    def matches(self, source_target: SourceTarget) -> bool:
        hostname = (urlparse(source_target.url).hostname or "").lower()
        return hostname.endswith("nba.com") and "/stats" in source_target.url.lower()

    async def fetch_payload_with_context(
        self,
        source_target: SourceTarget,
        context: Any,
    ) -> dict[str, Any] | None:
        endpoint, params = self._resolve_endpoint(source_target.url)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Origin": "https://www.nba.com",
            "Referer": source_target.url,
            "x-nba-stats-origin": "stats",
            "x-nba-stats-token": "true",
        }
        last_error: Exception | None = None
        body_text = ""
        for attempt in range(1, self.retry_attempts + 2):
            try:
                response = await context.send_request(
                    self._api_url(endpoint, params),
                    headers=headers,
                )
                body = await response.read()
                body_text = body.decode("utf-8", errors="replace")
                if response.status_code >= 400:
                    raise RuntimeError(f"http status {response.status_code}")
                break
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "NBAStatsAdapter attempt %d failed for %s: %s",
                    attempt,
                    source_target.url,
                    exc,
                )
                body_text = ""
        if not body_text:
            LOGGER.warning("NBAStatsAdapter exhausted retries for %s: %s", source_target.url, last_error)
            return None

        try:
            payload = json.loads(body_text)
        except ValueError as exc:
            LOGGER.warning("NBAStatsAdapter returned non-JSON for %s: %s", source_target.url, exc)
            return None

        return {
            "adapter": "nba_stats",
            "source_url": source_target.url,
            "api_url": self._api_url(endpoint, params),
            "endpoint": endpoint,
            "params": params,
            "payload": payload,
        }

    def _resolve_endpoint(self, url: str) -> tuple[str, dict[str, str]]:
        lowered = url.lower()
        season = self._extract_season(url)
        season_type = self._extract_season_type(url)

        if "/players/" in lowered or "player" in lowered:
            endpoint = "https://stats.nba.com/stats/leaguedashplayerstats"
            params = {
                "College": "",
                "Conference": "",
                "Country": "",
                "DateFrom": "",
                "DateTo": "",
                "Division": "",
                "DraftPick": "",
                "DraftYear": "",
                "GameScope": "",
                "GameSegment": "",
                "Height": "",
                "LastNGames": "0",
                "LeagueID": "00",
                "Location": "",
                "MeasureType": "Base",
                "Month": "0",
                "OpponentTeamID": "0",
                "Outcome": "",
                "PORound": "0",
                "PaceAdjust": "N",
                "PerMode": "PerGame",
                "Period": "0",
                "PlayerExperience": "",
                "PlayerPosition": "",
                "PlusMinus": "N",
                "Rank": "N",
                "Season": season,
                "SeasonSegment": "",
                "SeasonType": season_type,
                "ShotClockRange": "",
                "StarterBench": "",
                "TeamID": "0",
                "TwoWay": "0",
                "VsConference": "",
                "VsDivision": "",
            }
            return endpoint, params

        endpoint = "https://stats.nba.com/stats/leaguedashteamstats"
        params = {
            "Conference": "",
            "DateFrom": "",
            "DateTo": "",
            "Division": "",
            "GameScope": "",
            "GameSegment": "",
            "LastNGames": "0",
            "LeagueID": "00",
            "Location": "",
            "MeasureType": "Base",
            "Month": "0",
            "OpponentTeamID": "0",
            "Outcome": "",
            "PORound": "0",
            "PaceAdjust": "N",
            "PerMode": "PerGame",
            "Period": "0",
            "PlusMinus": "N",
            "Rank": "N",
            "Season": season,
            "SeasonSegment": "",
            "SeasonType": season_type,
            "ShotClockRange": "",
            "TeamID": "0",
            "TwoWay": "0",
            "VsConference": "",
            "VsDivision": "",
        }
        return endpoint, params

    def _extract_season(self, url: str) -> str:
        match = re.search(r"[?&]season=(\d{4}-\d{2})", url, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        year_match = re.search(r"\b(20\d{2})\b", url)
        if year_match:
            year = int(year_match.group(1))
            return f"{year - 1}-{str(year)[-2:]}"
        return "2025-26"

    def _extract_season_type(self, url: str) -> str:
        match = re.search(r"[?&]seasontype=([^&]+)", url, flags=re.IGNORECASE)
        if match:
            return match.group(1).replace("+", " ")
        return "Regular Season"

    def _api_url(self, endpoint: str, params: dict[str, str]) -> str:
        prepared = requests.PreparedRequest()
        prepared.prepare_url(endpoint, params)
        return str(prepared.url)


def build_domain_adapters() -> list[DomainAdapter]:
    """Return the currently supported domain adapter registry."""
    return [NBAStatsAdapter()]
