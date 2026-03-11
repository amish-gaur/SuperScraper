"""Domain-aware extraction adapters for known high-value sites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import re
from typing import Any
from urllib.parse import urlparse

import requests

from architect import SourceTarget


LOGGER = logging.getLogger(__name__)


class DomainAdapter(ABC):
    """Base interface for domain-specific extraction shortcuts."""

    @abstractmethod
    def matches(self, source_target: SourceTarget) -> bool:
        """Return whether this adapter can handle the given source target."""

    @abstractmethod
    def fetch_payload(self, source_target: SourceTarget) -> dict[str, Any] | None:
        """Return a raw JSON-like payload ready for downstream synthesis."""


@dataclass(slots=True)
class NBAStatsAdapter(DomainAdapter):
    """Proof-of-concept adapter for nba.com/stats hidden JSON endpoints."""

    timeout_seconds: float = 20.0
    retry_attempts: int = 2
    session: requests.Session = field(default_factory=requests.Session)

    def matches(self, source_target: SourceTarget) -> bool:
        hostname = (urlparse(source_target.url).hostname or "").lower()
        return hostname.endswith("nba.com") and "/stats" in source_target.url.lower()

    def fetch_payload(self, source_target: SourceTarget) -> dict[str, Any] | None:
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
        last_error: requests.RequestException | None = None
        response: requests.Response | None = None
        for attempt in range(1, self.retry_attempts + 2):
            try:
                response = self.session.get(
                    endpoint,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                LOGGER.warning(
                    "NBAStatsAdapter attempt %d failed for %s: %s",
                    attempt,
                    source_target.url,
                    exc,
                )
                response = None
        if response is None:
            LOGGER.warning("NBAStatsAdapter exhausted retries for %s: %s", source_target.url, last_error)
            return None

        try:
            payload = response.json()
        except ValueError as exc:
            LOGGER.warning("NBAStatsAdapter returned non-JSON for %s: %s", source_target.url, exc)
            return None

        return {
            "adapter": "nba_stats",
            "source_url": source_target.url,
            "api_url": response.url,
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


def build_domain_adapters() -> list[DomainAdapter]:
    """Return the currently supported domain adapter registry."""
    return [NBAStatsAdapter()]
