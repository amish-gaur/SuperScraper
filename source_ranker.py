"""Heuristics for ranking candidate source URLs before extraction."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import urlparse, urlsplit, urlunsplit

from source_health import REGISTRY


@dataclass(frozen=True, slots=True)
class RankedSource:
    """A candidate source URL with a deterministic quality score."""

    url: str
    score: int
    reasons: tuple[str, ...]


class SourceRanker:
    """Score candidate URLs for data density and extraction quality."""

    def rank(self, goal: str, urls: list[str]) -> list[RankedSource]:
        ranked: list[RankedSource] = []
        seen: set[str] = set()
        goal_tokens = set(re.findall(r"[a-z0-9]+", goal.lower()))

        for raw_url in urls:
            url = _normalize_url(raw_url)
            if not url or url in seen or not url.startswith(("http://", "https://")):
                continue
            seen.add(url)
            score, reasons = self._score_url(url, goal_tokens=goal_tokens)
            ranked.append(RankedSource(url=url, score=score, reasons=tuple(reasons)))

        ranked.sort(key=lambda item: (-item.score, item.url))
        return ranked

    def _score_url(self, url: str, *, goal_tokens: set[str]) -> tuple[int, list[str]]:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()
        full = f"{host}{path}"
        score = 0
        reasons: list[str] = []

        high_trust_hosts = {
            "wikipedia.org": 5,
            "ncaa.com": 5,
            "sec.gov": 5,
            "data.gov": 5,
            "census.gov": 5,
            "notebookcheck.net": 5,
            "nanoreview.net": 4,
            "sports-reference.com": 4,
            "fortune.com": 3,
            "forbes.com": 2,
        }
        for domain, domain_score in high_trust_hosts.items():
            if domain in host:
                score += domain_score
                reasons.append(f"trusted_domain:{domain}")
                break

        path_signals = {
            "list": 4,
            "stats": 5,
            "ranking": 4,
            "rankings": 4,
            "table": 4,
            "directory": 4,
            "fortune500": 5,
            "fortune-500": 5,
            "season": 3,
            "companies": 3,
            "laptop": 3,
            "notebook": 3,
        }
        for token, token_score in path_signals.items():
            if token in full:
                score += token_score
                reasons.append(f"path_signal:{token}")

        bad_signals = {
            "search": -4,
            "login": -6,
            "signup": -6,
            "subscribe": -4,
            "video": -3,
            "podcast": -3,
            "reviews": -2,
        }
        for token, penalty in bad_signals.items():
            if token in full:
                score += penalty
                reasons.append(f"bad_signal:{token}")

        token_matches = sum(1 for token in goal_tokens if token and token in full)
        if token_matches:
            score += token_matches
            reasons.append(f"goal_match:{token_matches}")

        health_penalty = REGISTRY.domain_penalty(url)
        if health_penalty:
            score -= health_penalty
            reasons.append(f"health_penalty:{health_penalty}")

        if REGISTRY.should_cooldown(url):
            score -= 20
            reasons.append("cooldown_domain")

        return score, reasons


def _normalize_url(raw_url: str) -> str:
    cleaned = raw_url.strip()
    if not cleaned:
        return ""
    parts = urlsplit(cleaned)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return ""
    return urlunsplit((scheme, parts.netloc.lower(), parts.path, parts.query, parts.fragment))
