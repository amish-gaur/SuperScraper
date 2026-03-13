"""Goal-aware source discovery helpers for generalized dataset requests."""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import quote_plus

from goal_intent import GoalDecomposition, decompose_goal
from source_adapters import adapter_urls_for_goal
from source_memory import SourceMemory

KNOWN_STALE_URL_FRAGMENTS = (
    "statbunker.com",
    "nanoreview.net/en/laptop-list",
)


@dataclass(frozen=True, slots=True)
class DiscoveryCandidate:
    """A source candidate enriched with family-level context."""

    url: str
    expected_source_type: str
    family: str
    rationale: str


class SourceDiscoveryEngine:
    """Generate higher-quality seed URLs and search prompts for broad goals."""

    def __init__(self, source_memory: SourceMemory | None = None) -> None:
        self.source_memory = source_memory or SourceMemory()

    def discover(
        self,
        goal: str,
        *,
        forbidden_domains: set[str] | None = None,
        limit: int = 12,
    ) -> list[DiscoveryCandidate]:
        decomposition = decompose_goal(goal)
        forbidden_domains = {domain.lower() for domain in (forbidden_domains or set()) if domain}

        candidates: list[DiscoveryCandidate] = []
        candidates.extend(self._memory_candidates(goal))
        candidates.extend(self._adapter_candidates(goal))
        candidates.extend(self._pattern_candidates(decomposition))

        deduped: list[DiscoveryCandidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.url.strip()
            if not normalized or normalized in seen:
                continue
            if any(fragment in normalized.lower() for fragment in KNOWN_STALE_URL_FRAGMENTS):
                continue
            if any(domain in normalized.lower() for domain in forbidden_domains):
                continue
            seen.add(normalized)
            deduped.append(candidate)
            if len(deduped) >= limit:
                break
        return deduped

    def search_queries(self, goal: str, *, limit: int = 8) -> list[str]:
        decomposition = decompose_goal(goal)
        entity_phrase = decomposition.row_granularity.replace("_", " ")
        target_phrase = (decomposition.target_hint or "target").replace("_", " ")
        feature_phrases = [feature.replace("_", " ") for feature in decomposition.feature_hints[:2]]
        domain_phrase = decomposition.domain_intent.replace("-", " ")
        temporal_phrase = {
            "historical": "historical",
            "current": "current",
            "recent": "latest",
        }.get(decomposition.temporal_scope, "")

        query_templates = [
            f"{goal} public table",
            f"{domain_phrase} {entity_phrase} {target_phrase} stats table".strip(),
            f"{domain_phrase} {entity_phrase} list {temporal_phrase}".strip(),
            f"{entity_phrase} {target_phrase} directory".strip(),
            f"{goal} wikipedia list",
        ]
        if feature_phrases:
            query_templates.append(
                f"{domain_phrase} {entity_phrase} {' '.join(feature_phrases)} dataset".strip()
            )
        queries: list[str] = []
        for query in query_templates:
            cleaned = re.sub(r"\s+", " ", query).strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
            if len(queries) >= limit:
                break
        return queries

    def _memory_candidates(self, goal: str) -> list[DiscoveryCandidate]:
        return [
            DiscoveryCandidate(
                url=url,
                expected_source_type="html_table",
                family="memory",
                rationale="reused_successful_source",
            )
            for url in self.source_memory.similar_urls(goal)
        ]

    def _adapter_candidates(self, goal: str) -> list[DiscoveryCandidate]:
        candidates: list[DiscoveryCandidate] = []
        for url in adapter_urls_for_goal(goal):
            source_type = "react_state" if any(token in url for token in ("nba.com/stats", "transfermarkt")) else "html_table"
            candidates.append(
                DiscoveryCandidate(
                    url=url,
                    expected_source_type=source_type,
                    family="adapter",
                    rationale="domain_specific_adapter",
                )
            )
        return candidates

    def _pattern_candidates(self, decomposition: GoalDecomposition) -> list[DiscoveryCandidate]:
        query = quote_plus(decomposition.raw_goal)
        list_query = quote_plus(f"{decomposition.raw_goal} list")
        candidates = [
            DiscoveryCandidate(
                url=f"https://en.wikipedia.org/w/index.php?search={list_query}",
                expected_source_type="html_table",
                family="reference_list",
                rationale="wikipedia_list_fallback",
            ),
            DiscoveryCandidate(
                url=f"https://www.wikidata.org/w/index.php?search={query}",
                expected_source_type="html_table",
                family="reference_list",
                rationale="wikidata_entity_search",
            ),
        ]

        if decomposition.domain_intent == "public-data" or decomposition.entity_intent == "state":
            candidates.extend(
                [
                    DiscoveryCandidate(
                        url="https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html",
                        expected_source_type="html_table",
                        family="official_dataset",
                        rationale="official_public_table",
                    ),
                    DiscoveryCandidate(
                        url="https://www.bea.gov/data/gdp/gdp-state",
                        expected_source_type="html_table",
                        family="official_dataset",
                        rationale="official_public_table",
                    ),
                ]
            )
        if decomposition.domain_intent in {"startup", "bank"} or decomposition.entity_intent == "company":
            candidates.extend(
                [
                    DiscoveryCandidate(
                        url="https://www.forbes.com/lists/",
                        expected_source_type="browser_heavy",
                        family="publisher_list",
                        rationale="publisher_rankings",
                    ),
                ]
            )
        if decomposition.domain_intent == "nba":
            candidates.extend(
                [
                    DiscoveryCandidate(
                        url="https://www.basketball-reference.com/leagues/NBA_2025_per_game.html",
                        expected_source_type="html_table",
                        family="aggregator",
                        rationale="sports_aggregator",
                    ),
                ]
            )
        elif decomposition.domain_intent == "ncaa-basketball":
            candidates.extend(
                [
                    DiscoveryCandidate(
                        url="https://www.sports-reference.com/cbb/seasons/",
                        expected_source_type="html_table",
                        family="aggregator",
                        rationale="sports_aggregator",
                    ),
                ]
            )
        elif decomposition.domain_intent == "soccer":
            candidates.extend(
                [
                    DiscoveryCandidate(
                        url="https://www.transfermarkt.com/",
                        expected_source_type="browser_heavy",
                        family="aggregator",
                        rationale="sports_aggregator",
                    ),
                ]
            )
        if decomposition.domain_intent == "laptop":
            candidates.extend(
                [
                    DiscoveryCandidate(
                        url="https://www.notebookcheck.net/",
                        expected_source_type="html_table",
                        family="aggregator",
                        rationale="hardware_rankings",
                    ),
                ]
            )
        return candidates
