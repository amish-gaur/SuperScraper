"""Async swarm orchestration for parallel research agents."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
import re

from pydantic import BaseModel

from agent import ResearchAgent
from architect import DatasetBlueprint
from browser import BrowserController
from llm import LLMGateway


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SwarmDispatcher:
    """Launch multiple research agents with complementary search strategies."""

    goal: str
    blueprint: DatasetBlueprint
    row_model: type[BaseModel]
    agent_count: int = 4
    llm_gateway: LLMGateway = field(default_factory=LLMGateway)

    async def run(self) -> list[BaseModel]:
        """Run the research swarm and aggregate all discovered rows."""
        queries = self._build_queries()
        per_agent_target = max(5, self.blueprint.target_record_count // max(len(queries), 1))
        agents = [
            ResearchAgent(
                name=f"agent-{index + 1}",
                goal=self.goal,
                row_model=self.row_model,
                search_query=query,
                target_record_count=per_agent_target,
                browser=BrowserController(extra_args=("--session", f"swarm-{index + 1}")),
                llm_gateway=self.llm_gateway,
            )
            for index, query in enumerate(queries)
        ]

        LOGGER.info("Launching %d parallel research agents", len(agents))
        results = await asyncio.gather(
            *(agent.run() for agent in agents),
            return_exceptions=True,
        )

        records: list[BaseModel] = []
        seen: set[str] = set()
        for index, result in enumerate(results, start=1):
            if isinstance(result, Exception):
                LOGGER.warning("Agent %d failed: %s", index, result)
                continue
            for record in result:
                fingerprint = record.model_dump_json(exclude_none=True, exclude_defaults=True)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                records.append(record)

        LOGGER.info("Swarm aggregated %d raw records", len(records))
        return records

    def _build_queries(self) -> list[str]:
        """Create diverse search queries from the blueprint strategies."""
        domain_filter = " OR ".join(f"site:{domain}" for domain in self._select_domains())
        strategies = list(dict.fromkeys(self.blueprint.search_strategies))
        base_query = self.blueprint.dataset_name.replace("_", " ")
        field_terms = " ".join(
            name.replace("_", " ") for name in self.row_model.model_fields.keys()
        )
        field_driven = [
            field_terms,
            f"{base_query} statistics",
            f"{base_query} official roster",
        ]
        combined = strategies + field_driven

        queries: list[str] = []
        for strategy in combined:
            if len(queries) >= min(5, max(3, self.agent_count)):
                break
            query = f"{_compress_query(base_query)} {_compress_query(strategy)} ({domain_filter})".strip()
            queries.append(query)

        if not queries:
            queries.append(f"{base_query} {field_terms} ({domain_filter})")
        return queries

    def _select_domains(self) -> tuple[str, ...]:
        """Choose a tighter domain whitelist based on the user goal."""
        goal = self.goal.lower()
        if any(keyword in goal for keyword in ("basketball", "ncaa", "football", "sports")):
            return (
                "247sports.com",
                "on3.com",
                "espn.com",
                "ncaa.com",
                "sports-reference.com",
                "cbssports.com",
            )
        return (
            "techcrunch.com",
            "venturebeat.com",
            "prweb.com",
            "businesswire.com",
            "globenewswire.com",
            "crunchbase.com",
        )


def _compress_query(value: str) -> str:
    """Trim overly wordy natural-language strategy text into search keywords."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", value.lower())
    tokens = [token for token in cleaned.split() if token not in {"the", "a", "an", "and", "or", "to", "for", "of", "from", "with", "i", "need", "dataset"}]
    return " ".join(tokens[:10])
