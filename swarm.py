"""Async swarm orchestration for parallel research agents."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urlparse

from pydantic import BaseModel

from agent import ResearchAgent
from architect import DatasetArchitect, DatasetBlueprint, SourceTarget
from browser import BrowserController
from checkpoint import CheckpointManager
from crawlee_fetcher import CrawleeStaticRequestProcessor
from extraction_router import ExtractionRouter
from llm import LLMGateway
from source_ranker import SourceRanker


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StaticStageResult:
    """Deterministic Crawlee stage output."""

    records: list[BaseModel]
    browser_targets: list[SourceTarget]


@dataclass(slots=True)
class BrowserStageResult:
    """Browser fallback stage output."""

    records: list[BaseModel]
    attempted_urls: int = 0
    failed_urls: int = 0


@dataclass(slots=True)
class SwarmDispatcher:
    """Launch multiple research agents with direct domain routing and checkpoint resume."""

    goal: str
    blueprint: DatasetBlueprint
    row_model: type[BaseModel]
    agent_count: int = 2
    llm_gateway: LLMGateway | None = None
    architect: DatasetArchitect | None = None
    domain_blacklist: set[str] = field(default_factory=set)
    checkpoint_manager: CheckpointManager = field(default_factory=CheckpointManager)
    source_ranker: SourceRanker = field(default_factory=SourceRanker)

    async def run(self) -> list[BaseModel]:
        """Run the research swarm and aggregate all discovered rows."""
        cached_records = self.checkpoint_manager.load_records(
            goal=self.goal,
            dataset_name=self.blueprint.dataset_name,
            row_model=self.row_model,
        )
        deduped_records = self._dedupe_records(cached_records)
        remaining_target = max(self.blueprint.target_record_count - len(deduped_records), 0)

        if remaining_target == 0:
            LOGGER.info("Checkpoint already satisfies target with %d raw records", len(deduped_records))
            return deduped_records

        starting_targets = self._select_source_targets(self.blueprint.source_targets)
        if not starting_targets:
            LOGGER.warning("No starting URLs available; returning %d cached records", len(deduped_records))
            return deduped_records

        refreshed_once = False
        static_processor = CrawleeStaticRequestProcessor()
        router = ExtractionRouter(
            goal=self.goal,
            row_model=self.row_model,
            llm_gateway=self.llm_gateway,
            domain_blacklist=self.domain_blacklist,
            static_processor=static_processor,
        )
        while True:
            static_result = await self._run_static_stage(
                source_targets=starting_targets,
                router=router,
                processor=static_processor,
            )
            if static_result.records:
                deduped_records.extend(static_result.records)
                deduped_records = self._dedupe_records(deduped_records)
                remaining_target = max(self.blueprint.target_record_count - len(deduped_records), 0)
                LOGGER.info(
                    "Static Crawlee stage produced %d records before browser fallback",
                    len(static_result.records),
                )
                if self._goal_satisfied(deduped_records):
                    return deduped_records

            browser_result = await self._run_browser_stage(
                browser_targets=static_result.browser_targets,
                existing_records=deduped_records,
                remaining_target=remaining_target,
            )
            if browser_result.records:
                deduped_records.extend(browser_result.records)
                deduped_records = self._dedupe_records(deduped_records)
                remaining_target = max(self.blueprint.target_record_count - len(deduped_records), 0)
                if self._goal_satisfied(deduped_records):
                    return deduped_records
            else:
                if self.llm_gateway is None:
                    LOGGER.info("No LLM gateway available; skipping browser fallback stage")

            if deduped_records or refreshed_once:
                break
            refreshed_targets = self._refresh_source_targets_from_architect()
            if not refreshed_targets:
                break
            refreshed_once = True
            starting_targets = refreshed_targets

        LOGGER.info("Swarm aggregated %d raw records", len(deduped_records))
        return deduped_records

    async def _run_static_stage(
        self,
        *,
        source_targets: list[SourceTarget],
        router: ExtractionRouter,
        processor: CrawleeStaticRequestProcessor,
    ) -> StaticStageResult:
        """Route static sources through Crawlee before browser fallback."""
        records: list[BaseModel] = []
        browser_targets: list[SourceTarget] = []

        static_targets: list[SourceTarget] = []
        remaining_targets: list[SourceTarget] = []
        for source_target in source_targets:
            adapter = router.select_adapter(source_target)
            if router.should_route_with_crawlee(source_target, adapter=adapter):
                static_targets.append(source_target)
            else:
                remaining_targets.append(source_target)

        if static_targets:
            fetched_targets = await asyncio.to_thread(
                processor.fetch_sync,
                static_targets,
                adapters=router.adapters,
            )
            for fetched_target in fetched_targets:
                decision = router.route_prefetched(
                    fetched_target.source_target,
                    fetched_target.fetch_result,
                    adapter=fetched_target.adapter,
                )
                if decision.records:
                    LOGGER.info(
                        "Extraction router produced %d records at %s via %s",
                        len(decision.records),
                        fetched_target.source_target.url,
                        decision.strategy,
                    )
                    records.extend(decision.records)
                elif decision.requires_browser:
                    browser_targets.append(fetched_target.source_target)

        browser_targets.extend(remaining_targets)
        LOGGER.info(
            (
                "Static stage summary routed=%d browser_fallback=%d input=%d unique=%d "
                "duplicates=%d successful=%d failed=%d failed_by_reason=%s artifacts=%d"
            ),
            len(records),
            len(browser_targets),
            processor.stats.input_urls,
            processor.stats.unique_urls,
            processor.stats.deduped_urls,
            processor.stats.handled_urls,
            processor.stats.failed_urls,
            processor.stats.failed_urls_by_reason or {},
            processor.stats.artifacts_written,
        )
        return StaticStageResult(records=records, browser_targets=browser_targets)

    async def _run_browser_stage(
        self,
        *,
        browser_targets: list[SourceTarget],
        existing_records: list[BaseModel],
        remaining_target: int,
    ) -> BrowserStageResult:
        """Run browser agents only for URLs that the static stage could not satisfy."""
        if self.llm_gateway is None or not browser_targets:
            return BrowserStageResult(records=[])

        attempted_urls = 0
        failed_urls = 0
        collected_records: list[BaseModel] = []
        per_agent_target = max(1, -(-remaining_target // max(self.agent_count, 1)))
        browser_urls = [target.url for target in browser_targets]
        LOGGER.info("Browser fallback stage starting for %d candidate URLs", len(browser_urls))
        for batch_start in range(0, len(browser_urls), self.agent_count):
            if len(existing_records) + len(collected_records) >= self.blueprint.target_record_count:
                break

            batch_urls = [
                url for url in browser_urls[batch_start : batch_start + self.agent_count]
                if _root_domain(url) not in self.domain_blacklist
            ]
            if not batch_urls:
                continue
            attempted_urls += len(batch_urls)
            agents = [
                ResearchAgent(
                    name=f"agent-{batch_start + index + 1}",
                    goal=self.goal,
                    dataset_name=self.blueprint.dataset_name,
                    row_model=self.row_model,
                    starting_url=url,
                    target_record_count=per_agent_target,
                    browser=BrowserController(extra_args=("--session", f"swarm-{batch_start + index + 1}")),
                    llm_gateway=self.llm_gateway,
                    checkpoint_manager=self.checkpoint_manager,
                    existing_records=existing_records + collected_records,
                    domain_blacklist=self.domain_blacklist,
                )
                for index, url in enumerate(batch_urls)
            ]

            LOGGER.info("Launching %d browser agents", len(agents))
            results = await asyncio.gather(*(agent.run() for agent in agents), return_exceptions=True)
            for index, result in enumerate(results, start=batch_start + 1):
                if isinstance(result, Exception):
                    failed_urls += 1
                    LOGGER.warning("Browser agent %d failed: %s", index, result)
                    continue
                collected_records.extend(result)
            if collected_records:
                break

        LOGGER.info(
            "Browser fallback stage summary attempted=%d failed=%d produced=%d",
            attempted_urls,
            failed_urls,
            len(collected_records),
        )
        return BrowserStageResult(
            records=collected_records,
            attempted_urls=attempted_urls,
            failed_urls=failed_urls,
        )

    def _select_source_targets(self, source_targets: list[SourceTarget]) -> list[SourceTarget]:
        """Choose the direct source targets that seed deterministic extraction and browser agents."""
        normalized: list[SourceTarget] = []
        domain_counts: dict[str, int] = {}
        seen_urls: set[str] = set()

        ranked_targets = self.source_ranker.rank(self.goal, [target.url for target in source_targets])
        target_by_url = {target.url: target for target in source_targets}
        for ranked in ranked_targets:
            cleaned = ranked.url.strip()
            if not cleaned.startswith(("http://", "https://")):
                continue
            if cleaned in seen_urls:
                continue
            if self._is_blocked_domain(cleaned):
                continue
            root_domain = _root_domain(cleaned)
            if root_domain in self.domain_blacklist:
                continue
            current_count = domain_counts.get(root_domain, 0)
            if current_count >= self._max_urls_per_domain():
                continue
            domain_counts[root_domain] = current_count + 1
            seen_urls.add(cleaned)
            target = target_by_url.get(cleaned, SourceTarget(url=cleaned, expected_source_type="unknown"))
            normalized.append(target)
        if normalized:
            return normalized
        return [target for target in self._fallback_source_targets() if not self._is_blocked_domain(target.url)]

    def _max_urls_per_domain(self) -> int:
        if self.blueprint.target_record_count <= 60:
            return 4
        return 2

    def _refresh_source_targets_from_architect(self) -> list[SourceTarget]:
        """Ask the architect for new seed domains when the current set is blocked."""
        if self.architect is None or not self.domain_blacklist:
            return []
        LOGGER.warning(
            "All current swarm URLs failed. Requesting new starting URLs while forbidding domains: %s",
            sorted(self.domain_blacklist),
        )
        refreshed_targets = self.architect.refresh_source_targets(self.goal, self.domain_blacklist)
        selected = self._select_source_targets(refreshed_targets)
        if selected:
            self.blueprint = self.blueprint.model_copy(update={"source_targets": selected})
        return selected

    def _fallback_source_targets(self) -> list[SourceTarget]:
        """Provide deterministic URLs when the architect fails to emit usable ones."""
        return [
            SourceTarget(
                url=f"https://en.wikipedia.org/w/index.php?search={quote_plus(self.goal + ' list')}",
                expected_source_type="html_table",
            ),
            SourceTarget(
                url=f"https://www.wikidata.org/w/index.php?search={quote_plus(self.goal)}",
                expected_source_type="unknown",
            ),
        ]

    def _is_blocked_domain(self, url: str) -> bool:
        blocked_domains = (
            "sports-reference.com",
        )
        root_domain = _root_domain(url)
        return (
            root_domain in self.domain_blacklist
            or any(domain in url for domain in blocked_domains)
        )

    def _dedupe_records(self, records: list[BaseModel]) -> list[BaseModel]:
        deduped: list[BaseModel] = []
        seen: set[str] = set()
        for record in records:
            if self._record_looks_suspicious(record):
                continue
            fingerprint = record.model_dump_json(exclude_none=True, exclude_defaults=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(record)
        return deduped

    def _record_looks_suspicious(self, record: BaseModel) -> bool:
        payload = record.model_dump(mode="json")
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
            "request that a redirect be created",
        )
        for value in payload.values():
            if isinstance(value, str) and any(phrase in value.lower() for phrase in blocked_phrases):
                return True
        return False

    def _goal_satisfied(self, records: list[BaseModel]) -> bool:
        if len(records) < self.blueprint.target_record_count:
            return False
        if self._average_schema_coverage(records[: self.blueprint.target_record_count]) >= 2.0:
            return True
        LOGGER.info(
            "Row target reached but schema coverage is still sparse; continuing to gather more sources"
        )
        return False

    def _average_schema_coverage(self, records: list[BaseModel]) -> float:
        if not records:
            return 0.0
        ignored_fields = {"source_url", "reference_url", "entity_name", "raw_entity_name", "name", "state", "school", "team_name"}
        populated_counts: list[int] = []
        for record in records:
            payload = record.model_dump(mode="json")
            populated = 0
            for field_name, value in payload.items():
                if field_name in ignored_fields:
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                if isinstance(value, list) and not value:
                    continue
                populated += 1
            populated_counts.append(populated)
        return sum(populated_counts) / len(populated_counts)


def _root_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    if not hostname:
        return ""
    parts = hostname.split(".")
    if len(parts) <= 2:
        return hostname
    if parts[-2] in {"co", "com", "org", "gov", "ac"} and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])
