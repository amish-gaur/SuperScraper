"""Asynchronous autonomous research agent."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote_plus

from pydantic import BaseModel, Field, create_model

from browser import BrowserController, BrowserControllerError
from llm import LLMError, LLMGateway, StructuredEnvelope


LOGGER = logging.getLogger(__name__)


class ResearchAgentError(RuntimeError):
    """Raised when an autonomous research agent fails."""


class ResearchDecisionBase(StructuredEnvelope):
    """Base decision contract for a research step."""

    status: str = Field(description='One of "searching", "extracting", or "complete".')
    thought_process: str = Field(description="Brief reasoning for the next action.")
    next_command: str | None = Field(
        default=None,
        description="A single executable agent-browser command.",
    )
    extracted_records: list[BaseModel] | None = Field(
        default=None,
        description="Records extracted from the current page, or null.",
    )


@dataclass(slots=True)
class ResearchAgent:
    """Async research worker that navigates the web and extracts structured rows."""

    name: str
    goal: str
    row_model: type[BaseModel]
    search_query: str
    target_record_count: int
    browser: BrowserController
    llm_gateway: LLMGateway = field(default_factory=LLMGateway)
    max_steps: int = 12
    search_engine_url_template: str = "https://www.bing.com/search?q={query}"
    post_action_delay_seconds: float = 1.5
    memory: list[BaseModel] = field(default_factory=list, init=False)
    last_browser_error: str | None = field(default=None, init=False)
    stall_counter: int = field(default=0, init=False)
    pivot_suffixes: tuple[str, ...] = (
        "statistics",
        "official roster",
        "press release",
        "scouting report",
    )
    pivot_index: int = field(default=0, init=False)

    async def run(self) -> list[BaseModel]:
        """Run the research loop asynchronously and return gathered rows."""
        await self._open_search(self.search_query)
        try:
            for step in range(1, self.max_steps + 1):
                if len(self.memory) >= self.target_record_count:
                    break

                if self.stall_counter >= 3:
                    await self._pivot_search()

                LOGGER.info(
                    "[%s] Step %d/%d records=%d/%d",
                    self.name,
                    step,
                    self.max_steps,
                    len(self.memory),
                    self.target_record_count,
                )
                snapshot = await self._snapshot()
                decision = await self._decide(snapshot)
                self._log_decision(step, decision)

                added = 0
                if decision.extracted_records:
                    added = self._append_records(decision.extracted_records)
                    LOGGER.info("[%s] Added %d new records", self.name, added)

                if added > 0:
                    self.stall_counter = 0
                else:
                    self.stall_counter += 1

                if decision.status == "complete":
                    break

                if decision.next_command:
                    await self._execute(decision.next_command)

            return self.memory[: self.target_record_count]
        finally:
            await asyncio.to_thread(self.browser.close)

    async def _open_search(self, query: str) -> None:
        """Open the search engine with the given query."""
        url = self.search_engine_url_template.format(query=quote_plus(query))
        await asyncio.to_thread(self.browser.open, url)

    async def _pivot_search(self) -> None:
        """Force a fresh search query when the agent stalls."""
        suffix = self.pivot_suffixes[self.pivot_index % len(self.pivot_suffixes)]
        self.pivot_index += 1
        LOGGER.info("[%s] Stall detected, pivoting search with suffix '%s'", self.name, suffix)
        await self._open_search(f"{self.search_query} {suffix}")
        self.stall_counter = 0
        self.last_browser_error = None

    async def _snapshot(self) -> str:
        """Capture a browser snapshot asynchronously."""
        try:
            return await asyncio.to_thread(self.browser.snapshot)
        except BrowserControllerError as exc:
            self.last_browser_error = str(exc)
            raise ResearchAgentError(str(exc)) from exc

    async def _execute(self, command: str) -> None:
        """Execute a single browser command asynchronously."""
        normalized_command = self._normalize_command(command)
        try:
            await asyncio.to_thread(self.browser.execute, normalized_command)
            self.last_browser_error = None
            await asyncio.sleep(self.post_action_delay_seconds)
        except BrowserControllerError as exc:
            self.last_browser_error = str(exc)
            LOGGER.warning("[%s] Browser command failed: %s", self.name, exc)

    async def _decide(self, snapshot: str) -> BaseModel:
        """Ask the LLM what to do next."""
        decision_model = create_model(
            f"{self.name.title().replace('-', '')}Decision",
            __base__=ResearchDecisionBase,
            extracted_records=(list[self.row_model] | None, Field(default=None)),
        )
        existing_records = json.dumps(
            [record.model_dump(mode="json") for record in self.memory],
            indent=2,
            sort_keys=True,
        )
        row_schema = json.dumps(self.row_model.model_json_schema(), indent=2, sort_keys=True)
        system_prompt = (
            "You are an autonomous web scraping agent working inside a larger data engineering swarm. "
            "You receive a page snapshot, a research goal, and the rows already collected. "
            "You must either navigate, extract validated rows, or finish. "
            "Return valid JSON only. Use exactly one browser command when next_command is provided. "
            "Only extract records grounded in visible page text. "
            "If some fields are missing but the record is otherwise strong, use null rather than inventing data."
        )
        user_prompt = (
            f"Goal:\n{self.goal}\n\n"
            f"Agent search strategy:\n{self.search_query}\n\n"
            f"Current record count: {len(self.memory)} / {self.target_record_count}\n\n"
            f"Existing records:\n{existing_records}\n\n"
            f"Last browser error:\n{self.last_browser_error or 'None'}\n\n"
            f"Row schema:\n{row_schema}\n\n"
            f"Browser snapshot:\n{snapshot}\n\n"
            "Use status='searching' to navigate, status='extracting' when you can emit records, "
            "and status='complete' when this agent is no longer likely to find better records."
        )
        try:
            return await asyncio.to_thread(
                self.llm_gateway.complete_structured,
                response_model=decision_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name=f"{self.name}_decision",
                max_tokens=1200,
            )
        except LLMError as exc:
            raise ResearchAgentError(str(exc)) from exc

    def _append_records(self, records: list[BaseModel]) -> int:
        """Append deduplicated, minimally usable records."""
        existing = {
            record.model_dump_json(exclude_none=True, exclude_defaults=True)
            for record in self.memory
        }
        added = 0
        for record in records:
            if not self._record_is_usable(record):
                continue
            fingerprint = record.model_dump_json(exclude_none=True, exclude_defaults=True)
            if fingerprint in existing:
                continue
            self.memory.append(record)
            existing.add(fingerprint)
            added += 1
            if len(self.memory) >= self.target_record_count:
                break
        return added

    def _record_is_usable(self, record: BaseModel) -> bool:
        """Reject rows with too little usable signal."""
        payload = record.model_dump(mode="json")
        populated = 0
        for value in payload.values():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, list) and not value:
                continue
            populated += 1
        return populated >= 2

    def _log_decision(self, step: int, decision: BaseModel) -> None:
        """Emit a concise decision log."""
        status = getattr(decision, "status", "unknown")
        thought = getattr(decision, "thought_process", "")
        LOGGER.info("[%s] Step %d decision=%s thought=%s", self.name, step, status, thought)
        next_command = getattr(decision, "next_command", None)
        if next_command:
            LOGGER.info("[%s] Step %d next command=%s", self.name, step, next_command)

    def _normalize_command(self, command: str) -> str:
        """Coerce fuzzy model output into a valid single agent-browser command."""
        cleaned = command.strip()
        lowered = cleaned.lower()

        if re.fullmatch(r"https?://\S+", cleaned):
            return f"open {cleaned}"

        if lowered.startswith("navigate to "):
            return f"open {cleaned[12:].strip()}"

        if lowered.startswith("navigate "):
            return f"open {cleaned[9:].strip()}"

        if lowered.startswith("goto "):
            return f"open {cleaned[5:].strip()}"

        if lowered.startswith("open https://") or lowered.startswith("open http://"):
            return cleaned

        click_match = re.fullmatch(r"click\s+@?(e\d+)", cleaned, flags=re.IGNORECASE)
        if click_match:
            return f"click @{click_match.group(1).lower()}"

        ref_click_match = re.fullmatch(
            r"click element link\[ref=(e\d+)\]",
            cleaned,
            flags=re.IGNORECASE,
        )
        if ref_click_match:
            return f"click @{ref_click_match.group(1).lower()}"

        selector_click_match = re.fullmatch(
            r"click element with selector ['\"](.+?)['\"]",
            cleaned,
            flags=re.IGNORECASE,
        )
        if selector_click_match:
            return f"click {selector_click_match.group(1)}"

        if lowered.startswith(("input ", "type ", "fill ")):
            query = self._extract_query_from_input_command(cleaned)
            if query:
                url = self.search_engine_url_template.format(query=quote_plus(query))
                return f"open {url}"

        query = self._extract_query_from_search_phrase(cleaned)
        if query:
            url = self.search_engine_url_template.format(query=quote_plus(query))
            return f"open {url}"

        if lowered.startswith("click the first search result"):
            url = self.search_engine_url_template.format(query=quote_plus(self.search_query))
            return f"open {url}"

        return cleaned

    def _extract_query_from_input_command(self, command: str) -> str | None:
        """Extract the intended search query from a fuzzy input command."""
        if "\"" in command:
            parts = command.split("\"")
            if len(parts) >= 3:
                return parts[1].strip()

        match = re.match(r"^(?:input|type|fill)\s+@?e\d+\s+(.+)$", command, flags=re.IGNORECASE)
        if not match:
            return None

        query = match.group(1)
        for separator in (" then click ", " and click ", " then press ", " and press "):
            separator_index = query.lower().find(separator)
            if separator_index != -1:
                query = query[:separator_index]
                break
        return query.strip()

    def _extract_query_from_search_phrase(self, command: str) -> str | None:
        """Extract a search query from natural-language search instructions."""
        quoted = re.search(r'"([^"]+)"', command)
        if quoted:
            return quoted.group(1).strip()

        value_match = re.search(r"value=['\"]([^'\"]+)['\"]", command, flags=re.IGNORECASE)
        if value_match:
            return value_match.group(1).strip()

        if command.lower().startswith("search using query "):
            return command[len("search using query ") :].strip()

        return None
