"""Asynchronous autonomous research agent."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
import shlex
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, Field, create_model, field_validator, model_validator

from browser import BrowserController, BrowserControllerError
from checkpoint import CheckpointManager
from list_page_extractor import ListPageExtractor
from page_state import PageState, PageStateParser
from llm import ActionType, LLMError, LLMGateway, StructuredEnvelope
from source_health import FailureReason, FetchOutcome, REGISTRY
from step_logger import StepArtifactLogger


LOGGER = logging.getLogger(__name__)
RESULT_LINK_RE = re.compile(r'link\s+"([^"]+)"\s+\[ref=(e\d+)\]', flags=re.IGNORECASE)


class ResearchAgentError(RuntimeError):
    """Raised when an autonomous research agent fails."""


class AgentDecisionBase(StructuredEnvelope):
    """Strict decision contract for a research step."""

    status: Literal["navigating", "extracting", "complete"] = Field(
        default="navigating",
        description='One of "navigating", "extracting", or "complete".'
    )
    thought_process: str | None = Field(
        default="",
        description="Brief reasoning for the next action.",
    )
    high_level_action: Literal[
        "extract_visible",
        "scroll_for_more",
        "open_candidate",
        "open_direct_url",
        "type_into_field",
        "wait_for_load",
        "switch_source",
        "finish_source",
    ] = Field(
        default="finish_source",
        description="Planner-level action intent compiled into a low-level browser command.",
    )
    action_type: ActionType = Field(
        default="none",
        description='One of "click", "type", "open_url", "wait", "scroll_down", "scroll_up", or "none".',
    )
    action_target: str | None = Field(
        default=None,
        description='Optional action target such as "@e1" or "https://example.com".',
    )
    action_value: str | None = Field(
        default=None,
        description="Optional action value such as text to type or seconds to wait.",
    )

    @field_validator("action_target", mode="before")
    @classmethod
    def normalize_action_target(cls, value: object) -> object:
        """Normalize near-valid element refs before strict validation runs."""
        if not isinstance(value, str):
            return value

        cleaned = value.strip()
        if not cleaned:
            return cleaned
        if cleaned.startswith("@") or cleaned.startswith(("http://", "https://")):
            return cleaned
        if cleaned.startswith("e") and cleaned[1:].isdigit():
            return f"@{cleaned}"
        if cleaned.isdigit():
            return f"@{cleaned}"
        return cleaned

    @field_validator("action_type", mode="before")
    @classmethod
    def normalize_action_type(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        lowered = value.strip().lower()
        high_level_map = {
            "extract_visible": "none",
            "scroll_for_more": "scroll_down",
            "open_candidate": "click",
            "open_direct_url": "open_url",
            "type_into_field": "type",
            "wait_for_load": "wait",
            "switch_source": "none",
            "finish_source": "none",
        }
        return high_level_map.get(lowered, lowered)

    @field_validator("action_value", mode="before")
    @classmethod
    def normalize_action_value(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return value

    @model_validator(mode="after")
    def validate_action(self) -> "AgentDecisionBase":
        """Reject malformed action payloads before they reach the browser CLI."""
        if self.action_value == "":
            self.action_value = None
        if self.action_type in {"scroll_down", "scroll_up"} and self.action_value is not None:
            try:
                if int(float(self.action_value)) <= 0:
                    self.action_value = None
            except ValueError:
                pass

        if (
            self.action_type == "open_url"
            and self.action_target is None
            and isinstance(self.action_value, str)
            and self.action_value.startswith(("http://", "https://"))
        ):
            self.action_target = self.action_value
            self.action_value = None

        self._normalize_from_high_level_action()

        if self.action_type == "none":
            if self.action_target is not None or self.action_value is not None:
                raise ValueError("action_target/action_value must be null when action_type='none'")
            return self

        if self.action_type == "click":
            if not self.action_target or not self.action_target.startswith("@e"):
                raise ValueError("click actions require an @eN action_target")
            if self.action_value is not None:
                raise ValueError("click actions do not accept action_value")
            return self

        if self.action_type == "type":
            if not self.action_target or not self.action_target.startswith("@e"):
                raise ValueError("type actions require an @eN action_target")
            if not self.action_value:
                raise ValueError("type actions require action_value text")
            return self

        if self.action_type == "open_url":
            if not self.action_target or not self.action_target.startswith(("http://", "https://")):
                raise ValueError("open_url actions require an http(s) action_target")
            if self.action_value is not None:
                raise ValueError("open_url actions do not accept action_value")
            return self

        if self.action_type == "wait":
            if self.action_target is not None:
                raise ValueError("wait actions do not accept action_target")
            if self.action_value is not None:
                try:
                    wait_seconds = float(self.action_value)
                except ValueError as exc:
                    raise ValueError("wait action_value must be numeric seconds") from exc
                if wait_seconds < 0:
                    raise ValueError("wait action_value must be non-negative")
            return self

        if self.action_type in {"scroll_down", "scroll_up"}:
            if self.action_target in {"page", "document"}:
                self.action_target = None
            if self.action_target is not None:
                raise ValueError(f"{self.action_type} actions do not accept action_target")
            if self.action_value is not None:
                try:
                    scroll_pixels = int(float(self.action_value))
                except ValueError as exc:
                    raise ValueError(f"{self.action_type} action_value must be numeric pixels") from exc
                if scroll_pixels <= 0:
                    raise ValueError(f"{self.action_type} action_value must be positive")
            return self

        raise ValueError(f"Unsupported action_type: {self.action_type}")

    def _normalize_from_high_level_action(self) -> None:
        if self.high_level_action == "extract_visible":
            self.action_type = "none"
            self.action_target = None
            self.action_value = None
        elif self.high_level_action == "scroll_for_more":
            self.action_type = "scroll_down"
            self.action_target = None
            if self.action_value is None:
                self.action_value = "700"
        elif self.high_level_action == "open_candidate":
            self.action_type = "click"
            self.action_value = None
        elif self.high_level_action == "open_direct_url":
            self.action_type = "open_url"
            self.action_value = None
        elif self.high_level_action == "type_into_field":
            self.action_type = "type"
        elif self.high_level_action == "wait_for_load":
            self.action_type = "wait"
            self.action_target = None
            if self.action_value is None:
                self.action_value = "3"
        elif self.high_level_action in {"switch_source", "finish_source"}:
            self.action_type = "none"
            self.action_target = None
            self.action_value = None


@dataclass(slots=True)
class ResearchAgent:
    """Async research worker that navigates the web and extracts structured rows."""

    name: str
    goal: str
    dataset_name: str
    row_model: type[BaseModel]
    starting_url: str
    target_record_count: int
    browser: BrowserController
    llm_gateway: LLMGateway | None = None
    checkpoint_manager: CheckpointManager = field(default_factory=CheckpointManager)
    max_steps: int = 12
    post_action_delay_seconds: float = 3.5
    existing_records: list[BaseModel] = field(default_factory=list)
    memory: list[BaseModel] = field(default_factory=list, init=False)
    baseline_record_count: int = field(default=0, init=False)
    last_browser_error: str | None = field(default=None, init=False)
    stall_counter: int = field(default=0, init=False)
    cloudflare_strikes: int = field(default=0, init=False)
    current_url: str | None = field(default=None, init=False)
    loop_warning: str | None = field(default=None, init=False)
    last_action_signature: str | None = field(default=None, init=False)
    repeated_action_count: int = field(default=0, init=False)
    last_snapshot_signature: str | None = field(default=None, init=False)
    repeated_snapshot_count: int = field(default=0, init=False)
    low_density_snapshot_count: int = field(default=0, init=False)
    no_progress_scroll_count: int = field(default=0, init=False)
    search_escape_attempted: bool = field(default=False, init=False)
    domain_blacklist: set[str] = field(default_factory=set)
    list_page_extractor: ListPageExtractor = field(init=False)
    page_state_parser: PageStateParser = field(init=False)
    artifact_logger: StepArtifactLogger = field(init=False)
    run_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.memory = list(self.existing_records)
        self.baseline_record_count = len(self.memory)
        self.current_url = self.starting_url
        self.list_page_extractor = ListPageExtractor(self.row_model)
        self.page_state_parser = PageStateParser()
        self.run_id = f"{self.name}_{uuid4().hex[:8]}"
        self.artifact_logger = StepArtifactLogger(
            root_dir=Path("artifacts") / "browser_runs",
            run_id=self.run_id,
        )

    async def run(self) -> list[BaseModel]:
        """Run the research loop asynchronously and return gathered rows."""
        if self._new_record_count() >= self.target_record_count:
            return self.memory

        await self._open_starting_url()
        try:
            for step in range(1, self.max_steps + 1):
                if self._new_record_count() >= self.target_record_count:
                    break

                if self.stall_counter >= 3 and await self._handle_stall():
                    break

                LOGGER.info(
                    "[%s] Step %d/%d records=%d/%d",
                    self.name,
                    step,
                    self.max_steps,
                    self._new_record_count(),
                    self.target_record_count,
                )
                snapshot = await self._snapshot()
                page_state = self.page_state_parser.parse(
                    snapshot,
                    current_url=self.current_url or self.starting_url,
                )
                self.artifact_logger.log_step(
                    step=step,
                    snapshot=snapshot,
                    page_state=page_state.model_dump(),
                    metadata={
                        "agent": self.name,
                        "goal": self.goal,
                        "records_collected": self._new_record_count(),
                    },
                )
                self._track_snapshot_pattern(snapshot)
                self._track_snapshot_density(snapshot)
                if self._snapshot_has_verification(snapshot):
                    self.cloudflare_strikes += 1
                    if self.cloudflare_strikes >= 2:
                        REGISTRY.record_fetch(
                            FetchOutcome(
                                url=self.current_url or self.starting_url,
                                ok=False,
                                reason=FailureReason.ANTI_BOT,
                                detail="browser verification loop",
                            )
                        )
                        self._blacklist_current_domain(reason="Cloudflare strike limit")
                        LOGGER.warning(
                            "[%s] Domain appears blocked by anti-bot verification at %s; stopping agent after %d consecutive strikes",
                            self.name,
                            self.starting_url,
                            self.cloudflare_strikes,
                        )
                        break
                else:
                    self.cloudflare_strikes = 0

                if self._should_abort_snapshot(snapshot):
                    self._blacklist_current_domain(reason="static or low-density browser snapshot")
                    break

                if self.llm_gateway is None:
                    added = self._extract_visible_rows(snapshot)
                    if added > 0:
                        LOGGER.info("[%s] Heuristically extracted %d rows from current page", self.name, added)
                        self.stall_counter = 0
                        if self._new_record_count() >= self.target_record_count:
                            break
                        continue

                if self.llm_gateway is None:
                    deterministic_action = self._deterministic_navigation(page_state)
                    if deterministic_action:
                        LOGGER.info("[%s] Applying deterministic navigation: %s", self.name, deterministic_action)
                        self.artifact_logger.log_step(
                            step=step,
                            snapshot=snapshot,
                            page_state=page_state.model_dump(),
                            decision=deterministic_action.model_dump(mode="json"),
                            metadata={"decision_source": "deterministic"},
                        )
                        if deterministic_action.status == "complete":
                            break
                        await self._execute_action(deterministic_action, prior_snapshot=snapshot)
                        self._track_action_pattern(deterministic_action)
                        self.stall_counter += 1
                        continue

                decision = await self._decide(page_state, snapshot)
                self._log_decision(step, decision)
                self.artifact_logger.log_step(
                    step=step,
                    snapshot=snapshot,
                    page_state=page_state.model_dump(),
                    decision=decision.model_dump(mode="json"),
                    metadata={"decision_source": "llm"},
                )

                added = 0
                if getattr(decision, "extracted_records", None):
                    added = self._append_records(decision.extracted_records)
                    LOGGER.info("[%s] Added %d new records", self.name, added)

                if added > 0:
                    self.stall_counter = 0
                else:
                    self.stall_counter += 1

                if decision.status == "complete" or getattr(decision, "high_level_action", "") in {"switch_source", "finish_source"}:
                    break

                await self._execute_action(decision, prior_snapshot=snapshot)
                self._track_action_pattern(decision)

            return self.memory
        finally:
            await asyncio.to_thread(self.browser.close)

    async def _open_starting_url(self) -> None:
        """Open the assigned direct source URL."""
        await asyncio.to_thread(self.browser.open, self.starting_url)
        self.current_url = self.starting_url

    async def _handle_stall(self) -> bool:
        """Break repeated no-progress loops by terminating bad sources early."""
        if self.current_url == self.starting_url:
            self._blacklist_current_domain(reason="stalled on starting URL")
            LOGGER.warning("[%s] Starting URL stalled without visible progress; ending agent", self.name)
            return True

        if self.repeated_snapshot_count >= 2 or self.no_progress_scroll_count >= 2:
            self._blacklist_current_domain(reason="stalled after leaving starting URL")
            return True

        LOGGER.info("[%s] Stall detected, reopening %s", self.name, self.starting_url)
        await self._open_starting_url()
        self.stall_counter = 0
        self.last_browser_error = None
        self.loop_warning = None
        return False

    async def _snapshot(self) -> str:
        """Capture a browser snapshot asynchronously."""
        try:
            return await asyncio.to_thread(self.browser.snapshot)
        except BrowserControllerError as exc:
            self.last_browser_error = str(exc)
            REGISTRY.record_fetch(
                FetchOutcome(
                    url=self.current_url or self.starting_url,
                    ok=False,
                    reason=FailureReason.BROWSER_ERROR,
                    detail=str(exc),
                )
            )
            raise ResearchAgentError(str(exc)) from exc

    async def _execute_action(self, decision: BaseModel, *, prior_snapshot: str | None = None) -> None:
        """Compile a validated action and apply it safely."""
        action_type = getattr(decision, "action_type", "none")
        if action_type == "none":
            return

        if action_type == "wait":
            wait_seconds = float(getattr(decision, "action_value", None) or self.post_action_delay_seconds)
            await asyncio.sleep(wait_seconds)
            self.last_browser_error = None
            return

        command = self._compile_browser_command(decision)
        before_scroll_signature: str | None = None
        if action_type in {"scroll_down", "scroll_up"}:
            before_scroll_signature = self._snapshot_material_signature(prior_snapshot or await self._snapshot())
        try:
            await asyncio.to_thread(self.browser.execute, command)
            self.last_browser_error = None
            if action_type == "open_url":
                self.current_url = getattr(decision, "action_target", self.current_url)
            elif action_type == "click":
                self.current_url = None
            await asyncio.sleep(self.post_action_delay_seconds)
            if action_type in {"scroll_down", "scroll_up"} and before_scroll_signature is not None:
                post_scroll_snapshot = await self._snapshot()
                after_scroll_signature = self._snapshot_material_signature(post_scroll_snapshot)
                if after_scroll_signature == before_scroll_signature:
                    self.no_progress_scroll_count += 1
                    self.loop_warning = (
                        "SYSTEM WARNING: Scrolling did not reveal new data. The page is static or blocked. "
                        "Choose a different action or navigate away."
                    )
                    self.stall_counter = max(self.stall_counter, 3)
                    LOGGER.warning("[%s] Scroll action did not reveal new data; forcing stall", self.name)
                else:
                    self.no_progress_scroll_count = 0
        except BrowserControllerError as exc:
            self.last_browser_error = str(exc)
            LOGGER.warning("[%s] Browser command failed: %s", self.name, exc)

    async def _decide(self, page_state: PageState, snapshot: str) -> BaseModel:
        """Ask the LLM what to do next."""
        if self.llm_gateway is None:
            return AgentDecisionBase(
                status="complete",
                thought_process="No LLM gateway available for autonomous navigation.",
                high_level_action="finish_source",
                action_type="none",
                action_target=None,
                action_value=None,
            )
        decision_model = create_model(
            f"{self.name.title().replace('-', '')}Decision",
            __base__=AgentDecisionBase,
            extracted_records=(list[self.row_model] | None, Field(default=None)),
        )
        existing_records = json.dumps(
            [record.model_dump(mode="json") for record in self.memory[-5:]],
            separators=(",", ":"),
            sort_keys=True,
        )
        row_schema = self._prompt_schema_summary()
        system_prompt = (
            "You are an autonomous web scraping agent working inside a larger data engineering swarm. "
            "You receive a structured page state, a research goal, and the rows already collected. "
            "You must either navigate, extract validated rows, or finish. "
            "Return valid JSON only. "
            "Choose the highest-level viable action first. "
            "Prefer these planner actions over raw browser micromanagement: extract_visible, scroll_for_more, open_candidate, open_direct_url, wait_for_load, switch_source, finish_source. "
            "Only extract records grounded in visible page state. "
            "When you land on a page, look for lists, tables, or directories that match the schema. "
            "If the page has row-like content, prefer extract_visible. "
            "If the page looks incomplete or below the fold, prefer scroll_for_more. "
            "Prioritize extracting data from visible tables or directories rather than clicking into individual profile links. "
            "Your goal is to extract as many valid rows as possible from the current page before navigating away. "
            "If some fields are missing but the record is otherwise strong, use null rather than inventing data. "
            "If your goal is to build a directory or list of entities, prefer the current index/list page over detail pages."
        )
        user_prompt = (
            f"Goal:\n{self.goal}\n\n"
            f"Dataset name:\n{self.dataset_name}\n\n"
            f"Assigned starting URL:\n{self.starting_url}\n\n"
            f"Current new record count: {self._new_record_count()} / {self.target_record_count}\n\n"
            f"Recent records:\n{existing_records}\n\n"
            f"Last browser error:\n{self.last_browser_error or 'None'}\n\n"
            f"System warning:\n{self.loop_warning or 'None'}\n\n"
            f"Row schema:\n{row_schema}\n\n"
            f"Structured page state:\n{page_state.compact_summary()}\n\n"
            f"Raw snapshot excerpt:\n{self._trim_snapshot(snapshot, limit=7000)}\n\n"
            "Rules:\n"
            "- Use status='navigating' when moving around the site.\n"
            "- Use status='extracting' when you can emit records, including from directory/list pages.\n"
            "- Use status='complete' when this agent is unlikely to add better records.\n"
            "- Set high_level_action='extract_visible' when the current page already contains row-like content.\n"
            "- Set high_level_action='scroll_for_more' before navigating away if row-like content may be below the fold.\n"
            "- Set high_level_action='open_candidate' only with action_target=@eN from clickable elements listed in page state.\n"
            "- Set high_level_action='open_direct_url' only for explicit http(s) targets.\n"
            "- Set high_level_action='wait_for_load' with numeric seconds when needed.\n"
            "- Set high_level_action='switch_source' or 'finish_source' when this page is a poor source.\n"
            "- If the page already lists many teams with visible fields, extract them immediately rather than insisting on a detail page.\n"
            "- For directory-style goals, prefer extracting team names, schools, conferences, and visible team links from listing pages.\n"
            "- Do not repeat the same click or reopen the same URL if it did not change the page."
        )
        try:
            decision = await asyncio.to_thread(
                self.llm_gateway.complete_structured,
                response_model=decision_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name=f"{self.name}_decision",
                max_tokens=1200,
            )
            self.loop_warning = None
            return decision
        except LLMError as exc:
            LOGGER.warning("[%s] Structured decision fallback triggered: %s", self.name, exc)
            try:
                fallback_text = await asyncio.to_thread(
                    self.llm_gateway.complete_text,
                    system_prompt=(
                        f"{system_prompt} "
                        "Provider strict schema generation failed. Return JSON only, with no markdown or prose."
                    ),
                    user_prompt=(
                        f"{user_prompt}\n\n"
                        "Return a single JSON object with exactly these keys: "
                        'status, thought_process, high_level_action, action_type, action_target, action_value, extracted_records.'
                    ),
                    max_tokens=900,
                )
                decision = decision_model.model_validate_json(_extract_json_object(fallback_text))
                self.loop_warning = None
                return decision
            except (LLMError, ValueError) as fallback_exc:
                raise ResearchAgentError(str(fallback_exc)) from fallback_exc

    def _append_records(self, records: list[BaseModel]) -> int:
        """Append deduplicated, minimally usable records and checkpoint them."""
        existing = {
            record.model_dump_json(exclude_none=True, exclude_defaults=True)
            for record in self.memory
        }
        added = 0
        for record in records:
            record = self._enrich_record(record)
            if not self._record_is_usable(record):
                continue
            fingerprint = record.model_dump_json(exclude_none=True, exclude_defaults=True)
            if fingerprint in existing:
                continue
            self.memory.append(record)
            existing.add(fingerprint)
            self.checkpoint_manager.append_record(
                goal=self.goal,
                dataset_name=self.dataset_name,
                row_model=self.row_model,
                record=record,
            )
            added += 1
            if self._new_record_count() >= self.target_record_count:
                break
        return added

    def _new_record_count(self) -> int:
        return max(len(self.memory) - self.baseline_record_count, 0)

    def _extract_visible_rows(self, snapshot: str) -> int:
        """Extract rows directly from dense list pages before asking the LLM to navigate."""
        if not self.list_page_extractor.should_extract(snapshot):
            return 0
        records = self.list_page_extractor.extract(
            snapshot,
            source_url=self.current_url or self.starting_url,
        )
        if not records:
            return 0
        return self._append_records(records)

    def _deterministic_navigation(self, page_state: PageState) -> AgentDecisionBase | None:
        """Use simple rules to escape search result pages without another LLM call."""
        lowered = page_state.visible_text_summary.lower()
        if "search results" not in lowered and "search results" not in " ".join(page_state.blocked_signals):
            return None

        goal_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", self.goal.lower())
            if token not in {"build", "dataset", "data", "of", "the", "and", "for", "a"}
        }

        best_ref: str | None = None
        best_score = -1
        for element in page_state.clickable_elements:
            label = element.label.strip()
            ref = element.ref.removeprefix("@")
            score = 0
            label_lower = label.lower()
            if "list of" in label_lower:
                score += 5
            score += sum(1 for token in goal_tokens if token in label_lower)
            if any(blocked in label_lower for blocked in ("donate", "help", "create account", "log in")):
                score = -1
            if score > best_score:
                best_score = score
                best_ref = ref

        if not best_ref or best_score <= 0:
            return None

        signature = f"click|@{best_ref}|None"
        if self.last_action_signature == signature and self.repeated_snapshot_count >= 1:
            self.loop_warning = (
                "SYSTEM WARNING: The same search-result click did not change the page. "
                "This search page is not helping. Finish or switch sources."
            )
            return AgentDecisionBase(
                status="complete",
                thought_process="Search results are repeating without progress; ending this source.",
                high_level_action="finish_source",
                action_type="none",
                action_target=None,
                action_value=None,
            )

        return AgentDecisionBase(
            status="navigating",
            thought_process="Deterministically clicking the most relevant search result.",
            high_level_action="open_candidate",
            action_type="click",
            action_target=f"@{best_ref}",
            action_value=None,
        )

    def _track_action_pattern(self, decision: BaseModel) -> None:
        """Track repeated actions so we can break obvious no-progress loops."""
        action_type = getattr(decision, "action_type", "none")
        action_target = getattr(decision, "action_target", None)
        action_value = getattr(decision, "action_value", None)
        signature = f"{action_type}|{action_target}|{action_value}"

        if signature == self.last_action_signature:
            self.repeated_action_count += 1
        else:
            self.last_action_signature = signature
            self.repeated_action_count = 0

        if self.repeated_action_count >= 2:
            self.loop_warning = (
                "SYSTEM WARNING: You repeated the same action multiple times without progress. "
                "Extract from the current page, click a different link, or finish."
            )

    def _track_snapshot_pattern(self, snapshot: str) -> None:
        """Track whether the visible page is materially changing between steps."""
        signature = self._snapshot_signature(snapshot)
        if signature == self.last_snapshot_signature:
            self.repeated_snapshot_count += 1
        else:
            self.last_snapshot_signature = signature
            self.repeated_snapshot_count = 0

        if self.repeated_snapshot_count >= 2:
            self.loop_warning = (
                "SYSTEM WARNING: The page snapshot has not materially changed across multiple steps. "
                "Extract from the current page, navigate elsewhere, or finish."
            )

    def _track_snapshot_density(self, snapshot: str) -> None:
        """Track whether the snapshot contains enough row-like signals to justify more browsing."""
        if self._snapshot_row_signal_count(snapshot) >= 5:
            self.low_density_snapshot_count = 0
            return
        self.low_density_snapshot_count += 1
        if self.low_density_snapshot_count >= 2:
            self.loop_warning = (
                "SYSTEM WARNING: The snapshot contains little visible row data. "
                "This page is likely blocked, empty, or a poor source. Switch sources or finish."
            )

    def _should_abort_snapshot(self, snapshot: str) -> bool:
        if self.repeated_snapshot_count >= 2 and self.low_density_snapshot_count >= 2:
            REGISTRY.record_extraction(
                self.current_url or self.starting_url,
                records_extracted=0,
                success=False,
                reason=FailureReason.SNAPSHOT_LOW_DENSITY,
            )
            return True
        if self.no_progress_scroll_count >= 2 and self._snapshot_row_signal_count(snapshot) < 3:
            REGISTRY.record_extraction(
                self.current_url or self.starting_url,
                records_extracted=0,
                success=False,
                reason=FailureReason.SNAPSHOT_STATIC,
            )
            return True
        return False

    def _prompt_schema_summary(self) -> str:
        """Compress the row schema so provider message limits are less likely to trip."""
        parts: list[str] = []
        required_fields = set(self.row_model.model_json_schema().get("required", []))
        for field_name, model_field in self.row_model.model_fields.items():
            annotation = getattr(model_field.annotation, "__name__", str(model_field.annotation))
            required_flag = "required" if field_name in required_fields else "optional"
            description = model_field.description or ""
            extra = model_field.json_schema_extra or {}
            ml_role = extra.get("x-ml-role")
            role_prefix = f"{ml_role} " if ml_role in {"target", "feature"} else ""
            parts.append(f"{field_name} ({role_prefix}{annotation}, {required_flag}): {description}".strip())
        return "\n".join(parts)

    def _trim_snapshot(self, snapshot: str, limit: int = 5000) -> str:
        """Trim large browser snapshots to stay under provider message limits."""
        normalized = snapshot.strip()
        if len(normalized) <= limit:
            return normalized
        head = normalized[: limit // 2]
        tail = normalized[-limit // 2 :]
        return f"{head}\n...[truncated]...\n{tail}"

    def _snapshot_signature(self, snapshot: str) -> str:
        lines = [line.strip() for line in snapshot.splitlines() if line.strip()]
        return "\n".join(lines[:25])

    def _snapshot_material_signature(self, snapshot: str) -> str:
        normalized = "\n".join(line.strip() for line in snapshot.splitlines() if line.strip())
        return f"{len(normalized)}:{hash(normalized)}"

    def _enrich_record(self, record: BaseModel) -> BaseModel:
        """Fill obvious missing fields from agent context before usability checks."""
        payload = record.model_dump(mode="json")
        changed = False

        if "source_url" in self.row_model.model_fields and not payload.get("source_url"):
            payload["source_url"] = self.current_url or self.starting_url
            changed = True

        if "season_year" in self.row_model.model_fields and payload.get("season_year") is None:
            inferred_year = self._infer_season_year()
            if inferred_year is not None:
                payload["season_year"] = inferred_year
                changed = True

        if not changed:
            return record
        return self.row_model.model_validate(payload)

    def _infer_season_year(self) -> int | None:
        """Infer a season year from the goal or URL when the page omits it."""
        for candidate in (self.goal, self.current_url or "", self.starting_url):
            for token in candidate.split("/"):
                if token.isdigit() and len(token) == 4:
                    year = int(token)
                    if 1900 <= year <= 2100:
                        return year
        return None

    def _snapshot_has_verification(self, snapshot: str) -> bool:
        """Detect common anti-bot interstitials before spending more LLM calls."""
        lowered = snapshot.lower()
        phrases = (
            "cloudflare",
            "verify you are human",
            "just a moment",
            "enable javascript and cookies",
        )
        return any(phrase in lowered for phrase in phrases)

    def _snapshot_row_signal_count(self, snapshot: str) -> int:
        return self.list_page_extractor.candidate_count(snapshot)

    def _record_is_usable(self, record: BaseModel) -> bool:
        """Reject rows with too little usable signal."""
        payload = record.model_dump(mode="json")
        if self._record_looks_like_ui_chrome(payload):
            return False
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

    def _record_looks_like_ui_chrome(self, payload: dict[str, object]) -> bool:
        """Reject records that clearly came from navigation or page chrome instead of entities."""
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
            "create a draft and submit it for review",
        )
        for value in payload.values():
            if not isinstance(value, str):
                continue
            lowered = value.lower()
            if any(phrase in lowered for phrase in blocked_phrases):
                return True
        return False

    def _log_decision(self, step: int, decision: BaseModel) -> None:
        """Emit a concise decision log."""
        status = getattr(decision, "status", "unknown")
        thought = getattr(decision, "thought_process", "")
        high_level_action = getattr(decision, "high_level_action", "unknown")
        LOGGER.info(
            "[%s] Step %d decision=%s planner_action=%s thought=%s",
            self.name,
            step,
            status,
            high_level_action,
            thought,
        )
        action_type = getattr(decision, "action_type", "none")
        if action_type != "none":
            LOGGER.info(
                "[%s] Step %d action=%s target=%s",
                self.name,
                step,
                action_type,
                getattr(decision, "action_target", None),
            )

    def _compile_browser_command(self, decision: BaseModel) -> str:
        """Compile the structured action fields into one agent-browser command."""
        action_type = getattr(decision, "action_type")
        action_target = getattr(decision, "action_target", None)
        action_value = getattr(decision, "action_value", None)

        if action_type == "open_url":
            return f"open {action_target}"
        if action_type == "click":
            return f"click {action_target}"
        if action_type == "type":
            return f"type {action_target} {shlex.quote(action_value)}"
        if action_type == "scroll_down":
            scroll_amount = int(float(action_value)) if action_value is not None else 700
            return f"scroll down {scroll_amount}"
        if action_type == "scroll_up":
            scroll_amount = int(float(action_value)) if action_value is not None else 500
            return f"scroll up {scroll_amount}"
        raise ResearchAgentError(f"Cannot compile unsupported action_type={action_type}")

    def _blacklist_current_domain(self, *, reason: str) -> None:
        current = self.current_url or self.starting_url
        domain = _root_domain(current)
        if not domain:
            return
        self.domain_blacklist.add(domain)
        REGISTRY.record_fetch(
            FetchOutcome(
                url=current,
                ok=False,
                reason=FailureReason.BROWSER_ERROR,
                detail=reason,
            )
        )
        LOGGER.warning("[%s] Blacklisting domain %s due to %s", self.name, domain, reason)


def _extract_json_object(value: str) -> str:
    """Extract the outermost JSON object from a free-form model response."""
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Fallback LLM response did not contain a JSON object")
    return value[start : end + 1]


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
