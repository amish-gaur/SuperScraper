"""Intermediate extraction router between raw fetches and browser automation."""

from __future__ import annotations

from dataclasses import dataclass, field
import html
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from architect import SourceTarget
from domain_adapters import DomainAdapter, build_domain_adapters
from html_table_extractor import HtmlTableExtractor
from source_health import FailureReason, FetchOutcome, fetch_url
from synthesizer import DataSynthesizer


LOGGER = logging.getLogger(__name__)

SCRIPT_TAG_RE = re.compile(
    r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>",
    flags=re.IGNORECASE | re.DOTALL,
)
SCRIPT_ATTR_RE = re.compile(r'([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*["\'](.*?)["\']', flags=re.DOTALL)
WINDOW_STATE_MARKERS = (
    "window.__INITIAL_STATE__",
    "window.__PRELOADED_STATE__",
)


@dataclass(slots=True)
class RouteDecision:
    """Outcome of routing one source target through the intermediate layer."""

    source_target: SourceTarget
    strategy: str
    records: list[BaseModel] = field(default_factory=list)
    requires_browser: bool = False
    fetch_outcome: FetchOutcome | None = None
    raw_payload: dict[str, Any] | None = None


@dataclass(slots=True)
class StateSniffer:
    """Extract hydration and preloaded JSON blobs from raw HTML."""

    def sniff(self, html_text: str) -> dict[str, Any] | None:
        payloads: list[dict[str, Any]] = []

        for attrs, body in self._iter_script_tags(html_text):
            script_id = str(attrs.get("id", "")).strip()
            script_type = str(attrs.get("type", "")).strip().lower()
            if script_id == "__NEXT_DATA__" and script_type == "application/json":
                payload = self._safe_json_loads(html.unescape(body.strip()))
                if payload is not None:
                    payloads.append({"kind": "__NEXT_DATA__", "data": payload})

        for marker in WINDOW_STATE_MARKERS:
            extracted = self._extract_javascript_value(html_text, marker)
            if isinstance(extracted, (dict, list)):
                payloads.append({"kind": marker, "data": extracted})

        if not payloads:
            return None

        combined: dict[str, Any] = {}
        flattened: dict[str, Any] = {}
        candidate_collections: list[dict[str, Any]] = []
        for index, payload in enumerate(payloads):
            key = f"payload_{index}_{payload['kind']}"
            combined[key] = payload["data"]
            flattened.update(_flatten_json(payload["data"], prefix=payload["kind"]))
            candidate_collections.extend(_extract_candidate_collections(payload["data"], prefix=payload["kind"]))
        return {
            "payloads": payloads,
            "combined": combined,
            "flattened": flattened,
            "candidate_collections": candidate_collections[:25],
        }

    def _iter_script_tags(self, html_text: str) -> list[tuple[dict[str, str], str]]:
        scripts: list[tuple[dict[str, str], str]] = []
        for match in SCRIPT_TAG_RE.finditer(html_text):
            attrs = {
                key.lower(): value
                for key, value in SCRIPT_ATTR_RE.findall(match.group("attrs") or "")
            }
            scripts.append((attrs, match.group("body") or ""))
        return scripts

    def _safe_json_loads(self, value: str) -> Any | None:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                import ast

                python_like = self._replace_javascript_literals(value)
                parsed = ast.literal_eval(python_like)
            except Exception:
                return None
        return parsed

    def _extract_javascript_value(self, html_text: str, marker: str) -> Any | None:
        marker_index = html_text.find(marker)
        if marker_index == -1:
            return None
        equals_index = html_text.find("=", marker_index)
        if equals_index == -1:
            return None
        start_index = self._find_value_start(html_text, equals_index + 1)
        if start_index == -1:
            return None
        candidate = self._extract_balanced_value(html_text, start_index)
        if candidate is None:
            return None
        normalized = self._normalize_javascript_object(candidate)
        return self._safe_json_loads(normalized)

    def _find_value_start(self, html_text: str, start_index: int) -> int:
        for index in range(start_index, len(html_text)):
            if not html_text[index].isspace():
                return index
        return -1

    def _extract_balanced_value(self, html_text: str, start_index: int) -> str | None:
        opening = html_text[start_index]
        pairs = {"{": "}", "[": "]"}
        closing = pairs.get(opening)
        if closing is None:
            return None
        depth = 0
        in_string = False
        escape = False
        quote_char = ""
        for index in range(start_index, len(html_text)):
            char = html_text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote_char:
                    in_string = False
                continue

            if char in {'"', "'"}:
                in_string = True
                quote_char = char
                continue
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return html_text[start_index : index + 1]
        return None

    def _normalize_javascript_object(self, value: str) -> str:
        cleaned = value.strip()
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        cleaned = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_$]*)(\s*:)', r'\1"\2"\3', cleaned)
        cleaned = re.sub(r":\s*undefined\b", ": null", cleaned)
        return cleaned

    def _replace_javascript_literals(self, value: str) -> str:
        replacements = {
            "undefined": "None",
            "null": "None",
            "true": "True",
            "false": "False",
        }
        characters: list[str] = []
        index = 0
        in_string = False
        escape = False
        quote_char = ""
        while index < len(value):
            char = value[index]
            if in_string:
                characters.append(char)
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote_char:
                    in_string = False
                index += 1
                continue
            if char in {'"', "'"}:
                in_string = True
                quote_char = char
                characters.append(char)
                index += 1
                continue
            if char.isalpha():
                end_index = index + 1
                while end_index < len(value) and (value[end_index].isalnum() or value[end_index] == "_"):
                    end_index += 1
                token = value[index:end_index]
                characters.append(replacements.get(token, token))
                index = end_index
                continue
            characters.append(char)
            index += 1
        return "".join(characters)


@dataclass(slots=True)
class ExtractionRouter:
    """Route sources through adapters, hydration sniffing, deterministic tables, then browser."""

    goal: str
    row_model: type[BaseModel]
    llm_gateway: Any | None = None
    timeout_seconds: float = 20.0
    domain_blacklist: set[str] | None = None
    adapters: list[DomainAdapter] = field(default_factory=build_domain_adapters)
    state_sniffer: StateSniffer = field(default_factory=StateSniffer)

    def route(self, source_target: SourceTarget) -> RouteDecision:
        adapter = self._select_adapter(source_target)
        if adapter is not None:
            payload = adapter.fetch_payload(source_target)
            if payload:
                records = self._synthesize_payload(source_target, payload, strategy="domain_adapter")
                if records:
                    return RouteDecision(
                        source_target=source_target,
                        strategy="domain_adapter",
                        records=records,
                        raw_payload=payload,
                    )

        outcome = self._fetch_html(source_target.url)
        if not outcome.ok or not outcome.text:
            return RouteDecision(
                source_target=source_target,
                strategy="fetch_failed",
                requires_browser=self._should_browser_fallback(source_target),
                fetch_outcome=outcome,
            )

        sniffed = self.state_sniffer.sniff(outcome.text)
        if sniffed:
            records = self._synthesize_payload(source_target, sniffed, strategy="react_state")
            if records:
                return RouteDecision(
                    source_target=source_target,
                    strategy="react_state",
                    records=records,
                    fetch_outcome=outcome,
                    raw_payload=sniffed,
                )

        html_records = HtmlTableExtractor(
            self.row_model,
            timeout_seconds=self.timeout_seconds,
            domain_blacklist=self.domain_blacklist,
        ).extract_from_html(source_target.url, outcome.text)
        if html_records:
            return RouteDecision(
                source_target=source_target,
                strategy="html_table",
                records=html_records,
                fetch_outcome=outcome,
            )

        return RouteDecision(
            source_target=source_target,
            strategy="browser",
            requires_browser=self._should_browser_fallback(source_target),
            fetch_outcome=outcome,
            raw_payload=sniffed,
        )

    def _fetch_html(self, url: str) -> FetchOutcome:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; WebScraperPrototype/1.0; "
                "+https://example.com/web-scraper)"
            )
        }
        outcome = fetch_url(
            url,
            headers=headers,
            timeout_seconds=self.timeout_seconds,
            verify=False,
        )
        if outcome.reason in {FailureReason.HTTP_403, FailureReason.HTTP_429, FailureReason.ANTI_BOT}:
            self._blacklist_domain(url)
        return outcome

    def _blacklist_domain(self, url: str) -> None:
        if self.domain_blacklist is None:
            return
        domain = _root_domain(url)
        if domain:
            self.domain_blacklist.add(domain)

    def _select_adapter(self, source_target: SourceTarget) -> DomainAdapter | None:
        for adapter in self.adapters:
            if adapter.matches(source_target):
                return adapter
        return None

    def _synthesize_payload(
        self,
        source_target: SourceTarget,
        payload: dict[str, Any],
        *,
        strategy: str,
    ) -> list[BaseModel]:
        synthesizer = DataSynthesizer(row_model=self.row_model, llm_gateway=self.llm_gateway)
        enriched_payload = self._enrich_payload(payload)
        try:
            return synthesizer.synthesize_state_payload(
                goal=self.goal,
                source_url=source_target.url,
                state_payload=enriched_payload,
                strategy=strategy,
            )
        except Exception as exc:
            LOGGER.warning("State payload synthesis failed for %s via %s: %s", source_target.url, strategy, exc)
            return []

    def _should_browser_fallback(self, source_target: SourceTarget) -> bool:
        return source_target.expected_source_type in {
            "browser_heavy",
            "unknown",
            "react_state",
            "json_api",
            "html_table",
        }

    def _enrich_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        if "flattened" not in enriched:
            enriched["flattened"] = _flatten_json(payload)
        if "candidate_collections" not in enriched:
            enriched["candidate_collections"] = _extract_candidate_collections(payload)[:25]
        return enriched


def _flatten_json(value: Any, *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_json(child, prefix=child_prefix))
        return flattened
    if isinstance(value, list):
        for index, child in enumerate(value[:50]):
            child_prefix = f"{prefix}[{index}]"
            flattened.update(_flatten_json(child, prefix=child_prefix))
        return flattened
    flattened[prefix or "value"] = value
    return flattened


def _extract_candidate_collections(value: Any, *, prefix: str = "") -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            candidates.extend(_extract_candidate_collections(child, prefix=child_prefix))
        return candidates
    if isinstance(value, list):
        dict_items = [item for item in value if isinstance(item, dict)]
        primitive_items = [item for item in value if not isinstance(item, dict)]
        if len(dict_items) >= 2:
            keys: set[str] = set()
            for item in dict_items[:10]:
                keys.update(item.keys())
            candidates.append(
                {
                    "path": prefix or "root",
                    "length": len(value),
                    "sample_keys": sorted(keys)[:20],
                    "sample_items": dict_items[:3],
                }
            )
        elif len(primitive_items) >= 3 and prefix:
            candidates.append(
                {
                    "path": prefix,
                    "length": len(value),
                    "sample_values": primitive_items[:10],
                }
            )
        for index, child in enumerate(value[:20]):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            candidates.extend(_extract_candidate_collections(child, prefix=child_prefix))
        return candidates
    return candidates


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
