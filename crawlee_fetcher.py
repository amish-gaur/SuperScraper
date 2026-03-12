"""Crawlee-backed fetch orchestration for deterministic extraction stages."""

from __future__ import annotations

import asyncio
import atexit
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import timedelta
import json
import logging
import threading
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from crawlee import Request
from crawlee.crawlers import BeautifulSoupCrawler, PlaywrightCrawler
from crawlee.errors import HttpClientStatusCodeError, HttpStatusCodeError
from crawlee.storages import Dataset, RequestQueue

from architect import SourceTarget
from domain_adapters import DomainAdapter
from source_health import FailureReason, FetchOutcome, REGISTRY


LOGGER = logging.getLogger(__name__)

ANTI_BOT_MARKERS = (
    "verify you are human",
    "enable javascript and cookies",
    "just a moment",
    "access denied",
    "cf-browser-verification",
    "cf-challenge",
    "why do i have to complete a captcha",
)

JS_RENDER_SOURCE_TYPES = {"browser_heavy", "react_state"}
NON_RETRYABLE_HTTP_STATUS_CODES = [404, 410]
MAX_ARTIFACT_HTML_CHARS = 200000
MAX_ARTIFACT_JSON_CHARS = 50000


@dataclass(slots=True)
class CrawleeFetchResult:
    """Structured result for a Crawlee-managed fetch."""

    fetch_outcome: FetchOutcome
    html_text: str = ""
    json_payload: dict[str, Any] | list[Any] | None = None
    adapter_payload: dict[str, Any] | None = None


@dataclass(slots=True)
class CrawleeFetchedTarget:
    """A fetched source target plus its Crawlee-managed artifacts."""

    source_target: SourceTarget
    fetch_result: CrawleeFetchResult
    adapter: DomainAdapter | None = None


@dataclass(slots=True)
class StaticProcessorStats:
    """Basic counters for the batch static Crawlee stage."""

    input_urls: int = 0
    unique_urls: int = 0
    queued_urls: int = 0
    deduped_urls: int = 0
    handled_urls: int = 0
    failed_urls: int = 0
    artifacts_written: int = 0
    failed_urls_by_reason: dict[str, int] | None = None


class _CrawleeLoopRunner:
    """Run sync Crawlee fetches on one dedicated event loop to avoid loop-bound storage state bugs."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    def run(self, coroutine: Any) -> Any:
        loop = self._ensure_loop()
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._thread is not None and self._thread.is_alive():
                return self._loop

            self._ready.clear()
            self._thread = threading.Thread(
                target=self._thread_main,
                name="crawlee-sync-loop",
                daemon=True,
            )
            self._thread.start()
            self._ready.wait()
            if self._loop is None:
                raise RuntimeError("Failed to initialize Crawlee background event loop")
            return self._loop

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    def shutdown(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


_SYNC_LOOP_RUNNER = _CrawleeLoopRunner()
atexit.register(_SYNC_LOOP_RUNNER.shutdown)


class CrawleeFetcher:
    """Small synchronous wrapper around Crawlee's async crawlers."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        static_processor: CrawleeStaticRequestProcessor | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.static_processor = static_processor

    def fetch_target(
        self,
        source_target: SourceTarget,
        *,
        adapter: Any | None = None,
    ) -> CrawleeFetchResult:
        return self._run_async(self._fetch_target(source_target, adapter=adapter))

    def fetch_html(self, url: str, *, expected_source_type: str = "unknown") -> FetchOutcome:
        source_target = SourceTarget(url=url, expected_source_type=expected_source_type)
        return self.fetch_target(source_target).fetch_outcome

    async def _fetch_target(
        self,
        source_target: SourceTarget,
        *,
        adapter: Any | None = None,
    ) -> CrawleeFetchResult:
        request_queue = await RequestQueue.open(name=f"router-{uuid4().hex}")
        result: dict[str, Any] = {}
        crawler = self._build_crawler(
            source_target=source_target,
            adapter=adapter,
            result=result,
            request_queue=request_queue,
        )
        request = self._build_request(source_target, adapter=adapter)

        try:
            await request_queue.add_request(request)
            await crawler.run(purge_request_queue=False)
        except Exception as exc:
            LOGGER.warning("Crawlee fetch failed for %s: %s", source_target.url, exc)
            outcome = _build_failure_outcome(
                url=source_target.url,
                exc=exc,
                browser_mode=self._should_render_with_playwright(source_target, adapter),
            )
            REGISTRY.record_fetch(outcome)
            return CrawleeFetchResult(fetch_outcome=outcome)
        finally:
            try:
                await request_queue.drop()
            except Exception:
                LOGGER.debug("Failed to drop Crawlee request queue for %s", source_target.url, exc_info=True)

        outcome = _build_fetch_outcome(
            url=str(result.get("url") or source_target.url),
            text=str(result.get("html_text") or ""),
            status_code=result.get("status_code"),
        )
        return CrawleeFetchResult(
            fetch_outcome=outcome,
            html_text=str(result.get("html_text") or ""),
            json_payload=result.get("json_payload"),
            adapter_payload=result.get("adapter_payload"),
        )

    def _build_crawler(
        self,
        *,
        source_target: SourceTarget,
        adapter: Any | None,
        result: dict[str, Any],
        request_queue: RequestQueue,
    ) -> BeautifulSoupCrawler | PlaywrightCrawler:
        common_kwargs = {
            "request_manager": request_queue,
            "request_handler_timeout": timedelta(seconds=self.timeout_seconds),
            "max_request_retries": 3,
            "retry_on_blocked": True,
            "ignore_http_error_status_codes": NON_RETRYABLE_HTTP_STATUS_CODES,
            "respect_robots_txt_file": False,
            "configure_logging": False,
        }
        if self._should_render_with_playwright(source_target, adapter):
            crawler = PlaywrightCrawler(
                browser_type="chromium",
                headless=True,
                **common_kwargs,
            )

            @crawler.router.default_handler
            async def handle(context: Any) -> None:
                html_text = await context.page.content()
                result["html_text"] = html_text
                result["url"] = context.request.loaded_url or context.request.url
                result["status_code"] = getattr(context.response, "status", None)
                parsed_json = _safe_json_loads(html_text)
                if parsed_json is not None:
                    result["json_payload"] = parsed_json
                if adapter is not None:
                    payload = await adapter.fetch_payload_with_context(source_target, context)
                    if payload is not None:
                        result["adapter_payload"] = payload

            return crawler

        crawler = BeautifulSoupCrawler(**common_kwargs)

        @crawler.router.default_handler
        async def handle(context: Any) -> None:
            body = await context.http_response.read()
            html_text = body.decode("utf-8", errors="replace")
            result["html_text"] = html_text
            result["url"] = context.request.loaded_url or context.request.url
            result["status_code"] = context.http_response.status_code
            parsed_json = _safe_json_loads(html_text)
            if parsed_json is not None:
                result["json_payload"] = parsed_json
            if adapter is not None:
                payload = await adapter.fetch_payload_with_context(source_target, context)
                if payload is not None:
                    result["adapter_payload"] = payload

        return crawler

    def _build_request(self, source_target: SourceTarget, *, adapter: Any | None) -> Request:
        if adapter is not None and hasattr(adapter, "build_request"):
            return adapter.build_request(source_target)
        return Request.from_url(
            source_target.url,
            user_data={
                "source_target_url": source_target.url,
                "expected_source_type": source_target.expected_source_type,
            },
        )

    def _should_render_with_playwright(self, source_target: SourceTarget, adapter: Any | None) -> bool:
        if adapter is not None and hasattr(adapter, "requires_javascript"):
            try:
                if adapter.requires_javascript(source_target):
                    return True
            except Exception:
                LOGGER.debug("Adapter JS detection failed for %s", source_target.url, exc_info=True)
        return source_target.expected_source_type in JS_RENDER_SOURCE_TYPES

    def _run_async(self, coroutine: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return _SYNC_LOOP_RUNNER.run(coroutine)
        raise RuntimeError("CrawleeFetcher cannot be invoked from an active asyncio event loop")


class CrawleeStaticRequestProcessor:
    """Batch static fetch stage backed by Crawlee request queue and dataset storage."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        request_queue_name: str | None = None,
        dataset_name: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.request_queue_name = request_queue_name or f"swarm-static-{uuid4().hex}"
        self.dataset_name = dataset_name or f"swarm-static-artifacts-{uuid4().hex}"
        self.stats = StaticProcessorStats(failed_urls_by_reason={})

    async def fetch(
        self,
        source_targets: list[SourceTarget],
        *,
        adapters: list[DomainAdapter],
    ) -> list[CrawleeFetchedTarget]:
        if not source_targets:
            return []

        self.stats = StaticProcessorStats(failed_urls_by_reason={})
        self.stats.input_urls = len(source_targets)

        adapter_by_url = {
            source_target.url: self._select_adapter(source_target, adapters)
            for source_target in source_targets
        }
        request_queue = await RequestQueue.open(name=self.request_queue_name)
        dataset = await Dataset.open(name=self.dataset_name)
        fetched_targets: dict[str, CrawleeFetchedTarget] = {}

        crawler = BeautifulSoupCrawler(
            request_manager=request_queue,
            request_handler_timeout=timedelta(seconds=self.timeout_seconds),
            max_request_retries=3,
            retry_on_blocked=True,
            ignore_http_error_status_codes=NON_RETRYABLE_HTTP_STATUS_CODES,
            respect_robots_txt_file=False,
            configure_logging=False,
        )

        @crawler.router.default_handler
        async def handle(context: Any) -> None:
            source_target = _source_target_from_request(context.request)
            adapter = adapter_by_url.get(source_target.url)
            body = await context.http_response.read()
            html_text = body.decode("utf-8", errors="replace")
            adapter_payload = None
            if adapter is not None:
                adapter_payload = await adapter.fetch_payload_with_context(source_target, context)

            fetch_result = CrawleeFetchResult(
                fetch_outcome=_build_fetch_outcome(
                    url=context.request.loaded_url or context.request.url,
                    text=html_text,
                    status_code=context.http_response.status_code,
                ),
                html_text=html_text,
                json_payload=_safe_json_loads(html_text),
                adapter_payload=adapter_payload,
            )
            fetched_targets[source_target.url] = CrawleeFetchedTarget(
                source_target=source_target,
                fetch_result=fetch_result,
                adapter=adapter,
            )
            self.stats.handled_urls += 1
            await _push_artifact(
                dataset,
                source_target=source_target,
                fetch_result=fetch_result,
            )
            self.stats.artifacts_written += 1

        @crawler.failed_request_handler
        async def handle_failed_request(context: Any, exc: Exception) -> None:
            source_target = _source_target_from_request(context.request)
            outcome = _build_failure_outcome(url=source_target.url, exc=exc)
            REGISTRY.record_fetch(outcome)
            fetch_result = CrawleeFetchResult(fetch_outcome=outcome)
            fetched_targets[source_target.url] = CrawleeFetchedTarget(
                source_target=source_target,
                fetch_result=fetch_result,
                adapter=adapter_by_url.get(source_target.url),
            )
            self.stats.failed_urls += 1
            self.stats.failed_urls_by_reason[outcome.reason.value] = (
                self.stats.failed_urls_by_reason.get(outcome.reason.value, 0) + 1
            )
            await _push_artifact(
                dataset,
                source_target=source_target,
                fetch_result=fetch_result,
            )
            self.stats.artifacts_written += 1

        try:
            seen_enqueued: set[str] = set()
            for source_target in source_targets:
                if source_target.url in seen_enqueued:
                    self.stats.deduped_urls += 1
                    continue
                seen_enqueued.add(source_target.url)
                self.stats.unique_urls += 1
                adapter = adapter_by_url.get(source_target.url)
                self.stats.queued_urls += 1
                await request_queue.add_request(_build_request(source_target, adapter=adapter))
            await crawler.run(purge_request_queue=False)
        finally:
            try:
                await request_queue.drop()
            except Exception:
                LOGGER.debug("Failed to drop Crawlee request queue %s", self.request_queue_name, exc_info=True)

        ordered_results: list[CrawleeFetchedTarget] = []
        seen_urls: set[str] = set()
        for source_target in source_targets:
            if source_target.url in seen_urls:
                continue
            seen_urls.add(source_target.url)
            fetched = fetched_targets.get(source_target.url)
            if fetched is not None:
                ordered_results.append(fetched)
        return ordered_results

    def fetch_sync(
        self,
        source_targets: list[SourceTarget],
        *,
        adapters: list[DomainAdapter],
    ) -> list[CrawleeFetchedTarget]:
        """Run the async static fetch stage on the dedicated Crawlee loop."""
        return _SYNC_LOOP_RUNNER.run(self.fetch(source_targets, adapters=adapters))

    def _select_adapter(
        self,
        source_target: SourceTarget,
        adapters: list[DomainAdapter],
    ) -> DomainAdapter | None:
        for adapter in adapters:
            if adapter.matches(source_target):
                return adapter
        return None


def _safe_json_loads(value: str) -> dict[str, Any] | list[Any] | None:
    stripped = value.strip()
    if not stripped or stripped[0] not in {"{", "["}:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _build_request(source_target: SourceTarget, *, adapter: DomainAdapter | None) -> Request:
    if adapter is not None:
        return adapter.build_request(source_target)
    return Request.from_url(
        source_target.url,
        user_data={
            "source_target_url": source_target.url,
            "expected_source_type": source_target.expected_source_type,
        },
    )


def _source_target_from_request(request: Request) -> SourceTarget:
    return SourceTarget(
        url=str(request.user_data.get("source_target_url") or request.loaded_url or request.url),
        expected_source_type=str(request.user_data.get("expected_source_type") or "unknown"),
    )


def _build_fetch_outcome(*, url: str, text: str, status_code: int | None) -> FetchOutcome:
    lowered = text.lower()
    if status_code == 403:
        outcome = FetchOutcome(
            url=url,
            ok=False,
            status_code=status_code,
            reason=FailureReason.HTTP_403,
            detail="403 response",
        )
    elif status_code == 429:
        outcome = FetchOutcome(
            url=url,
            ok=False,
            status_code=status_code,
            reason=FailureReason.HTTP_429,
            detail="429 response",
        )
    elif any(marker in lowered for marker in ANTI_BOT_MARKERS):
        outcome = FetchOutcome(
            url=url,
            ok=False,
            text=text,
            status_code=status_code,
            reason=FailureReason.ANTI_BOT,
            detail="anti-bot text detected",
        )
    elif status_code is not None and status_code >= 400:
        outcome = FetchOutcome(
            url=url,
            ok=False,
            status_code=status_code,
            reason=FailureReason.HTTP_ERROR,
            detail=f"http status {status_code}",
        )
    elif not text.strip():
        outcome = FetchOutcome(
            url=url,
            ok=False,
            status_code=status_code,
            reason=FailureReason.EMPTY_CONTENT,
            detail="empty response body",
        )
    else:
        outcome = FetchOutcome(
            url=url,
            ok=True,
            text=text,
            status_code=status_code,
            reason=FailureReason.SUCCESS,
            detail="",
        )

    REGISTRY.record_fetch(outcome)
    return outcome


def _build_failure_outcome(
    *,
    url: str,
    exc: Exception,
    browser_mode: bool = False,
) -> FetchOutcome:
    status_code = _extract_status_code(exc)
    if status_code is not None:
        return FetchOutcome(
            url=url,
            ok=False,
            status_code=status_code,
            reason=_status_code_to_failure_reason(status_code),
            detail=str(exc),
        )
    return FetchOutcome(
        url=url,
        ok=False,
        reason=FailureReason.BROWSER_ERROR if browser_mode else FailureReason.NETWORK_ERROR,
        detail=str(exc),
    )


def _extract_status_code(exc: Exception) -> int | None:
    if isinstance(exc, (HttpClientStatusCodeError, HttpStatusCodeError)):
        return int(exc.status_code)
    status_code = getattr(exc, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _status_code_to_failure_reason(status_code: int) -> FailureReason:
    if status_code == 403:
        return FailureReason.HTTP_403
    if status_code == 429:
        return FailureReason.HTTP_429
    return FailureReason.HTTP_ERROR


async def _push_artifact(
    dataset: Dataset,
    *,
    source_target: SourceTarget,
    fetch_result: CrawleeFetchResult,
) -> None:
    artifact = _sanitize_artifact_for_storage(source_target=source_target, fetch_result=fetch_result)
    if artifact is None:
        LOGGER.warning("Skipping invalid Crawlee artifact for %s", source_target.url)
        return
    await dataset.push_data(artifact)


def _sanitize_artifact_for_storage(
    *,
    source_target: SourceTarget,
    fetch_result: CrawleeFetchResult,
) -> dict[str, Any] | None:
    source_url = str(source_target.url).strip()
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    status_code = fetch_result.fetch_outcome.status_code
    if not isinstance(status_code, int) or not 100 <= status_code <= 599:
        status_code = None

    fetch_reason = getattr(fetch_result.fetch_outcome.reason, "value", "unknown")
    expected_source_type = str(source_target.expected_source_type or "unknown").strip().lower()
    if expected_source_type not in {"html_table", "json_api", "react_state", "browser_heavy", "unknown"}:
        expected_source_type = "unknown"

    artifact = {
        "source_url": source_url,
        "expected_source_type": expected_source_type,
        "status_code": status_code,
        "fetch_ok": bool(fetch_result.fetch_outcome.ok),
        "fetch_reason": str(fetch_reason),
        "fetch_detail": _truncate_text(fetch_result.fetch_outcome.detail, limit=4000),
        "html_length": len(fetch_result.html_text or ""),
        "html_text": _truncate_text(fetch_result.html_text, limit=MAX_ARTIFACT_HTML_CHARS),
        "json_payload": _truncate_json_blob(fetch_result.json_payload),
        "adapter_payload": _truncate_json_blob(fetch_result.adapter_payload),
        "artifact_truncated": False,
    }
    artifact["artifact_truncated"] = (
        artifact["html_length"] > len(artifact["html_text"])
        or artifact["json_payload"] != fetch_result.json_payload
        or artifact["adapter_payload"] != fetch_result.adapter_payload
    )
    return artifact


def _truncate_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    return text[:limit]


def _truncate_json_blob(value: Any) -> Any:
    if value is None:
        return None
    try:
        serialized = json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return _truncate_text(value, limit=MAX_ARTIFACT_JSON_CHARS)
    if len(serialized) <= MAX_ARTIFACT_JSON_CHARS:
        return value
    return {
        "_truncated": True,
        "_preview": serialized[:MAX_ARTIFACT_JSON_CHARS],
    }
