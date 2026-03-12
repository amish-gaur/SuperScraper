"""Live local-server integration checks for the Crawlee static stage."""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import io
import json
import os
from pathlib import Path
import socketserver
import tempfile
import threading
from uuid import uuid4

os.environ.setdefault("CRAWLEE_STORAGE_DIR", tempfile.mkdtemp(prefix="crawlee-live-storage-"))

from pydantic import BaseModel

from architect import SourceTarget
from crawlee_fetcher import CrawleeStaticRequestProcessor
from extraction_router import ExtractionRouter


class TableRow(BaseModel):
    state: str | None = None
    gdp: float | None = None
    source_url: str | None = None


class JsonRow(BaseModel):
    name: str | None = None
    source_url: str | None = None


class JsonRouter(ExtractionRouter):
    def _synthesize_payload(self, source_target: SourceTarget, payload: dict, *, strategy: str) -> list[BaseModel]:
        if strategy == "json_payload":
            return [
                self.row_model.model_validate({"name": item["name"], "source_url": source_target.url})
                for item in payload["payload"]["items"]
            ]
        return super()._synthesize_payload(source_target, payload, strategy=strategy)


def test_live_static_processor_and_router() -> None:
    html = """
    <html><body>
    <table>
      <tr><th>State</th><th>GDP</th></tr>
      <tr><td>California</td><td>4103124</td></tr>
      <tr><td>Texas</td><td>2709393</td></tr>
    </table>
    </body></html>
    """.strip()
    payload = json.dumps({"items": [{"name": "json-a"}, {"name": "json-b"}]})

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "table.html").write_text(html, encoding="utf-8")
        (root / "data.json").write_text(payload, encoding="utf-8")

        request_counts: dict[str, int] = {}

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)

            def do_GET(self) -> None:  # noqa: N802
                request_counts[self.path] = request_counts.get(self.path, 0) + 1
                super().do_GET()

        with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
            port = server.server_address[1]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                table_url = f"http://127.0.0.1:{port}/table.html"
                json_url = f"http://127.0.0.1:{port}/data.json"
                missing_url = f"http://127.0.0.1:{port}/missing.html"

                processor = CrawleeStaticRequestProcessor(
                    timeout_seconds=15.0,
                    request_queue_name=f"live-queue-{uuid4().hex}",
                    dataset_name=f"live-dataset-{uuid4().hex}",
                )
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    fetched = asyncio.run(
                        processor.fetch(
                            [
                                SourceTarget(url=table_url, expected_source_type="html_table"),
                                SourceTarget(url=table_url, expected_source_type="html_table"),
                                SourceTarget(url=json_url, expected_source_type="json_api"),
                                SourceTarget(url=missing_url, expected_source_type="html_table"),
                            ],
                            adapters=[],
                        )
                    )

                fetched_by_url = {item.source_target.url: item for item in fetched}
                table_router = ExtractionRouter(goal="state gdp table", row_model=TableRow, adapters=[])
                json_router = JsonRouter(goal="json items", row_model=JsonRow, adapters=[])

                table_decision = table_router.route_prefetched(
                    fetched_by_url[table_url].source_target,
                    fetched_by_url[table_url].fetch_result,
                )
                json_decision = json_router.route_prefetched(
                    fetched_by_url[json_url].source_target,
                    fetched_by_url[json_url].fetch_result,
                )
                missing_outcome = fetched_by_url[missing_url].fetch_result.fetch_outcome

                assert request_counts["/table.html"] == 1
                assert request_counts["/data.json"] == 1
                assert request_counts["/missing.html"] == 1
                assert len(fetched) == 3

                assert processor.stats.input_urls == 4
                assert processor.stats.unique_urls == 3
                assert processor.stats.queued_urls == 3
                assert processor.stats.deduped_urls == 1
                assert processor.stats.handled_urls == 3
                assert processor.stats.failed_urls == 0
                assert processor.stats.artifacts_written == 3
                assert processor.stats.failed_urls_by_reason == {}

                assert table_decision.strategy == "html_table"
                assert len(table_decision.records) == 2
                assert json_decision.strategy == "json_payload"
                assert len(json_decision.records) == 2
                assert missing_outcome.reason == "http_error" or missing_outcome.reason.value == "http_error"
                assert missing_outcome.status_code == 404
                assert "Traceback" not in stderr.getvalue()
            finally:
                server.shutdown()
                thread.join(timeout=2)


def main(*, verbose: bool = True) -> int:
    tests = [test_live_static_processor_and_router]
    for test in tests:
        test()
        if verbose:
            print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
