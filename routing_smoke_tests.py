"""Focused local smoke tests for the extraction router architecture."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from architect import DatasetArchitect, DatasetBlueprint, SourceTarget
from domain_adapters import DomainAdapter
from extraction_router import ExtractionRouter, StateSniffer
from source_health import FailureReason, FetchOutcome


class SimpleRow(BaseModel):
    name: str | None = None
    source_url: str | None = None


@dataclass(slots=True)
class FakeAdapter(DomainAdapter):
    payload: dict

    def matches(self, source_target: SourceTarget) -> bool:
        return "adapter.test" in source_target.url

    def fetch_payload(self, source_target: SourceTarget) -> dict | None:
        return self.payload


class TestRouter(ExtractionRouter):
    def __init__(self, *args, fetch_outcome: FetchOutcome | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_fetch_outcome = fetch_outcome

    def _fetch_html(self, url: str) -> FetchOutcome:
        if self._test_fetch_outcome is None:
            raise AssertionError("HTML fetch should not have been called")
        return self._test_fetch_outcome

    def _synthesize_payload(self, source_target: SourceTarget, payload: dict, *, strategy: str) -> list[BaseModel]:
        if strategy == "domain_adapter":
            return [self.row_model.model_validate({"name": "adapter", "source_url": source_target.url})]
        if strategy == "react_state":
            return [self.row_model.model_validate({"name": "state", "source_url": source_target.url})]
        return []


def test_source_target_coerces_invalid_type() -> None:
    target = SourceTarget(url="https://example.com", expected_source_type="BAD_TYPE")
    assert target.expected_source_type == "unknown"


def test_blueprint_accepts_legacy_starting_urls() -> None:
    architect = DatasetArchitect(llm_gateway=None)
    payload = architect._sanitize_recovered_blueprint(
        {
            "dataset_name": "x",
            "dataset_description": "y",
            "target_record_count": 1,
            "starting_urls": ["https://example.com/a"],
            "row_schema": {
                "title": "Row",
                "description": "d",
                "fields": [{"name": "label", "type": "string", "description": "d", "ml_role": "target"}],
            },
        }
    )
    blueprint = DatasetBlueprint.model_validate(payload)
    assert blueprint.source_targets[0].url == "https://example.com/a"
    assert blueprint.source_targets[0].expected_source_type == "unknown"


def test_state_sniffer_extracts_multiple_payloads() -> None:
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"items":[{"name":"A"},{"name":"B"}]}}}</script>'
        "<script>window.__PRELOADED_STATE__ = {foo: {bar: 1}, baz: true};</script>"
    )
    payload = StateSniffer().sniff(html)
    assert payload is not None
    assert "__NEXT_DATA__.props.pageProps.items[0].name" in payload["flattened"]
    assert "window.__PRELOADED_STATE__.foo.bar" in payload["flattened"]
    assert any(candidate["path"].endswith("items") for candidate in payload["candidate_collections"])


def test_state_sniffer_handles_next_data_attribute_order_variants() -> None:
    html = '<script type="application/json" id="__NEXT_DATA__">{"props":{"pageProps":{"items":[{"name":"A"}]}}}</script>'
    payload = StateSniffer().sniff(html)
    assert payload is not None
    assert payload["flattened"]["__NEXT_DATA__.props.pageProps.items[0].name"] == "A"


def test_state_sniffer_handles_javascript_style_objects() -> None:
    html = "<script>window.__INITIAL_STATE__ = {'foo': {'bar': 'x'}, trailing: undefined, active: false};</script>"
    payload = StateSniffer().sniff(html)
    assert payload is not None
    assert payload["flattened"]["window.__INITIAL_STATE__.foo.bar"] == "x"
    assert payload["flattened"]["window.__INITIAL_STATE__.trailing"] is None
    assert payload["flattened"]["window.__INITIAL_STATE__.active"] is False


def test_state_sniffer_handles_root_array_payloads() -> None:
    html = '<script>window.__INITIAL_STATE__ = [{"name":"A"},{"name":"B"}];</script>'
    payload = StateSniffer().sniff(html)
    assert payload is not None
    assert payload["flattened"]["window.__INITIAL_STATE__[0].name"] == "A"
    assert any(candidate["path"] == "window.__INITIAL_STATE__" for candidate in payload["candidate_collections"])


def test_state_sniffer_preserves_literal_strings_when_normalizing() -> None:
    html = "<script>window.__INITIAL_STATE__ = {message: 'null value', trailing: undefined};</script>"
    payload = StateSniffer().sniff(html)
    assert payload is not None
    assert payload["flattened"]["window.__INITIAL_STATE__.message"] == "null value"
    assert payload["flattened"]["window.__INITIAL_STATE__.trailing"] is None


def test_router_uses_adapter_before_fetching_html() -> None:
    router = TestRouter(
        goal="test",
        row_model=SimpleRow,
        adapters=[FakeAdapter(payload={"rows": [{"name": "adapter"}]})],
    )
    decision = router.route(SourceTarget(url="https://adapter.test/stats", expected_source_type="json_api"))
    assert decision.strategy == "domain_adapter"
    assert len(decision.records) == 1
    assert decision.records[0].name == "adapter"


def test_router_enriches_payloads_with_candidate_collections() -> None:
    router = TestRouter(
        goal="test",
        row_model=SimpleRow,
        adapters=[],
        fetch_outcome=FetchOutcome(url="https://example.test", ok=True, text="<html></html>"),
    )
    enriched = router._enrich_payload({"payload": {"items": [{"name": "A"}, {"name": "B"}]}})
    assert "flattened" in enriched
    assert "candidate_collections" in enriched
    assert any(candidate["path"] == "payload.items" for candidate in enriched["candidate_collections"])


def test_router_uses_state_sniffer_before_browser() -> None:
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"items":[{"name":"A"}]}}}</script>'
    )
    router = TestRouter(
        goal="test",
        row_model=SimpleRow,
        fetch_outcome=FetchOutcome(url="https://state.test", ok=True, text=html),
        adapters=[],
    )
    decision = router.route(SourceTarget(url="https://state.test", expected_source_type="react_state"))
    assert decision.strategy == "react_state"
    assert len(decision.records) == 1
    assert decision.records[0].name == "state"


def test_router_falls_back_to_browser_after_fetch_failure() -> None:
    router = TestRouter(
        goal="test",
        row_model=SimpleRow,
        fetch_outcome=FetchOutcome(
            url="https://broken.test",
            ok=False,
            reason=FailureReason.NETWORK_ERROR,
            detail="simulated",
        ),
        adapters=[],
    )
    decision = router.route(SourceTarget(url="https://broken.test", expected_source_type="html_table"))
    assert decision.requires_browser is True
    assert decision.strategy == "fetch_failed"


def test_router_falls_back_to_browser_when_state_cannot_be_synthesized() -> None:
    class EmptySynthRouter(TestRouter):
        def _synthesize_payload(self, source_target: SourceTarget, payload: dict, *, strategy: str) -> list[BaseModel]:
            return []

    html = '<script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{"items":[{"name":"A"}]}}}</script>'
    router = EmptySynthRouter(
        goal="test",
        row_model=SimpleRow,
        fetch_outcome=FetchOutcome(url="https://state.test", ok=True, text=html),
        adapters=[],
    )
    decision = router.route(SourceTarget(url="https://state.test", expected_source_type="react_state"))
    assert decision.requires_browser is True
    assert decision.strategy == "browser"


def main(*, verbose: bool = True) -> int:
    tests = [
        test_source_target_coerces_invalid_type,
        test_blueprint_accepts_legacy_starting_urls,
        test_state_sniffer_extracts_multiple_payloads,
        test_state_sniffer_handles_next_data_attribute_order_variants,
        test_state_sniffer_handles_javascript_style_objects,
        test_state_sniffer_handles_root_array_payloads,
        test_state_sniffer_preserves_literal_strings_when_normalizing,
        test_router_uses_adapter_before_fetching_html,
        test_router_enriches_payloads_with_candidate_collections,
        test_router_uses_state_sniffer_before_browser,
        test_router_falls_back_to_browser_after_fetch_failure,
        test_router_falls_back_to_browser_when_state_cannot_be_synthesized,
    ]
    for test in tests:
        test()
        if verbose:
            print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
