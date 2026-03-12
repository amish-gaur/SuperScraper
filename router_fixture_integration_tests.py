"""Fixture-backed integration tests for routing and paginated builder flows."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from architect import SourceTarget
from crawlee_fetcher import CrawleeFetchResult
from extraction_router import ExtractionRouter
from predictive_dataset_builder import PredictiveDatasetBuilder
from source_health import FetchOutcome


FIXTURE_ROOT = Path(__file__).parent / "fixtures"


class TeamRow(BaseModel):
    name: str | None = None
    source_url: str | None = None


class FixtureRouter(ExtractionRouter):
    fixture_urls: dict[str, str]

    def __init__(self, *args, fixture_urls: dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self.fixture_urls = fixture_urls

    def _fetch_html(self, url: str, *, expected_source_type: str = "unknown") -> FetchOutcome:
        fixture_name = self.fixture_urls.get(url)
        if fixture_name is None:
            raise AssertionError(f"Unexpected fixture URL: {url}")
        text = (FIXTURE_ROOT / fixture_name).read_text(encoding="utf-8")
        return FetchOutcome(url=url, ok=True, text=text)

    def _fetch_target(self, source_target: SourceTarget) -> CrawleeFetchResult:
        outcome = self._fetch_html(
            source_target.url,
            expected_source_type=source_target.expected_source_type,
        )
        return CrawleeFetchResult(fetch_outcome=outcome)

    def _synthesize_payload(self, source_target: SourceTarget, payload: dict, *, strategy: str) -> list[BaseModel]:
        if strategy != "react_state":
            return []
        rows = payload.get("payloads", [])
        for row_group in rows:
            data = row_group.get("data", {})
            items = data.get("props", {}).get("pageProps", {}).get("items", [])
            if items:
                return [self.row_model.model_validate(item) for item in items]
        return []


class FixturePaginatedBuilder(PredictiveDatasetBuilder):
    fixture_urls: dict[str, str]

    def __init__(self, *args, fixture_urls: dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self.fixture_urls = fixture_urls

    def _fetch_html(self, url: str) -> str:
        fixture_name = self.fixture_urls.get(url)
        if fixture_name is None:
            raise AssertionError(f"Unexpected fixture URL: {url}")
        return (FIXTURE_ROOT / fixture_name).read_text(encoding="utf-8")

    def _expand_urls(self) -> list[str]:
        return list(self.starting_urls)


def test_fixture_router_extracts_records_from_state_payload() -> None:
    router = FixtureRouter(
        goal="fixture state extraction",
        row_model=TeamRow,
        adapters=[],
        fixture_urls={"https://fixture.test/state": "router_state_fixture.html"},
    )
    decision = router.route(SourceTarget(url="https://fixture.test/state", expected_source_type="react_state"))
    assert decision.strategy == "react_state"
    assert len(decision.records) == 2
    assert decision.records[0].name == "Denver Nuggets"
    assert decision.records[1].source_url == "https://fixture.test/teams/boston"


def test_fixture_builder_extracts_paginated_ncaa_tables() -> None:
    builder = FixturePaginatedBuilder(
        goal="NCAA men's basketball programs",
        dataset_name="fixture_ncaa_stats",
        starting_urls=["https://www.ncaa.com/stats/basketball-men/d1/current/team/145"],
        minimum_rows=2,
        fixture_urls={
            "https://www.ncaa.com/stats/basketball-men/d1/current/team/145": "ncaa_stat_page1.html",
            "https://www.ncaa.com/stats/basketball-men/d1/current/team/145/p2": "ncaa_stat_page2.html",
        },
    )
    frames = builder._extract_tables("https://www.ncaa.com/stats/basketball-men/d1/current/team/145")
    assert len(frames) == 1
    frame = frames[0]
    assert len(frame) == 4
    assert "scoring_offense_g" in frame.columns, frame.columns
    assert "scoring_offense_avg" in frame.columns, frame.columns
    assert set(frame["team"]) == {"Houston", "Duke", "Gonzaga", "Arizona"}


def main(*, verbose: bool = True) -> int:
    tests = [
        test_fixture_router_extracts_records_from_state_payload,
        test_fixture_builder_extracts_paginated_ncaa_tables,
    ]
    for test in tests:
        test()
        if verbose:
            print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
