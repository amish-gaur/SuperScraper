"""Fixture-backed integration tests for deterministic builder flows."""

from __future__ import annotations

from pathlib import Path

from predictive_dataset_builder import PredictiveDatasetBuilder


FIXTURE_ROOT = Path(__file__).parent / "fixtures"


class FixturePredictiveDatasetBuilder(PredictiveDatasetBuilder):
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


def test_fixture_builder_merges_metric_tables_from_disk() -> None:
    builder = FixturePredictiveDatasetBuilder(
        goal="Build a predictive dataset of NBA teams with historical team performance features and playoff outcome target",
        dataset_name="fixture_nba_team_metrics",
        starting_urls=[
            "https://www.teamrankings.com/nba/stat/points-per-game",
            "https://www.teamrankings.com/nba/stat/offensive-efficiency",
        ],
        minimum_rows=2,
        minimum_columns=5,
        max_candidate_urls=2,
        fixture_urls={
            "https://www.teamrankings.com/nba/stat/points-per-game": "teamrankings_points_per_game.html",
            "https://www.teamrankings.com/nba/stat/offensive-efficiency": "teamrankings_offensive_efficiency.html",
        },
    )
    result = builder.build()
    assert result is not None
    assert len(result.dataframe) == 2, result.dataframe
    assert "points_per_game_2025" in result.dataframe.columns, result.dataframe.columns
    assert "offensive_efficiency_2025" in result.dataframe.columns, result.dataframe.columns
    denver = result.dataframe.loc[result.dataframe["entity_name"] == "denver nuggets"].iloc[0]
    assert denver["points_per_game_2025"] == 120.3
    assert denver["offensive_efficiency_2025"] == 119.8


def test_fixture_builder_promotes_header_rows_for_salary_fixture() -> None:
    builder = FixturePredictiveDatasetBuilder(
        goal="Build a predictive dataset of NBA players with salary as the target and performance features",
        dataset_name="fixture_nba_player_salary",
        starting_urls=["https://www.espn.com/nba/salaries"],
        minimum_rows=2,
        minimum_columns=3,
        max_candidate_urls=1,
        fixture_urls={
            "https://www.espn.com/nba/salaries": "espn_salaries_fixture.html",
        },
    )
    frames = builder._extract_tables("https://www.espn.com/nba/salaries")
    assert len(frames) == 1
    frame = frames[0]
    assert list(frame.columns) == ["player", "team", "salary_salary", "_source_url"], frame.columns
    assert frame.loc[0, "salary_salary"] == "$51.4M"


def main(*, verbose: bool = True) -> int:
    tests = [
        test_fixture_builder_merges_metric_tables_from_disk,
        test_fixture_builder_promotes_header_rows_for_salary_fixture,
    ]
    for test in tests:
        test()
        if verbose:
            print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
