"""Small smoke tests for structural pipeline regressions."""

from __future__ import annotations

import sys

import pandas as pd
from pydantic import BaseModel

from html_table_extractor import HtmlTableExtractor
from predictive_dataset_builder import PredictiveDatasetBuilder, _parse_money_value
from source_health import FailureReason, FetchOutcome, REGISTRY
from source_ranker import SourceRanker
from synthesizer import DataSynthesizer


def test_source_penalty() -> None:
    REGISTRY.domains.clear()
    blocked = "https://www.basketball-reference.com/leagues/NBA_2024.html"
    safe = "https://www.teamrankings.com/nba/stat/points-per-game"
    REGISTRY.record_fetch(
        FetchOutcome(url=blocked, ok=False, reason=FailureReason.HTTP_403, detail="simulated 403")
    )
    ranked = SourceRanker().rank("nba team stats", [blocked, safe])
    assert ranked[0].url == safe, ranked


def test_metric_prefixing() -> None:
    builder = PredictiveDatasetBuilder(goal="nba team stats", dataset_name="x", starting_urls=[])
    frame = pd.DataFrame(
        {
            "rank": [1, 2],
            "team": ["Denver", "Boston"],
            "2025": [120.3, 114.5],
            "home": [117.4, 114.6],
        }
    )
    prefixed = builder._prefix_metric_columns(frame, "https://www.teamrankings.com/nba/stat/points-per-game")
    assert "points_per_game_2025" in prefixed.columns, prefixed.columns
    assert "points_per_game_home" in prefixed.columns, prefixed.columns


def test_metric_prefixing_ignores_query_strings() -> None:
    builder = PredictiveDatasetBuilder(goal="nba team stats", dataset_name="x", starting_urls=[])
    frame = pd.DataFrame(
        {
            "team": ["Denver", "Boston"],
            "value": [120.3, 114.5],
        }
    )
    prefixed = builder._prefix_metric_columns(
        frame,
        "https://www.teamrankings.com/nba/stat/points-per-game?date=2026-03-01",
    )
    assert "points_per_game_value" in prefixed.columns, prefixed.columns
    assert all("date_2026_03_01" not in column for column in prefixed.columns), prefixed.columns


def test_parse_money_value_handles_capitalized_loan_fee() -> None:
    assert _parse_money_value("Loan fee: €250k") == 250000.0


class ProgramRow(BaseModel):
    school: str | None = None
    conference: str | None = None
    tournament_appearances: str | None = None
    source_url: str | None = None


def test_html_table_extractor_promotes_header_row_and_extracts_records() -> None:
    html = """
    <table>
      <tr><th>0</th><th>1</th><th>2</th></tr>
      <tr><td>School</td><td>Conference</td><td>Appearances</td></tr>
      <tr><td>Gonzaga</td><td>WCC</td><td>26</td></tr>
      <tr><td>UConn</td><td>Big East</td><td>37</td></tr>
    </table>
    """
    extractor = HtmlTableExtractor(ProgramRow)
    records = extractor.extract_from_html("https://example.com", html)
    assert len(records) == 2
    assert records[0].school == "Gonzaga"
    assert records[1].conference == "Big East"


class StateGrowthRow(BaseModel):
    state: str | None = None
    population_growth_rate: float | None = None
    population: float | None = None
    gdp: float | None = None
    gdp_growth_rate: float | None = None
    source_url: str | None = None


def test_html_table_extractor_maps_generic_schema_fields() -> None:
    html = """
    <table>
      <tr>
        <th>State or territory</th>
        <th>Population growth rate</th>
        <th>GDP</th>
      </tr>
      <tr><td>Texas</td><td>1.8%</td><td>$2.7T</td></tr>
      <tr><td>Florida</td><td>1.6%</td><td>$1.6T</td></tr>
    </table>
    """
    extractor = HtmlTableExtractor(StateGrowthRow)
    records = extractor.extract_from_html("https://example.com/states", html)
    assert len(records) == 2
    assert records[0].state == "Texas"
    assert records[0].population_growth_rate == 1.8
    assert records[1].gdp == 1600000000000


def test_html_table_extractor_handles_state_growth_style_headers() -> None:
    html = """
    <table>
      <tr>
        <th>State/federal district/territory/division/region</th>
        <th>2020 pop.</th>
        <th>2010-2020 change</th>
      </tr>
      <tr><td>Massachusetts</td><td>7029917</td><td>7.4%</td></tr>
      <tr><td>Connecticut</td><td>3605944</td><td>0.9%</td></tr>
      <tr><td>South Region</td><td>126266107</td><td>10.2%</td></tr>
    </table>
    """
    extractor = HtmlTableExtractor(StateGrowthRow)
    records = extractor.extract_from_html("https://example.com/state-growth", html)
    assert len(records) == 2, records
    assert records[0].state == "Massachusetts"
    assert records[0].population == 7029917
    assert records[0].population_growth_rate == 7.4


def test_html_table_extractor_handles_gdp_style_headers() -> None:
    html = """
    <table>
      <tr>
        <th>State or federal district</th>
        <th>Nominal GDP at current prices 2024</th>
        <th>Real GDP growth rate (2023-2024)</th>
      </tr>
      <tr><td>California</td><td>4103124</td><td>3.6%</td></tr>
      <tr><td>Texas</td><td>2709393</td><td>3.6%</td></tr>
      <tr><td>United States</td><td>28912345</td><td>2.8%</td></tr>
    </table>
    """
    extractor = HtmlTableExtractor(StateGrowthRow)
    records = extractor.extract_from_html("https://example.com/state-gdp", html)
    assert len(records) == 2, records
    assert records[0].state == "California"
    assert records[0].gdp == 4103124
    assert records[0].gdp_growth_rate == 3.6


def test_deterministic_synthesizer_coalesces_partial_rows() -> None:
    synthesizer = DataSynthesizer(row_model=StateGrowthRow, llm_gateway=None)
    raw_records = [
        StateGrowthRow(state="Texas", population_growth_rate=1.8, source_url="growth"),
        StateGrowthRow(state="Texas", population=31709821, source_url="population"),
        StateGrowthRow(state="Texas", gdp=2709393, gdp_growth_rate=7.7, source_url="gdp"),
        StateGrowthRow(state="Florida", population_growth_rate=1.6, source_url="growth"),
        StateGrowthRow(state="Florida", population=23932841, source_url="population"),
    ]
    clean_records = synthesizer.synthesize("Build a dataset of U.S. states with population growth", raw_records)
    assert len(clean_records) == 2, clean_records
    texas = next(record for record in clean_records if record["state"] == "Texas")
    assert texas["population_growth_rate"] == 1.8
    assert texas["population"] == 31709821
    assert texas["gdp"] == 2709393
    assert texas["gdp_growth_rate"] == 7.7


def test_prepare_frame_drops_repeated_header_rows() -> None:
    builder = PredictiveDatasetBuilder(
        goal="nba team stats",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
    )
    frame = pd.DataFrame(
        {
            "team": ["Team", "Denver", "Boston", "Phoenix"],
            "wins": ["Wins", "57", "64", "49"],
            "losses": ["Losses", "25", "18", "33"],
            "_source_url": ["src", "src", "src", "src"],
        }
    )
    prepared = builder._prepare_frame(frame)
    assert prepared is not None
    assert len(prepared) == 3, prepared
    assert "Denver" in prepared["raw_entity_name"].tolist()


def test_merge_frames_coalesces_overlapping_columns() -> None:
    builder = PredictiveDatasetBuilder(
        goal="nba team stats",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
    )
    left = pd.DataFrame(
        {
            "entity_name": ["denver nuggets", "boston celtics"],
            "raw_entity_name": ["Denver Nuggets", "Boston Celtics"],
            "wins": [57, None],
            "pace": [98.1, 99.2],
        }
    )
    right = pd.DataFrame(
        {
            "entity_name": ["denver nuggets", "boston celtics"],
            "raw_entity_name": ["Denver Nuggets", "Boston Celtics"],
            "wins": [58, 64],
            "off_rating": [119.5, 121.2],
        }
    )
    merged = builder._merge_frames(left, right, how="left")
    assert len(merged) == 2
    assert "off_rating" in merged.columns
    assert merged.loc[merged["entity_name"] == "boston celtics", "wins"].iloc[0] == 64
    assert merged.loc[merged["entity_name"] == "denver nuggets", "wins"].iloc[0] == 57


def test_concat_frames_preserves_union_of_columns_and_rows() -> None:
    builder = PredictiveDatasetBuilder(
        goal="startup companies",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
    )
    left = pd.DataFrame(
        {
            "entity_name": ["stripe", "databricks"],
            "raw_entity_name": ["Stripe", "Databricks"],
            "valuation": [65.0, 43.0],
        }
    )
    right = pd.DataFrame(
        {
            "entity_name": ["openai", "anthropic"],
            "raw_entity_name": ["OpenAI", "Anthropic"],
            "funding": [11.3, 7.3],
        }
    )
    combined = builder._concat_frames(left, right)
    assert len(combined) == 4
    assert "valuation" in combined.columns
    assert "funding" in combined.columns
    assert set(combined["entity_name"]) == {"stripe", "databricks", "openai", "anthropic"}


def test_nba_predictive_builder() -> None:
    goal = "Build a predictive dataset of NBA teams with historical team performance features and playoff outcome target"
    builder = PredictiveDatasetBuilder(goal=goal, dataset_name="x", starting_urls=[])
    result = builder.build()
    assert result is not None
    assert len(result.dataframe) >= 25, result.dataframe.shape
    assert len(result.dataframe.columns) >= 20, result.dataframe.shape
    assert "points_per_game_2025" in result.dataframe.columns, result.dataframe.columns


def test_nba_player_predictive_builder() -> None:
    goal = "Build a predictive dataset of NBA players with salary as the target and performance features"
    builder = PredictiveDatasetBuilder(goal=goal, dataset_name="x", starting_urls=[])
    result = builder.build()
    assert result is not None
    assert len(result.dataframe) >= 40, result.dataframe.shape
    assert "salary_salary" in result.dataframe.columns, result.dataframe.columns
    assert int(result.dataframe["salary_salary"].notna().sum()) >= 15


def test_startup_predictive_builder() -> None:
    goal = "Build a predictive dataset of startup companies with valuation as the target and funding features"
    builder = PredictiveDatasetBuilder(goal=goal, dataset_name="x", starting_urls=[])
    result = builder.build()
    assert result is not None
    assert len(result.dataframe) >= 100, result.dataframe.shape
    assert any("valuation" in column for column in result.dataframe.columns), result.dataframe.columns


def test_soccer_club_predictive_builder() -> None:
    goal = "Build a predictive dataset of European soccer clubs with transfer spend as the target and squad features"
    builder = PredictiveDatasetBuilder(goal=goal, dataset_name="x", starting_urls=[])
    result = builder.build()
    assert result is not None
    assert len(result.dataframe) >= 50, result.dataframe.shape
    assert "incoming_transfer_spend" in result.dataframe.columns, result.dataframe.columns


def main() -> int:
    tests = [
        test_source_penalty,
        test_metric_prefixing,
        test_metric_prefixing_ignores_query_strings,
        test_parse_money_value_handles_capitalized_loan_fee,
        test_html_table_extractor_promotes_header_row_and_extracts_records,
        test_html_table_extractor_maps_generic_schema_fields,
        test_html_table_extractor_handles_state_growth_style_headers,
        test_html_table_extractor_handles_gdp_style_headers,
        test_deterministic_synthesizer_coalesces_partial_rows,
        test_prepare_frame_drops_repeated_header_rows,
        test_merge_frames_coalesces_overlapping_columns,
        test_concat_frames_preserves_union_of_columns_and_rows,
        test_nba_predictive_builder,
        test_nba_player_predictive_builder,
        test_startup_predictive_builder,
        test_soccer_club_predictive_builder,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
