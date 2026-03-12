"""Small smoke tests for structural pipeline regressions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from goal_intent import infer_goal_cardinality
from html_table_extractor import HtmlTableExtractor
from predictive_dataset_builder import PredictiveDatasetBuilder, _parse_money_value
from source_health import FailureReason, FetchOutcome, REGISTRY
from source_ranker import SourceRanker
from synthesizer import DataSynthesizer


FIXTURES_DIR = Path(__file__).parent / "fixtures"


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


class LaptopSpecRow(BaseModel):
    name: str | None = None
    price_usd: float | None = None
    cpu_model: str | None = None
    gpu_model: str | None = None
    ram_gb: float | None = None
    display_size_inches: float | None = None
    weight_kg: float | None = None
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


def test_html_table_extractor_maps_laptop_spec_headers() -> None:
    html = """
    <table>
      <tr>
        <th>Laptop / Model</th>
        <th>Price</th>
        <th>CPU</th>
        <th>GPU</th>
        <th>RAM</th>
        <th>Display</th>
        <th>Weight</th>
      </tr>
      <tr><td>Lenovo Yoga Pro 9i</td><td>$1799</td><td>Intel Core Ultra 9 185H</td><td>NVIDIA RTX 4050</td><td>32 GB</td><td>16.0</td><td>1.7 kg</td></tr>
      <tr><td>Asus Zenbook 14 OLED</td><td>$1299</td><td>Intel Core Ultra 7 155H</td><td>Intel Arc</td><td>16 GB</td><td>14.0</td><td>1.2 kg</td></tr>
    </table>
    """
    extractor = HtmlTableExtractor(LaptopSpecRow)
    records = extractor.extract_from_html("https://example.com/laptops", html)
    assert len(records) == 2, records
    assert records[0].name == "Lenovo Yoga Pro 9i"
    assert records[0].price_usd == 1799
    assert records[0].ram_gb == 32
    assert records[1].weight_kg == 1.2


def test_html_table_extractor_derives_laptop_specs_from_name_blob() -> None:
    html = """
    <table>
      <tr>
        <th>Laptop / Model</th>
        <th>Price</th>
        <th>Display</th>
      </tr>
      <tr>
        <td>Acer Aspire 16 A16-51GM-77G2 Intel Core 7 150U ⎘ NVIDIA GeForce RTX 2050 Mobile ⎘ 16 GB Memory, 1024 GB SSD</td>
        <td>$929</td>
        <td>16.0</td>
      </tr>
    </table>
    """
    extractor = HtmlTableExtractor(LaptopSpecRow)
    records = extractor.extract_from_html("https://example.com/laptops", html)
    assert len(records) == 1, records
    assert records[0].name == "Acer Aspire 16 A16-51GM-77G2 Intel Core 7 150U"
    assert records[0].cpu_model == "Intel Core 7 150U"
    assert records[0].gpu_model == "NVIDIA GeForce RTX 2050 Mobile"
    assert records[0].ram_gb == 16


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


def test_prepare_frame_coalesces_duplicate_entities_with_partial_values() -> None:
    builder = PredictiveDatasetBuilder(
        goal="nba team stats",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
    )
    frame = pd.DataFrame(
        {
            "team": ["Denver", "Denver", "Boston"],
            "wins": ["57", None, "64"],
            "pace": [None, "98.1", "99.2"],
            "_source_url": ["a", "b", "a"],
        }
    )
    prepared = builder._prepare_frame(frame)
    assert prepared is not None
    assert len(prepared) == 2, prepared
    denver = prepared.loc[prepared["entity_name"] == "denver"]
    assert denver["wins"].iloc[0] == 57
    assert denver["pace"].iloc[0] == 98.1
    assert denver["source_url"].iloc[0] == "a | b"


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


def test_merge_frames_preserves_combined_source_urls() -> None:
    builder = PredictiveDatasetBuilder(
        goal="nba team stats",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
    )
    left = pd.DataFrame(
        {
            "entity_name": ["denver nuggets"],
            "raw_entity_name": ["Denver Nuggets"],
            "wins": [57],
            "source_url": ["https://example.com/wins"],
        }
    )
    right = pd.DataFrame(
        {
            "entity_name": ["denver nuggets"],
            "off_rating": [119.5],
            "source_url": ["https://example.com/off-rating"],
        }
    )
    merged = builder._merge_frames(left, right, how="left")
    assert merged["source_url"].iloc[0] == "https://example.com/wins | https://example.com/off-rating"


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


def test_finalize_frame_cleans_column_labels_and_ordering() -> None:
    builder = PredictiveDatasetBuilder(
        goal="Build a predictive dataset of NBA players with salary as the target and performance features",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
        minimum_columns=4,
    )
    frame = pd.DataFrame(
        {
            "entity_name": ["nikola jokic", "jayson tatum"],
            "raw_entity_name": ["Nikola Jokic", "Jayson Tatum"],
            "salary_salary": [51400000, 34800000],
            "points_per_game_value": [29.6, 27.1],
            "source_url": ["https://example.com/salaries", "https://example.com/salaries"],
        }
    )
    finalized = builder._finalize_frame(frame)
    assert list(finalized.columns) == ["name", "entity_name", "salary", "points_per_game", "source"], finalized.columns


def test_finalize_frame_adds_schema_aliases_for_state_goal() -> None:
    builder = PredictiveDatasetBuilder(
        goal="Build a predictive dataset of U.S. states with population growth as the target and economic features",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
        minimum_columns=5,
        target_field="population_growth_rate",
        core_feature_fields=["state", "population", "gdp", "gdp_growth_rate"],
    )
    frame = pd.DataFrame(
        {
            "entity_name": ["texas", "florida"],
            "raw_entity_name": ["Texas", "Florida"],
            "2010_2020_change": [15.9, 14.6],
            "census_population_8_9_a_july_1_2025_est": [31709821, 23932841],
            "nominal_gdp_at_current_prices_2024_millions_of_u_s_dollars_1_2024": [2709393, 1649876],
            "real_gdp_growth_rate_2023_2024_1_real_gdp_growth_rate_2023_2024_1": [7.7, 3.2],
            "source_url": ["https://example.com/states", "https://example.com/states"],
        }
    )
    finalized = builder._finalize_frame(frame)
    assert "state" in finalized.columns, finalized.columns
    assert "population_growth_rate" in finalized.columns, finalized.columns
    assert "population" in finalized.columns, finalized.columns
    assert "gdp" in finalized.columns, finalized.columns
    assert "gdp_growth_rate" in finalized.columns, finalized.columns
    assert finalized.loc[0, "state"] == "Texas"
    assert finalized.loc[0, "population_growth_rate"] == 15.9


def test_prepare_frame_preserves_rank_when_schema_requires_it() -> None:
    builder = PredictiveDatasetBuilder(
        goal="Build a predictive dataset of Fortune 500 companies",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
        target_field="revenue_usd_millions",
        core_feature_fields=["company_name", "rank", "revenue_growth", "employees", "headquarters"],
    )
    frame = pd.DataFrame(
        {
            "rank": [1, 2],
            "name": ["Walmart", "Amazon"],
            "revenue": [680985, 637959],
            "revenue_growth": [5.1, 11.0],
            "employees": [2100000, 1556000],
            "_source_url": ["https://example.com/fortune", "https://example.com/fortune"],
        }
    )
    prepared = builder._prepare_frame(frame)
    assert prepared is not None
    assert "rank" in prepared.columns, prepared.columns
    finalized = builder._finalize_frame(prepared)
    assert "company_name" in finalized.columns, finalized.columns
    assert "rank" in finalized.columns, finalized.columns


def test_finalize_frame_trims_excess_width_using_signal() -> None:
    builder = PredictiveDatasetBuilder(
        goal="Build a predictive dataset of NBA teams with playoff outcome target",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
        minimum_columns=4,
    )
    frame = pd.DataFrame(
        {
            "entity_name": [f"team_{index}" for index in range(5)],
            "raw_entity_name": [f"Team {index}" for index in range(5)],
            "playoff_outcome": [1, 0, 1, 1, 0],
            "wins": [60, 55, 50, 48, 44],
            "net_rating": [8.2, 6.5, 4.4, 2.1, 0.4],
            "home_rating": [7.9, 6.0, 4.0, 1.9, 0.2],
            "away_rating": [7.1, 5.5, 3.2, 1.1, -0.3],
            "pace": [98.7, 97.1, 99.5, 96.4, 95.8],
            "turnovers": [11.2, 12.0, 13.1, 13.4, 14.2],
            "assist_ratio": [1.9, 1.8, 1.6, 1.5, 1.4],
            "bench_points": [35.2, 32.1, 28.4, 27.2, 24.9],
            "travel_miles": [12000, 11000, 10500, 9900, 9700],
            "city": ["Denver", "Boston", "Phoenix", "Dallas", "Miami"],
            "conference": ["West", "East", "West", "West", "East"],
            "division": ["NW", "Atlantic", "Pacific", "SW", "SE"],
            "source_url": ["https://example.com/teams"] * 5,
        }
    )
    finalized = builder._finalize_frame(frame)
    assert len(finalized.columns) == 12, finalized.columns
    assert "playoff_outcome" in finalized.columns
    assert "wins" in finalized.columns
    assert "source" not in finalized.columns


def test_finalize_frame_drops_duplicate_value_columns() -> None:
    builder = PredictiveDatasetBuilder(
        goal="Build a predictive dataset of NBA players with salary as the target and performance features",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
        minimum_columns=4,
    )
    frame = pd.DataFrame(
        {
            "entity_name": ["nikola jokic", "jayson tatum"],
            "raw_entity_name": ["Nikola Jokic", "Jayson Tatum"],
            "salary_salary": [51400000, 34800000],
            "salary": [51400000, 34800000],
            "points_per_game_value": [29.6, 27.1],
        }
    )
    finalized = builder._finalize_frame(frame)
    assert list(finalized.columns).count("salary") == 1, finalized.columns


def test_goal_aware_row_count_prefers_expected_table_size() -> None:
    builder = PredictiveDatasetBuilder(
        goal="Build a predictive dataset of NBA teams with playoff outcome target",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=2,
        minimum_columns=4,
    )
    near_expected = pd.DataFrame(
        {
            "entity_name": [f"team_{index}" for index in range(30)],
            "wins": list(range(30)),
            "playoff_outcome": [index % 2 for index in range(30)],
        }
    )
    too_large = pd.DataFrame(
        {
            "entity_name": [f"team_{index}" for index in range(80)],
            "wins": list(range(80)),
            "playoff_outcome": [index % 2 for index in range(80)],
        }
    )
    assert builder._row_count_score(len(near_expected)) > builder._row_count_score(len(too_large))
    assert builder._frame_row_count_is_reasonable(near_expected) is True
    assert builder._frame_row_count_is_reasonable(too_large) is False


def test_discover_ncaa_team_stat_urls_uses_filtered_option_values() -> None:
    class DiscoveryBuilder(PredictiveDatasetBuilder):
        def _fetch_html(self, url: str) -> str:
            return """
            <select>
              <option value='/stats/basketball-men/d1/current/team/145'>Scoring Offense</option>
              <option value="/stats/basketball-men/d1/current/team/932">Rebounds</option>
              <option value="/stats/basketball-men/d1/current/team/999">Ignore</option>
            </select>
            """

    builder = DiscoveryBuilder(goal="ncaa basketball team statistics", dataset_name="x", starting_urls=[], minimum_rows=2)
    urls = builder._discover_ncaa_team_stat_urls("https://www.ncaa.com/stats/basketball-men/d1/current/team/145")
    assert urls == [
        "https://www.ncaa.com/stats/basketball-men/d1/current/team/145",
        "https://www.ncaa.com/stats/basketball-men/d1/current/team/932",
    ]


def test_extract_transfermarkt_club_summary_uses_shared_fetch_path() -> None:
    class TransfermarktBuilder(PredictiveDatasetBuilder):
        def _fetch_html(self, url: str) -> str:
            return """
            <html><body>
              <a href="#to-1"><img title="Arsenal"/></a>
              <table><tr><th>placeholder</th></tr><tr><td>x</td></tr></table>
              <table>
                <tr><th>Player</th><th>Fee</th><th>Market value</th></tr>
                <tr><td>A</td><td>EUR 10m</td><td>EUR 12m</td></tr>
              </table>
              <table>
                <tr><th>Player</th><th>Fee</th><th>Market value</th></tr>
                <tr><td>B</td><td>EUR 7m</td><td>EUR 8m</td></tr>
              </table>
            </body></html>
            """

    builder = TransfermarktBuilder(
        goal="Build a predictive dataset of soccer clubs with transfer spend target",
        dataset_name="x",
        starting_urls=[],
        minimum_rows=1,
    )
    frames = builder._extract_transfermarkt_club_summary(
        "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/plus/?saison_id=2025"
    )
    assert len(frames) == 1
    frame = frames[0]
    assert len(frame) == 1
    assert frame["club"].iloc[0] == "Arsenal"
    assert frame["incoming_transfer_spend"].iloc[0] == 10_000_000.0
    assert frame["outgoing_transfer_income"].iloc[0] == 7_000_000.0


def test_historical_nba_team_goal_skips_exact_row_cardinality() -> None:
    goal = "Build a predictive dataset of NBA teams with historical team performance features and playoff outcome target"
    assert infer_goal_cardinality(goal) is None


def _fixture_text(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _build_from_html_map(
    goal: str,
    html_by_url: dict[str, str],
    *,
    minimum_rows: int = 2,
    minimum_columns: int = 4,
) -> PredictiveDatasetBuilder:
    class FixtureBackedBuilder(PredictiveDatasetBuilder):
        def _expand_urls(self) -> list[str]:
            return list(html_by_url)

        def _fetch_html(self, url: str) -> str:
            return html_by_url.get(url, "")

    return FixtureBackedBuilder(
        goal=goal,
        dataset_name="x",
        starting_urls=[],
        minimum_rows=minimum_rows,
        minimum_columns=minimum_columns,
        max_candidate_urls=len(html_by_url),
    )


def test_nba_predictive_builder() -> None:
    goal = "Build a predictive dataset of NBA teams with historical team performance features and playoff outcome target"
    builder = _build_from_html_map(
        goal,
        {
            "https://www.teamrankings.com/nba/stat/points-per-game": _fixture_text(
                "teamrankings_points_per_game.html"
            ),
            "https://www.teamrankings.com/nba/stat/offensive-efficiency": _fixture_text(
                "teamrankings_offensive_efficiency.html"
            ),
            "https://example.com/nba/playoff-outcome": """
            <table>
              <thead>
                <tr><th>Team</th><th>Playoff outcome</th></tr>
              </thead>
              <tbody>
                <tr><td>Denver Nuggets</td><td>1</td></tr>
                <tr><td>Boston Celtics</td><td>0</td></tr>
              </tbody>
            </table>
            """,
        },
        minimum_columns=5,
    )
    result = builder.build()
    assert result is not None
    assert len(result.dataframe) == 2, result.dataframe.shape
    assert len(result.dataframe.columns) >= 5, result.dataframe.shape
    assert "points_per_game_2025" in result.dataframe.columns, result.dataframe.columns
    assert "offensive_efficiency_2025" in result.dataframe.columns, result.dataframe.columns
    assert "playoff_outcome" in result.dataframe.columns, result.dataframe.columns


def test_nba_player_predictive_builder() -> None:
    goal = "Build a predictive dataset of NBA players with salary as the target and performance features"
    builder = _build_from_html_map(
        goal,
        {
            "https://example.com/nba/player-performance": """
            <table>
              <thead>
                <tr><th>Player</th><th>Points per game</th><th>Assists per game</th></tr>
              </thead>
              <tbody>
                <tr><td>Nikola Jokic</td><td>29.6</td><td>10.2</td></tr>
                <tr><td>Jayson Tatum</td><td>27.1</td><td>4.9</td></tr>
              </tbody>
            </table>
            """,
            "https://example.com/nba/player-salaries": _fixture_text("espn_salaries_fixture.html"),
        },
        minimum_columns=5,
    )
    result = builder.build()
    assert result is not None
    assert len(result.dataframe) == 2, result.dataframe.shape
    assert "salary" in result.dataframe.columns, result.dataframe.columns
    assert int(result.dataframe["salary"].notna().sum()) == 2
    assert "points_per_game" in result.dataframe.columns, result.dataframe.columns


def test_startup_predictive_builder() -> None:
    goal = "Build a predictive dataset of startup companies with valuation as the target and funding features"
    builder = _build_from_html_map(
        goal,
        {
            "https://example.com/startups/valuations": """
            <table>
              <thead>
                <tr><th>Company</th><th>Valuation</th></tr>
              </thead>
              <tbody>
                <tr><td>Stripe</td><td>$65B</td></tr>
                <tr><td>Databricks</td><td>$43B</td></tr>
              </tbody>
            </table>
            """,
            "https://example.com/startups/funding": """
            <table>
              <thead>
                <tr><th>Company</th><th>Funding</th><th>Employees</th></tr>
              </thead>
              <tbody>
                <tr><td>Stripe</td><td>$8.7B</td><td>7000</td></tr>
                <tr><td>Databricks</td><td>$4.0B</td><td>8000</td></tr>
              </tbody>
            </table>
            """,
        },
        minimum_columns=5,
    )
    result = builder.build()
    assert result is not None
    assert len(result.dataframe) == 2, result.dataframe.shape
    assert "valuation" in result.dataframe.columns, result.dataframe.columns
    assert "funding" in result.dataframe.columns, result.dataframe.columns


def test_soccer_club_predictive_builder() -> None:
    goal = "Build a predictive dataset of European soccer clubs with transfer spend as the target and squad features"
    builder = _build_from_html_map(
        goal,
        {
            "https://example.com/soccer/transfer-spend": """
            <table>
              <thead>
                <tr><th>Club</th><th>Incoming transfer spend</th></tr>
              </thead>
              <tbody>
                <tr><td>Arsenal</td><td>€210m</td></tr>
                <tr><td>Liverpool</td><td>€185m</td></tr>
              </tbody>
            </table>
            """,
            "https://example.com/soccer/squad-features": """
            <table>
              <thead>
                <tr><th>Club</th><th>Squad size</th><th>Average age</th></tr>
              </thead>
              <tbody>
                <tr><td>Arsenal</td><td>25</td><td>26.4</td></tr>
                <tr><td>Liverpool</td><td>24</td><td>27.1</td></tr>
              </tbody>
            </table>
            """,
        },
        minimum_columns=5,
    )
    result = builder.build()
    assert result is not None
    assert len(result.dataframe) == 2, result.dataframe.shape
    assert "incoming_transfer_spend" in result.dataframe.columns, result.dataframe.columns
    assert "squad_size" in result.dataframe.columns, result.dataframe.columns


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
        test_html_table_extractor_maps_laptop_spec_headers,
        test_html_table_extractor_derives_laptop_specs_from_name_blob,
        test_deterministic_synthesizer_coalesces_partial_rows,
        test_prepare_frame_drops_repeated_header_rows,
        test_prepare_frame_coalesces_duplicate_entities_with_partial_values,
        test_merge_frames_coalesces_overlapping_columns,
        test_merge_frames_preserves_combined_source_urls,
        test_concat_frames_preserves_union_of_columns_and_rows,
        test_finalize_frame_cleans_column_labels_and_ordering,
        test_finalize_frame_adds_schema_aliases_for_state_goal,
        test_prepare_frame_preserves_rank_when_schema_requires_it,
        test_finalize_frame_trims_excess_width_using_signal,
        test_finalize_frame_drops_duplicate_value_columns,
        test_goal_aware_row_count_prefers_expected_table_size,
        test_discover_ncaa_team_stat_urls_uses_filtered_option_values,
        test_extract_transfermarkt_club_summary_uses_shared_fetch_path,
        test_historical_nba_team_goal_skips_exact_row_cardinality,
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
