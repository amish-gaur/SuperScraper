"""Deterministic demo datasets for hosted prototype fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from goal_intent import infer_domain_intent, infer_entity_intent


@dataclass(frozen=True, slots=True)
class DemoDataset:
    records: list[dict[str, Any]]
    provenance_map: dict[str, str]
    source_label: str


def demo_dataset_for_goal(goal: str) -> DemoDataset | None:
    domain = infer_domain_intent(goal)
    entity = infer_entity_intent(goal)
    lowered = goal.lower()

    if domain == "nba" and entity == "player":
        return _nba_salary_demo()
    if domain == "public-data" and entity == "state":
        return _state_growth_demo()
    if domain == "ncaa-basketball" and any(
        phrase in lowered for phrase in ("team statistics", "team strength", "teams")
    ):
        return _ncaa_team_demo()
    return None


def _nba_salary_demo() -> DemoDataset:
    source = "demo_fallback://nba_salary"
    records = [
        {"name": "Stephen Curry", "salary": 59606817, "points_per_game": 26.4, "assists_per_game": 6.1, "three_point_pct": 40.8, "player_efficiency_rating": 24.7, "team": "Warriors", "position": "G", "source_url": source},
        {"name": "Nikola Jokic", "salary": 55224526, "points_per_game": 26.2, "assists_per_game": 9.0, "three_point_pct": 35.7, "player_efficiency_rating": 31.0, "team": "Nuggets", "position": "C", "source_url": source},
        {"name": "Joel Embiid", "salary": 55224526, "points_per_game": 33.1, "assists_per_game": 5.6, "three_point_pct": 38.8, "player_efficiency_rating": 29.4, "team": "76ers", "position": "C", "source_url": source},
        {"name": "Kevin Durant", "salary": 54708608, "points_per_game": 27.1, "assists_per_game": 5.3, "three_point_pct": 41.3, "player_efficiency_rating": 25.6, "team": "Suns", "position": "F", "source_url": source},
        {"name": "Anthony Davis", "salary": 54126450, "points_per_game": 24.7, "assists_per_game": 3.5, "three_point_pct": 27.1, "player_efficiency_rating": 26.5, "team": "Lakers", "position": "F-C", "source_url": source},
        {"name": "Jayson Tatum", "salary": 54126450, "points_per_game": 27.0, "assists_per_game": 4.9, "three_point_pct": 37.6, "player_efficiency_rating": 23.8, "team": "Celtics", "position": "F", "source_url": source},
        {"name": "Jimmy Butler", "salary": 54126450, "points_per_game": 20.8, "assists_per_game": 5.3, "three_point_pct": 41.4, "player_efficiency_rating": 22.1, "team": "Heat", "position": "F", "source_url": source},
        {"name": "Damian Lillard", "salary": 48787976, "points_per_game": 24.3, "assists_per_game": 7.0, "three_point_pct": 35.4, "player_efficiency_rating": 21.8, "team": "Bucks", "position": "G", "source_url": source},
        {"name": "Luka Doncic", "salary": 43031940, "points_per_game": 33.9, "assists_per_game": 9.8, "three_point_pct": 38.2, "player_efficiency_rating": 30.1, "team": "Mavericks", "position": "G", "source_url": source},
        {"name": "Shai Gilgeous-Alexander", "salary": 35859950, "points_per_game": 30.1, "assists_per_game": 6.2, "three_point_pct": 35.3, "player_efficiency_rating": 28.6, "team": "Thunder", "position": "G", "source_url": source},
        {"name": "Devin Booker", "salary": 49205000, "points_per_game": 27.1, "assists_per_game": 6.9, "three_point_pct": 36.4, "player_efficiency_rating": 22.8, "team": "Suns", "position": "G", "source_url": source},
        {"name": "Donovan Mitchell", "salary": 35410000, "points_per_game": 26.6, "assists_per_game": 6.1, "three_point_pct": 36.8, "player_efficiency_rating": 22.6, "team": "Cavaliers", "position": "G", "source_url": source},
    ]
    return DemoDataset(
        records=records,
        provenance_map={column: source for column in records[0].keys() if column != "name"},
        source_label=source,
    )


def _state_growth_demo() -> DemoDataset:
    source = "demo_fallback://us_states"
    records = [
        {"state": "Texas", "population_growth_rate": 1.6, "gdp": 2660.0, "median_income": 76000, "unemployment_rate": 4.1, "population": 30500000, "source_url": source},
        {"state": "Florida", "population_growth_rate": 1.9, "gdp": 1590.0, "median_income": 71000, "unemployment_rate": 3.4, "population": 22600000, "source_url": source},
        {"state": "California", "population_growth_rate": 0.4, "gdp": 3890.0, "median_income": 92000, "unemployment_rate": 5.2, "population": 38900000, "source_url": source},
        {"state": "North Carolina", "population_growth_rate": 1.3, "gdp": 840.0, "median_income": 69000, "unemployment_rate": 3.8, "population": 10900000, "source_url": source},
        {"state": "Georgia", "population_growth_rate": 1.2, "gdp": 910.0, "median_income": 72000, "unemployment_rate": 3.6, "population": 11100000, "source_url": source},
        {"state": "Arizona", "population_growth_rate": 1.4, "gdp": 560.0, "median_income": 74000, "unemployment_rate": 3.9, "population": 7440000, "source_url": source},
        {"state": "Tennessee", "population_growth_rate": 1.1, "gdp": 560.0, "median_income": 67000, "unemployment_rate": 3.2, "population": 7130000, "source_url": source},
        {"state": "New York", "population_growth_rate": -0.2, "gdp": 2280.0, "median_income": 84000, "unemployment_rate": 4.4, "population": 19400000, "source_url": source},
        {"state": "Illinois", "population_growth_rate": -0.1, "gdp": 1090.0, "median_income": 78000, "unemployment_rate": 4.7, "population": 12500000, "source_url": source},
        {"state": "Washington", "population_growth_rate": 1.0, "gdp": 820.0, "median_income": 94000, "unemployment_rate": 4.5, "population": 7820000, "source_url": source},
        {"state": "Colorado", "population_growth_rate": 0.9, "gdp": 520.0, "median_income": 89000, "unemployment_rate": 3.7, "population": 5900000, "source_url": source},
        {"state": "Nevada", "population_growth_rate": 1.5, "gdp": 240.0, "median_income": 69000, "unemployment_rate": 5.1, "population": 3200000, "source_url": source},
    ]
    return DemoDataset(
        records=records,
        provenance_map={column: source for column in records[0].keys() if column != "state"},
        source_label=source,
    )


def _ncaa_team_demo() -> DemoDataset:
    source = "demo_fallback://ncaa_teams"
    records = [
        {"team_name": "Houston", "scoring_offense": 74.2, "scoring_defense": 58.3, "rebound_margin": 7.8, "turnover_margin": 3.4, "effective_fg_pct": 55.1, "winning_pct": 0.892, "source_url": source},
        {"team_name": "UConn", "scoring_offense": 81.4, "scoring_defense": 63.2, "rebound_margin": 8.1, "turnover_margin": 2.0, "effective_fg_pct": 56.4, "winning_pct": 0.865, "source_url": source},
        {"team_name": "Purdue", "scoring_offense": 83.0, "scoring_defense": 69.0, "rebound_margin": 6.2, "turnover_margin": 1.8, "effective_fg_pct": 57.8, "winning_pct": 0.838, "source_url": source},
        {"team_name": "Tennessee", "scoring_offense": 79.1, "scoring_defense": 63.8, "rebound_margin": 5.7, "turnover_margin": 3.1, "effective_fg_pct": 54.4, "winning_pct": 0.811, "source_url": source},
        {"team_name": "North Carolina", "scoring_offense": 81.0, "scoring_defense": 68.4, "rebound_margin": 7.1, "turnover_margin": 1.6, "effective_fg_pct": 53.8, "winning_pct": 0.784, "source_url": source},
        {"team_name": "Auburn", "scoring_offense": 82.2, "scoring_defense": 70.2, "rebound_margin": 6.5, "turnover_margin": 2.1, "effective_fg_pct": 54.9, "winning_pct": 0.757, "source_url": source},
        {"team_name": "Iowa State", "scoring_offense": 75.4, "scoring_defense": 61.9, "rebound_margin": 4.8, "turnover_margin": 4.2, "effective_fg_pct": 52.7, "winning_pct": 0.784, "source_url": source},
        {"team_name": "Arizona", "scoring_offense": 85.1, "scoring_defense": 71.0, "rebound_margin": 8.9, "turnover_margin": 1.1, "effective_fg_pct": 55.0, "winning_pct": 0.784, "source_url": source},
        {"team_name": "Duke", "scoring_offense": 80.0, "scoring_defense": 67.2, "rebound_margin": 5.2, "turnover_margin": 1.4, "effective_fg_pct": 55.3, "winning_pct": 0.757, "source_url": source},
        {"team_name": "Creighton", "scoring_offense": 79.5, "scoring_defense": 68.0, "rebound_margin": 4.0, "turnover_margin": 0.8, "effective_fg_pct": 56.1, "winning_pct": 0.757, "source_url": source},
        {"team_name": "Illinois", "scoring_offense": 84.5, "scoring_defense": 73.4, "rebound_margin": 5.0, "turnover_margin": -0.1, "effective_fg_pct": 55.5, "winning_pct": 0.730, "source_url": source},
        {"team_name": "Gonzaga", "scoring_offense": 84.1, "scoring_defense": 71.6, "rebound_margin": 7.0, "turnover_margin": 0.6, "effective_fg_pct": 57.0, "winning_pct": 0.730, "source_url": source},
    ]
    return DemoDataset(
        records=records,
        provenance_map={column: source for column in records[0].keys() if column != "team_name"},
        source_label=source,
    )
