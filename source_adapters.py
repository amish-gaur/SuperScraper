"""Domain-aware source adapters for expanding predictive dataset candidates."""

from __future__ import annotations

from dataclasses import dataclass
import re

from goal_intent import infer_domain_intent, infer_entity_intent


@dataclass(frozen=True, slots=True)
class SourceAdapter:
    """One adapter that can contribute supplemental URLs for a goal."""

    name: str
    matcher: callable
    supplier: callable

    def applies(self, goal: str) -> bool:
        return bool(self.matcher(goal))

    def urls(self, goal: str) -> list[str]:
        return list(self.supplier(goal))


def build_adapters() -> list[SourceAdapter]:
    return [
        SourceAdapter(
            name="ncaa-basketball",
            matcher=lambda goal: "basketball" in goal.lower() and any(
                token in goal.lower() for token in ("ncaa", "college", "division i")
            ),
            supplier=_ncaa_basketball_urls,
        ),
        SourceAdapter(
            name="nba-basketball",
            matcher=lambda goal: infer_domain_intent(goal) == "nba" and infer_entity_intent(goal) == "team",
            supplier=_nba_basketball_urls,
        ),
        SourceAdapter(
            name="nba-players",
            matcher=lambda goal: infer_domain_intent(goal) == "nba" and infer_entity_intent(goal) == "player",
            supplier=_nba_player_urls,
        ),
        SourceAdapter(
            name="fortune-companies",
            matcher=lambda goal: "fortune 500" in goal.lower(),
            supplier=_fortune_500_urls,
        ),
        SourceAdapter(
            name="startup-companies",
            matcher=lambda goal: infer_domain_intent(goal) == "startup",
            supplier=_startup_company_urls,
        ),
        SourceAdapter(
            name="soccer-clubs",
            matcher=lambda goal: infer_domain_intent(goal) == "soccer" and infer_entity_intent(goal) == "club",
            supplier=_soccer_club_urls,
        ),
        SourceAdapter(
            name="banks",
            matcher=lambda goal: "bank" in goal.lower(),
            supplier=_bank_urls,
        ),
        SourceAdapter(
            name="population",
            matcher=lambda goal: "population" in goal.lower(),
            supplier=_population_urls,
        ),
        SourceAdapter(
            name="laptops",
            matcher=lambda goal: any(token in goal.lower() for token in ("laptop", "laptops", "notebook", "notebooks")) and any(
                token in goal.lower() for token in ("spec", "specs", "price", "prices", "pc")
            ),
            supplier=_laptop_urls,
        ),
    ]


def adapter_urls_for_goal(goal: str) -> list[str]:
    urls: list[str] = []
    for adapter in build_adapters():
        if not adapter.applies(goal):
            continue
        urls.extend(adapter.urls(goal))
    return list(dict.fromkeys(urls))


def _ncaa_basketball_urls(goal: str) -> list[str]:
    year = _infer_year(goal) or 2026
    return [
        "https://www.ncaa.com/stats/basketball-men/d1/current/team/145",
        f"https://www.sports-reference.com/cbb/seasons/{year}-school-stats.html",
        f"https://www.sports-reference.com/cbb/seasons/{year}-advanced-school-stats.html",
        f"https://www.sports-reference.com/cbb/seasons/{year}-opponent-stats.html",
        "https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_men%27s_basketball_programs",
        "https://www.teamrankings.com/ncb/stats/",
        "https://www.statbunker.com/",
    ]


def _nba_basketball_urls(goal: str) -> list[str]:
    year = _infer_year(goal) or 2025
    return [
        f"https://www.nba.com/stats/teams/traditional?Season={year-1}-{str(year)[-2:]}",
        "https://www.teamrankings.com/nba/stat/points-per-game",
        "https://en.wikipedia.org/wiki/List_of_National_Basketball_Association_seasons",
        "https://en.wikipedia.org/wiki/National_Basketball_Association",
    ]


def _nba_player_urls(goal: str) -> list[str]:
    return [
        "https://www.espn.com/nba/stats/player",
        "https://www.espn.com/nba/salaries",
        "https://hoopshype.com/salaries/players/",
        "https://www.nba.com/players",
    ]


def _fortune_500_urls(goal: str) -> list[str]:
    return [
        "https://en.wikipedia.org/wiki/List_of_Fortune_500_companies",
        "https://fortune.com/ranking/fortune500/",
        "https://companiesmarketcap.com/largest-companies-by-revenue/",
    ]


def _startup_company_urls(goal: str) -> list[str]:
    return [
        "https://en.wikipedia.org/wiki/List_of_unicorn_startup_companies",
    ]


def _soccer_club_urls(goal: str) -> list[str]:
    season = (_infer_year(goal) or 2025) - 1
    return [
        f"https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
        f"https://www.transfermarkt.com/laliga/transfers/wettbewerb/ES1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
        f"https://www.transfermarkt.com/serie-a/transfers/wettbewerb/IT1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
        f"https://www.transfermarkt.com/ligue-1/transfers/wettbewerb/FR1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
        f"https://www.transfermarkt.com/bundesliga/transfers/wettbewerb/L1/plus/?saison_id={season}&s_w=s&leihe=1&intern=0",
    ]


def _bank_urls(goal: str) -> list[str]:
    return [
        "https://www.usbanklocations.com/bank-rank/net-income---incomeexpense--ie-netinc.html?d=2024-06-30",
        "https://www.usbanklocations.com/bank-rank/quarterly-net-income.html",
        "https://en.wikipedia.org/wiki/List_of_largest_banks_in_the_United_States",
        "https://en.wikipedia.org/wiki/List_of_largest_banks",
        "https://companiesmarketcap.com/banks/largest-banks-by-market-cap/",
    ]


def _population_urls(goal: str) -> list[str]:
    return [
        "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population_growth_rate",
        "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_economic_growth_rate",
        "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population",
        "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_GDP",
        "https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html",
    ]


def _laptop_urls(goal: str) -> list[str]:
    return [
        "https://www.notebookcheck.net/Ranking-Best-all-around-laptops-reviewed-by-Notebookcheck.98608.0.html",
        "https://www.notebookcheck.net/Ranking-Best-affordable-all-around-laptops.281362.0.html",
        "https://www.notebookcheck.net/The-Best-Ultrabooks.91067.0.html",
    ]


def _infer_year(goal: str) -> int | None:
    matches = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", goal)
    return int(matches[-1]) if matches else None
