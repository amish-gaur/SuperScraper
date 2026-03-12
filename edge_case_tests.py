"""Focused deterministic edge-case regressions for base utility behavior."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile

from crawlee.errors import HttpClientStatusCodeError
import pandas as pd
from pydantic import BaseModel

from agent import AgentDecisionBase
from architect import DatasetArchitect, DatasetBlueprint, SourceTarget
from crawlee_fetcher import CrawleeFetchResult, _build_failure_outcome, _sanitize_artifact_for_storage
from data_validation import SemanticDataValidator
from entity_resolver import EntityResolver
from goal_intent import decompose_goal, infer_entity_intent, infer_goal_cardinality
from list_page_extractor import ListPageExtractor
from page_state import PageStateParser
from predictive_dataset_builder import DataQualityError, PredictiveDatasetBuilder
from source_discovery import SourceDiscoveryEngine
from source_health import FailureReason, FetchOutcome, SourceHealthRegistry
from source_memory import SourceMemory
from source_ranker import SourceRanker
from synthesizer import DataSynthesizer
from text_cleaner import TextCleaningUtility


class TeamRow(BaseModel):
    team_name: str | None = None
    school: str | None = None
    source_url: str | None = None
    reference_url: str | None = None


class LaptopRow(BaseModel):
    name: str | None = None
    price_usd: float | None = None
    cpu_model: str | None = None
    gpu_model: str | None = None
    ram_gb: float | None = None


class NumericLaptopRow(BaseModel):
    name: str | None = None
    price_usd: float | None = None
    ram_gb: int | None = None


class CompensationRow(BaseModel):
    name: str | None = None
    salary: float | None = None
    performance_rating: float | None = None
    source_url: str | None = None


class RangeRow(BaseModel):
    name: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    source_url: str | None = None


def test_entity_resolver_handles_nullish_values() -> None:
    resolver = EntityResolver()
    assert resolver.canonical_key(None) == ""
    assert resolver.canonical_key(math.nan) == ""
    assert resolver.canonical_key("nan") == ""


def test_list_page_extractor_derives_multiword_school_name() -> None:
    extractor = ListPageExtractor(TeamRow)
    records = extractor.extract(
        '"Duke Blue Devils" @e1 https://example.com/duke',
        source_url="https://example.com",
    )
    assert len(records) == 1
    assert records[0].school == "Duke"


def test_architect_infers_score_as_target_field() -> None:
    architect = DatasetArchitect(llm_gateway=None)
    blueprint = DatasetBlueprint.model_validate(
        architect._sanitize_recovered_blueprint(
            {
                "dataset_name": "x",
                "dataset_description": "y",
                "target_record_count": 10,
                "source_targets": [{"url": "https://valid.test/a", "expected_source_type": "html_table"}],
                "row_schema": {
                    "title": "Row",
                    "description": "d",
                    "fields": [
                        {"name": "name", "type": "string", "description": "entity name"},
                        {"name": "score", "type": "number", "description": "target score"},
                    ],
                },
            }
        )
    )
    normalized = architect._normalize_blueprint(blueprint, goal="predict score from name")
    assert normalized.row_schema.target_field == "score"
    roles = {field.name: field.ml_role for field in normalized.row_schema.fields}
    assert roles["score"] == "target"
    assert roles["name"] == "feature"


def test_architect_uses_deterministic_blueprint_for_weird_nba_salary_goal() -> None:
    architect = DatasetArchitect(llm_gateway=None)
    blueprint = architect.design(
        "I want data to predict how much an NBA player gets paid using on-court performance stats."
    )
    assert blueprint.row_schema.target_field == "salary"
    assert any("hoopshype.com/salaries/players" in target.url for target in blueprint.source_targets)
    assert any("espn.com/nba/stats/player" in target.url for target in blueprint.source_targets)
    assert any("espn.com/nba/salaries" in target.url for target in blueprint.source_targets)


def test_architect_uses_deterministic_blueprint_for_startup_valuation_goal() -> None:
    architect = DatasetArchitect(llm_gateway=None)
    blueprint = architect.design(
        "Put together a machine-learning table for startup companies where valuation is the label and funding is a key predictor."
    )
    assert blueprint.row_schema.target_field == "valuation"
    assert any("wikipedia.org/wiki/List_of_unicorn_startup_companies" in target.url for target in blueprint.source_targets)


def test_architect_uses_deterministic_blueprint_for_fortune_500_goal() -> None:
    architect = DatasetArchitect(llm_gateway=None)
    blueprint = architect.design(
        "Build a predictive dataset of Fortune 500 companies"
    )
    assert blueprint.dataset_name == "Fortune 500 Company Financials"
    assert blueprint.row_schema.target_field == "revenue_usd_millions"
    assert any("wikipedia.org/wiki/List_of_Fortune_500_companies" in target.url for target in blueprint.source_targets)


def test_goal_intent_treats_ncaa_programs_as_school_entities() -> None:
    assert infer_entity_intent("NCAA men's basketball programs") == "school"


def test_goal_cardinality_skips_exact_nba_team_count_for_historical_goals() -> None:
    assert infer_goal_cardinality(
        "Build a predictive dataset of NBA teams with playoff outcome target"
    ).count == 30
    assert infer_goal_cardinality(
        "Build a predictive dataset of NBA teams with historical team performance features and playoff outcome target"
    ) is None


def test_goal_decomposition_extracts_target_features_and_temporal_scope() -> None:
    decomposition = decompose_goal(
        "Build a historical dataset of startup companies to predict valuation from funding and revenue"
    )
    assert decomposition.domain_intent == "startup"
    assert decomposition.entity_intent == "company"
    assert decomposition.target_hint == "valuation"
    assert "funding" in decomposition.feature_hints
    assert decomposition.temporal_scope == "historical"


def test_source_memory_reuses_similar_goal_sources() -> None:
    with tempfile.TemporaryDirectory() as directory:
        memory = SourceMemory(Path(directory) / "memory.json")
        memory.record_success(
            "Predict NBA player salary from performance stats",
            ["https://www.espn.com/nba/salaries", "https://www.espn.com/nba/stats/player"],
        )
        reused = memory.similar_urls(
            "Build an NBA player dataset with salary as target and stats as features"
        )
    assert "https://www.espn.com/nba/salaries" in reused


def test_source_discovery_generates_non_search_seed_candidates() -> None:
    with tempfile.TemporaryDirectory() as directory:
        memory = SourceMemory(Path(directory) / "memory.json")
        discovery = SourceDiscoveryEngine(source_memory=memory)
        candidates = discovery.discover(
            "Build a dataset of U.S. states with population growth as the target and GDP as a feature"
        )
    urls = [candidate.url for candidate in candidates]
    assert any("census.gov" in url for url in urls)
    assert any("wikipedia.org" in url for url in urls)
    assert all("google.com/search" not in url for url in urls)


def test_predictive_builder_normalizes_lowercase_numeric_suffixes() -> None:
    builder = PredictiveDatasetBuilder(goal="x", dataset_name="x", starting_urls=[])
    assert builder._normalize_numeric_like("1.5m") == "1500000.0"
    assert builder._normalize_numeric_like("250k") == "250000.0"
    assert builder._normalize_numeric_like("4.1bn") == "4100000000.0"


def test_predictive_builder_accepts_estimate_wording_for_bank_goal() -> None:
    builder = PredictiveDatasetBuilder(
        goal="Give me a dataset for the biggest U.S. banks so I can estimate market value from asset size and capital strength.",
        dataset_name="x",
        starting_urls=[],
    )
    assert builder.is_applicable() is True


def test_semantic_validator_rejects_negative_and_inverted_ranges() -> None:
    validator = SemanticDataValidator(row_model=RangeRow)
    frame = pd.DataFrame(
        [
            {"name": "Valid", "salary_min": 100.0, "salary_max": 200.0, "source_url": "https://example.com/a"},
            {"name": "Negative", "salary_min": -5.0, "salary_max": 100.0, "source_url": "https://example.com/b"},
            {"name": "Inverted", "salary_min": 300.0, "salary_max": 100.0, "source_url": "https://example.com/c"},
        ]
    )
    result = validator.validate(frame, dataset_name="Comp")
    assert list(result.dataframe["name"]) == ["Valid"]
    assert result.report.dropped_row_count == 2


def test_semantic_validator_rejects_cross_source_conflicts_on_target() -> None:
    validator = SemanticDataValidator(row_model=CompensationRow)
    frame = pd.DataFrame(
        [
            {"name": "Alice", "salary": 100.0, "performance_rating": 8.0, "source_url": "https://a.example.com/alice"},
            {"name": "Bob", "salary": 120.0, "performance_rating": 7.5, "source_url": "https://a.example.com/bob"},
        ]
    )
    raw_records = [
        CompensationRow(name="Alice", salary=100.0, performance_rating=8.0, source_url="https://espn.com/alice"),
        CompensationRow(name="Alice", salary=200.0, performance_rating=8.1, source_url="https://hoopshype.com/alice"),
        CompensationRow(name="Bob", salary=120.0, performance_rating=7.5, source_url="https://espn.com/bob"),
    ]
    result = validator.validate(frame, dataset_name="Salaries", raw_records=raw_records)
    assert list(result.dataframe["name"]) == ["Bob"]
    assert result.report.cross_source_entity_count == 1


def test_crawlee_artifact_sanitizer_truncates_large_payloads() -> None:
    source_target = SourceTarget(url="https://example.com/data", expected_source_type="html_table")
    fetch_result = CrawleeFetchResult(
        fetch_outcome=FetchOutcome(url="https://example.com/data", ok=True, status_code=200),
        html_text="A" * 250000,
        json_payload={"payload": "B" * 60000},
        adapter_payload={"payload": "C" * 10},
    )
    artifact = _sanitize_artifact_for_storage(source_target=source_target, fetch_result=fetch_result)
    assert artifact is not None
    assert artifact["artifact_truncated"] is True
    assert len(artifact["html_text"]) == 200000
    assert artifact["json_payload"]["_truncated"] is True


def test_source_ranker_normalizes_mixed_case_urls() -> None:
    ranked = SourceRanker().rank("example", [" HTTPS://EXAMPLE.COM/A ", "https://example.com/a"])
    assert len(ranked) == 2
    assert {item.url for item in ranked} == {"https://example.com/A", "https://example.com/a"}


def test_source_ranker_prefers_discovery_backed_table_sources_over_homepages() -> None:
    ranked = SourceRanker().rank(
        "Predict NBA player salary from performance stats",
        [
            "https://www.espn.com/",
            "https://www.espn.com/nba/salaries",
        ],
        context={
            "source_family_by_url": {
                "https://www.espn.com/": "publisher_list",
                "https://www.espn.com/nba/salaries": "adapter",
            },
            "preferred_domains": ("espn.com",),
            "query_terms": ("nba player salary stats table",),
        },
    )
    assert ranked[0].url == "https://www.espn.com/nba/salaries"


def test_source_health_tracks_fetch_and_extraction_success_separately() -> None:
    registry = SourceHealthRegistry()
    url = "https://example.com/data"
    registry.record_fetch(FetchOutcome(url=url, ok=True, text="<html>ok</html>"))
    registry.record_extraction(url, records_extracted=50, success=True, reason=FailureReason.SUCCESS)
    stats = registry.domains["example.com"]
    assert stats.fetch_successes == 1
    assert stats.successes == 1
    assert registry.domain_penalty(url) == -6


def test_crawlee_failure_outcome_preserves_http_status_code() -> None:
    outcome = _build_failure_outcome(
        url="https://example.test/missing",
        exc=HttpClientStatusCodeError("Client error status code returned", 404),
    )
    assert outcome.ok is False
    assert outcome.status_code == 404
    assert outcome.reason == FailureReason.HTTP_ERROR


def test_page_state_parser_surfaces_high_signal_lines_beyond_page_chrome() -> None:
    parser = PageStateParser()
    chrome = "\n".join(f'button "Navigation item {index}" [ref=e{index}]' for index in range(1, 26))
    rows = "\n".join(
        [
            'heading "2025 Team Salaries"',
            'row "Boston Celtics $192.3M 15 players" [ref=e101]',
            'row "Denver Nuggets $181.1M 14 players" [ref=e102]',
            'row "Phoenix Suns $209.4M 15 players" [ref=e103]',
        ]
    )
    snapshot = f"{chrome}\n{rows}"
    page_state = parser.parse(snapshot, current_url="https://example.com/salaries")
    assert "Boston Celtics" in page_state.visible_text_summary
    assert "Denver Nuggets" in page_state.visible_text_summary
    assert "2025 Team Salaries" in page_state.visible_text_summary


def test_synthesizer_prepares_document_text_by_stripping_html_noise() -> None:
    synthesizer = DataSynthesizer(row_model=TeamRow, llm_gateway=None)
    prepared = synthesizer._prepare_document_text_for_prompt(
        """
        <html>
          <head>
            <style>.hidden { display:none; }</style>
            <script>window.__STATE__ = {"ignore": true};</script>
          </head>
          <body>
            <nav>Log in</nav>
            <main>
              <h1>2025 Team Salaries</h1>
              <div>Boston Celtics $192.3M</div>
              <div>Denver Nuggets $181.1M</div>
            </main>
          </body>
        </html>
        """
    )
    assert "Boston Celtics $192.3M" in prepared
    assert "Denver Nuggets $181.1M" in prepared
    assert "window.__STATE__" not in prepared
    assert "Log in" not in prepared


def test_agent_decision_normalizes_high_level_fallback_actions() -> None:
    decision = AgentDecisionBase.model_validate(
        {
            "status": "navigating",
            "thought_process": "",
            "high_level_action": "scroll_for_more",
            "action_type": "scroll_for_more",
            "action_target": None,
            "action_value": 500,
        }
    )
    assert decision.action_type == "scroll_down"
    assert decision.action_value == "500"


def test_architect_uses_deterministic_laptop_blueprint() -> None:
    architect = DatasetArchitect(llm_gateway=None)
    blueprint = architect.design("Build a dataset of pc specs of modern laptops")
    assert blueprint.dataset_name == "Modern Laptop Specs for Price Prediction"
    assert blueprint.row_schema.target_field == "price_usd"
    assert any("notebookcheck.net" in target.url for target in blueprint.source_targets)


def test_synthesizer_prefers_deterministic_merge_for_clean_records() -> None:
    records = [
        LaptopRow(name="Laptop A", price_usd=999, cpu_model="Intel Core 7 150U", gpu_model="Intel Arc", ram_gb=16),
        LaptopRow(name="Laptop B", price_usd=1299, cpu_model="Intel Core Ultra 7 155H", gpu_model="RTX 4050", ram_gb=32),
    ]
    synthesizer = DataSynthesizer(row_model=LaptopRow, llm_gateway=None)
    assert synthesizer._records_are_structurally_clean(records) is True
    assert synthesizer._should_use_llm_merge(records) is False


def test_text_cleaner_parses_us_and_eu_prices() -> None:
    assert TextCleaningUtility.clean_price("$1,299.99") == 1299.99
    assert TextCleaningUtility.clean_price("29,99 €") == 29.99


def test_text_cleaner_extracts_laptop_specs() -> None:
    specs = TextCleaningUtility.extract_laptop_specs("16GB RAM DDR5, 512GB SSD, RTX 4060")
    assert specs == {"ram_gb": 16, "storage_gb": 512, "gpu_model": "RTX 4060"}


def test_synthesizer_cleans_numeric_fields_from_text() -> None:
    synthesizer = DataSynthesizer(row_model=NumericLaptopRow, llm_gateway=None)
    cleaned = synthesizer._clean_payload_for_schema(
        {
            "name": "Laptop A",
            "price_usd": "$1,299.99",
            "ram_gb": "16GB RAM DDR5",
        }
    )
    assert cleaned["price_usd"] == 1299.99
    assert cleaned["ram_gb"] == 16


def test_predictive_builder_rejects_low_fill_rate() -> None:
    builder = PredictiveDatasetBuilder(
        goal="predict laptop price",
        dataset_name="x",
        starting_urls=[],
        target_field="price_usd",
        core_feature_fields=["cpu_model", "ram_gb"],
    )
    frame = pd.DataFrame(
        {
            "name": ["A", "B", "C", "D", "E"],
            "entity_name": ["a", "b", "c", "d", "e"],
            "price_usd": [1000, None, None, None, None],
            "cpu_model": ["Intel", None, None, None, None],
            "ram_gb": [16, None, None, 32, None],
        }
    )
    try:
        builder._enforce_fill_rate(frame)
    except DataQualityError as exc:
        assert "low fill rate" in str(exc)
    else:
        raise AssertionError("Expected DataQualityError for low fill rate")


def main(*, verbose: bool = True) -> int:
    tests = [
        test_entity_resolver_handles_nullish_values,
        test_list_page_extractor_derives_multiword_school_name,
        test_architect_infers_score_as_target_field,
        test_architect_uses_deterministic_blueprint_for_weird_nba_salary_goal,
        test_architect_uses_deterministic_blueprint_for_startup_valuation_goal,
        test_architect_uses_deterministic_blueprint_for_fortune_500_goal,
        test_goal_intent_treats_ncaa_programs_as_school_entities,
        test_goal_cardinality_skips_exact_nba_team_count_for_historical_goals,
        test_goal_decomposition_extracts_target_features_and_temporal_scope,
        test_source_memory_reuses_similar_goal_sources,
        test_source_discovery_generates_non_search_seed_candidates,
        test_predictive_builder_normalizes_lowercase_numeric_suffixes,
        test_predictive_builder_accepts_estimate_wording_for_bank_goal,
        test_semantic_validator_rejects_negative_and_inverted_ranges,
        test_semantic_validator_rejects_cross_source_conflicts_on_target,
        test_crawlee_artifact_sanitizer_truncates_large_payloads,
        test_source_ranker_normalizes_mixed_case_urls,
        test_source_ranker_prefers_discovery_backed_table_sources_over_homepages,
        test_source_health_tracks_fetch_and_extraction_success_separately,
        test_page_state_parser_surfaces_high_signal_lines_beyond_page_chrome,
        test_synthesizer_prepares_document_text_by_stripping_html_noise,
        test_agent_decision_normalizes_high_level_fallback_actions,
        test_architect_uses_deterministic_laptop_blueprint,
        test_synthesizer_prefers_deterministic_merge_for_clean_records,
        test_text_cleaner_parses_us_and_eu_prices,
        test_text_cleaner_extracts_laptop_specs,
        test_synthesizer_cleans_numeric_fields_from_text,
        test_predictive_builder_rejects_low_fill_rate,
    ]
    for test in tests:
        test()
        if verbose:
            print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
