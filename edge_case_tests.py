"""Focused deterministic edge-case regressions for base utility behavior."""

from __future__ import annotations

import math

import pandas as pd
from pydantic import BaseModel

from agent import AgentDecisionBase
from architect import DatasetArchitect, DatasetBlueprint
from entity_resolver import EntityResolver
from goal_intent import infer_entity_intent, infer_goal_cardinality
from list_page_extractor import ListPageExtractor
from page_state import PageStateParser
from predictive_dataset_builder import DataQualityError, PredictiveDatasetBuilder
from source_health import FailureReason, FetchOutcome, SourceHealthRegistry
from source_ranker import SourceRanker
from synthesizer import DataSynthesizer
from text_cleaner import TextCleaningUtility
from synthesizer import DataSynthesizer


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


def test_architect_uses_deterministic_blueprint_for_startup_valuation_goal() -> None:
    architect = DatasetArchitect(llm_gateway=None)
    blueprint = architect.design(
        "Put together a machine-learning table for startup companies where valuation is the label and funding is a key predictor."
    )
    assert blueprint.row_schema.target_field == "valuation"
    assert any("wikipedia.org/wiki/List_of_unicorn_startup_companies" in target.url for target in blueprint.source_targets)


def test_goal_intent_treats_ncaa_programs_as_school_entities() -> None:
    assert infer_entity_intent("NCAA men's basketball programs") == "school"


def test_goal_cardinality_skips_exact_nba_team_count_for_historical_goals() -> None:
    assert infer_goal_cardinality(
        "Build a predictive dataset of NBA teams with playoff outcome target"
    ).count == 30
    assert infer_goal_cardinality(
        "Build a predictive dataset of NBA teams with historical team performance features and playoff outcome target"
    ) is None


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


def test_source_ranker_normalizes_mixed_case_urls() -> None:
    ranked = SourceRanker().rank("example", [" HTTPS://EXAMPLE.COM/A ", "https://example.com/a"])
    assert len(ranked) == 2
    assert {item.url for item in ranked} == {"https://example.com/A", "https://example.com/a"}


def test_source_health_tracks_fetch_and_extraction_success_separately() -> None:
    registry = SourceHealthRegistry()
    url = "https://example.com/data"
    registry.record_fetch(FetchOutcome(url=url, ok=True, text="<html>ok</html>"))
    registry.record_extraction(url, records_extracted=50, success=True, reason=FailureReason.SUCCESS)
    stats = registry.domains["example.com"]
    assert stats.fetch_successes == 1
    assert stats.successes == 1
    assert registry.domain_penalty(url) == -6


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
        test_goal_intent_treats_ncaa_programs_as_school_entities,
        test_goal_cardinality_skips_exact_nba_team_count_for_historical_goals,
        test_predictive_builder_normalizes_lowercase_numeric_suffixes,
        test_predictive_builder_accepts_estimate_wording_for_bank_goal,
        test_source_ranker_normalizes_mixed_case_urls,
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
