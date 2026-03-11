"""Focused deterministic edge-case regressions for base utility behavior."""

from __future__ import annotations

import math

from pydantic import BaseModel

from architect import DatasetArchitect, DatasetBlueprint
from entity_resolver import EntityResolver
from goal_intent import infer_entity_intent
from list_page_extractor import ListPageExtractor
from predictive_dataset_builder import PredictiveDatasetBuilder
from source_ranker import SourceRanker


class TeamRow(BaseModel):
    team_name: str | None = None
    school: str | None = None
    source_url: str | None = None
    reference_url: str | None = None


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


def test_goal_intent_treats_ncaa_programs_as_school_entities() -> None:
    assert infer_entity_intent("NCAA men's basketball programs") == "school"


def test_predictive_builder_normalizes_lowercase_numeric_suffixes() -> None:
    builder = PredictiveDatasetBuilder(goal="x", dataset_name="x", starting_urls=[])
    assert builder._normalize_numeric_like("1.5m") == "1500000.0"
    assert builder._normalize_numeric_like("250k") == "250000.0"
    assert builder._normalize_numeric_like("4.1bn") == "4100000000.0"


def test_source_ranker_normalizes_mixed_case_urls() -> None:
    ranked = SourceRanker().rank("example", [" HTTPS://EXAMPLE.COM/A ", "https://example.com/a"])
    assert len(ranked) == 1
    assert ranked[0].url == "https://example.com/a"


def main(*, verbose: bool = True) -> int:
    tests = [
        test_entity_resolver_handles_nullish_values,
        test_list_page_extractor_derives_multiword_school_name,
        test_architect_infers_score_as_target_field,
        test_goal_intent_treats_ncaa_programs_as_school_entities,
        test_predictive_builder_normalizes_lowercase_numeric_suffixes,
        test_source_ranker_normalizes_mixed_case_urls,
    ]
    for test in tests:
        test()
        if verbose:
            print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
