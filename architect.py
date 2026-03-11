"""Dataset architecture stage for dynamic schema generation."""

from __future__ import annotations

import logging
import ast
import json
import re
from typing import Any, Literal
from urllib.parse import quote_plus, urlparse

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from source_adapters import adapter_urls_for_goal
from goal_intent import infer_goal_cardinality
from llm import LLMError, LLMGateway, StructuredEnvelope
from source_health import REGISTRY
from source_ranker import SourceRanker


LOGGER = logging.getLogger(__name__)


class SchemaPropertySpec(StructuredEnvelope):
    """A constrained JSON Schema property description for a dataset row."""

    name: str = Field(description="Column name in snake_case.")
    type: str = Field(description='JSON Schema type: "string", "integer", "number", "boolean", or "array".')
    description: str = Field(description="Description of the field.")
    ml_role: str = Field(
        default="feature",
        description='ML role for the column: "target" for the label or "feature" for predictive inputs.',
    )
    nullable: bool = Field(default=False, description="Whether the field may be null.")
    enum: list[str] | None = Field(default=None, description="Optional enum values for the field.")
    items_type: str | None = Field(
        default=None,
        description='Array item type when type="array".',
    )


class GeneratedRowSchema(StructuredEnvelope):
    """Structured row schema emitted by the LLM."""

    title: str = Field(description="Short title for the row schema.")
    description: str = Field(description="Description of what a dataset row represents.")
    target_field: str | None = Field(
        default=None,
        description="Primary supervised learning target variable for the row.",
    )
    fields: list[SchemaPropertySpec] = Field(
        description="Schema fields for the dataset row."
    )
    required: list[str] = Field(
        default_factory=list,
        description="Required property names. If omitted, they will be inferred from non-nullable fields.",
    )


class SourceTarget(StructuredEnvelope):
    """Seed source plus the architect's best guess for how it should be extracted."""

    url: str = Field(description="Direct source URL to target.")
    expected_source_type: Literal["html_table", "json_api", "react_state", "browser_heavy", "unknown"] = Field(
        description="Expected extraction strategy for this source."
    )

    @field_validator("expected_source_type", mode="before")
    @classmethod
    def _coerce_expected_source_type(cls, value: Any) -> str:
        allowed = {"html_table", "json_api", "react_state", "browser_heavy", "unknown"}
        normalized = str(value or "unknown").strip().lower()
        return normalized if normalized in allowed else "unknown"


class DatasetBlueprint(StructuredEnvelope):
    """Blueprint describing the dataset to collect."""

    dataset_name: str = Field(description="Human-readable dataset name.")
    dataset_description: str = Field(description="What the dataset is intended to model.")
    target_record_count: int = Field(description="Recommended number of rows to gather.")
    source_targets: list[SourceTarget] = Field(
        description="Direct high-probability source targets for the scraping swarm to route first."
    )
    row_schema: GeneratedRowSchema = Field(description="JSON Schema for a single dataset row.")

    @property
    def starting_urls(self) -> list[str]:
        """Compatibility shim for older callers that only expect URL strings."""
        return [target.url for target in self.source_targets]


class DynamicRowBase(BaseModel):
    """Base model for dynamically generated row schemas."""

    model_config = ConfigDict(extra="forbid")


class DatasetArchitect:
    """Use an LLM to design a dynamic dataset schema from a high-level ML goal."""

    def __init__(self, llm_gateway: LLMGateway | None = None) -> None:
        self.llm_gateway = llm_gateway
        self.source_ranker = SourceRanker()

    def design(self, goal: str, forbidden_domains: set[str] | None = None) -> DatasetBlueprint:
        """Generate a dataset blueprint and row schema for the requested goal."""
        if self._should_use_deterministic_blueprint(goal):
            blueprint = self._deterministic_blueprint_from_goal(goal)
            blueprint = self._normalize_blueprint(blueprint, goal=goal)
            LOGGER.info("Architect generated dataset blueprint: %s", blueprint.dataset_name)
            return blueprint

        forbidden_domains = {domain.lower() for domain in (forbidden_domains or set()) if domain}
        forbidden_domains_text = (
            f"\nForbidden domains:\n{', '.join(sorted(forbidden_domains))}\n"
            if forbidden_domains
            else ""
        )
        system_prompt = (
            "You are a Senior Machine Learning Engineer designing supervised-learning datasets for web scraping. "
            "You must respond with a valid JSON object matching the requested structure. "
            "Given a high-level ML goal, first identify the single Target Variable that should be predicted, then identify 3 to 5 high-signal Predictive Features that are likely to help predict that target. "
            "Favor observable columns that are causally or statistically useful for prediction and likely to appear together on a single public list, directory, aggregator, table, or index page. "
            "Strictly exclude low-signal, decorative, or administrative fields such as jersey colors, website URLs, logos, random IDs, UUIDs, slugs, internal keys, and other metadata that would not materially help a supervised model. "
            "The row schema must contain exactly one target field plus 3 to 5 predictive feature fields. "
            "Use strict primitive types only: numerical or continuous values must be integer or number, binary indicators must be boolean, and categorical/text values must be string. "
            "Every field description must explain why the value is being extracted for modeling, not just what it is. "
            "Do not plan around search engines. Instead, identify likely public source domains and return direct source targets "
            "for pages or site sections with a high probability of containing the needed rows. "
            "Prioritize high-density directory pages, aggregator pages, public databases, and Wikipedia 'List of...' pages related to the goal. "
            "When generating source_targets, you MUST provide URLs from at least 3 distinct root domains. Do not provide multiple URLs from the same website. Always include at least one high-density aggregator or Wikipedia link as a safe fallback. "
            "For each source target, set expected_source_type to one of: html_table, json_api, react_state, browser_heavy, or unknown. "
            "Avoid generic homepages when a more specific index or listing page exists. "
            "Avoid highly restricted or anti-bot-heavy domains whenever possible. "
            "Keep the row schema simple and list-friendly: 4 to 6 flat fields total, no deep nesting, and no fields that would require opening individual profile pages unless absolutely necessary."
        )
        user_prompt = (
            f"Machine learning goal:\n{goal}\n\n"
            "Return:\n"
            "1. A concise dataset name.\n"
            "2. A short dataset description.\n"
            "3. A recommended target_record_count between 25 and 250.\n"
            "4. Four to eight source_targets using full https:// URLs from likely source sites.\n"
            "5. A JSON Schema for a single dataset row with simple JSON types only.\n\n"
            "For the row schema:\n"
            "- First decide the primary target variable and set row_schema.target_field to that field name.\n"
            "- Include exactly one field with ml_role=\"target\".\n"
            "- Include 3 to 5 additional fields with ml_role=\"feature\".\n"
            "- Each field description must state why that field matters for prediction.\n"
            "- Do not include URLs, IDs, decorative attributes, or provenance-only columns in the row schema.\n\n"
            "The source targets should be direct entry points to data-rich list pages such as roster pages, statistics indexes, public directories, "
            "Wikipedia list pages, database listings, rankings tables, or official archive indexes. "
            "Do not return generic homepages if a category or list page exists. "
            "Each source target must include expected_source_type so the downstream router can prefer hidden APIs and frontend hydration JSON before browser automation. "
            "The row schema must be simple enough that the swarm can extract multiple rows from a single listing page without clicking into detail pages. "
            "Focus on emitting complete field definitions. If you are unsure about row_schema.required, you may omit it.\n\n"
            f"{forbidden_domains_text}"
            "Return JSON in exactly this shape:\n"
            f"{self._blueprint_json_example()}"
        )
        try:
            blueprint = self._generate_blueprint_with_retries(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1800,
            )
        except (LLMError, json.JSONDecodeError) as exc:
            LOGGER.warning("Architect fallback triggered: %s", exc)
            blueprint = self._recover_or_retry_blueprint(goal, str(exc), forbidden_domains=forbidden_domains)
        blueprint = self._normalize_blueprint(blueprint, goal=goal)
        LOGGER.info("Architect generated dataset blueprint: %s", blueprint.dataset_name)
        return blueprint

    def _normalize_blueprint(self, blueprint: DatasetBlueprint, *, goal: str) -> DatasetBlueprint:
        """Fill in deterministic schema details the model may omit."""
        simplified_fields = self._simplify_fields(blueprint.row_schema.fields)
        target_field = self._resolve_target_field(
            declared_target=blueprint.row_schema.target_field,
            fields=simplified_fields,
        )
        normalized_fields = self._normalize_ml_fields(
            fields=simplified_fields,
            target_field=target_field,
        )
        declared_fields = [field.name for field in normalized_fields]
        inferred_required = [
            field.name for field in normalized_fields if not field.nullable
        ]

        raw_required = blueprint.row_schema.required or inferred_required
        normalized_required: list[str] = []
        seen: set[str] = set()
        for field_name in raw_required:
            if field_name not in declared_fields or field_name in seen:
                continue
            seen.add(field_name)
            normalized_required.append(field_name)

        normalized_schema = blueprint.row_schema.model_copy(
            update={
                "fields": normalized_fields,
                "required": normalized_required,
                "target_field": target_field,
            }
        )
        normalized_targets = self._normalize_source_targets(blueprint.source_targets, goal=goal)
        normalized_record_count = self._normalize_target_record_count(
            blueprint.target_record_count,
            goal=goal,
        )
        return blueprint.model_copy(
            update={
                "row_schema": normalized_schema,
                "source_targets": normalized_targets,
                "target_record_count": normalized_record_count,
            }
        )

    def _simplify_fields(self, fields: list[SchemaPropertySpec]) -> list[SchemaPropertySpec]:
        """Keep schemas flat and list-friendly so index pages can satisfy them."""
        simplified: list[SchemaPropertySpec] = []
        for field in fields:
            normalized_type = self._normalize_field_type(field)
            if normalized_type not in {"string", "integer", "number", "boolean"}:
                continue
            if self._is_low_signal_field(field):
                continue
            simplified.append(field.model_copy(update={"type": normalized_type}))
            if len(simplified) >= 6:
                break
        return simplified or fields[:6]

    def _resolve_target_field(
        self,
        *,
        declared_target: str | None,
        fields: list[SchemaPropertySpec],
    ) -> str:
        if declared_target and any(field.name == declared_target for field in fields):
            return declared_target

        for field in fields:
            if field.ml_role == "target":
                return field.name

        target_keywords = (
            "target",
            "label",
            "outcome",
            "destination",
            "status",
            "price",
            "score",
            "points",
            "salary",
            "winner",
            "result",
            "default",
            "churn",
            "transfer",
            "admit",
            "conversion",
            "survival",
        )
        for field in fields:
            lowered_name = field.name.lower()
            if any(keyword in lowered_name for keyword in target_keywords):
                return field.name

        return fields[0].name if fields else "target"

    def _normalize_ml_fields(
        self,
        *,
        fields: list[SchemaPropertySpec],
        target_field: str,
    ) -> list[SchemaPropertySpec]:
        normalized: list[SchemaPropertySpec] = []
        feature_count = 0
        for field in fields:
            ml_role = "target" if field.name == target_field else "feature"
            if ml_role == "feature" and feature_count >= 5:
                continue
            updated_field = field.model_copy(
                update={
                    "ml_role": ml_role,
                    "description": self._ensure_ml_description(field, ml_role=ml_role),
                    "type": self._normalize_field_type(field),
                }
            )
            normalized.append(updated_field)
            if ml_role == "feature":
                feature_count += 1

        if not any(field.ml_role == "target" for field in normalized) and normalized:
            normalized[0] = normalized[0].model_copy(
                update={
                    "ml_role": "target",
                    "description": self._ensure_ml_description(normalized[0], ml_role="target"),
                }
            )

        target_records = [field for field in normalized if field.ml_role == "target"]
        feature_records = [field for field in normalized if field.ml_role == "feature"]
        if target_records and len(feature_records) >= 3:
            return target_records[:1] + feature_records[:5]
        return normalized

    def _normalize_field_type(self, field: SchemaPropertySpec) -> str:
        field_type = field.type.lower()
        description = field.description.lower()
        name = field.name.lower()
        if field_type in {"integer", "number", "boolean", "string"}:
            return field_type
        if field_type == "array":
            return "string"
        if any(token in name for token in ("flag", "is_", "has_", "was_", "did_")):
            return "boolean"
        if any(token in description for token in ("yes/no", "binary", "indicator", "flag")):
            return "boolean"
        return "string"

    def _is_low_signal_field(self, field: SchemaPropertySpec) -> bool:
        field_name = field.name.lower()
        segments = [segment for segment in re.split(r"[^a-z0-9]+", field_name) if segment]
        low_signal_segments = {
            "url",
            "website",
            "link",
            "logo",
            "image",
            "photo",
            "thumbnail",
            "color",
            "colour",
            "id",
            "uuid",
            "slug",
            "handle",
            "username",
            "source",
            "reference",
        }
        return bool(low_signal_segments.intersection(segments))

    def _ensure_ml_description(self, field: SchemaPropertySpec, *, ml_role: str) -> str:
        base_description = re.sub(r"\s+", " ", field.description).strip().rstrip(".")
        if not base_description:
            base_description = field.name.replace("_", " ")

        rationale = (
            "This is the supervised learning target that each row must contain so the model has a label to predict."
            if ml_role == "target"
            else "This predictive feature should be extracted because it is likely to carry signal that helps estimate the target variable."
        )
        if any(token in base_description.lower() for token in ("predict", "target", "signal", "model")):
            return f"{base_description}."
        return f"{base_description}. {rationale}"

    def _normalize_source_targets(self, source_targets: list[SourceTarget], *, goal: str) -> list[SourceTarget]:
        blocked_domains = ("sports-reference.com", "espn.com")
        normalized: list[SourceTarget] = []
        seen_domains: set[str] = set()
        allow_duplicate_domains = self._allow_duplicate_domains(goal)
        has_safe_fallback = False
        ranked_candidates = self.source_ranker.rank(goal, [target.url for target in source_targets])
        target_by_url = {target.url: target for target in source_targets}
        for ranked in ranked_candidates:
            original = target_by_url.get(ranked.url, SourceTarget(url=ranked.url, expected_source_type="unknown"))
            url = self._normalize_seed_url(original.url, goal=goal)
            cleaned = url.strip()
            if not cleaned.startswith(("http://", "https://")):
                continue
            if any(domain in cleaned for domain in blocked_domains):
                continue
            if REGISTRY.should_cooldown(cleaned):
                continue
            root_domain = _root_domain(cleaned)
            if root_domain in seen_domains and not allow_duplicate_domains:
                continue
            if not allow_duplicate_domains:
                seen_domains.add(root_domain)
            normalized.append(original.model_copy(update={"url": cleaned}))
            if "wikipedia.org" in root_domain or any(
                token in root_domain for token in ("kaggle.com", "statbunker.com", "fbref.com", "transfermarkt.", "basketball-reference.com")
            ):
                has_safe_fallback = True

        if not has_safe_fallback:
            wikipedia_fallback = f"https://en.wikipedia.org/w/index.php?search={quote_plus(goal + ' list')}"
            normalized.append(SourceTarget(url=wikipedia_fallback, expected_source_type="html_table"))
            seen_domains.add(_root_domain(wikipedia_fallback))
        if len(normalized) < 3:
            for fallback in self.source_ranker.rank(goal, self._deterministic_fallback_urls(goal)):
                root_domain = _root_domain(fallback.url)
                if (root_domain in seen_domains and not allow_duplicate_domains) or REGISTRY.should_cooldown(fallback.url):
                    continue
                normalized.append(SourceTarget(url=fallback.url, expected_source_type="unknown"))
                if not allow_duplicate_domains:
                    seen_domains.add(root_domain)
                if len(normalized) >= 3:
                    break
        return normalized or [SourceTarget(url=f"https://en.wikipedia.org/w/index.php?search={quote_plus(goal + ' list')}", expected_source_type="html_table")]

    def _deterministic_fallback_urls(self, goal: str) -> list[str]:
        return [
            f"https://en.wikipedia.org/w/index.php?search={quote_plus(goal + ' list')}",
            f"https://www.wikidata.org/w/index.php?search={quote_plus(goal)}",
        ]

    def _normalize_seed_url(self, url: str, *, goal: str) -> str:
        """Repair common low-yield URLs into more stable, data-dense alternatives."""
        lowered = url.lower()
        if "nba.com/stats/standings" in lowered:
            season_match = re.search(r"[?&]season=(\d{4})\b", lowered)
            season_year = int(season_match.group(1)) if season_match else self._infer_goal_year(goal)
            if season_year is not None:
                return f"https://www.nba.com/stats/teams/traditional?Season={season_year - 1}-{str(season_year)[-2:]}"
            return "https://www.nba.com/stats/teams/traditional"
        return url

    def _infer_goal_year(self, goal: str) -> int | None:
        matches = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", goal)
        return int(matches[-1]) if matches else None

    def _retry_list_first_blueprint(self, goal: str, *, forbidden_domains: set[str] | None = None) -> DatasetBlueprint:
        """Retry the architect call with stronger instructions for data-dense list pages."""
        forbidden_domains = {domain.lower() for domain in (forbidden_domains or set()) if domain}
        forbidden_domains_text = (
            f"\nYou are strictly forbidden from using these domains: {', '.join(sorted(forbidden_domains))}.\n"
            if forbidden_domains
            else ""
        )
        retry_system_prompt = (
            "You are a Senior Machine Learning Engineer repairing a failed supervised-learning dataset blueprint. "
            "You must respond with a valid JSON object matching the requested structure. "
            "Your retry must prioritize pages that list many rows on one screen. "
            "Find a Wikipedia 'List of...' page or a public database directory related to the goal, plus a few other high-density list pages. "
            "Do not use generic homepages when a list page exists. "
            "When generating source_targets, you MUST provide URLs from at least 3 distinct root domains. Do not provide multiple URLs from the same website. Always include at least one high-density aggregator or Wikipedia link as a safe fallback. "
            "Keep the row schema flat, simple, and list-friendly with exactly one target variable and 3 to 5 predictive features. "
            "Exclude URLs, IDs, and decorative metadata. "
            "Each source target must include expected_source_type so the extraction router can prioritize hidden APIs and hydration state."
        )
        retry_user_prompt = (
            f"Find a Wikipedia 'List of...' page or a public database directory related to: {goal}\n\n"
            "Return:\n"
            "1. A concise dataset name.\n"
            "2. A short dataset description.\n"
            "3. A recommended target_record_count between 25 and 250.\n"
            "4. Four to six source_targets that are high-density list or directory pages.\n"
            "5. A flat row schema with 4 to 6 simple fields that can be extracted from those list pages without opening detail pages.\n\n"
            "The schema must have exactly one target field and 3 to 5 predictive features, with descriptions that explain the modeling value of each field.\n"
            "Generate 3 NEW source targets. "
            f"{forbidden_domains_text}\n"
            "Return JSON in exactly this shape:\n"
            f"{self._blueprint_json_example()}"
        )
        try:
            return self._generate_blueprint_with_retries(
                system_prompt=retry_system_prompt,
                user_prompt=retry_user_prompt,
                max_tokens=1400,
            )
        except (LLMError, json.JSONDecodeError) as exc:
            LOGGER.warning("Architect retry fallback failed: %s", exc)
            return self._deterministic_blueprint_from_goal(goal)

    def _generate_blueprint_with_retries(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> DatasetBlueprint:
        """Call the LLM in json_object mode and locally parse the blueprint with retries."""
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                gateway = self._get_llm_gateway()
                try:
                    raw_response = gateway.complete_json_object(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=max_tokens,
                    )
                except LLMError as exc:
                    if "status 400" not in str(exc).lower():
                        raise
                    LOGGER.warning(
                        "Architect json_object generation failed on attempt %d/3; retrying with plain text JSON prompt: %s",
                        attempt,
                        exc,
                    )
                    raw_response = gateway.complete_text(
                        system_prompt=system_prompt,
                        user_prompt=(
                            f"{user_prompt}\n\n"
                            "Return JSON only. Do not use markdown fences. Do not add commentary before or after the JSON object."
                        ),
                        max_tokens=max_tokens,
                    )
                return self._parse_blueprint_response(raw_response)
            except json.JSONDecodeError as exc:
                last_error = exc
                LOGGER.warning("Architect JSON parse failed on attempt %d/3: %s", attempt, exc)
            except LLMError as exc:
                last_error = exc
                LOGGER.warning("Architect generation failed on attempt %d/3: %s", attempt, exc)
                if "status 400" not in str(exc).lower() and attempt >= 1:
                    break

        if last_error is None:
            raise LLMError("Architect generation failed without an explicit error")
        raise last_error

    def _parse_blueprint_response(self, raw_response: str) -> DatasetBlueprint:
        """Strip markdown noise, parse JSON, and validate the resulting blueprint locally."""
        cleaned_response = self._strip_markdown_json_fences(raw_response)
        data = json.loads(cleaned_response)
        try:
            return DatasetBlueprint.model_validate(data)
        except Exception:
            return DatasetBlueprint.model_validate(self._sanitize_recovered_blueprint(data))

    def _strip_markdown_json_fences(self, raw_response: str) -> str:
        """Remove common Markdown wrappers around model JSON responses."""
        cleaned = raw_response.strip()
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end >= start:
            cleaned = cleaned[start : end + 1]
        return cleaned.strip()

    def _blueprint_json_example(self) -> str:
        """Return a concrete JSON example showing the exact response structure."""
        example = {
            "dataset_name": "College Basketball Transfer Destinations",
            "dataset_description": "Supervised dataset for predicting a player's transfer destination from public roster and season summary pages.",
            "target_record_count": 120,
            "source_targets": [
                {
                    "url": "https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_men%27s_basketball_programs",
                    "expected_source_type": "html_table",
                },
                {
                    "url": "https://www.ncaa.com/stats/basketball-men/d1/current/team/145",
                    "expected_source_type": "html_table",
                },
            ],
            "row_schema": {
                "title": "TransferPredictionRow",
                "description": "One supervised-learning row representing a player or team outcome to model.",
                "target_field": "transfer_destination",
                "fields": [
                    {
                        "name": "transfer_destination",
                        "type": "string",
                        "description": "The player's actual transfer destination. This is the supervised learning target that each row must contain so the model has a label to predict.",
                        "ml_role": "target",
                        "nullable": False,
                        "enum": None,
                        "items_type": None,
                    },
                    {
                        "name": "previous_season_minutes",
                        "type": "number",
                        "description": "Minutes played in the previous season. This predictive feature is extracted because usage often carries strong signal for transfer outcomes.",
                        "ml_role": "feature",
                        "nullable": True,
                        "enum": None,
                        "items_type": None,
                    },
                    {
                        "name": "coach_fired_flag",
                        "type": "boolean",
                        "description": "Whether the player's coach was fired. This predictive feature is extracted because staff instability can influence transfer decisions.",
                        "ml_role": "feature",
                        "nullable": True,
                        "enum": None,
                        "items_type": None,
                    },
                    {
                        "name": "team_win_percentage",
                        "type": "number",
                        "description": "Team win percentage from the prior season. This predictive feature is extracted because team performance can affect transfer behavior.",
                        "ml_role": "feature",
                        "nullable": True,
                        "enum": None,
                        "items_type": None,
                    },
                ],
                "required": [
                    "transfer_destination",
                    "previous_season_minutes",
                    "coach_fired_flag",
                    "team_win_percentage",
                ],
            },
        }
        return json.dumps(example, indent=2)

    def refresh_source_targets(self, goal: str, forbidden_domains: set[str]) -> list[SourceTarget]:
        """Generate a fresh source target set when the initial domains are blocked."""
        blueprint = self._retry_list_first_blueprint(goal, forbidden_domains=forbidden_domains)
        normalized = self._normalize_blueprint(blueprint, goal=goal)
        return normalized.source_targets

    def refresh_starting_urls(self, goal: str, forbidden_domains: set[str]) -> list[str]:
        """Compatibility wrapper for older call sites."""
        return [target.url for target in self.refresh_source_targets(goal, forbidden_domains)]

    def _recover_or_retry_blueprint(
        self,
        goal: str,
        error_message: str,
        *,
        forbidden_domains: set[str] | None = None,
    ) -> DatasetBlueprint:
        """Recover a usable blueprint from provider failed_generation payloads before retrying."""
        recovered = self._recover_blueprint_from_error(error_message)
        if recovered is not None:
            LOGGER.info("Architect recovered blueprint from failed_generation payload")
            return recovered
        return self._retry_list_first_blueprint(goal, forbidden_domains=forbidden_domains)

    def _recover_blueprint_from_error(self, error_message: str) -> DatasetBlueprint | None:
        """Extract and validate provider failed_generation JSON when available."""
        marker = "'failed_generation': "
        marker_index = error_message.find(marker)
        if marker_index == -1:
            return None

        fragment = error_message[marker_index + len(marker) :].strip()
        if fragment.endswith("}}"):
            fragment = fragment[:-1]
        try:
            raw_json = ast.literal_eval(fragment)
        except (SyntaxError, ValueError):
            return None
        if not isinstance(raw_json, str):
            return None
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return None

        try:
            return DatasetBlueprint.model_validate(self._sanitize_recovered_blueprint(data))
        except Exception:
            try:
                return DatasetBlueprint.model_validate(self._sanitize_recovered_blueprint(data))
            except Exception:
                return None

    def _sanitize_recovered_blueprint(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize provider output into the strict schema accepted by DatasetBlueprint."""
        row_schema = data.get("row_schema") if isinstance(data.get("row_schema"), dict) else {}
        raw_fields = row_schema.get("fields") if isinstance(row_schema.get("fields"), list) else []

        cleaned_fields: list[dict[str, Any]] = []
        for field in raw_fields:
            if not isinstance(field, dict):
                continue
            cleaned_fields.append(
                {
                    "name": field.get("name"),
                    "type": field.get("type", "string"),
                    "description": field.get("description", ""),
                    "nullable": bool(field.get("nullable", False)),
                    "enum": field.get("enum"),
                    "items_type": field.get("items_type"),
                    "ml_role": field.get("ml_role") or "feature",
                }
            )

        if not cleaned_fields:
            cleaned_fields = [
                {
                    "name": "outcome_label",
                    "type": "string",
                    "description": "Observed outcome or class label for the entity. This target is required so the dataset remains usable for supervised learning.",
                    "nullable": False,
                    "enum": None,
                    "items_type": None,
                    "ml_role": "target",
                },
                {
                    "name": "entity_category",
                    "type": "string",
                    "description": "Category or segment of the entity. This predictive feature captures structural differences that may explain the target.",
                    "nullable": False,
                    "enum": None,
                    "items_type": None,
                    "ml_role": "feature",
                },
                {
                    "name": "historical_performance",
                    "type": "number",
                    "description": "Historical performance metric visible on the source page. This predictive feature captures prior strength relevant to the target.",
                    "nullable": True,
                    "enum": None,
                    "items_type": None,
                    "ml_role": "feature",
                },
                {
                    "name": "experience_level",
                    "type": "number",
                    "description": "Visible tenure, age, or experience signal. This predictive feature often correlates with future outcomes.",
                    "nullable": True,
                    "enum": None,
                    "items_type": None,
                    "ml_role": "feature",
                },
            ]

        target_field = row_schema.get("target_field")
        if not target_field:
            for field in cleaned_fields:
                if field.get("ml_role") == "target":
                    target_field = field["name"]
                    break

        required = row_schema.get("required") or [
            field["name"] for field in cleaned_fields if field.get("name") and not field.get("nullable", False)
        ]

        raw_source_targets = data.get("source_targets")
        if not isinstance(raw_source_targets, list):
            raw_source_targets = [
                {"url": url, "expected_source_type": "unknown"}
                for url in (data.get("starting_urls") or [])
                if isinstance(url, str)
            ]

        cleaned_targets: list[dict[str, Any]] = []
        allowed_source_types = {"html_table", "json_api", "react_state", "browser_heavy", "unknown"}
        for target in raw_source_targets:
            if isinstance(target, str):
                cleaned_targets.append({"url": target, "expected_source_type": "unknown"})
                continue
            if not isinstance(target, dict):
                continue
            expected_source_type = str(target.get("expected_source_type") or "unknown")
            if expected_source_type not in allowed_source_types:
                expected_source_type = "unknown"
            cleaned_targets.append(
                {
                    "url": target.get("url"),
                    "expected_source_type": expected_source_type,
                }
            )

        return {
            "dataset_name": data.get("dataset_name") or "Recovered Dataset",
            "dataset_description": data.get("dataset_description") or "Recovered dataset blueprint from provider output.",
            "target_record_count": int(data.get("target_record_count") or 50),
            "source_targets": cleaned_targets,
            "row_schema": {
                "title": row_schema.get("title") or "RecoveredDatasetRow",
                "description": row_schema.get("description") or "Recovered row schema.",
                "target_field": target_field,
                "fields": cleaned_fields,
                "required": required,
            },
        }

    def _generic_directory_blueprint(self, goal: str) -> DatasetBlueprint:
        """Build a generic list-friendly blueprint when all LLM attempts fail."""
        adapter_urls = adapter_urls_for_goal(goal)
        if adapter_urls:
            ranked = self.source_ranker.rank(goal, adapter_urls)[:3]
            source_targets = [
                SourceTarget(url=item.url, expected_source_type="html_table")
                for item in ranked
            ]
        else:
            source_targets = [
                SourceTarget(
                    url=f"https://en.wikipedia.org/w/index.php?search={quote_plus(goal + ' list')}",
                    expected_source_type="html_table",
                ),
            ]
        return DatasetBlueprint(
            dataset_name=_sanitize_dataset_name(goal),
            dataset_description=f"General-purpose directory dataset for: {goal}",
            target_record_count=50,
            source_targets=source_targets,
            row_schema=GeneratedRowSchema(
                title="SupervisedLearningRecord",
                description="Flat supervised-learning record extracted from a public directory, list, or table.",
                target_field="outcome_label",
                fields=[
                    SchemaPropertySpec(
                        name="outcome_label",
                        type="string",
                        description="Observed outcome or class label to be predicted for each entity. This target is required so the dataset can be used for supervised training.",
                        ml_role="target",
                    ),
                    SchemaPropertySpec(
                        name="entity_category",
                        type="string",
                        description="Category or segment of the entity. This predictive feature can capture structural differences tied to the target.",
                    ),
                    SchemaPropertySpec(
                        name="historical_performance",
                        type="number",
                        description="Historical performance metric visible on the listing page. This predictive feature captures prior strength or momentum relevant to the target.",
                        nullable=True,
                    ),
                    SchemaPropertySpec(
                        name="experience_level",
                        type="number",
                        description="Visible tenure, age, or experience signal. This predictive feature often correlates with future outcomes.",
                        nullable=True,
                    ),
                    SchemaPropertySpec(
                        name="region",
                        type="string",
                        description="Location or region associated with the entity. This predictive feature may capture environmental or market effects relevant to the target.",
                        nullable=True,
                    ),
                ],
                required=["outcome_label", "entity_category"],
            ),
        )

    def _get_llm_gateway(self) -> LLMGateway:
        if self.llm_gateway is None:
            self.llm_gateway = LLMGateway(max_tokens=1800)
        return self.llm_gateway

    def _allow_duplicate_domains(self, goal: str) -> bool:
        lowered = goal.lower()
        return "state" in lowered and "population" in lowered

    def _normalize_target_record_count(self, proposed_count: int, *, goal: str) -> int:
        cardinality = infer_goal_cardinality(goal)
        if cardinality is not None:
            return cardinality.count
        return max(25, min(int(proposed_count or 50), 1000))

    def _should_use_deterministic_blueprint(self, goal: str) -> bool:
        lowered = goal.lower()
        return (
            (
                "ncaa" in lowered
                and "basketball" in lowered
                and any(token in lowered for token in ("team statistics", "programs", "teams"))
            )
            or (
                "state" in lowered
                and "population" in lowered
                and any(token in lowered for token in ("growth", "gdp", "economic"))
            )
        )

    def _deterministic_blueprint_from_goal(self, goal: str) -> DatasetBlueprint:
        """Produce a usable blueprint for common list-oriented goals without an LLM."""
        lowered = goal.lower()

        if "ncaa" in lowered and "basketball" in lowered:
            return DatasetBlueprint(
                dataset_name="NCAA Men S Basketball Programs",
                dataset_description="Directory of NCAA Division I men's basketball programs and core public metadata.",
                target_record_count=365,
                source_targets=[
                    SourceTarget(
                        url="https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_men%27s_basketball_programs",
                        expected_source_type="html_table",
                    ),
                ],
                row_schema=GeneratedRowSchema(
                    title="NCAABasketballProgram",
                    description="One NCAA men's basketball program represented as a supervised-learning row.",
                    target_field="tournament_appearances",
                    fields=[
                        SchemaPropertySpec(
                            name="tournament_appearances",
                            type="string",
                            description="Visible NCAA tournament appearance summary to serve as the target outcome for each program.",
                            ml_role="target",
                        ),
                        SchemaPropertySpec(
                            name="school",
                            type="string",
                            description="School or university identity. This predictive feature distinguishes programs whose historical outcomes may differ systematically.",
                        ),
                        SchemaPropertySpec(
                            name="conference",
                            type="string",
                            description="Athletic conference membership. This predictive feature proxies for schedule strength and competitive context.",
                            nullable=True,
                        ),
                        SchemaPropertySpec(
                            name="home_arena_capacity",
                            type="number",
                            description="Home arena capacity when visible. This predictive feature approximates program resources and fan support.",
                            nullable=True,
                        ),
                        SchemaPropertySpec(
                            name="program_age_years",
                            type="number",
                            description="Program age or years in operation when visible. This predictive feature captures institutional stability and legacy effects.",
                            nullable=True,
                        ),
                    ],
                    required=["tournament_appearances", "school"],
                ),
            )

        if "state" in lowered and "population" in lowered:
            state_urls = adapter_urls_for_goal(goal) or [
                "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population_growth_rate",
                "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_economic_growth_rate",
                "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_population",
                "https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_GDP",
            ]
            return DatasetBlueprint(
                dataset_name="US State Population Growth",
                dataset_description="Dataset of U.S. states with population growth and economic features.",
                target_record_count=50,
                source_targets=[
                    SourceTarget(url=url, expected_source_type="html_table")
                    for url in state_urls[:4]
                ],
                row_schema=GeneratedRowSchema(
                    title="USStatePopulationGrowthRow",
                    description="One U.S. state represented as a supervised-learning row.",
                    target_field="population_growth_rate",
                    fields=[
                        SchemaPropertySpec(
                            name="population_growth_rate",
                            type="number",
                            description="Population growth rate for the state. This is the supervised learning target.",
                            ml_role="target",
                        ),
                        SchemaPropertySpec(
                            name="state",
                            type="string",
                            description="State name. This predictive feature distinguishes geographic units.",
                        ),
                        SchemaPropertySpec(
                            name="population",
                            type="number",
                            description="Current state population. This predictive feature captures scale effects.",
                            nullable=True,
                        ),
                        SchemaPropertySpec(
                            name="gdp",
                            type="number",
                            description="State GDP when visible. This predictive feature captures economic scale.",
                            nullable=True,
                        ),
                        SchemaPropertySpec(
                            name="gdp_growth_rate",
                            type="number",
                            description="State economic growth rate when visible. This predictive feature captures momentum related to population change.",
                            nullable=True,
                        ),
                    ],
                    required=["population_growth_rate", "state"],
                ),
            )

        return self._generic_directory_blueprint(goal)


def schema_to_json_schema(schema: GeneratedRowSchema) -> dict[str, Any]:
    """Convert the constrained row schema into a standard JSON Schema dict."""
    properties: dict[str, Any] = {}
    for spec in schema.fields:
        field_name = spec.name
        property_schema: dict[str, Any] = {
            "description": spec.description,
            "x-ml-role": spec.ml_role,
        }
        if spec.enum:
            property_schema["enum"] = spec.enum
            property_schema["type"] = "string"
        elif spec.type == "array":
            property_schema["type"] = "array"
            property_schema["items"] = {"type": spec.items_type or "string"}
        else:
            property_schema["type"] = spec.type

        if spec.nullable:
            property_schema = {"anyOf": [property_schema, {"type": "null"}]}

        properties[field_name] = property_schema

    return {
        "title": schema.title,
        "description": schema.description,
        "type": "object",
        "properties": properties,
        "required": schema.required,
        "additionalProperties": False,
        "x-target-field": schema.target_field,
    }


def json_schema_to_pydantic_model(
    json_schema: dict[str, Any],
    model_name: str | None = None,
) -> type[BaseModel]:
    """Convert a standard JSON Schema object into a runtime Pydantic model."""
    schema_title = model_name or _sanitize_model_name(json_schema.get("title", "DynamicDatasetRow"))
    required_fields = set(json_schema.get("required", []))
    model_fields: dict[str, tuple[Any, Any]] = {}

    for field_name, property_schema in json_schema.get("properties", {}).items():
        annotation, default = _schema_fragment_to_annotation(
            property_schema,
            field_name=field_name,
            model_name=schema_title,
            required=field_name in required_fields,
        )
        model_fields[field_name] = (annotation, default)

    return create_model(schema_title, __base__=DynamicRowBase, **model_fields)


def relax_pydantic_model(
    row_model: type[BaseModel],
    *,
    model_name: str | None = None,
) -> type[BaseModel]:
    """Create a scrape-time variant of a strict row model with all fields optional.

    The swarm often needs to collect partial records across multiple pages before the
    synthesizer can merge them into training-ready rows. Field types and schema
    metadata are preserved, but requiredness is relaxed so stage 2 does not discard
    useful partial evidence prematurely.
    """
    relaxed_name = model_name or f"{row_model.__name__}Partial"
    relaxed_fields: dict[str, tuple[Any, Any]] = {}
    for field_name, model_field in row_model.model_fields.items():
        annotation = model_field.annotation
        try:
            relaxed_annotation = annotation | None
        except TypeError:
            relaxed_annotation = annotation

        relaxed_fields[field_name] = (
            relaxed_annotation,
            Field(
                default=None,
                description=model_field.description or "",
                json_schema_extra=model_field.json_schema_extra,
            ),
        )

    return create_model(relaxed_name, __base__=DynamicRowBase, **relaxed_fields)


def _schema_fragment_to_annotation(
    schema: dict[str, Any],
    *,
    field_name: str,
    model_name: str,
    required: bool,
) -> tuple[Any, Field]:
    """Convert a JSON Schema field fragment into a Pydantic field annotation."""
    nullable = False
    working_schema = dict(schema)

    if "anyOf" in working_schema:
        variants = working_schema["anyOf"]
        non_null_variants = [variant for variant in variants if variant.get("type") != "null"]
        nullable = len(non_null_variants) != len(variants)
        working_schema = non_null_variants[0] if non_null_variants else {"type": "string"}

    annotation = _json_type_to_python_type(
        working_schema,
        field_name=field_name,
        model_name=model_name,
    )
    if nullable or not required:
        annotation = annotation | None

    default = ... if required and not nullable else None
    return (
        annotation,
        Field(
            default=default,
            description=working_schema.get("description", ""),
            json_schema_extra={
                "x-ml-role": working_schema.get("x-ml-role"),
                "x-target-field": working_schema.get("x-target-field"),
            },
        ),
    )


def _json_type_to_python_type(
    schema: dict[str, Any],
    *,
    field_name: str,
    model_name: str,
) -> Any:
    """Map JSON Schema primitives into Python/Pydantic annotations."""
    schema_type = schema.get("type", "string")

    if schema.get("enum"):
        return str
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        item_schema = schema.get("items", {"type": "string"})
        item_type = _json_type_to_python_type(
            item_schema,
            field_name=f"{field_name}_item",
            model_name=model_name,
        )
        return list[item_type]
    if schema_type == "object":
        nested_name = _sanitize_model_name(f"{model_name}_{field_name}")
        return json_schema_to_pydantic_model(
            {
                "title": nested_name,
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
                "additionalProperties": False,
            },
            model_name=nested_name,
        )
    return str


def _sanitize_model_name(value: str) -> str:
    """Normalize an arbitrary string into a valid model class name."""
    sanitized = re.sub(r"[^0-9a-zA-Z]+", " ", value).title().replace(" ", "")
    return sanitized or "DynamicDatasetRow"


def _sanitize_dataset_name(value: str) -> str:
    """Normalize an arbitrary goal into a readable dataset name."""
    cleaned = re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z]+", " ", value)).strip()
    return cleaned.title() or "General Directory Dataset"


def _root_domain(url: str) -> str:
    """Extract a coarse root domain for diversity and blacklist checks."""
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    if not hostname:
        return ""
    parts = hostname.split(".")
    if len(parts) <= 2:
        return hostname
    if parts[-2] in {"co", "com", "org", "gov", "ac"} and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])
