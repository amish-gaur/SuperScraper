"""Dataset architecture stage for dynamic schema generation."""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from llm import LLMGateway, StructuredEnvelope


LOGGER = logging.getLogger(__name__)


class SchemaPropertySpec(StructuredEnvelope):
    """A constrained JSON Schema property description for a dataset row."""

    name: str = Field(description="Column name in snake_case.")
    type: str = Field(description='JSON Schema type: "string", "integer", "number", "boolean", or "array".')
    description: str = Field(description="Description of the field.")
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
    fields: list[SchemaPropertySpec] = Field(
        description="Schema fields for the dataset row."
    )
    required: list[str] = Field(description="Required property names.")


class DatasetBlueprint(StructuredEnvelope):
    """Blueprint describing the dataset to collect."""

    dataset_name: str = Field(description="Human-readable dataset name.")
    dataset_description: str = Field(description="What the dataset is intended to model.")
    target_record_count: int = Field(description="Recommended number of rows to gather.")
    search_strategies: list[str] = Field(
        description="Diverse search angles for the scraping swarm."
    )
    row_schema: GeneratedRowSchema = Field(description="JSON Schema for a single dataset row.")


class DynamicRowBase(BaseModel):
    """Base model for dynamically generated row schemas."""

    model_config = ConfigDict(extra="forbid")


class DatasetArchitect:
    """Use an LLM to design a dynamic dataset schema from a high-level ML goal."""

    def __init__(self, llm_gateway: LLMGateway | None = None) -> None:
        self.llm_gateway = llm_gateway or LLMGateway(max_tokens=1800)

    def design(self, goal: str) -> DatasetBlueprint:
        """Generate a dataset blueprint and row schema for the requested goal."""
        system_prompt = (
            "You are a senior data architect designing machine-learning datasets. "
            "Given a high-level ML goal, produce a pragmatic row schema that can be collected from the public web. "
            "Favor columns that are observable, relevant to prediction, and likely to appear in articles, rosters, stats pages, or press releases. "
            "Search strategies should cover different evidence sources and subproblems."
        )
        user_prompt = (
            f"Machine learning goal:\n{goal}\n\n"
            "Return:\n"
            "1. A concise dataset name.\n"
            "2. A short dataset description.\n"
            "3. A recommended target_record_count between 25 and 250.\n"
            "4. Three to five search strategies.\n"
            "5. A JSON Schema for a single dataset row with simple JSON types only."
        )
        blueprint = self.llm_gateway.complete_structured(
            response_model=DatasetBlueprint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="dataset_blueprint",
            max_tokens=1800,
        )
        LOGGER.info("Architect generated dataset blueprint: %s", blueprint.dataset_name)
        return blueprint


def schema_to_json_schema(schema: GeneratedRowSchema) -> dict[str, Any]:
    """Convert the constrained row schema into a standard JSON Schema dict."""
    properties: dict[str, Any] = {}
    for spec in schema.fields:
        field_name = spec.name
        property_schema: dict[str, Any] = {"description": spec.description}
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
