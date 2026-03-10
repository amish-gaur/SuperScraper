"""Shared LLM utilities for the autonomous data engineering pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any, TypeVar

from openai import APIStatusError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model


LOGGER = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM request fails or returns invalid structured data."""


class StructuredEnvelope(BaseModel):
    """Base configuration for strict structured outputs."""

    model_config = ConfigDict(extra="forbid")


class LLMGateway:
    """Thin wrapper around the OpenAI-compatible SDK with structured output helpers."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1600,
    ) -> None:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        resolved_base_url = os.getenv("OPENAI_BASE_URL")
        default_headers: dict[str, str] | None = None

        if resolved_api_key and resolved_api_key.startswith("sk-or-v1") and not resolved_base_url:
            resolved_base_url = "https://openrouter.ai/api/v1"
            default_headers = {
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "Autonomous Data Engineer"),
            }

        default_model = "openai/gpt-4.1-mini" if resolved_base_url else "gpt-4.1-mini"
        self.model = model or os.getenv("OPENAI_MODEL", default_model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            default_headers=default_headers,
        )

    def complete_structured(
        self,
        *,
        response_model: type[T],
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        max_tokens: int | None = None,
    ) -> T:
        """Request a strict structured response and validate it with Pydantic."""
        schema = response_model.model_json_schema()
        _normalize_json_schema(schema)
        LOGGER.debug("Requesting structured output with schema %s", schema_name)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except APIStatusError as exc:
            raise LLMError(f"LLM request failed with status {exc.status_code}: {exc.message}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("LLM returned an empty response")

        try:
            return response_model.model_validate_json(content)
        except ValidationError as exc:
            LOGGER.error("Structured response validation failed: %s", content)
            raise LLMError(f"Structured response validation failed: {exc}") from exc


def build_record_list_model(row_model: type[BaseModel], model_name: str) -> type[BaseModel]:
    """Create a wrapper model containing a list of runtime row models."""
    return create_model(
        model_name,
        __base__=StructuredEnvelope,
        records=(list[row_model], Field(description="Normalized records matching the row schema.")),
    )


def _normalize_json_schema(schema: dict[str, Any]) -> None:
    """Normalize JSON schema for strict providers."""
    if "$ref" in schema:
        ref_value = schema["$ref"]
        schema.clear()
        schema["$ref"] = ref_value
        return

    if schema.get("type") == "object" and "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
        schema.setdefault("additionalProperties", False)

    for key in ("properties", "$defs"):
        for child in schema.get(key, {}).values():
            if isinstance(child, dict):
                _normalize_json_schema(child)

    items = schema.get("items")
    if isinstance(items, dict):
        _normalize_json_schema(items)

    for key in ("anyOf", "allOf", "oneOf"):
        for child in schema.get(key, []):
            if isinstance(child, dict):
                _normalize_json_schema(child)
