"""Shared LLM utilities for the autonomous data engineering pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal, TypeVar

from env_utils import API_KEY_ENV_VARS, env_value_is_usable
from openai import APIStatusError, OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from settings import get_settings
from tracing import instrument_openai_client


LOGGER = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
ActionType = Literal["click", "type", "open_url", "wait", "scroll_down", "scroll_up", "none"]


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
        resolved_api_key = api_key or _first_configured_api_key()
        resolved_base_url = os.getenv("OPENAI_BASE_URL")
        default_headers: dict[str, str] | None = None

        if not resolved_api_key:
            raise LLMError(
                "No API key configured. Set OPENAI_API_KEY, or set GEMINI_API_KEY for Gemini's OpenAI-compatible API."
            )

        if resolved_api_key.startswith("AI") and not resolved_base_url:
            resolved_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif resolved_api_key.startswith("gsk_") and not resolved_base_url:
            resolved_base_url = "https://api.groq.com/openai/v1"
        elif resolved_api_key.startswith("sk-or-v1") and not resolved_base_url:
            resolved_base_url = "https://openrouter.ai/api/v1"
            default_headers = {
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "Autonomous Data Engineer"),
            }

        if "generativelanguage.googleapis.com" in (resolved_base_url or ""):
            default_model = "gemini-2.0-flash-lite"
        elif "api.groq.com/openai/v1" in (resolved_base_url or ""):
            default_model = "openai/gpt-oss-20b"
        else:
            default_model = "openai/gpt-4.1-mini" if resolved_base_url else "gpt-4.1-mini"
        self.model = model or os.getenv("OPENAI_MODEL", default_model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            client = OpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
                default_headers=default_headers,
            )
            self.client = instrument_openai_client(client, get_settings())
        except OpenAIError as exc:
            raise LLMError(f"Failed to initialize LLM client: {exc}") from exc

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

        if self._prefer_prompt_only_json():
            content = self.complete_text(
                system_prompt=(
                    f"{system_prompt} "
                    "Return a single JSON object only. Do not call tools. Do not emit markdown."
                ),
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            return _validate_or_recover_structured_response(
                content=_extract_json_candidate(content),
                response_model=response_model,
            )

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
            raise LLMError(_format_api_error(exc, model=self.model)) from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("LLM returned an empty response")

        return _validate_or_recover_structured_response(
            content=content,
            response_model=response_model,
        )

    def complete_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Request an unconstrained text response for local post-processing."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                tool_choice="none",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except APIStatusError as exc:
            raise LLMError(_format_api_error(exc, model=self.model)) from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("LLM returned an empty response")
        return content

    def complete_json_object(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Request a JSON object response without provider-side schema enforcement."""
        if self._prefer_prompt_only_json():
            return self.complete_text(
                system_prompt=(
                    f"{system_prompt} "
                    "Return a single JSON object only. Do not call tools. Do not emit markdown."
                ),
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except APIStatusError as exc:
            raise LLMError(_format_api_error(exc, model=self.model)) from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("LLM returned an empty response")
        return content

    def _prefer_prompt_only_json(self) -> bool:
        lowered_model = self.model.lower()
        lowered_base_url = str(getattr(self.client, "base_url", "")).lower()
        return "gpt-oss" in lowered_model or "api.groq.com" in lowered_base_url


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


def _format_api_error(exc: APIStatusError, *, model: str) -> str:
    """Convert provider-specific API failures into short actionable messages."""
    base_message = str(exc.message).strip()

    if exc.status_code == 402:
        return (
            f"LLM billing error for model '{model}': the provider reported insufficient credits. "
            "Add credits or switch to a different API key/provider."
        )

    if exc.status_code == 429:
        lowered = base_message.lower()
        if "quota" in lowered or "resource_exhausted" in lowered:
            return (
                f"LLM quota exhausted for model '{model}'. The provider rate limit or free-tier quota is depleted. "
                "Wait for quota reset, reduce request volume, or use another key/provider."
            )
        return (
            f"LLM rate limit reached for model '{model}'. Retry after a short delay or use another provider."
        )

    if exc.status_code in {401, 403}:
        return (
            f"LLM authentication failed for model '{model}'. Check that the API key is valid and allowed to use this endpoint."
        )

    if exc.status_code == 400 and "api key expired" in base_message.lower():
        return "LLM authentication failed: the provider says this API key has expired. Generate a fresh key and try again."

    return f"LLM request failed with status {exc.status_code} for model '{model}': {base_message}"


def _first_configured_api_key() -> str | None:
    for env_var in API_KEY_ENV_VARS:
        value = os.getenv(env_var)
        if not value:
            continue
        if not env_value_is_usable(value, key=env_var):
            continue
        return value.strip()
    return None


def _extract_json_payload(value: str) -> str:
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError("LLM response did not contain a JSON object")
    return value[start : end + 1]


def _validate_or_recover_structured_response(
    *,
    content: str,
    response_model: type[T],
) -> T:
    try:
        return response_model.model_validate_json(content)
    except ValidationError as exc:
        recovered_payload = _recover_structured_payload(content, response_model)
        if recovered_payload is None:
            LOGGER.error("Structured response validation failed: %s", content)
            raise LLMError(f"Structured response validation failed: {exc}") from exc
        try:
            return response_model.model_validate(recovered_payload)
        except ValidationError as recovered_exc:
            LOGGER.error("Structured response validation failed: %s", content)
            raise LLMError(
                f"Structured response validation failed: {recovered_exc}"
            ) from recovered_exc


def _recover_structured_payload(
    content: str,
    response_model: type[BaseModel],
) -> Any | None:
    parsed = _parse_json_candidate(content)
    if parsed is None:
        return None
    return _coerce_payload_to_model(parsed, response_model)


def _parse_json_candidate(content: str) -> Any | None:
    candidates = [_strip_control_characters(content).strip()]
    extracted = _extract_json_candidate(content)
    if extracted not in candidates:
        candidates.append(extracted)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _coerce_payload_to_model(parsed: Any, response_model: type[BaseModel]) -> Any:
    field_names = set(response_model.model_fields)

    if "records" in field_names:
        if isinstance(parsed, list):
            return {"records": parsed}
        if isinstance(parsed, dict):
            if "records" in parsed:
                return parsed
            for key in ("canonical_records", "extracted_records", "rows", "items", "data"):
                value = parsed.get(key)
                if isinstance(value, list):
                    return {"records": value}
            for value in parsed.values():
                if isinstance(value, list):
                    return {"records": value}

    if isinstance(parsed, dict):
        normalized = dict(parsed)
        if "inferred_target_column" in field_names and "inferred_target_column" not in normalized:
            if isinstance(normalized.get("target"), str):
                normalized["inferred_target_column"] = normalized["target"]
        if "ml_task_type" in field_names and "ml_task_type" not in normalized:
            if isinstance(normalized.get("task_type"), str):
                normalized["ml_task_type"] = normalized["task_type"]
        return {key: value for key, value in normalized.items() if key in field_names}

    return parsed


def _extract_json_candidate(value: str) -> str:
    cleaned = _strip_control_characters(value).strip()
    object_start = cleaned.find("{")
    object_end = cleaned.rfind("}")
    array_start = cleaned.find("[")
    array_end = cleaned.rfind("]")

    object_candidate = (
        cleaned[object_start : object_end + 1]
        if object_start != -1 and object_end != -1 and object_end > object_start
        else ""
    )
    array_candidate = (
        cleaned[array_start : array_end + 1]
        if array_start != -1 and array_end != -1 and array_end > array_start
        else ""
    )

    if array_candidate and (
        not object_candidate or array_start < object_start
    ):
        return array_candidate
    if object_candidate:
        return object_candidate
    return cleaned


def _strip_control_characters(value: str) -> str:
    return "".join(character for character in value if character in ("\n", "\r", "\t") or ord(character) >= 32)
