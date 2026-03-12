"""Data synthesis stage for entity and conflict resolution."""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import BaseModel, ValidationError

from llm import LLMError, LLMGateway, build_record_list_model
from text_cleaner import TextCleaningUtility


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DataSynthesizer:
    """Use an LLM to merge duplicates and normalize record values."""

    row_model: type[BaseModel]
    llm_gateway: LLMGateway | None = None
    batch_size: int = 20
    llm_merge_record_limit: int = 40

    def synthesize(self, goal: str, raw_records: Sequence[BaseModel]) -> list[dict[str, Any]]:
        """Normalize and deduplicate raw swarm records."""
        if not raw_records:
            return []

        if not self._should_use_llm_merge(raw_records):
            LOGGER.info("Using deterministic dedupe for synthesis")
            merged_payloads = self._merge_records_deterministically(raw_records)
            filtered_payloads = self._filter_payloads_with_target(merged_payloads)
            validated_records: list[BaseModel] = []
            for payload in filtered_payloads:
                try:
                    validated_records.append(
                        self.row_model.model_validate(self._clean_payload_for_schema(payload))
                    )
                except ValidationError:
                    continue
            return [record.model_dump(mode="json") for record in validated_records]

        LOGGER.info("Synthesizing %d raw records", len(raw_records))
        canonical: list[BaseModel] = []
        for start in range(0, len(raw_records), self.batch_size):
            batch = raw_records[start : start + self.batch_size]
            canonical = self._merge_batch(goal, canonical, batch)

        final_records = self._merge_batch(goal, [], canonical)
        filtered_records = self._filter_records_with_target(final_records)
        LOGGER.info("Synthesizer produced %d canonical records and retained %d rows with target labels", len(final_records), len(filtered_records))
        return [record.model_dump(mode="json") for record in filtered_records]

    def synthesize_state_payload(
        self,
        *,
        goal: str,
        source_url: str,
        state_payload: dict[str, Any],
        strategy: str,
    ) -> list[BaseModel]:
        """Extract schema-aligned records from a raw JSON state payload."""
        if not state_payload:
            return []
        if not self._llm_available():
            LOGGER.info("No LLM gateway available; skipping %s synthesis for %s", strategy, source_url)
            return []

        response_model = build_record_list_model(self.row_model, "StatePayloadRecordBatch")
        system_prompt = (
            "You are a data extraction engine operating on raw JSON from a webpage's hidden API or frontend hydration state. "
            "Identify repeated entities inside the payload and emit rows that match the target schema exactly. "
            "Use only values explicitly grounded in the provided JSON. "
            "Prefer high-density arrays or nested collections over singleton metadata. "
            "If candidate_collections are provided, treat them as the strongest hints about where row-like entities live. "
            "If a field is unavailable, leave it null instead of inventing values. "
            "If source_url or reference_url exists in the schema, populate it with the provided source URL."
        )
        user_prompt = (
            f"Dataset goal:\n{goal}\n\n"
            f"Source URL:\n{source_url}\n\n"
            f"Routing strategy:\n{strategy}\n\n"
            f"Target row schema:\n{json.dumps(self.row_model.model_json_schema(), indent=2, sort_keys=True)}\n\n"
            f"JSON payload:\n{self._truncate_json_for_prompt(state_payload)}\n\n"
            "Return the extracted record list."
        )
        gateway = self._get_llm_gateway()
        response = gateway.complete_structured(
            response_model=response_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="state_payload_record_batch",
            max_tokens=2200,
        )
        return response.records

    def synthesize_document_text(
        self,
        *,
        goal: str,
        source_url: str,
        document_text: str,
        strategy: str,
    ) -> list[BaseModel]:
        """Extract schema-aligned rows from visible page text when structure-specific paths fail."""
        prepared_text = self._prepare_document_text_for_prompt(document_text)
        if not prepared_text:
            return []
        if not self._llm_available():
            LOGGER.info("No LLM gateway available; skipping %s synthesis for %s", strategy, source_url)
            return []

        response_model = build_record_list_model(self.row_model, "DocumentTextRecordBatch")
        system_prompt = (
            "You are a data extraction engine operating on webpage text captured from a browser or raw HTML. "
            "Infer repeated row-like entities from semi-structured text such as tables, cards, directories, rankings, or lists. "
            "Use only values explicitly grounded in the provided text. "
            "Prefer dense repeated patterns over page chrome, ads, or navigation text. "
            "If a field is unavailable, leave it null instead of inventing values. "
            "If source_url or reference_url exists in the schema, populate it with the provided source URL when no better URL is visible."
        )
        user_prompt = (
            f"Dataset goal:\n{goal}\n\n"
            f"Source URL:\n{source_url}\n\n"
            f"Routing strategy:\n{strategy}\n\n"
            f"Target row schema:\n{json.dumps(self.row_model.model_json_schema(), indent=2, sort_keys=True)}\n\n"
            f"Document text:\n{prepared_text}\n\n"
            "Return the extracted record list."
        )
        gateway = self._get_llm_gateway()
        response = gateway.complete_structured(
            response_model=response_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="document_text_record_batch",
            max_tokens=2200,
        )
        return response.records

    def _merge_batch(
        self,
        goal: str,
        canonical_records: Sequence[BaseModel],
        incoming_records: Sequence[BaseModel],
    ) -> list[BaseModel]:
        """Merge a batch of records into the canonical set."""
        response_model = build_record_list_model(self.row_model, "SynthesizedRecordBatch")
        system_prompt = (
            "You are a data synthesis engine performing entity resolution and conflict resolution. "
            "Merge duplicate entities, choose the most specific and plausible values, and coerce values into the schema's target types. "
            "If two records are clearly the same entity, output only one merged row. "
            "Prefer non-null values and the most recent or detailed statement when conflicts appear. "
            "The schema marks one field as the supervised learning target. Rows missing that target, rows with empty target values, and rows whose target is 'Not Disclosed' are unusable for training and should be removed from the output."
        )
        user_prompt = (
            f"Dataset goal:\n{goal}\n\n"
            f"Canonical records so far:\n{json.dumps([row.model_dump(mode='json') for row in canonical_records], indent=2, sort_keys=True)}\n\n"
            f"Incoming batch:\n{json.dumps([row.model_dump(mode='json') for row in incoming_records], indent=2, sort_keys=True)}\n\n"
            f"Target row schema:\n{json.dumps(self.row_model.model_json_schema(), indent=2, sort_keys=True)}\n\n"
            "Return the merged canonical record list."
        )
        try:
            gateway = self._get_llm_gateway()
            response = gateway.complete_structured(
                response_model=response_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name="synthesized_record_batch",
                max_tokens=2200,
            )
        except LLMError as exc:
            LOGGER.warning("Synthesizer fallback triggered: %s", exc)
            return list(canonical_records) + [
                row for row in incoming_records if row not in canonical_records
            ]

        return response.records

    def _filter_records_with_target(self, records: Sequence[BaseModel]) -> list[BaseModel]:
        """Drop records that do not contain a usable supervised-learning target."""
        target_field = self._identify_target_field()
        if target_field is None:
            LOGGER.warning("No target field metadata found in row schema; returning records without label filtering")
            return list(records)

        missing_target_records: list[BaseModel] = []
        kept_records: list[BaseModel] = []
        for record in records:
            value = getattr(record, target_field, None)
            if self._is_missing_target_value(value):
                missing_target_records.append(record)
                continue
            kept_records.append(record)

        if missing_target_records and not kept_records:
            LOGGER.warning(
                "All %d records are missing target '%s'; keeping deduped records instead of returning 0 rows",
                len(missing_target_records),
                target_field,
            )
            return list(records)

        if missing_target_records:
            LOGGER.warning(
                "Dropping %d records because they are missing the target variable: '%s'",
                len(missing_target_records),
                target_field,
            )
        return kept_records

    def _identify_target_field(self) -> str | None:
        for field_name, model_field in self.row_model.model_fields.items():
            extra = model_field.json_schema_extra or {}
            if extra.get("x-ml-role") == "target":
                return field_name

        schema = self.row_model.model_json_schema()
        for field_name, property_schema in schema.get("properties", {}).items():
            if property_schema.get("x-ml-role") == "target":
                return field_name

        target_keywords = (
            "target",
            "label",
            "outcome",
            "destination",
            "status",
            "result",
            "winner",
            "transfer",
            "price",
            "salary",
            "churn",
            "default",
        )
        for field_name in self.row_model.model_fields:
            if any(keyword in field_name.lower() for keyword in target_keywords):
                return field_name
        return None

    def _is_missing_target_value(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return True
            if normalized.casefold() in {"not disclosed", "n/a", "na", "none", "null", "unknown", "undisclosed"}:
                return True
        return False

    def _get_llm_gateway(self) -> LLMGateway:
        if self.llm_gateway is None:
            self.llm_gateway = LLMGateway(max_tokens=2000)
        return self.llm_gateway

    def _llm_available(self) -> bool:
        try:
            self._get_llm_gateway()
        except LLMError:
            return False
        return True

    def _should_use_llm_merge(self, records: Sequence[BaseModel]) -> bool:
        if not self._llm_available():
            return False
        if len(records) > self.llm_merge_record_limit:
            return False
        if self._records_are_structurally_clean(records):
            return False
        return True

    def _records_are_structurally_clean(self, records: Sequence[BaseModel]) -> bool:
        if not records:
            return True
        populated_counts: list[int] = []
        identity_keys: set[str] = set()
        duplicate_identities = 0
        field_count = max(len(self.row_model.model_fields), 1)

        for record in records:
            payload = record.model_dump(mode="json")
            populated_counts.append(
                sum(0 if self._is_missing_value(value) else 1 for value in payload.values())
            )
            identity_key = self._identity_key(record)
            if identity_key is None:
                continue
            if identity_key in identity_keys:
                duplicate_identities += 1
            else:
                identity_keys.add(identity_key)

        avg_populated = sum(populated_counts) / len(populated_counts)
        completeness_ratio = avg_populated / field_count
        duplicate_ratio = duplicate_identities / max(len(records), 1)
        return completeness_ratio >= 0.6 and duplicate_ratio <= 0.1

    def _merge_records_deterministically(self, records: Sequence[BaseModel]) -> list[dict[str, Any]]:
        merged_records: list[dict[str, Any]] = []
        merged_by_key: dict[str, int] = {}
        seen: set[str] = set()
        for record in records:
            fingerprint = record.model_dump_json(exclude_none=True, exclude_defaults=True)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            merge_key = self._identity_key(record)
            if merge_key is None:
                merged_records.append(self._clean_payload_for_schema(record.model_dump(mode="json")))
                continue
            existing_index = merged_by_key.get(merge_key)
            if existing_index is None:
                merged_by_key[merge_key] = len(merged_records)
                merged_records.append(self._clean_payload_for_schema(record.model_dump(mode="json")))
                continue
            merged_records[existing_index] = self._merge_record_pair(
                merged_records[existing_index],
                self._clean_payload_for_schema(record.model_dump(mode="json")),
            )
        return merged_records

    def _identity_key(self, record: BaseModel) -> str | None:
        payload = record.model_dump(mode="json")
        for field_name in (
            "state",
            "school",
            "team_name",
            "name",
            "entity_name",
            "organization_name",
            "player_name",
            "company_name",
        ):
            value = payload.get(field_name)
            if isinstance(value, str) and value.strip():
                return f"{field_name}:{value.strip().casefold()}"
        return None

    def _merge_record_pair(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        merged_payload = dict(left)
        incoming_payload = dict(right)
        for field_name in self.row_model.model_fields:
            preferred = self._prefer_value(
                existing=merged_payload.get(field_name),
                incoming=incoming_payload.get(field_name),
            )
            if preferred is not None or field_name in merged_payload:
                merged_payload[field_name] = preferred
        return merged_payload

    def _filter_payloads_with_target(self, payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        target_field = self._identify_target_field()
        if target_field is None:
            LOGGER.warning("No target field metadata found in row schema; returning records without label filtering")
            return list(payloads)

        missing_target_payloads: list[dict[str, Any]] = []
        kept_payloads: list[dict[str, Any]] = []
        for payload in payloads:
            value = payload.get(target_field)
            if self._is_missing_target_value(value):
                missing_target_payloads.append(payload)
                continue
            kept_payloads.append(payload)

        if missing_target_payloads and not kept_payloads:
            LOGGER.warning(
                "All %d records are missing target '%s'; keeping deduped records instead of returning 0 rows",
                len(missing_target_payloads),
                target_field,
            )
            return list(payloads)

        if missing_target_payloads:
            LOGGER.warning(
                "Dropping %d records because they are missing the target variable: '%s'",
                len(missing_target_payloads),
                target_field,
            )
        return kept_payloads

    def _prefer_value(self, *, existing: Any, incoming: Any) -> Any:
        if self._is_missing_value(existing):
            return incoming
        if self._is_missing_value(incoming):
            return existing
        if isinstance(existing, (int, float)) and isinstance(incoming, (int, float)):
            if existing == 0 and incoming != 0:
                return incoming
            return existing
        if isinstance(existing, str) and isinstance(incoming, str):
            if existing.strip().casefold() in {"unknown", "n/a", "na", "none", "null"}:
                return incoming
            return existing
        return existing

    def _clean_payload_for_schema(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(payload)
        text_values = [
            str(value).strip()
            for value in cleaned.values()
            if isinstance(value, str) and str(value).strip()
        ]
        derived_specs = TextCleaningUtility.extract_laptop_specs(", ".join(text_values))
        for field_name, value in derived_specs.items():
            if field_name in self.row_model.model_fields and self._is_missing_value(cleaned.get(field_name)):
                cleaned[field_name] = value

        for field_name in self.row_model.model_fields:
            value = cleaned.get(field_name)
            if value is None:
                continue
            if isinstance(value, str):
                normalized = value.strip()
                cleaned[field_name] = normalized or None
                value = cleaned[field_name]
            if value is None or not self._field_expects_numeric(field_name):
                continue
            coerced = self._coerce_numeric_field(field_name, value)
            cleaned[field_name] = coerced
        return cleaned

    def _field_expects_numeric(self, field_name: str) -> bool:
        property_schema = self.row_model.model_json_schema().get("properties", {}).get(field_name, {})
        if property_schema.get("type") in {"integer", "number"}:
            return True
        return any(branch.get("type") in {"integer", "number"} for branch in property_schema.get("anyOf", []))

    def _field_expects_integer(self, field_name: str) -> bool:
        property_schema = self.row_model.model_json_schema().get("properties", {}).get(field_name, {})
        if property_schema.get("type") == "integer":
            return True
        return any(branch.get("type") == "integer" for branch in property_schema.get("anyOf", []))

    def _coerce_numeric_field(self, field_name: str, value: Any) -> Any:
        if isinstance(value, (int, float)):
            if self._field_expects_integer(field_name) and float(value).is_integer():
                return int(value)
            return value

        text = str(value).strip()
        if not text:
            return None

        numeric_value = TextCleaningUtility.clean_price(text)
        if numeric_value is None:
            spec_value = TextCleaningUtility.extract_laptop_specs(text).get(field_name)
            if isinstance(spec_value, (int, float)):
                numeric_value = float(spec_value)
        if numeric_value is None:
            return value
        if self._field_expects_integer(field_name):
            return int(round(numeric_value))
        return float(numeric_value)

    def _is_missing_value(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return True
            if normalized.casefold() in {"unknown", "n/a", "na", "none", "null"}:
                return True
        if isinstance(value, list) and not value:
            return True
        return False

    def _truncate_json_for_prompt(self, payload: dict[str, Any], *, max_chars: int = 40000) -> str:
        serialized = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if len(serialized) <= max_chars:
            return serialized
        return serialized[:max_chars] + "\n... [truncated]"

    def _prepare_document_text_for_prompt(self, document_text: str, *, max_chars: int = 24000) -> str:
        """Collapse raw HTML or snapshot text into a denser text block for LLM extraction."""
        if not document_text.strip():
            return ""

        cleaned = re.sub(r"<script\b[^>]*>.*?</script>", " ", document_text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"<style\b[^>]*>.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"<[^>]+>", "\n", cleaned)
        cleaned = html.unescape(cleaned)

        lines: list[str] = []
        seen: set[str] = set()
        for raw_line in cleaned.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if len(line) < 3:
                continue
            lowered = line.lower()
            if lowered in seen:
                continue
            if any(
                phrase in lowered
                for phrase in (
                    "cookie policy",
                    "privacy policy",
                    "accept cookies",
                    "sign in",
                    "log in",
                    "jump to content",
                    "main menu",
                )
            ):
                continue
            seen.add(lowered)
            lines.append(line)

        prepared = "\n".join(lines)
        if len(prepared) <= max_chars:
            return prepared
        head = prepared[: max_chars // 2]
        tail = prepared[-max_chars // 2 :]
        return f"{head}\n...[truncated]...\n{tail}"
