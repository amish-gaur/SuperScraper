"""Data synthesis stage for entity and conflict resolution."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from pydantic import BaseModel, ValidationError

from llm import LLMError, LLMGateway, build_record_list_model


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
                    validated_records.append(self.row_model.model_validate(payload))
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
        return True

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
                merged_records.append(record.model_dump(mode="json"))
                continue
            existing_index = merged_by_key.get(merge_key)
            if existing_index is None:
                merged_by_key[merge_key] = len(merged_records)
                merged_records.append(record.model_dump(mode="json"))
                continue
            merged_records[existing_index] = self._merge_record_pair(
                merged_records[existing_index],
                record.model_dump(mode="json"),
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
