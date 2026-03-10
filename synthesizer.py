"""Data synthesis stage for entity and conflict resolution."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import BaseModel

from llm import LLMError, LLMGateway, build_record_list_model


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DataSynthesizer:
    """Use an LLM to merge duplicates and normalize record values."""

    row_model: type[BaseModel]
    llm_gateway: LLMGateway = field(default_factory=lambda: LLMGateway(max_tokens=2000))
    batch_size: int = 20

    def synthesize(self, goal: str, raw_records: Sequence[BaseModel]) -> list[dict[str, Any]]:
        """Normalize and deduplicate raw swarm records."""
        if not raw_records:
            return []

        LOGGER.info("Synthesizing %d raw records", len(raw_records))
        canonical: list[BaseModel] = []
        for start in range(0, len(raw_records), self.batch_size):
            batch = raw_records[start : start + self.batch_size]
            canonical = self._merge_batch(goal, canonical, batch)

        final_records = self._merge_batch(goal, [], canonical)
        LOGGER.info("Synthesizer produced %d canonical records", len(final_records))
        return [record.model_dump(mode="json") for record in final_records]

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
            "Prefer non-null values and the most recent or detailed statement when conflicts appear."
        )
        user_prompt = (
            f"Dataset goal:\n{goal}\n\n"
            f"Canonical records so far:\n{json.dumps([row.model_dump(mode='json') for row in canonical_records], indent=2, sort_keys=True)}\n\n"
            f"Incoming batch:\n{json.dumps([row.model_dump(mode='json') for row in incoming_records], indent=2, sort_keys=True)}\n\n"
            f"Target row schema:\n{json.dumps(self.row_model.model_json_schema(), indent=2, sort_keys=True)}\n\n"
            "Return the merged canonical record list."
        )
        try:
            response = self.llm_gateway.complete_structured(
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
