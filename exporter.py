"""CSV export utilities for research results."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel


class CSVWriter:
    """Write validated Pydantic records to a CSV file."""

    def __init__(self, output_dir: str | Path = ".") -> None:
        self.output_dir = Path(output_dir)

    def write(self, records: Sequence[BaseModel], filename: str | None = None) -> Path:
        """Write records to CSV using schema field names as headers."""
        if not records:
            raise ValueError("records must contain at least one item")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        schema_type = type(records[0])
        headers = list(schema_type.model_fields.keys())
        output_path = self.output_dir / (
            filename or f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for record in records:
                writer.writerow(record.model_dump(mode="json"))

        return output_path
