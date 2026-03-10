"""CLI entrypoint for the autonomous data engineering pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from architect import DatasetArchitect, json_schema_to_pydantic_model, schema_to_json_schema
from formatter import MLFormatter
from synthesizer import DataSynthesizer
from swarm import SwarmDispatcher


def configure_logging() -> None:
    """Configure INFO-level logging for pipeline progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Autonomous Data Engineer pipeline."
    )
    parser.add_argument(
        "--goal",
        required=True,
        help="High-level machine learning data goal.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the four-stage autonomous data engineering pipeline."""
    args = parse_args(argv or sys.argv[1:])
    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("Stage 1/4: Designing dataset schema")
    architect = DatasetArchitect()
    blueprint = architect.design(args.goal)
    row_model = json_schema_to_pydantic_model(
        schema_to_json_schema(blueprint.row_schema),
        model_name=blueprint.row_schema.title,
    )

    logger.info("Stage 2/4: Dispatching scraping swarm")
    swarm = SwarmDispatcher(goal=args.goal, blueprint=blueprint, row_model=row_model)
    raw_records = asyncio.run(swarm.run())
    logger.info("Swarm returned %d raw records", len(raw_records))

    logger.info("Stage 3/4: Synthesizing clean entities")
    synthesizer = DataSynthesizer(row_model=row_model)
    clean_records = synthesizer.synthesize(args.goal, raw_records)
    logger.info("Synthesizer returned %d clean records", len(clean_records))

    logger.info("Stage 4/4: Formatting ML-ready dataset")
    formatter = MLFormatter(clean_records, blueprint.dataset_name)
    formatter.handle_missing_values()
    formatter.encode_categoricals()
    csv_path, parquet_path = formatter.export()

    print(f"Dataset name: {blueprint.dataset_name}")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path}")
    print(f"Rows: {len(clean_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
