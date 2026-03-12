# Web Scraper

Turn a vague machine-learning data request into a usable dataset.

This project takes goals like:

- "predict NBA player salary from performance stats"
- "forecast U.S. state population growth from GDP"
- "estimate startup valuation from funding"

and tries to produce:

- a row schema
- a set of likely public sources
- a cleaned dataset
- `csv`, `parquet`, and profile artifacts

It is built for list-heavy public web data, not arbitrary full-site crawling.

## How it works

The pipeline has four stages:

1. `architect`
   Turns a user goal into a dataset blueprint: target field, feature fields, and starting URLs.
2. `predictive builder`
   Tries the fast path first by merging public HTML tables directly for supported goal families.
3. `swarm`
   Falls back to routed extraction when the direct table path is not enough.
4. `synthesizer + formatter`
   Cleans records, exports files, and writes a dataset profile.

The project is strongest when the target data exists on public rankings, directories, stat tables, or "List of..." pages.

## Current strengths

- Predictive datasets from public stat tables and list pages
- Deterministic handling for several common goal families
- API mode, worker mode, and local CLI mode
- Regression coverage for routing, heuristics, and dataset assembly

## Current limits

- It is not a general-purpose crawler for any website
- Anti-bot-heavy sources can still be inconsistent
- Some goals depend on LLM-backed schema design and recovery
- Browser fallback requires `agent-browser`

If a goal can be satisfied from public tables, this repo performs much better than if it has to infer data from scattered detail pages.

## Quick start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Set one API key if you want LLM-backed planning and synthesis:

```bash
export OPENAI_API_KEY=...
```

Run a build:

```bash
python3 main.py --goal "Build a predictive dataset of NBA players with salary as the target and performance features" --max-agents 3
```

Output is written under the working directory or configured artifact directory.

## Good example goals

These are the kinds of requests the system handles best:

```text
Build a predictive dataset of NCAA men's basketball team statistics
Build a predictive dataset of U.S. states with population growth as the target and GDP features
I want data to predict how much an NBA player gets paid using on-court performance stats
Put together a machine-learning table for startup companies where valuation is the label and funding is a key predictor
Give me a dataset for the biggest U.S. banks so I can estimate market value from asset size and capital strength
I need laptop pricing data where the thing to predict is price and the inputs are hardware specs
```

## CLI

Main entrypoint:

```bash
python3 main.py --goal "..." --max-agents 3
```

Useful commands:

```bash
python3 main.py --setup
python3 main.py --api-key-status
python3 main.py --model-status
python3 main.py --doctor
```

`--doctor` is the fastest preflight check before local use or deployment.

## API

Run the stack:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8000/docs
```

Main endpoints:

- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/download/csv`
- `GET /jobs/{job_id}/download/parquet`
- `GET /jobs/{job_id}/download/profile`

## Important files

- [`main.py`](main.py)
  CLI entrypoint and setup flow.
- [`api.py`](api.py)
  FastAPI service for queued jobs.
- [`worker.py`](worker.py)
  Celery worker for background execution.
- [`pipeline_service.py`](pipeline_service.py)
  Shared orchestration for CLI, API, and worker paths.
- [`architect.py`](architect.py)
  Goal interpretation, source selection, and schema planning.
- [`predictive_dataset_builder.py`](predictive_dataset_builder.py)
  Deterministic wide-table assembly from compatible public tables.
- [`smoke_tests.py`](smoke_tests.py)
  Deterministic structural smoke tests.
- [`edge_case_tests.py`](edge_case_tests.py)
  Regression tests for heuristics and weird phrasing.
- [`DEPLOYMENT.md`](DEPLOYMENT.md)
  Deployment notes and readiness guidance.

## Environment

Minimum practical setup:

- Python `3.12+`
- one of:
  - `OPENAI_API_KEY`
  - `GEMINI_API_KEY`
  - `GROQ_API_KEY`

For browser fallback:

- `agent-browser` installed and available on `PATH`

Optional:

- `BROWSERBASE_API_KEY`
- `BROWSERBASE_PROJECT_ID`
- tracing config for LangSmith or Phoenix

## Testing

Run the local regression set:

```bash
python3 config_tests.py
python3 edge_case_tests.py
python3 routing_smoke_tests.py
python3 smoke_tests.py
```

Or run the broader preflight:

```bash
python3 main.py --doctor
```

## Deployment

For service-oriented usage, see [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Positioning

This repo sits between a brittle scraper script and a full data platform.

It is best viewed as a dataset-generation engine for public, semi-structured web data:

- more automated than hand-written scrapers
- more opinionated than a generic crawling framework
- most effective when the requested dataset can be assembled from a small number of public tables or list pages
