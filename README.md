# Web Scraper Deployment

## API

Run the stack with:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000/docs`.

Primary endpoints:

- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/download/csv`
- `GET /jobs/{job_id}/download/parquet`
- `GET /jobs/{job_id}/download/profile`

## Required environment variables

Set one of `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `GROQ_API_KEY`.

If you use Browserbase, set both `BROWSERBASE_API_KEY` and `BROWSERBASE_PROJECT_ID`.

The container installs `agent-browser` and runs `agent-browser install` during image build so the worker fails at startup, not mid-crawl, if the browser runtime is missing.

Optional tracing:

- `LLM_TRACING_BACKEND=langsmith`
- `LANGSMITH_API_KEY=...`

or

- `LLM_TRACING_BACKEND=phoenix`
- `PHOENIX_COLLECTOR_ENDPOINT=http://host.docker.internal:6006/v1/traces`
