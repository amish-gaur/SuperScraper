# Deployment

This document covers service readiness, required runtime configuration, and the normal commands for running the CLI, API, and worker in a deployment-oriented environment.

## Recommended split

For the easiest ongoing updates:

- deploy the frontend from [`frontend/`](frontend) to Vercel
- deploy the Python API from the repo root to Render as a single web service

This project can run without a separate worker for hobby/demo use because the API already falls back to in-process job execution when Redis is unavailable.

## Cost reality

This can be close to free for a demo, but not a durable production setup.

- Vercel frontend: usually free on the Hobby plan
- Render backend: free tier may work for a demo, but it can sleep, has limits, and uses ephemeral disk

That means:

- the frontend is a good long-term fit for Vercel
- the backend is acceptable on a free Render service for a demo
- exported artifacts and job history can disappear after restarts or redeploys on the free backend because `ARTIFACT_ROOT` is set to `/tmp/artifacts`

If you want durable artifacts or heavier usage, move the backend to a paid service or add persistent storage.

## Readiness check

Run a local readiness check before deploying or restarting:

```bash
python3 main.py --doctor
```

`--doctor` checks:

- Python files compile cleanly
- Required Python packages are importable
- Routing-layer smoke tests pass
- `.env.local` can be parsed
- An LLM API key is detected or deterministic-only mode is possible
- `agent-browser` is installed and executable
- The workspace is writable for checkpoint and dataset output files

## Typical commands

```bash
python3 main.py --goal "Build a predictive dataset of NBA players with salary as the target and performance features"
python3 main.py --goal "Build a predictive dataset of startup companies with valuation as the target and funding features" --log-level DEBUG
```

## Operational notes

- Browser fallback depends on the `agent-browser` binary. Set `AGENT_BROWSER_BIN` if it is not on `PATH`.
- LLM-backed schema design and synthesis require one of `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `GROQ_API_KEY`.
- If no API key is configured, deterministic routing still works, but LLM extraction from hidden JSON state is skipped.

## Render backend

This repo includes a Render blueprint in [`render.yaml`](render.yaml).

Suggested setup:

1. Create a new Render Blueprint service from the repo.
2. Use the root directory of the repo.
3. Set one API key:
   - `OPENAI_API_KEY`, or
   - `GEMINI_API_KEY`, or
   - `GROQ_API_KEY`
4. After Vercel gives you the frontend URL, set `FRONTEND_ORIGIN` to that exact origin.

Notes:

- `render.yaml` sets `ARTIFACT_ROOT=/tmp/artifacts` to avoid requiring a persistent disk for the first deploy.
- `REDIS_URL` is present for compatibility, but the API can still run in-process when Redis is unreachable.

## Vercel frontend

Suggested setup:

1. Import the same repo into Vercel.
2. Set the Root Directory to `frontend`.
3. Framework preset: `Vite`.
4. Set `VITE_API_BASE_URL` to your Render backend URL, for example:

```text
https://web-scraper-api.onrender.com
```

The frontend environment example is included at [`frontend/.env.example`](frontend/.env.example).

## Easy updates later

This deployment split is easy to maintain:

- push frontend changes to the repo and Vercel rebuilds from `frontend/`
- push backend changes to the repo and Render redeploys the API
- update the frontend API target with `VITE_API_BASE_URL` only if the backend URL changes
