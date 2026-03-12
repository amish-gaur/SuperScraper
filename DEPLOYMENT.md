# Deployment

This document covers service readiness, required runtime configuration, and the normal commands for running the CLI, API, and worker in a deployment-oriented environment.

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
