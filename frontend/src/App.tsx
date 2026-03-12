import { FormEvent, useEffect, useMemo, useState } from "react";

type JobStatus = "queued" | "running" | "completed" | "failed";

type JobPayload = {
  job_id: string;
  status: JobStatus;
  goal: string;
  max_agents: number;
  progress: {
    stage?: string;
    message?: string;
    detail?: Record<string, unknown> | null;
  } | null;
  artifacts: {
    dataset_name?: string;
    rows?: number;
    columns?: number;
  };
  error: string | null;
};

type ProfilePayload = {
  dataset_name: string;
  row_count: number;
  column_count: number;
  numeric_columns: string[];
  categorical_columns: string[];
  inferred_target_column: string | null;
  ml_task_type: string;
  dropped_features: string[];
  leakage_warnings: string[];
};

type PreviewPayload = {
  columns: string[];
  rows: Array<Record<string, unknown>>;
};

type JobLogPayload = {
  lines: string[];
};

const EXAMPLE_GOALS = [
  "Build a predictive dataset of NBA players with salary as the target and performance features",
  "Build a predictive dataset of NCAA men's basketball team statistics",
  "Build a predictive dataset of U.S. states with population growth as the target and GDP features",
];

const STAGES = [
  { id: "queued", label: "Queued" },
  { id: "architect", label: "Schema Design" },
  { id: "predictive_builder", label: "Direct Table Path" },
  { id: "swarm", label: "Swarm Extraction" },
  { id: "synthesizer", label: "Entity Cleanup" },
  { id: "formatter", label: "Dataset Packaging" },
  { id: "completed", label: "Ready" },
];

const CONFIGURED_API_BASE =
  typeof import.meta !== "undefined" && import.meta.env.VITE_API_BASE_URL
    ? String(import.meta.env.VITE_API_BASE_URL)
    : "";
const DEFAULT_API_BASE =
  CONFIGURED_API_BASE.trim() ||
  (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

function App() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [goal, setGoal] = useState(EXAMPLE_GOALS[0]);
  const [maxAgents, setMaxAgents] = useState(2);
  const [job, setJob] = useState<JobPayload | null>(null);
  const [profile, setProfile] = useState<ProfilePayload | null>(null);
  const [preview, setPreview] = useState<PreviewPayload | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") {
      return undefined;
    }

    const intervalId = window.setInterval(async () => {
      try {
        const nextJob = await fetchJson<JobPayload>(`${normalizedApiBase(apiBase)}/jobs/${job.job_id}`);
        setJob(nextJob);
      } catch (pollError) {
        setError(getErrorMessage(pollError));
      }
    }, 2500);

    return () => window.clearInterval(intervalId);
  }, [apiBase, job]);

  useEffect(() => {
    if (!job) {
      return undefined;
    }

    let cancelled = false;

    const loadLogs = async () => {
      try {
        const payload = await fetchJson<JobLogPayload>(
          `${normalizedApiBase(apiBase)}/jobs/${job.job_id}/logs?limit=160`,
        );
        if (!cancelled) {
          setLogs(payload.lines);
        }
      } catch (logError) {
        if (!cancelled) {
          setError(getErrorMessage(logError));
        }
      }
    };

    void loadLogs();
    const intervalId = window.setInterval(() => {
      void loadLogs();
    }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [apiBase, job]);

  useEffect(() => {
    if (job?.status !== "completed") {
      return;
    }

    const jobId = job.job_id;
    let cancelled = false;

    async function loadArtifacts() {
      setLoadingArtifacts(true);
      try {
        const [nextProfile, nextPreview] = await Promise.all([
          fetchJson<ProfilePayload>(`${normalizedApiBase(apiBase)}/jobs/${jobId}/profile`),
          fetchJson<PreviewPayload>(`${normalizedApiBase(apiBase)}/jobs/${jobId}/preview?limit=8`),
        ]);
        if (!cancelled) {
          setProfile(nextProfile);
          setPreview(nextPreview);
        }
      } catch (artifactError) {
        if (!cancelled) {
          setError(getErrorMessage(artifactError));
        }
      } finally {
        if (!cancelled) {
          setLoadingArtifacts(false);
        }
      }
    }

    void loadArtifacts();
    return () => {
      cancelled = true;
    };
  }, [apiBase, job?.job_id, job?.status]);

  const stageIndex = useMemo(() => {
    const currentStage = job?.progress?.stage ?? job?.status ?? "queued";
    const index = STAGES.findIndex((stage) => stage.id === currentStage);
    if (index >= 0) {
      return index;
    }
    if (job?.status === "failed") {
      return 0;
    }
    return 0;
  }, [job]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setJob(null);
    setProfile(null);
    setPreview(null);
    setStartedAt(Date.now());
    setLogs([]);

    try {
      const created = await fetchJson<{ job_id: string; status: string }>(
        `${normalizedApiBase(apiBase)}/jobs`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            goal,
            max_agents: maxAgents,
          }),
        },
      );
      const nextJob = await fetchJson<JobPayload>(
        `${normalizedApiBase(apiBase)}/jobs/${created.job_id}`,
      );
      setJob(nextJob);
    } catch (submitError) {
      setError(getErrorMessage(submitError));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Web Scraper</p>
          <h1>Build a dataset, watch the pipeline, inspect the output.</h1>
          <p className="lede">
            A local control room for the dataset-generation pipeline. Submit one goal, watch the
            live job log, and review the finished artifacts without leaving the page.
          </p>
        </div>
        <div className="hero-card">
          <div className="hero-metric">
            <span>Current stage</span>
            <strong>{job?.progress?.stage ?? "idle"}</strong>
          </div>
          <div className="hero-metric">
            <span>Run status</span>
            <strong>{job?.status ?? "waiting"}</strong>
          </div>
          <div className="hero-metric">
            <span>Elapsed</span>
            <strong>{startedAt ? formatElapsed(Date.now() - startedAt) : "0:00"}</strong>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel composer">
          <div className="panel-header">
            <h2>Start A Run</h2>
            <span className="badge badge-neutral">Local</span>
          </div>
          <form onSubmit={handleSubmit}>
            <label className="field">
              <span>Dataset Goal</span>
              <textarea
                rows={6}
                value={goal}
                onChange={(event) => setGoal(event.target.value)}
                placeholder="Describe the dataset you want to build"
              />
            </label>

            <div className="split-fields">
              <label className="field">
                <span>Max Agents</span>
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={maxAgents}
                  onChange={(event) => setMaxAgents(Number(event.target.value))}
                />
              </label>

              <label className="field">
                <span>API Base URL</span>
                <input
                  value={apiBase}
                  onChange={(event) => setApiBase(event.target.value)}
                  placeholder={DEFAULT_API_BASE}
                />
              </label>
            </div>

            <div className="actions">
              <button type="submit" disabled={submitting || goal.trim().length === 0}>
                {submitting ? "Submitting…" : "Generate Dataset"}
              </button>
            </div>
          </form>

          <div className="examples">
            <h3>Example Goals</h3>
            <div className="example-list">
              {EXAMPLE_GOALS.map((example) => (
                <button
                  key={example}
                  className="example-chip"
                  type="button"
                  onClick={() => setGoal(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="stack">
          <section className="panel status">
            <div className="panel-header">
              <h2>Pipeline Status</h2>
              {job ? <span className={`badge badge-${job.status}`}>{job.status}</span> : null}
            </div>

            {job ? (
              <>
                <p className="status-message">{job.progress?.message ?? "Waiting for updates"}</p>
                {job.progress?.detail && Object.keys(job.progress.detail).length > 0 ? (
                  <div className="detail-box">
                    {Object.entries(job.progress.detail).map(([key, value]) => (
                      <p key={key}>
                        <span>{humanizeKey(key)}:</span> {String(value)}
                      </p>
                    ))}
                  </div>
                ) : null}
                <div className="timeline">
                  {STAGES.map((stage, index) => {
                    const active = index === stageIndex;
                    const complete = index < stageIndex || job.status === "completed";
                    return (
                      <div
                        key={stage.id}
                        className={`timeline-step${complete ? " complete" : ""}${active ? " active" : ""}`}
                      >
                        <div className="timeline-dot" />
                        <div>
                          <strong>{stage.label}</strong>
                          <p>{stage.id}</p>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <dl className="job-meta">
                  <div>
                    <dt>Job ID</dt>
                    <dd>{job.job_id}</dd>
                  </div>
                  <div>
                    <dt>Goal</dt>
                    <dd>{job.goal}</dd>
                  </div>
                </dl>

                {job.error ? <p className="error-text">{job.error}</p> : null}
              </>
            ) : (
              <p className="empty-state">
                Start a run and this panel will track the job through schema design, source
                fetching, synthesis, and export.
              </p>
            )}
          </section>

          <section className="panel terminal-panel">
            <div className="panel-header">
              <h2>Run Log</h2>
              <span className="badge badge-neutral">{logs.length} lines</span>
            </div>
            {job ? (
              <div className="terminal">
                {logs.length > 0 ? (
                  logs.map((line, index) => <div key={`${index}-${line}`}>{line}</div>)
                ) : (
                  <div>Waiting for log output…</div>
                )}
              </div>
            ) : (
              <p className="empty-state">Job logs will stream here after you submit a run.</p>
            )}
          </section>
        </section>

        <section className="panel results">
          <div className="panel-header">
            <h2>Results</h2>
            {loadingArtifacts ? <span className="badge badge-running">Loading</span> : null}
          </div>

          {profile && job ? (
            <>
              <div className="stats-grid">
                <article>
                  <span>Dataset</span>
                  <strong>{profile.dataset_name}</strong>
                </article>
                <article>
                  <span>Rows</span>
                  <strong>{profile.row_count}</strong>
                </article>
                <article>
                  <span>Columns</span>
                  <strong>{profile.column_count}</strong>
                </article>
                <article>
                  <span>Task Type</span>
                  <strong>{profile.ml_task_type}</strong>
                </article>
              </div>

              <div className="result-grid">
                <article className="result-card">
                  <h3>Modeling Hints</h3>
                  <p>
                    Target column: <strong>{profile.inferred_target_column ?? "Not inferred"}</strong>
                  </p>
                  <p>Numeric features: {profile.numeric_columns.length}</p>
                  <p>Categorical features: {profile.categorical_columns.length}</p>
                  <p>
                    Dropped features:{" "}
                    {profile.dropped_features.length > 0
                      ? profile.dropped_features.join(", ")
                      : "None flagged"}
                  </p>
                </article>

                <article className="result-card">
                  <h3>Artifact Downloads</h3>
                  <div className="download-list">
                    <a href={`${normalizedApiBase(apiBase)}/jobs/${job.job_id}/download/csv`} target="_blank" rel="noreferrer">
                      Download CSV
                    </a>
                    <a href={`${normalizedApiBase(apiBase)}/jobs/${job.job_id}/download/parquet`} target="_blank" rel="noreferrer">
                      Download Parquet
                    </a>
                    <a href={`${normalizedApiBase(apiBase)}/jobs/${job.job_id}/download/profile`} target="_blank" rel="noreferrer">
                      Download Profile JSON
                    </a>
                  </div>
                </article>
              </div>

              <article className="result-card warnings">
                <h3>Leakage Warnings</h3>
                {profile.leakage_warnings.length > 0 ? (
                  <ul>
                    {profile.leakage_warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                ) : (
                  <p>No obvious leakage warnings were flagged.</p>
                )}
              </article>

              <article className="result-card preview">
                <h3>Preview Rows</h3>
                {preview ? (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          {preview.columns.map((column) => (
                            <th key={column}>{column}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {preview.rows.map((row, rowIndex) => (
                          <tr key={`row-${rowIndex}`}>
                            {preview.columns.map((column) => (
                              <td key={`${rowIndex}-${column}`}>{formatCell(row[column])}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p>Preview not loaded yet.</p>
                )}
              </article>
            </>
          ) : (
            <p className="empty-state">
              Completed runs will show profile metadata, a row preview, and artifact downloads here.
            </p>
          )}
        </section>
      </main>

      {error ? <aside className="error-banner">{error}</aside> : null}
    </div>
  );
}

function normalizedApiBase(value: string) {
  return value.trim().replace(/\/+$/, "");
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const detail = await safeDetail(response);
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

async function safeDetail(response: Response) {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? `Request failed with status ${response.status}`;
  } catch {
    return `Request failed with status ${response.status}`;
  }
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unexpected error";
}

function formatCell(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}

function humanizeKey(value: string) {
  return value.replace(/_/g, " ");
}

function formatElapsed(milliseconds: number) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

export default App;
