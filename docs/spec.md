# FDE Project 5: Eval and monitoring harness

Spec v1, 2026-11-10. Part of the FDE Portfolio (vault: `Projects/In progress/FDE Portfolio`). Same fictional customer: Larkfield Coffee Co.

## The customer problem
Larkfield now runs several AI features (support agent, ask-your-data, later document extraction and voice). Every prompt, model or data change can silently break one of them. Goal: one harness that runs each feature's golden set against any model, compares runs, blocks regressions in CI, and watches production logs for drift.

## Stack (local only, no installs)
Venv `.venv` (fastapi, httpx, psycopg, pydantic, PyYAML, pytest, ruff). Storage: SQLite (stdlib) `runs.db`. Reports: static HTML (inline CSS and inline SVG charts, no CDN, no JS libraries) plus Markdown. Models compared (owner's choice, local only): Ollama `fusion27b:iq3`, `fusion27b:q4ks`, `qwen3:14b`.

## Design
- **Adapters** (`harness/adapters/`): one per feature, behind a small interface: `cases()`, `run_case(case, model) -> output`, `score(case, output) -> {passed, checks, reasons}`. The harness owns timing, token counts, retries, timeouts and storage; adapters own domain logic.
  - `support_agent`: calls project 1 in-process (`../fde-support-agent`, import `app.agent.run_turn` via `sys.path`; set the model through its settings) and reuses its `evals/scoring.py` scoring. Must not modify project 1 files.
    - (audit 2026-11-10) `run_case` calls P1's own `evals.run.run_case(case, "agent", run_turn, timeout_s)`: it handles multi-turn cases (`turns`), a fresh session per case, reply merging and timeouts, and never raises. Do not re-implement it. `score` calls `evals.scoring.score_case(case, merged_reply)`. Model switch: set `app.config.settings.ollama_model` (read on every LLM call). Needs P1's Postgres on 5433 and `nomic-embed-text`.
    - (audit 2026-11-10) The gated metric is P1's exact scoring. P1's LLM judge uses the same `settings.ollama_model`, so a judge would change with the model under test. If the judge is used at all, pin one judge model for every candidate and store it in the run's metadata.
  - `ask_your_data`: same pattern for `../fde-ask-your-data` (being built in parallel; write the adapter against its spec `docs/spec.md` and its `evals/` once it exists; if not ready, ship the adapter with a clear TODO and tests on a fake). (audit 2026-11-10) Its spec now fixes the contract: `app.agent.ask(question) -> dict`, `evals.scoring.score_case(case, result)`, categories `simple`, `complex`, `refuse`.
  - (audit 2026-11-10) One suite per process. P1 and P4 both have top-level `app` and `evals` packages, so two feature repos must never be imported into one interpreter: `harness.run` loads exactly one adapter, and `harness.matrix` starts a subprocess per (suite, model). The harness's own packages must not be named `app` or `evals`.
- **Runs**: `python -m harness.run --suite support_agent --model fusion27b:iq3 [--limit N] [--resume]`. Each case result stored with model, git hash of the feature repo if available, prompt hash, latency, tokens, pass/fail and reasons. Resumable, per-case timeout, never crashes the whole run on one case.
- **Compare**: `python -m harness.compare RUN_A RUN_B`: per-category deltas, cases that flipped (pass to fail and back), latency and token deltas; flags statistically weak differences (with 60 cases, say how many flips are noise: compute a simple McNemar test or bootstrap CI in code).
- **Gate**: `python -m harness.gate --baseline RUN --candidate RUN --max-drop 2 --must-pass refuse`: exit 1 if overall drops more than N cases, or any must-pass category loses a case. GitHub Actions workflow that runs the gate on two recorded fixture runs (CI cannot run the models) plus the unit tests. (audit 2026-11-10) Exit 2 (not comparable) if the two runs differ in suite, case-id set or case-file hash, so a `--limit` run cannot pass the gate. Errored and timed-out cases count as failures, never as skipped. No GitHub repo exists, so the workflow's steps must also pass locally; claim no CI badge.
- (audit 2026-11-10) **Noise floor**: P1 runs at temperature 0.2, so identical configs flip cases. The orchestrator runs `fusion27b:iq3` twice; `compare` of those two runs is reported as the noise floor, and the default `--max-drop` is justified against it in the runbook.
- **Model matrix**: `python -m harness.matrix --suite support_agent --models fusion27b:iq3,fusion27b:q4ks,qwen3:14b` writes `reports/matrix.md` and `reports/matrix.html`: accuracy by category, p50/p95 latency, tokens, cost (INR 0 local; field ready for paid models). (audit 2026-11-10) Agent variant only (P1's baselines are not rerun). Models run one after another; after each switch, one untimed warm-up call so model load time does not land in the first case's latency.
- **Monitoring** (`harness/monitor.py`): reads project 1's `logs/requests.jsonl` and reports per day: requests, hand-off rate, guardrail flag rate, tool error rate, p50/p95 latency, citations per answer, top retrieved articles, and alerts when a metric moves past a threshold vs the trailing 7-day median. Output `reports/monitor.md`.
  - (audit 2026-11-10) The JSONL records carry no timestamp, so "per day" must come from P1's `request_log` table (`created_at`, `record` JSONB; Postgres on 5433, read-only queries). Use the JSONL only when the DB is down, and say so in the report.
  - (audit 2026-11-10) Test traffic is excluded by default by session-id prefix (`eval-`, `sim-`, `smoke`, `warmup`; `--include-test-traffic` to keep it). Today nearly all P1 log lines are test traffic. Definitions: hand-off = `escalated`; guardrail flag = non-empty `output_flags` or `injection_suspected`; tool error = a `tool_calls` entry whose `result` has an `error` key.
  - (audit 2026-11-10) Real traffic covers about two days, so the alert logic is proved on a seeded synthetic log with planted drift (a hand-off spike, a latency step) that must fire, and a stable stretch that must not.

## Proof
- Unit tests with fake adapters: storage, resume, compare, McNemar/bootstrap, gate exit codes, monitor aggregation and alert logic.
- (audit 2026-11-10) Parity test, no LLM: feed the replies in P1's latest `evals/results/*-agent.jsonl` through the `support_agent` adapter's `score`; pass/fail must equal P1's recorded `passed` for every case (use a results file scored by the current `scoring.py`; copy it into the harness's test fixtures so the test does not break when P1 reruns). This proves the harness measures what P1 measures.
- The orchestrator runs the live model matrix (GPU is shared; do not call Ollama during the build).

## Deliverables
README (matrix table on top once run; pending until then), `docs/runbook.md` (how to add a suite, how to read a compare, how the CI gate works), `docs/engagement.md` (audit 2026-11-10: it opens with discovery: who asked, the success metric agreed, assumptions made, what was left out). No emoji. No git commit by the builder.

## Done when (audit 2026-11-10; the spec had none)
ruff clean, pytest green (including the parity test and gate exit codes 0/1/2 on fixture runs), CI workflow present and its steps pass locally, `harness.monitor` writes `reports/monitor.md` from P1's real `request_log`, live matrix and the iq3 noise-floor rerun done by the orchestrator, README matches `reports/matrix.md`.
