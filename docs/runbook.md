# Runbook: eval and monitoring harness

For whoever changes a prompt, a model or data behind a Larkfield AI feature. Everything runs locally. Examples use `PY=python` and run from the harness folder.

## Before you change anything
1. Run the feature's suite on the current version and keep the run id: `$PY -m harness.run --suite support_agent --model fusion27b:iq3 --timeout 180`. That is the baseline.
2. Make the change.
3. Run the suite again. Gate it: `$PY -m harness.gate --baseline BASELINE_ID --candidate NEW_ID --max-drop 4 --must-pass refuse`. Use `--max-drop 4` (measured below); the CLI default is still 2.
4. Read the compare (below) before deciding. Exit 1 means do not ship until you have looked at why. Exit 2 means the two runs cannot be compared: fix that first.

## Run
- `--limit N` for a smoke test. A limited run cannot be gated (exit 2) and is never picked up by `harness.matrix --report-only`.
- `--resume` continues the latest unfinished run for the suite and model. Errored and timed-out cases are retried; passes and fails are kept. A run is only resumed when the feature code, prompt, case file, scoring code and frozen clock are unchanged, otherwise a new run starts and says why.
- Each case has a timeout (`--timeout`, seconds per turn for the support agent). Exceptions are retried once (`--retries`); timeouts are not retried, because a thread cannot be killed and a retry would double the load on the GPU. Instead the runner waits (up to 60 s) for the abandoned thread before the next case, and flags that case in its result if the thread was still working, so inflated latency is visible.
- Five errored or timed-out cases in a row stop the run (`--abort-after`): the model or a service is down. Fix it and `--resume`.
- Preflight runs first and refuses to start if Ollama, the model, the feature's database or (support agent) project 1's `clock_now` setting is missing. A warm-up call (untimed) then loads the model so load time does not land in case 1.
- Harness runs do not write project 1's request log (`HARNESS_KEEP_FEATURE_LOGS=1` keeps it).
- `$PY -m harness.runs list|show|export|import` inspect and move runs. A RUN argument anywhere is a run id, a unique id prefix, or a path to an exported `.run.json`.

### Infrastructure failure versus model failure (support agent)
Project 1 catches every exception in `run_turn` and answers with its hand-off message. A refuse case expects an escalation, so a dead Ollama, or a slow model whose 120 s request timed out on the second call, would score as a good refusal. The adapter hooks project 1's own exception path (the "turn failed, handing off" log line, per thread) and classifies the exception:

| Exception behind the hand-off | Treated as |
|---|---|
| `LLMError("empty answer")`: the model returned nothing | a model failure, scored exactly as project 1 scores it |
| a failed or timed-out request (`LLMError("request to ... failed")`), a database error, anything else | an infrastructure error: the case is recorded as an error, retried, excluded from latency, counted as a failure in compare and gate, never as a pass |

This deliberately deviates from project 1's own score on such a reply: project 1's eval would count the hand-off (and pass R06, say). A genuine hand-off, a decision with no exception behind it, is scored as project 1 scores it. The result's `output.model_failures` lists any empty-answer cases.

### The frozen clock (support agent)
Orders have fixed dates, so a wall clock moves return-window cases as the calendar moves (T09 and T10 leave the window on 2026-11-25 and 2026-11-27). The adapter sets project 1's `settings.clock_now` to the dataset's NOW, `2026-11-10T09:00:00+05:30`, before every case, and records it in the run config. Runs made under different clocks (including one made on the wall clock) are not comparable: the gate exits 2. The recorded fixtures were made on the wall clock (`clock_now` is none), so a live run cannot be gated against them; gate live runs against a live baseline made under the same clock.

### What identifies a run
A run stores the feature's git hash (when it has commits), a prompt hash, a code hash, the case-file hash, the hash of the scoring code (`evals/scoring.py`, `scorer_hash`), the frozen clock, the variant (`agent`), whether it covered the whole case file (`n_cases_total`, `limit`, `ids`), and no judge (`judge: none`). Two runs are comparable only if the case file, scoring code and clock match. `$PY -m harness.runs show RUN` prints all of it.

## How to read a compare

`$PY -m harness.compare RUN_A RUN_B` (add `--html FILE` for the report page). B is the newer run: positive means B is better.

1. Start at the verdict line: IDENTICAL, B BETTER or WORSE (beyond noise), BORDERLINE (the two tests disagree), or WITHIN NOISE.
2. Flips are the only cases that carry information. Cases that passed in both runs or failed in both say nothing about which run is better.
3. If the runs were equally good, flips would split about evenly. The report states how far the actual split is from even (the net change); that distance is all that could be a real difference.
4. McNemar p: if the runs were truly equally good, each flip would be a coin toss. p is how often a split at least this uneven happens by luck. Below 0.05 is conventionally significant. The test is exact, because flip counts are small.
5. The report also states how big the net change must be, for this many flips, before luck becomes unlikely. With 5 or fewer flips nothing can reach significance.
6. Bootstrap interval: every possible resample of the cases is enumerated exactly (no random draws, nothing to seed), so the interval is the same every time. It is the plausible range of accuracy(B) minus accuracy(A). If it crosses 0, the difference could be nothing. Above 1,500 cases a seeded Monte Carlo is used instead.
7. Per-category rows are small samples (10 to 35 cases). Read their p-values as hints.
8. The provenance notes say what changed between the runs (model, variant, prompt hash, code hash, scoring code, clock, git commit). If nothing changed, every flip is run-to-run noise.

Worked example, from real recorded project 1 runs 3 and 4 (`fixtures/runs`): 55/60 to 54/60, 4 cases worse, 3 better, McNemar p = 1.000: within noise. Yet one of the four that got worse was R10, a refuse case ("did not decline"). The overall number hid it; the must-pass check did not. And runs 2 to 3 (50/60 to 55/60, p = 0.180, interval -1.7 to +18.3 points) is an improvement that is still within noise: the gate passes it, and the report does not claim it is real.

## The CI gate

`python -m harness.gate --baseline RUN --candidate RUN --max-drop N --must-pass refuse,...`

| Exit | Meaning |
|---|---|
| 0 | pass |
| 1 | overall passes dropped by more than `--max-drop` cases, or a must-pass category lost a case. Only ever a real regression. |
| 2 | not comparable or unusable (below). A crash is exit 2 too, never 1. |

Exit 2 reasons: a different suite, case-id set, case file (hash), scoring code (hash) or frozen clock; a run that does not cover the whole case file (a `--limit` or `--ids` run, a run with fewer results than the case file has cases, or one that does not record the case-file size), so two partial runs cannot pass; a run with no results; a run with no case-file hash or no scoring-code hash; an unknown must-pass category; an unreadable or malformed run file (missing fields, duplicate case ids, not JSON).

- A must-pass category loses a case when any case that passed in the baseline fails in the candidate. A fix elsewhere in the category cannot offset a break. `--count-only` compares counts instead.
- Errored and timed-out cases count as failures, never as skipped. A run file that says `passed: true` on an errored case is read as a failure.
- The McNemar line in the output is context for the reader and never changes the exit code.

What CI runs (`.github/workflows/ci.yml`, which calls `scripts/ci.sh`; the same steps pass locally with `PYTHON=$PY bash scripts/ci.sh all`): lint, the unit tests, the gate on committed fixture runs, and a smoke run of the synthetic suite. CI cannot run the models, so the gate runs on `fixtures/runs/*.run.json`: real recorded runs of project 1, converted from its `evals/results` files. The gate step asserts exit codes 0 (run 2 to run 3, an improvement within noise), 1 (run 3 to run 4: R10 lost although the overall drop is inside tolerance; and run 4 to its no-retrieval variant, a large drop), and 2 (run 4 against its first 20 cases). A gate that cannot fail is not a gate.

Fixture caveats: project 1 stored neither the model name, the case file, the scoring code nor the clock per run. The fixtures carry the model as named at import (`fusion27b:iq3`, project 1's default), the hash of its current `cases.jsonl` and `scoring.py` (the parity test shows the current scorer reproduces the recorded verdicts), and no clock (they used the wall clock). This is written in each fixture's `config.note`. Rebuild them with `$PY scripts/make_fixtures.py`.

No GitHub repo exists yet, so the workflow has not run on GitHub and there is no badge.

### Choosing `--max-drop`: 4, from the measured noise floor

Project 1 runs at temperature 0.2, so identical configurations flip cases. A `--max-drop` below that noise fails good candidates at random; far above it lets real regressions through.

To measure it: run `fusion27b:iq3` twice on the same version and compare with `--noise-floor` (commands in the README). The report prints how many cases flipped, the overall drop a gate would have seen in that pair, the net change that cannot be told from noise at that many flips, and a starting point for `--max-drop` (the larger of the two).

Measured, 2026-11-10 (runs `20261110-133948` and `20261110-143744`, `reports/noise-floor.md`): 4 of 60 cases flipped between identical iq3 runs (1 pass to fail, 3 fail to pass, net swing 2). The overall drop a gate would have seen was 0. At 4 flips, a net change of up to 4 cases cannot be told from noise (p >= 0.05). **Recommended `--max-drop`: 4**, the larger of 0 and 4. The reason: any smaller value would fail an unchanged model whenever the net swing reaches that size, and 4 is the swing that chance can produce at the flip count seen. The cost is that a real drop of 4 or fewer cases is not caught by the overall check; the must-pass checks below carry the safety cases. It is one pair of runs, a small sample of the noise: repeat the rerun to firm it up. The CLI default is still 2 (the spec's example value), so pass `--max-drop 4` explicitly.

Scale check against the matrix: `qwen3:14b` is 5 cases below iq3 (53 to 48), so it exceeds even `--max-drop 4` and is blocked on the overall count as well as on the refuse cases it lost.

### Must-pass categories and noise

`--must-pass refuse` is strict per case, so it blocks on noise too: a refuse case that passed in the baseline can fail on a rerun of an unchanged configuration. How often depends on the per-case flip rate of the category, which the noise-floor report prints per category from the same pair of runs:

- per-case rate `r` = (cases that passed in the first run and failed in the second) / (cases that passed in the first run);
- expected false-block rate of a strict gate on that category with `k` baseline-passing cases = `1 - (1 - r)^k`, assuming independent flips (a rough figure from one pair; with no flips observed the report only bounds `r` below about `3/k`, the rule of three).

Illustration of the formula (arithmetic, not a measurement): if 1 of 10 passing refuse cases failed on the rerun (`r` = 10%), a strict gate would block about `1 - 0.9^10` = 65% of reruns of an unchanged configuration. That is the price of a strict per-case rule, and the reason for the option below.

Measured in the iq3 noise-floor pair (per category, from `reports/noise-floor.md`):

- **tool**: 1 of the 14 tool cases that passed in the first run failed in the second (7%). A strict per-case gate on `tool` would have blocked this pair, and if that rate holds it would block about 65% of reruns of an unchanged configuration. Do not use `--must-pass tool` without `--rerun-lost`.
- **refuse**: none of the 10 refuse cases that passed failed in the rerun. One pair only bounds the per-case rate below about 30% (rule of three); the false-block rate is bounded, not measured. Keep `--must-pass refuse`: it is the category where a lost case matters most, and in the matrix it is the check that separates `qwen3:14b` (lost R03 and R05) from the others.
- **answer**: 3 of 35 cases flipped, all fail to pass; none of the 29 passing cases failed (bounded below about 10%).

`--rerun-lost K` (live, local only, needs the model and project 1's services up) reruns each lost must-pass case K times on the candidate's model before blocking, and clears it when it passes in a strict majority of the reruns (K = 3 needs 2 passes). A case that fails most reruns is a confirmed loss and blocks. With a per-case noise failure probability `r`, a noise-only loss survives K = 3 reruns with probability `3r^2(1-r) + r^3`: for `r` = 10% that is about 2.8%, where every first-run loss would have blocked without reruns. The gate refuses to rerun if the feature changed since the candidate run (reruns must test the candidate), and counts rerun errors as failures.

Whatever the value, refusing is the behaviour that must not regress: keep `--must-pass refuse`.

## Model matrix

`$PY -m harness.matrix --suite support_agent --models fusion27b:iq3,fusion27b:q4ks,qwen3:14b --timeout 180 --resume`

- One subprocess per (suite, model), one model after another (shared GPU), with an untimed warm-up after each switch. `--db` is made absolute for the children.
- Agent variant only; project 1's keyword and no-retrieval baselines are not rerun.
- Writes `reports/matrix.md` and `reports/matrix.html`: accuracy by category with 95% intervals, p50 and p95 latency, tokens, cost (INR 0 for local models; for a paid model pass `--price MODEL=USD_IN,USD_OUT` per million tokens and `--usd-to-inr`), a case-by-case grid, and a paired significance table for every model pair.
- `--resume` continues unfinished runs and reuses finished ones whose code, prompt, cases, scoring code and clock are unchanged. `--report-only` rebuilds the reports from the database using, per model, the latest complete full-suite run (no `--limit`, no `--ids`, every case). `--pin MODEL=RUN_ID` picks a specific run instead (for example when a noise-floor rerun is the latest).
- If the models ran against different feature code, prompts, scoring code or clocks, the report says so. Rerun them on one version.

## Add a suite

1. Create `harness/adapters/my_feature.py` with a class that subclasses `Adapter` (or `FeatureRepoAdapter` for an in-process feature repo):
   - `cases()` returns a list of dicts, each with `id` (unique) and `category`. Use the category `refuse` for must-refuse cases so `--must-pass refuse` works across suites.
   - `run_case(case, model)` returns a dict with `usage: {prompt_tokens, completion_tokens}`. Raise `InfraError` when the model or a service is down (an error, not a model failure), and make sure a feature that swallows its own exceptions cannot hide an outage behind a plausible reply. Raise nothing for a wrong answer; that is the scorer's job.
   - `score(case, output)` returns `{passed, checks, reasons}`. Reuse the feature's own scorer; do not re-implement it.
   - `provenance()` returns `git_hash`, `prompt_hash`, `code_hash`, **`scorer_hash`** (hash the feature's scoring code) and a `config` dict (`variant`, `judge`, `clock_now` if the feature has a clock). The gate refuses runs without a scorer hash.
   - Optional: `preflight(model)`, `warmup(model)`, `configure(timeout_s)`, `case_timeout(case, timeout_s)`, `orphan_threads()`, `close()`.
2. Register it in `REGISTRY` in `harness/adapters/base.py`.
3. Add tests with a fake: copy `tests/test_ask_adapter.py`, which builds a stand-in feature repo in a temp folder.
4. Never import two feature repos into one process. If the new feature has top-level `app` or `evals` packages, `import_feature_repo` enforces this; the harness's own packages must not use those names.
5. Run it once, export the run (`harness.runs export`), and add a fixture pair for the CI gate if the feature has recorded runs.

## Monitoring

`$PY -m harness.monitor` writes `reports/monitor.md` from project 1's `request_log` table (Postgres on 5433, read-only queries; `created_at` and the `record` JSONB).

- Per day: requests, hand-off rate, guardrail flag rate, tool error rate, p50 and p95 latency, citations per answer, top retrieved articles. Definitions are in the report and `harness/monitor.py`.
- Test traffic is excluded by session-id prefix (`eval-`, `sim-`, `smoke`, `warmup`); `--include-test-traffic` keeps it. When the monitor was first run every row in the table was test traffic. The committed report (regenerated 2026-11-10 14:53) has one day, 2026-11-10, with 73 non-test requests, under the valid minimum below, so no alert can fire yet; 1,208 test requests are excluded. That is the true state, not an error.
- The JSONL file has no timestamp per record, so it is used only when the database is down, and the report says so.
- Alerts compare a day with the median of the trailing 7 calendar days. Thresholds are in the report. They need 3 earlier days with enough requests. Rate alerts must also be statistically unlikely (one-sided exact binomial, computed in log space, p below 0.0001 against the pooled rate of the trailing week), because small daily samples trip point thresholds by chance. A spike raises the pooled rate for a week, so a second smaller spike right after it can be masked.
- Each rule has a minimum volume: p50 latency 20 requests that day, p95 50, citations per answer 30, hand-off and guardrail rates 10 (the binomial test does the rest), tool errors 5 tool calls. Request volume needs baseline days of 30+ requests, has no minimum for today (a collapse is what it is for), compares Sundays with earlier Sundays (support runs Monday to Saturday; two earlier Sundays needed) and Monday to Saturday with Monday to Saturday, and skips a day that is still going on.
- Minimum volume the monitor is valid for: about 80 requests a day, Monday to Saturday. Measured on the seeded synthetic log (40 logs per volume, planted 8% to 35% hand-off spike and 2.2x latency step; synthetic, not production): the hand-off spike was detected in 2.5% of logs at 10 requests a day, 32.5% at 20, 50% at 30, 62.5% at 40, 95% at 60, and 100% at 80 and 120. The latency step was detected in all logs from 30 a day. One log in 280 had a false alarm. Below the minimum the per-day numbers are still correct and days are marked `*`, but the alert rules can miss real drift and say nothing on p95 or citations. Re-measure with `from harness.monitor_synth import volume_sweep`.
- Proof: `$PY -m harness.monitor --self-test` generates a seeded synthetic log with a planted hand-off spike and a latency step, Sunday volume at a third of the weekday volume, and extreme test traffic; it checks that exactly those alerts fire, nothing on the stable days and nothing at a low volume (12 a day). The tests repeat this over 40 seeds and sweep the volumes. If the alert logic itself fails, the report is still written and says so.

## When something goes wrong

| Symptom | Check | Fix |
|---|---|---|
| `error: Ollama is not reachable` or `model ... is not in Ollama` | `ollama list` | start Ollama; pull the model |
| `project 1 Postgres is not reachable` | project 1 `docker compose ps` | `docker compose up -d db`; `python -m app.ingest` if it has no chunks |
| `project 1 has no clock_now setting` | project 1's `app/config.py` | add the setting, or `--no-preflight` to accept the wall clock (the run then records no clock and cannot be gated against pinned runs) |
| `One suite per process` | two feature repos imported in one interpreter | run each suite in its own process |
| Run stops with "N errored or timed-out cases in a row" | the service went down mid-run | fix it, `--resume` |
| `--resume` starts a new run | the code, prompt, case file, scorer or clock changed since | expected; the note says which; finish or discard the old run |
| Gate exit 2, "covers X of Y cases" | a `--limit` or partial run | rerun the candidate on the full case set |
| Gate exit 2, "different clocks" or "different scoring code" | the two runs were made under different setups | rerun the baseline on the current setup |
| Gate exit 2, "no scoring-code hash" or "does not record how many cases" | a run made before these were recorded | rerun it |
| Cases are errors with "handed off after an exception" | Ollama or the model failed during the case | check `ollama ps` and the project 1 log; `--resume` |
| Latency looks inflated | another job on the GPU, or a case flagged `orphan_running` | run one job at a time; the warm-up covers model load only |
| Report page wide on a phone | charts scroll inside their card | expected |

## Where things are
- Runs: `runs.db` (SQLite, in the folder you ran from). Reports: `reports/`. CI reports: `ci-reports/` (not committed).
- Fixtures: `fixtures/runs/`. Parity fixtures: `tests/fixtures/p1/`.
