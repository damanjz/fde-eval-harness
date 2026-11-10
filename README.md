# FDE eval and monitoring harness

One harness for Larkfield Coffee Co.'s AI features (fictional customer, FDE Portfolio project 5). It runs each feature's golden set against any model, compares runs with honest statistics, blocks regressions in CI, and watches production logs for drift. Local only, standard library only, no installs.

## Results: live run, 2026-11-10

Three local models on project 1's 60 cases, one run each, project 1's exact scoring, the frozen dataset clock (2026-11-10T09:00:00+05:30), project 1 commit `8ae4bb9`. Tables below are copied from `reports/matrix.md` (generated 2026-11-10 14:37) and `reports/noise-floor.md`; open `reports/matrix.html` and `reports/noise-floor.html` for the charts. `runs.db` also holds two earlier runs that are not used: `20261110-052006` (iq3, 53/60, made before a project 1 commit) and `20261110-053950` (q4ks, incomplete, abandoned at a reboot).

### Model matrix

#### Accuracy

| model | passed | 95% interval (Wilson) | answer | refuse | tool | errored |
|---|---|---|---|---|---|---|
| fusion27b:iq3 | 53/60 (88%) | 78% to 94% | 29/35 (83%) | 10/10 (100%) | 14/15 (93%) | 0 |
| fusion27b:q4ks | 55/60 (92%) | 82% to 96% | 32/35 (91%) | 10/10 (100%) | 13/15 (87%) | 1 |
| qwen3:14b | 48/60 (80%) | 68% to 88% | 29/35 (83%) | 8/10 (80%) | 11/15 (73%) | 0 |

#### Latency, tokens, cost

| model | p50 | p95 | prompt tok/case | completion tok/case | cost/case | total cost |
|---|---|---|---|---|---|---|
| fusion27b:iq3 | 16.2s | 29.2s | 4276 | 92 | INR 0.00 | INR 0.00 |
| fusion27b:q4ks | 22.9s | 136.5s | 4419 | 98 | INR 0.00 | INR 0.00 |
| qwen3:14b | 5.9s | 9.1s | 3949 | 87 | INR 0.00 | INR 0.00 |

Latency is wall time per case measured by the harness, over cases that completed. Local models cost INR 0; set --price-in and --price-out for paid models.

Latency caveat: the first iq3 run shared the GPU with other work. Its p50 of 16.2 s (p95 29.2 s) falls to 7.1 s (12.6 s) in the noise-floor rerun of the same configuration, below, so the first run's iq3 latency is inflated and the rerun's is the cleaner iq3 figure. Token counts are not affected.

#### Are the differences real?

Paired by case. A flip is a case one model passes and the other fails. McNemar is an exact test on the flips; the bootstrap interval is for accuracy(B) minus accuracy(A). With 60 cases, a few points of difference is usually noise.

| A | B | A passed | B passed | flips (B better / B worse) | McNemar p | bootstrap 95% (points) | reading |
|---|---|---|---|---|---|---|---|
| fusion27b:iq3 | fusion27b:q4ks | 53 | 55 | 4 / 2 | 0.688 | -5.0 to +11.7 | within noise |
| fusion27b:iq3 | qwen3:14b | 53 | 48 | 3 / 8 | 0.227 | -18.3 to +1.7 | within noise |
| fusion27b:q4ks | qwen3:14b | 55 | 48 | 2 / 9 | 0.065 | -21.7 to -1.7 | borderline |

#### Runs

| model | run | status | git | prompt | code | started |
|---|---|---|---|---|---|---|
| fusion27b:iq3 | 20261110-133948-support-agent-fusion27b-iq3 | complete | 8ae4bb9 | f559ab1a9b7d | a30d709e628f | 2026-11-10 13:39 |
| fusion27b:q4ks | 20261110-135652-support-agent-fusion27b-q4ks | partial | 8ae4bb9 | f559ab1a9b7d | a30d709e628f | 2026-11-10 13:56 |
| qwen3:14b | 20261110-143043-support-agent-qwen3-14b | complete | 8ae4bb9 | f559ab1a9b7d | a30d709e628f | 2026-11-10 14:30 |

#### Notes

- fusion27b:q4ks: run 20261110-135652-support-agent-fusion27b-q4ks is partial; errored or missing cases count as failures.
- Not in the matrix or incomplete: fusion27b:q4ks (run incomplete; rerun with --resume)

The q4ks run has one timed-out case, counted as a failure (55/60 includes it); `--resume` would retry it.

### Reading the matrix

- `fusion27b:iq3` against `fusion27b:q4ks`: 53 against 55 passed, 4 cases better and 2 worse, McNemar p = 0.688: within noise. The larger q4ks model buys nothing measurable, and it was slower than iq3 in both iq3 runs (p50 22.9 s and p95 136.5 s, with one timeout).
- `qwen3:14b` is the fastest (p50 5.9 s, p95 9.1 s) and the least accurate (48/60). It fails 2 of the 10 refuse cases (R03 and R05) that iq3 passes, so a `--must-pass refuse` gate blocks it (checked: `harness.gate` on iq3 as baseline and qwen3:14b as candidate exits 1, lost R03 and R05, overall drop 5). Its speed advantage is smaller than the matrix suggests: about 1.2 times faster than iq3's clean p50 (5.9 s against 7.1 s), not 2.7 times faster (against the 16.2 s run that shared the GPU).
- iq3 against `qwen3:14b` is within noise on the overall count (p = 0.227); q4ks against `qwen3:14b` is borderline (p = 0.065; `qwen3:14b` is worse on 9 cases and better on 2): suggestive, not established. The refuse failures are what make `qwen3:14b` unsafe to swap in, not the overall count.
- With 60 cases and one run per model, differences of a few points are within noise, as the noise floor below shows.

### Noise floor

`fusion27b:iq3` run twice on the same version, code, scorer and clock: runs `20261110-133948` (the matrix run) and `20261110-143744` (label `noise-floor-rerun`), from `reports/noise-floor.md`.

WITHIN NOISE: the difference is too small to tell from chance.

| | A | B | change (B - A) |
|---|---|---|---|
| accuracy | 53/60 (88%) | 55/60 (92%) | +2 case(s), +3.3 points |
| 95% interval (Wilson) | 78% to 94% | 82% to 96% | bootstrap -3.3 to +10.0 points |
| flips | 1 worse | 3 better | McNemar p = 0.625 |
| errored or timed out | 0 | 0 | |

#### Flipped cases

| case | category | direction | why it failed |
|---|---|---|---|
| T11 | tool | pass -> fail | missing fact '30 days' |
| A13 | answer | fail -> pass | missing fact 'defective'; missing fact 'damaged' |
| A23 | answer | fail -> pass | missing fact 'not accepted' |
| A26 | answer | fail -> pass | missing fact 'checkout' |

#### What the noise floor says

- Same configuration, run twice: 4 of 60 cases flipped (1 pass to fail, 3 fail to pass), a net swing of 2 case(s). Nothing changed between the runs, so all of it is noise.
- Seen from a gate (first run as baseline, second as candidate), the overall drop was 0 case(s): a --max-drop of 0 or more would have let this pair through. Noise can swing the other way too.
- Statistically, with 4 flips a net change of up to 4 case(s) cannot be told from noise (p >= 0.05). Take the larger of 0 and 4 (4) as a starting point for --max-drop. It is one pair of runs, a small sample of the noise: repeat the rerun to firm it up.
- Per category (the cost of a strict per-case must-pass; compare with --rerun-lost K in harness.gate):
- answer: 3 of 35 cases flipped; none of the 29 cases that passed in the first run failed in the second; one pair only bounds the per-case rate below about 10% (rule of three), so the false-block rate is bounded, not measured.
- refuse: 0 of 10 cases flipped; none of the 10 cases that passed in the first run failed in the second; one pair only bounds the per-case rate below about 30% (rule of three), so the false-block rate is bounded, not measured.
- tool: 1 of 15 cases flipped; 1 of 14 cases that passed in the first run failed in the second (7%), so a strict per-case gate on this category would have blocked this pair; if that rate holds, about 65% of reruns of an unchanged configuration would be blocked.

Reading: 4 of 60 cases flipped between identical runs, so about 7% of the cases flip from run to run. The recommended `--max-drop` is 4: it is the net change that cannot be told from noise at 4 flips, and the observed overall drop in this pair was 0. It is one pair of runs, a small sample of the noise; repeat the rerun to firm it up. The gate CLI default is still 2 (the spec's example), so pass `--max-drop 4` explicitly. A strict per-case `--must-pass tool` gate would have blocked this pair (1 of 14 passing tool cases failed on the rerun, 65% of reruns blocked if that rate holds), so use `--rerun-lost K` for categories that flip. `--must-pass refuse` is the one to keep: no refuse case flipped in this pair (the rate is bounded below about 30%, not measured). Details: `docs/runbook.md`.

#### Latency between the two identical runs

| | A | B | change |
|---|---|---|---|
| p50 latency | 16.2s | 7.1s | -56% |
| p95 latency | 29.2s | 12.6s | -57% |
| prompt tokens / case | 4276 | 4342 | +2% |
| completion tokens / case | 92 | 88 | -4% |
| cost / case | INR 0.00 | INR 0.00 | 0.00 |

Latency is wall time per case measured by the harness, over cases that completed.

## What it does

| Piece | Command | Output |
|---|---|---|
| Run | `python -m harness.run --suite support_agent --model fusion27b:iq3 [--limit N] [--resume]` | `runs.db` (SQLite): every case with model, git hash, prompt hash, latency, tokens, pass or fail, reasons |
| Compare | `python -m harness.compare RUN_A RUN_B [--noise-floor] [--html FILE]` | per-category deltas, flipped cases, latency and token deltas, McNemar and bootstrap in plain words |
| Gate | `python -m harness.gate --baseline RUN --candidate RUN --max-drop 4 --must-pass refuse [--rerun-lost K]` | exit 0 pass, 1 regression, 2 not comparable or unusable (never a crash as 1) |
| Matrix | `python -m harness.matrix --suite support_agent --models fusion27b:iq3,fusion27b:q4ks,qwen3:14b` | `reports/matrix.md` and `reports/matrix.html` (inline SVG, no external assets) |
| Monitor | `python -m harness.monitor` | `reports/monitor.md`: per-day metrics from project 1's `request_log`, drift alerts |

Suites: `support_agent` (project 1), `ask_your_data` (project 4), `fake` (synthetic, for tests and CI; its reports carry a SYNTHETIC banner).

## Live commands (what produced the results above; rerun to refresh)

Prerequisites: Ollama running with `fusion27b:iq3`, `fusion27b:q4ks`, `qwen3:14b` and `nomic-embed-text`; project 1's Postgres up and ingested (`docker compose up -d db`, then `python -m app.ingest` in its repo); project 1 has its `clock_now` setting (preflight checks); the GPU free (project 1's own eval not running). Run from this folder.

```
PY=python

# 1. smoke: three cases, one model (preflight checks Ollama, the model and the database first)
$PY -m harness.run --suite support_agent --model fusion27b:iq3 --limit 3 --timeout 180 --db smoke.db

# 2. the matrix: one subprocess per model, an untimed warm-up after each switch; resumable
$PY -m harness.matrix --suite support_agent --models fusion27b:iq3,fusion27b:q4ks,qwen3:14b --timeout 180 --resume

# 3. noise floor: run iq3 a second time (a new run, not --resume), then compare the two iq3 runs
$PY -m harness.run --suite support_agent --model fusion27b:iq3 --timeout 180 --label noise-floor-rerun
$PY -m harness.runs list --suite support_agent
$PY -m harness.compare RUN_ID_OF_FIRST_IQ3 RUN_ID_OF_SECOND_IQ3 --noise-floor --md reports/noise-floor.md --html reports/noise-floor.html

# 4. monitoring from project 1's real request_log (read-only)
$PY -m harness.monitor
```

`--timeout 180` matches project 1's own eval. If a run is interrupted, rerun the same command with `--resume`. The matrix and the noise-floor rerun share `runs.db`: write `reports/matrix.*` before the rerun, or pass `--pin MODEL=RUN_ID` to `--report-only`: it otherwise uses the latest complete full-suite run per model. A run is a model on an unchanged feature version, scoring code and clock: if project 1's code, prompt, cases, scorer or clock change between models, the matrix says so, and the gate exits 2.


## Offline checks (no GPU, no Ollama)

```
PY=python
$PY -m pytest -q                      # 276 tests; 266 run anywhere, 10 need projects 1 or 4 locally
PYTHON=$PY bash scripts/ci.sh all     # the CI steps: lint, tests, gate on fixtures, smoke
$PY -m harness.monitor --self-test    # alert logic on a seeded synthetic log with planted drift
```

The CI workflow (`.github/workflows/ci.yml`) calls the same `scripts/ci.sh` steps. No GitHub repo exists yet, so there is no CI badge and the workflow has not run on GitHub; its steps pass locally.

## How it is built

- Adapters (`harness/adapters/`): `cases()`, `run_case(case, model)`, `score(case, output)`. The harness owns timing, tokens, retries, timeouts and storage.
- `support_agent` calls project 1 in-process: its own `evals.run.run_case(case, "agent", run_turn, timeout_s)` and `evals.scoring.score_case`, so the harness measures what project 1 measures. The model is switched through `app.config.settings.ollama_model`, and the clock is frozen to the dataset's NOW through `settings.clock_now` (recorded in the run). The gated metric is project 1's exact scoring; its LLM judge is not used (it would change with the model under test). Project 1 files are never modified.
- Infrastructure failures are not model answers. Project 1 turns every exception into its hand-off message, which can pass a refuse case; the adapter hooks that exception path and records a failed or timed-out request as an error (retried, counted as a failure, never a pass). This deliberately deviates from project 1's own score on that reply. An empty model answer stays a model failure.
- A run is identified by the feature version, prompt, case file, scoring code (`scorer_hash`), frozen clock and variant. The gate exits 2 when two runs differ in the case file, scoring code or clock, or when either run does not cover the whole case file.
- One suite per process: projects 1 and 4 both have top-level `app` and `evals` packages. `harness.run` loads one adapter, `harness.matrix` starts a subprocess per (suite, model), and a second repo in one process raises an error.
- Statistics (`harness/stats.py`): exact McNemar test on paired flips, exact paired bootstrap interval for the accuracy difference (every resample enumerated, no seed), Wilson intervals for single accuracies.
- Recorded fixtures (`fixtures/runs/`): real recorded runs of project 1 (`evals/results`), used by the CI gate. See `docs/runbook.md` for what they do and do not carry.

## Files

| Path | What |
|---|---|
| `harness/` | the package (`run`, `compare`, `gate`, `matrix`, `monitor`, `runs`, `report`, `stats`, `storage`, adapters) |
| `fixtures/runs/` | committed fixture runs for the CI gate |
| `tests/` | unit tests; `tests/fixtures/p1/` is a frozen copy of project 1's scoring code and one full results file, for the parity test |
| `scripts/ci.sh`, `scripts/make_fixtures.py` | CI steps; fixture builder (read-only on project 1) |
| `docs/runbook.md`, `docs/engagement.md`, `docs/spec.md` | operating guide, engagement write-up, spec |

## Known limits

- One run per model: with 60 cases a gap of a few points is usually noise. The compare and matrix reports say so per pair.
- A strict per-case must-pass blocks on noise too; the noise-floor report prints the per-category rate, and `--rerun-lost K` reruns lost cases before blocking (live only).
- The monitor is valid from about 80 requests a day, Monday to Saturday (measured on synthetic logs); below that it reports numbers but its alerts can miss real drift.
- Project 4's adapter matches its code and contract but has not run live (its warehouse database was not up during the build).
- The recorded fixtures carry the model name as given at import and the current case-file and scoring-code hashes, and no clock (wall clock); project 1 stored none of these per run. A live run (frozen clock) therefore cannot be gated against them.

## How this was built

Built with Claude Code as a pair programmer. Scope, design decisions and verification are mine. Larkfield Coffee Co. is a fictional customer; all data is synthetic. MIT licensed (see `LICENSE`).
