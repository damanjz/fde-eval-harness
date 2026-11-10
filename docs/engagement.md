# Engagement write-up: Larkfield eval and monitoring harness

Larkfield Coffee Co. is fictional. The code, tests and findings below are real. Every figure comes from this repo's tests, from project 1's recorded eval runs, from checks run during the build, or from the live runs of 2026-11-10 (`runs.db`, `reports/matrix.md`, `reports/noise-floor.md`).

## 1. Discovery

**Who asked.** Larkfield's owner of its AI features (support agent today; ask-your-data, document extraction and voice to come) and whoever ships prompt, model or data changes. The gate output is read by that engineer at merge time. The monitor report is read by the support lead.

**The problem.** Every prompt, model or data change can silently break a feature, and each feature measured itself in its own way. Nobody could say "is this change safe?" or "did something drift in production?" with one answer.

**Success metric agreed.**
1. One command runs any feature's golden set against any local model, and an interrupted run resumes.
2. A regression blocks the merge: the overall pass count drops by more than a tolerance, or a must-pass category (refuse) loses a case.
3. Every comparison says what is noise, in plain words, with a real statistical test.
4. Production drift shows per day, with alerts that are proved on a log where the drift is planted.
5. The harness measures what each feature's own eval measures (proved by parity on recorded results).

**Assumptions made.**
- Pass or fail is each feature's own exact scoring. No LLM judge in the gate: project 1's judge uses the model under test, so it would move with the candidate.
- Local models only (`fusion27b:iq3`, `fusion27b:q4ks`, `qwen3:14b`), one after another, because the GPU is shared.
- One run per model. Project 1 runs at temperature 0.2, so identical runs flip cases; the noise floor is measured, not assumed.
- Recorded project 1 runs are a fair CI fixture, because CI cannot run the models.
- Day boundaries are IST.
- Reports are written for an engineer who can read a table, not a statistician.

**Left out.**
- The LLM judge (above), and project 1's keyword and no-retrieval baselines in the matrix (agent variant only).
- A price table for paid models: the cost field and flags exist, with no invented prices.
- Alert delivery (email, Slack), dashboards beyond static pages, parallel runs.
- A GitHub repo, so no CI badge and no run of the workflow on GitHub.
- A live run of the ask-your-data adapter (its warehouse database was not up during the build).

## 2. What was built

| Piece | What it does |
|---|---|
| Adapters | `support_agent` calls project 1's own `evals.run.run_case` and `evals.scoring.score_case`; `ask_your_data` calls project 4's `app.agent.ask` and `evals.scoring.score_case`; `fake` is synthetic, for tests and CI |
| Run | per-case timeout, retries on exceptions, resume, preflight, untimed warm-up, stops after 5 errors in a row, waits for the thread of a timed-out case (flags the next case if it is still busy), frozen clock and scoring-code hash recorded |
| Storage | SQLite `runs.db`; runs export to `.run.json` files |
| Compare | per-category deltas, flips, latency and token deltas, exact McNemar and paired bootstrap, plain-words explanation, noise-floor mode, HTML report |
| Gate | exit 0, 1 or 2; strict per-case must-pass; errors count as failures; full-suite coverage, scorer and clock checks; optional rerun of lost must-pass cases |
| Matrix | one subprocess per model, `reports/matrix.md` and `.html` with inline SVG |
| Monitor | per-day metrics from `request_log`, alerts against the trailing 7-day median |
| CI | `scripts/ci.sh` run by `.github/workflows/ci.yml`: lint, tests, gate on fixtures with asserted exit codes, smoke |

## 3. What building it found

| Finding | What was done |
|---|---|
| Project 1 turns every exception into its hand-off message. A refuse case expects an escalation, so a dead Ollama, or a slow model whose 120 s request timed out on the second call, could pass refuse cases. A first fix keyed on zero LLM calls missed the second-call case (review, 2026-11-10). | The adapter hooks project 1's own exception path (the "turn failed, handing off" log line, per thread) and classifies the exception: an empty model answer is a model failure; a failed or timed-out request, a database error or anything else is an infrastructure error, recorded as an error and never a pass. This deliberately deviates from project 1's own score on that reply. Preflight and warm-up also refuse to start without a working model. |
| Orders have fixed dates but project 1's tools read the wall clock, so return-window cases (T09 and T10) move with the calendar between runs. | The adapter freezes project 1's `clock_now` to the dataset's NOW (2026-11-10T09:00:00+05:30), records it in the run, and the gate exits 2 when two runs' clocks differ. |
| Two runs could differ in scoring code without anyone noticing. | `scorer_hash` (hash of `evals/scoring.py`) is part of the run's fingerprint and the gate exits 2 on a mismatch or a missing hash. |
| The first gate compared case-id sets only, so two `--limit` runs with the same ids could pass. | Each run stores the case-file size and its limit and ids; the gate requires every run to cover the whole case file. Zero results, a missing hash, a malformed or duplicate-id run file are exit 2, and a crash is never exit 1. |
| Project 1's JSONL log has no timestamp, so "per day" is impossible from it. | The monitor reads `request_log` (`created_at`), and falls back to the file only when the database is down, saying so. |
| When first run, all 1,097 rows of `request_log` were test traffic (eval-, sim-, smoke, warmup). By 2026-11-10 14:53 there were 73 non-test requests on one day. | Test traffic is excluded by default (1,208 rows excluded in the latest report). The report shows the one day, marks it as under the valid minimum volume, and no alert can fire yet. |
| Between project 1's runs 3 and 4 the overall score moved 55/60 to 54/60 (4 cases worse, 3 better, McNemar p = 1.000: noise). But one of the four was refuse case R10. | This is the gate's design case: it fails on the lost refuse case although the overall drop is inside tolerance. It is a committed fixture pair with an asserted exit code. |
| Threshold-only alerts gave a false alarm on 3 of the first 12 seeded synthetic logs: small daily samples cross a 5 or 10 point threshold by chance. | Rate alerts must also be statistically unlikely (one-sided exact binomial, p below 0.0001, against the pooled rate of the trailing week, computed in log space). With per-rule minimum volumes, Sundays compared with Sundays and a day in progress skipped, 100 seeded logs at 120 requests a day (Sundays a third of that) had the planted drift fire every time and no stray alert. |
| Sweeping the daily volume showed where the monitor stops being reliable: the planted 8% to 35% hand-off spike was found in 2.5% of logs at 10 requests a day, 32.5% at 20, 62.5% at 40, 95% at 60, 100% from 80; the 2.2x latency step from 30 a day (40 synthetic logs per volume). | The report states a minimum valid volume of about 80 requests a day (Monday to Saturday), marks days below it, and stays quiet rather than guessing at low volume (tested at 12 a day). |
| A strict per-case must-pass blocks on noise as well as on regressions. | The noise-floor report prints the per-category flip rate and the false-block rate it implies; `--rerun-lost K` reruns lost must-pass cases before blocking. Measured on the iq3 rerun: 1 of 14 passing tool cases failed on the rerun (about 65% of reruns would be blocked if that holds), no refuse case flipped (rate bounded below about 30%, not measured). |
| Project 1 stored neither the model nor the case file per run. | Fixtures say so in their config; the case-file hash is the current `cases.jsonl`, stamped at import. |
| Project 1 and project 4 both have top-level `app` and `evals` packages. | One suite per process, enforced by an error, with a subprocess per model in the matrix. |

## 4. How the numbers were checked
- McNemar p-values equal an independent brute-force computation with exact fractions over a grid of flip counts.
- The bootstrap interval is exact and has no seed; it matches the normal approximation for a paired difference (within 0.02) and has zero width for identical runs.
- Wilson intervals match known values; the smallest significant net change was checked as a boundary for 6 to 39 flips.
- Parity: all 60 replies in project 1's latest complete agent results, scored through the adapter, reproduce its recorded pass or fail, checks and reasons, with a frozen copy of its scoring code (runs in CI) and with its current code (local).
- The bootstrap interval is exact (every resample enumerated) and is checked against brute-force enumeration of all 3^n draws for small n, and against a 200,000-resample Monte Carlo for a 50-case run.
- All 60 real cases ran through `harness.run` on project 1's real retrieval, tools and database with its LLM mocked before any live run: no errors, project 1's provenance recorded, and a case-file hash equal to the fixtures'. (Live runs use the frozen clock, so they are gated against live baselines, not the wall-clock fixtures.)
- The CI steps pass locally. The gate step asserts exit 0 on an improvement within noise, 1 on two kinds of regression, and 2 on a limit run.

## 5. Results

Live runs, 2026-11-10: project 1 commit `8ae4bb9`, the same 60 cases, project 1's exact scoring, frozen clock 2026-11-10T09:00:00+05:30, one run per model. Two earlier runs in `runs.db` (`20261110-052006` iq3, made before a project 1 commit, and `20261110-053950` q4ks, abandoned at a reboot) are not used.

| Model | Passed | 95% interval | answer / refuse / tool | p50 / p95 |
|---|---|---|---|---|
| `fusion27b:iq3` | 53/60 (88%) | 78% to 94% | 29/35, 10/10, 14/15 | 16.2s / 29.2s (first run, GPU shared with other work) |
| `fusion27b:q4ks` | 55/60 (92%), one case timed out and counts as a failure | 82% to 96% | 32/35, 10/10, 13/15 | 22.9s / 136.5s |
| `qwen3:14b` | 48/60 (80%) | 68% to 88% | 29/35, 8/10, 11/15 | 5.9s / 9.1s |

| Question | Answer |
|---|---|
| iq3 against q4ks | within noise (53 against 55; 4 better, 2 worse; p = 0.688): the larger, slower q4ks buys nothing measurable |
| iq3 against qwen3:14b | within noise on the overall count (p = 0.227), but `qwen3:14b` lost 2 refuse cases (R03, R05): a `--must-pass refuse` gate exits 1 (overall drop 5 as well) |
| q4ks against qwen3:14b | borderline (p = 0.065; `qwen3:14b` worse on 9 cases, better on 2) |
| How much faster is qwen3:14b? | about 1.2 times, not 2.7 times: 5.9s against the iq3 rerun's clean 7.1s p50; the 16.2s iq3 matrix figure is inflated by the shared GPU |
| Noise floor (iq3 twice: `20261110-133948`, `20261110-143744`) | 4 of 60 cases flipped (1 worse, 3 better), net swing 2, McNemar p = 0.625: within noise |
| Recommended `--max-drop` | 4: the net change that cannot be told from noise at 4 flips; one pair, so repeat to firm it up. The CLI default is still 2. |
| Per-case must-pass on noise | tool: 1 of 14 passing cases failed on the rerun, so a strict `tool` gate would block about 65% of reruns of an unchanged model; refuse: no flip (bounded below about 30%). Use `--rerun-lost` for categories that flip. |

## 6. Known limits
- With 60 cases, a few points of accuracy is usually noise. One run per model.
- A thread cannot be killed: a timed-out case keeps running in the background until the model client's own timeout ends it.
- The exception hook is per turn and covers multi-turn cases; it relies on project 1 keeping the "turn failed, handing off" log line and the `LLMError("empty answer")` text.
- Monitoring depends on real traffic. So far there is one day with 73 non-test requests, under the valid minimum of about 80 a day, so no alert can fire; the alert logic is proved on the synthetic log, not on production.
- Latency: the first iq3 run shared the GPU with other work (p50 16.2 s, against 7.1 s in the rerun of the same configuration), so latency from the matrix is indicative, and the rerun's iq3 figures are the clean ones.
- A recent spike raises the pooled baseline for a week, so a second smaller spike right after it can be masked.

## 7. What I would do next for a real customer
- Run each model three times and report the spread, instead of one run plus a noise-floor pair.
- Schedule the monitor and send alerts to where the team already looks.
- Add document extraction and voice as suites; each is an adapter and a fixture pair.
- Add a pinned judge model as a separate, non-gating column for answer quality.
- Keep the gate's must-pass list per suite in a file the CI reads, so the policy is reviewed like code.
