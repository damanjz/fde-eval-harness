# Monitor: support agent

Generated 2026-11-10 14:53.

- Source: request_log table (Postgres, read-only; created_at from the database)
- Requests read: 1281. Excluded as test traffic: 1208 (session ids starting eval-, sim-, smoke, warmup).
- Requests reported: 73 over 1 day(s).
- Excluded test traffic by session prefix: eval- 967, sim- 229, smoke 11, warmup 1; dates 2026-11-09 to 2026-11-10.

## Per day

| day | requests | hand-off | guardrail flags | tool errors | p50 | p95 | citations per answer | top retrieved articles |
|---|---|---|---|---|---|---|---|---|
| 2026-11-10 | 73* | 6.8% | 0.0% | 0.0% (60 calls) | 16.5s | 38.9s | 0.60 | KB-002 (36), KB-001 (26), KB-012 (22) |

\* Fewer than 80 requests: shown, but too few for the alert rules to be reliable (see Alerts).

## Alerts

None. An alert needs at least 3 earlier days with enough requests in the trailing 7 days (Sundays: 2 earlier Sundays), and enough requests on the day.

Valid for about 80+ requests a day. Below that the alert rules can miss real drift (a rate spike cannot be told from chance on a few dozen requests) and p95 and citation rules stay silent; the per-day numbers are still correct. Minimum volumes per rule: handoff_rate 10 (baseline days 10+); flag_rate 10 (baseline days 10+); tool_error_rate 10 (baseline days 10+); p50_ms 20 (baseline days 20+); p95_ms 50 (baseline days 50+); citations_per_answer 30 (baseline days 30+); requests any (baseline days 30+). Request volume compares Sundays with Sundays and Monday to Saturday with Monday to Saturday, and skips a day that is still going on.

Rules: hand-off rate up 10+ points; guardrail flag rate up 5+ points; tool error rate up 5+ points; p50 latency 1.5x the baseline; p95 latency 1.5x the baseline; citations per answer down 30%+; request volume doubled or halved. Rate rules also need the day's count to be unlikely by chance (one-sided binomial p below 0.0001).

## Definitions

- Hand-off: the reply was escalated (`escalated`). Guardrail flag: non-empty `output_flags` or `injection_suspected`.
- Tool error: a `tool_calls` entry whose `result` has an `error` key; the rate is errors over tool calls that day.
- Citations per answer: mean citations over requests that were not hand-offs.
- Baseline: median of the same metric over the previous 7 calendar days that have enough requests.
