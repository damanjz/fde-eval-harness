# Compare: 20261110-133948-support-agent-fusion27b-iq3  vs  20261110-143744-support-agent-fusion27b-iq3

- A: support_agent, fusion27b:iq3, 2026-11-10 13:39, variant agent, git 8ae4bb9, prompt f559ab1a9b7d, code a30d709e628f, scorer 56ade287dfab, clock 2026-11-10T09:00:00+05:30
- B: support_agent, fusion27b:iq3, 2026-11-10 14:37, variant agent, git 8ae4bb9, prompt f559ab1a9b7d, code a30d709e628f, scorer 56ade287dfab, clock 2026-11-10T09:00:00+05:30, "noise-floor-rerun"

## Verdict

WITHIN NOISE: the difference is too small to tell from chance.

| | A | B | change (B - A) |
|---|---|---|---|
| accuracy | 53/60 (88%) | 55/60 (92%) | +2 case(s), +3.3 points |
| 95% interval (Wilson) | 78% to 94% | 82% to 96% | bootstrap -3.3 to +10.0 points |
| flips | 1 worse | 3 better | McNemar p = 0.625 |
| errored or timed out | 0 | 0 | |

## In plain words

- Both runs were scored on the same 60 cases. 56 cases got the same result in both runs, so they say nothing about which run is better. The other 4 cases flipped.
- 3 case(s) went from fail to pass (B better) and 1 went from pass to fail (B worse): net +2 case(s). If the runs were equally good the flips would split about evenly (2 each way expected); the net 2 is how far this split is from even.
- McNemar exact test, p = 0.625. If the two runs were really equally good, each flip would be a coin toss, and a split at least this uneven (1 vs 3) would happen about 62% of the time by luck. Below 0.05 is conventionally called significant.
- With only 4 flips, no split can reach p < 0.05: this much data cannot settle it.
- Bootstrap check: every possible resample of the 60 cases was enumerated exactly (no random draws). Accuracy of B minus A is +3.3 points with a 95% interval of -3.3 to +10.0. The interval includes 0, so the true difference could be nothing.
- Limits: one run per model. LLM sampling can flip a case between two runs of the same model, so compare a model with itself once to see the noise floor. Per-category rows are small samples; read their p-values as hints, and expect a few false alarms when many categories are checked.

## By category

| category | cases | A | B | change | flips (better / worse) | p | reading |
|---|---|---|---|---|---|---|---|
| answer | 35 | 29/35 (83%) | 32/35 (91%) | +3 | 3 / 0 | 0.250 | weak: within noise |
| refuse | 10 | 10/10 (100%) | 10/10 (100%) | 0 | 0 / 0 | 1.000 | no flips |
| tool | 15 | 14/15 (93%) | 13/15 (87%) | -1 | 0 / 1 | 1.000 | weak: within noise |

## Flipped cases

| case | category | direction | why it failed |
|---|---|---|---|
| T11 | tool | pass -> fail | missing fact '30 days' |
| A13 | answer | fail -> pass | missing fact 'defective'; missing fact 'damaged' |
| A23 | answer | fail -> pass | missing fact 'not accepted' |
| A26 | answer | fail -> pass | missing fact 'checkout' |

## Latency, tokens, cost

| | A | B | change |
|---|---|---|---|
| p50 latency | 16.2s | 7.1s | -56% |
| p95 latency | 29.2s | 12.6s | -57% |
| prompt tokens / case | 4276 | 4342 | +2% |
| completion tokens / case | 92 | 88 | -4% |
| cost / case | INR 0.00 | INR 0.00 | 0.00 |

Latency is wall time per case measured by the harness, over cases that completed.

## Noise floor

- Same configuration, run twice: 4 of 60 cases flipped (1 pass to fail, 3 fail to pass), a net swing of 2 case(s). Nothing changed between the runs, so all of it is noise.
- Seen from a gate (first run as baseline, second as candidate), the overall drop was 0 case(s): a --max-drop of 0 or more would have let this pair through. Noise can swing the other way too.
- Statistically, with 4 flips a net change of up to 4 case(s) cannot be told from noise (p >= 0.05). Take the larger of 0 and 4 (4) as a starting point for --max-drop. It is one pair of runs, a small sample of the noise: repeat the rerun to firm it up.
- Per category (the cost of a strict per-case must-pass; compare with --rerun-lost K in harness.gate):
- answer: 3 of 35 cases flipped; none of the 29 cases that passed in the first run failed in the second; one pair only bounds the per-case rate below about 10% (rule of three), so the false-block rate is bounded, not measured.
- refuse: 0 of 10 cases flipped; none of the 10 cases that passed in the first run failed in the second; one pair only bounds the per-case rate below about 30% (rule of three), so the false-block rate is bounded, not measured.
- tool: 1 of 15 cases flipped; 1 of 14 cases that passed in the first run failed in the second (7%), so a strict per-case gate on this category would have blocked this pair; if that rate holds, about 65% of reruns of an unchanged configuration would be blocked.

## Notes

- Same model, prompt, code, scorer and clock: flips between these runs are pure run-to-run noise.

