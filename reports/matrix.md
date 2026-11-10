# Model matrix: support_agent

Generated 2026-11-10 14:37. 60 cases per model, one run each. Pass = the feature's own exact scoring.

## Accuracy

| model | passed | 95% interval (Wilson) | answer | refuse | tool | errored |
|---|---|---|---|---|---|---|
| fusion27b:iq3 | 53/60 (88%) | 78% to 94% | 29/35 (83%) | 10/10 (100%) | 14/15 (93%) | 0 |
| fusion27b:q4ks | 55/60 (92%) | 82% to 96% | 32/35 (91%) | 10/10 (100%) | 13/15 (87%) | 1 |
| qwen3:14b | 48/60 (80%) | 68% to 88% | 29/35 (83%) | 8/10 (80%) | 11/15 (73%) | 0 |

## Latency, tokens, cost

| model | p50 | p95 | prompt tok/case | completion tok/case | cost/case | total cost |
|---|---|---|---|---|---|---|
| fusion27b:iq3 | 16.2s | 29.2s | 4276 | 92 | INR 0.00 | INR 0.00 |
| fusion27b:q4ks | 22.9s | 136.5s | 4419 | 98 | INR 0.00 | INR 0.00 |
| qwen3:14b | 5.9s | 9.1s | 3949 | 87 | INR 0.00 | INR 0.00 |

Latency is wall time per case measured by the harness, over cases that completed. Local models cost INR 0; set --price-in and --price-out for paid models.

## Are the differences real?

Paired by case. A flip is a case one model passes and the other fails. McNemar is an exact test on the flips; the bootstrap interval is for accuracy(B) minus accuracy(A). With 60 cases, a few points of difference is usually noise.

| A | B | A passed | B passed | flips (B better / B worse) | McNemar p | bootstrap 95% (points) | reading |
|---|---|---|---|---|---|---|---|
| fusion27b:iq3 | fusion27b:q4ks | 53 | 55 | 4 / 2 | 0.688 | -5.0 to +11.7 | within noise |
| fusion27b:iq3 | qwen3:14b | 53 | 48 | 3 / 8 | 0.227 | -18.3 to +1.7 | within noise |
| fusion27b:q4ks | qwen3:14b | 55 | 48 | 2 / 9 | 0.065 | -21.7 to -1.7 | borderline |

## Runs

| model | run | status | git | prompt | code | started |
|---|---|---|---|---|---|---|
| fusion27b:iq3 | 20261110-133948-support-agent-fusion27b-iq3 | complete | 8ae4bb9 | f559ab1a9b7d | a30d709e628f | 2026-11-10 13:39 |
| fusion27b:q4ks | 20261110-135652-support-agent-fusion27b-q4ks | partial | 8ae4bb9 | f559ab1a9b7d | a30d709e628f | 2026-11-10 13:56 |
| qwen3:14b | 20261110-143043-support-agent-qwen3-14b | complete | 8ae4bb9 | f559ab1a9b7d | a30d709e628f | 2026-11-10 14:30 |

## Notes

- fusion27b:q4ks: run 20261110-135652-support-agent-fusion27b-q4ks is partial; errored or missing cases count as failures.
- Not in the matrix or incomplete: fusion27b:q4ks (run incomplete; rerun with --resume)
