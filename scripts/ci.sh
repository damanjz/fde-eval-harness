#!/usr/bin/env bash
# The CI steps, runnable locally and by .github/workflows/ci.yml (one source of truth).
#   bash scripts/ci.sh lint | tests | gate | smoke | all
# Set PYTHON to pick the interpreter (default: python). Reports land in ci-reports/.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
FX=fixtures/runs
RUN2="$FX/support_agent-p1-run2-agent.run.json"
RUN3="$FX/support_agent-p1-run3-agent.run.json"
RUN4="$FX/support_agent-p1-run4-agent.run.json"
RUN4_NONE="$FX/support_agent-p1-run4-no-retrieval.run.json"
RUN4_FIRST20="$FX/support_agent-p1-run4-agent-first20.run.json"
mkdir -p ci-reports

# expect EXIT_CODE command...: fail the step unless the command exits with exactly that code
expect() {
  local want="$1"; shift
  local got=0
  "$@" || got=$?
  if [ "$got" -ne "$want" ]; then
    echo "FAIL: expected exit $want, got $got: $*" >&2
    exit 1
  fi
  echo "ok: exit $got as expected"
}

summary() { if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then echo "--md $GITHUB_STEP_SUMMARY"; else echo "--md ci-reports/gate-summary.md"; fi; }

lint() {
  "$PYTHON" -m ruff check .
  "$PYTHON" -m ruff format --check .
}

tests() {
  "$PYTHON" -m pytest -q
}

# The gate on committed fixture runs: real recorded project 1 runs (CI cannot run the models).
gate() {
  local md; md=$(summary)
  echo "== 1. an improvement within noise passes (exit 0)"
  # shellcheck disable=SC2086
  expect 0 "$PYTHON" -m harness.gate --baseline "$RUN2" --candidate "$RUN3" --max-drop 2 --must-pass refuse $md
  echo "== 2. a refusal regression is blocked although the overall drop is inside --max-drop (exit 1)"
  # shellcheck disable=SC2086
  expect 1 "$PYTHON" -m harness.gate --baseline "$RUN3" --candidate "$RUN4" --max-drop 2 --must-pass refuse $md
  echo "== 3. the same pair without must-pass is tolerated (exit 0)"
  expect 0 "$PYTHON" -m harness.gate --baseline "$RUN3" --candidate "$RUN4" --max-drop 2
  echo "== 4. a large overall drop is blocked (exit 1)"
  # shellcheck disable=SC2086
  expect 1 "$PYTHON" -m harness.gate --baseline "$RUN4" --candidate "$RUN4_NONE" --max-drop 2 --must-pass refuse $md
  echo "== 5. a --limit style run cannot be gated against a full run (exit 2)"
  # shellcheck disable=SC2086
  expect 2 "$PYTHON" -m harness.gate --baseline "$RUN4" --candidate "$RUN4_FIRST20" --max-drop 2 $md
  echo "== compare report on the blocked pair"
  "$PYTHON" -m harness.compare "$RUN3" "$RUN4" --html ci-reports/compare-run3-vs-run4.html --md ci-reports/compare-run3-vs-run4.md > /dev/null
}

# End to end on the synthetic suite: run, resume, compare, matrix, monitor alert proof.
smoke() {
  local tmp; tmp=$(mktemp -d)
  local db="$tmp/smoke.db"
  "$PYTHON" -m harness.run --suite fake --model fake-strong --db "$db" --no-preflight
  "$PYTHON" -m harness.run --suite fake --model fake-mid --db "$db" --no-preflight
  "$PYTHON" -m harness.compare "$("$PYTHON" -m harness.runs --db "$db" list | awk '/fake-strong/ {print $1}')" \
    "$("$PYTHON" -m harness.runs --db "$db" list | awk '/fake-mid/ {print $1}')" --db "$db" > /dev/null
  "$PYTHON" -m harness.matrix --suite fake --models fake-strong,fake-mid,fake-weak --db "$db" --out ci-reports/smoke \
    --no-preflight --resume
  test -s ci-reports/smoke/matrix.md && test -s ci-reports/smoke/matrix.html
  "$PYTHON" -m harness.monitor --self-test
  rm -rf "$tmp"
}

case "${1:-all}" in
  lint) lint ;;
  tests) tests ;;
  gate) gate ;;
  smoke) smoke ;;
  all) lint; tests; gate; smoke ;;
  *) echo "usage: bash scripts/ci.sh lint|tests|gate|smoke|all" >&2; exit 2 ;;
esac
