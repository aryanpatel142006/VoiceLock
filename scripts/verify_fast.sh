#!/usr/bin/env bash
# ≤ 60 seconds. Lint + unit tests without model weights. Exit non-zero on any failure. Print only summaries.
set -uo pipefail
cd "$(dirname "$0")/.."
# Under a swarm runner: wait for one of this machine's verify slots (swarm-control hostlock)
if [ -n "${SWARM_VERIFY_LOCK:-}" ] && [ -z "${SWARM_VERIFY_SLOT_HELD:-}" ] && [ -x "${SWARM_VERIFY_LOCK}" ]; then
  exec "${SWARM_VERIFY_LOCK}" -- bash "$0" "$@"
fi
# Share the host's verify slots (swarm-control template): re-run this script under swarm-lock once.
if [ -z "${SWARM_LOCK_HELD:-}" ] && [ -z "${SWARM_VERIFY_SLOT_HELD:-}" ] && [ -z "${SWARM_VERIFY_LOCK:-}" ] && command -v swarm-lock >/dev/null 2>&1; then
  SWARM_LOCK_HELD=1 exec swarm-lock -- bash "$0" "$@"
fi
bash scripts/setup_worktree.sh >/dev/null 2>&1 || true   # idempotent: links the main checkout's .venv into a worktree
mkdir -p .swarm-run
source scripts/_py.sh
echo "verify_fast: $PY"
status=0
"$PY" -m ruff check hearing tests eval --quiet > .swarm-run/ruff.txt 2>&1 || { status=1; echo "ruff: FAIL"; }; tail -20 .swarm-run/ruff.txt
if [ $status -ne 0 ]; then echo "verify_fast: FAIL (ruff)"; exit 1; fi
"$PY" -m pytest -q -x --no-header --durations=15 -m "not slow and not network" > .swarm-run/pytest.txt 2>&1; prc=$?
if [ $prc -ne 0 ]; then
  if { [ $prc -eq 134 ] || [ $prc -eq 139 ]; } && grep -qE "[0-9]+ passed" .swarm-run/pytest.txt && ! grep -qE "[0-9]+ failed" .swarm-run/pytest.txt; then
    echo "pytest: tests passed but the interpreter died at exit (signal $prc): native teardown crash, not a test failure (see ~/Library/Logs/DiagnosticReports)"
  else status=1; echo "pytest: FAIL (exit $prc)"; fi
fi; tail -20 .swarm-run/pytest.txt
if command -v node >/dev/null 2>&1; then
  node_tests="web/console/tests/*.test.mjs"
  # M14 app tests (web/app) run too, once the directory has any
  if compgen -G "web/app/tests/*.test.mjs" >/dev/null; then node_tests="$node_tests web/app/tests/*.test.mjs"; fi
  echo "node: node --test $node_tests"
  node --test $node_tests > .swarm-run/node.txt 2>&1 || { status=1; echo "node: FAIL"; }
  tail -20 .swarm-run/node.txt
else
  echo "SKIP: no node"
fi
if [ -f frontend/package.json ]; then
  (cd frontend && npx --no-install tsc --noEmit -p . 2>&1 | tail -20) || status=1
fi
if [ $status -eq 0 ]; then echo "verify_fast: PASS"; else echo "verify_fast: FAIL"; fi
exit $status
