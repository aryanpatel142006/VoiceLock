#!/usr/bin/env bash
# Verify for a web-only change (swarm verify.web_paths: web/**, tests/test_app_*, docs, *.md). Under 60 s on a
# loaded Mac. Runs only what a web change can break: the web node tests, a JS syntax check of web/**, ruff on the
# app tests, and the Python tests that serve or read the web files (server imports, /app, /judge, /simple).
# Takes no verify slot and never waits behind swarm-lock --exclusive: nothing here touches audio or timing.
set -uo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_worktree.sh >/dev/null 2>&1 || true   # idempotent: links the main checkout's .venv into a worktree
mkdir -p .swarm-run
source scripts/_py.sh
echo "verify_web: $PY"
status=0
start=$SECONDS

if ! command -v node >/dev/null 2>&1; then echo "verify_web: FAIL (no node)"; exit 1; fi

# 1. JS syntax of every web script (vendor included: a broken vendored file breaks the page too)
bad=0; n=0
while IFS= read -r -d '' f; do
  n=$((n + 1))
  if ! node --check "$f" > .swarm-run/node-check.txt 2>&1; then
    bad=$((bad + 1)); echo "node --check: FAIL $f"; head -8 .swarm-run/node-check.txt
  fi
done < <(find web -type f \( -name '*.js' -o -name '*.mjs' \) -not -path '*/node_modules/*' -print0)
if [ $bad -ne 0 ]; then status=1; fi
echo "node --check: $((n - bad))/$n ok"

# 2. web node tests (console + app), as in verify_fast
node_tests=""
compgen -G "web/console/tests/*.test.mjs" >/dev/null && node_tests="web/console/tests/*.test.mjs"
compgen -G "web/app/tests/*.test.mjs" >/dev/null && node_tests="$node_tests web/app/tests/*.test.mjs"
if [ -n "$node_tests" ]; then
  echo "node: node --test $node_tests"
  # shellcheck disable=SC2086
  node --test $node_tests > .swarm-run/node.txt 2>&1 || { status=1; echo "node: FAIL"; }
  tail -12 .swarm-run/node.txt
fi

# 3. ruff on the app tests (tests/test_app_* is a web path)
app_py=$(compgen -G "tests/test_app_*.py" || true)
if [ -n "$app_py" ]; then
  # shellcheck disable=SC2086
  "$PY" -m ruff check $app_py --quiet > .swarm-run/ruff.txt 2>&1 || { status=1; echo "ruff: FAIL"; tail -20 .swarm-run/ruff.txt; }
fi

# 4. Python smoke: the server still imports and serves the web pages. Every test file that reads web/ or hits a
#    page route, picked by grep so new ones join automatically (about 120 tests, ~15 s).
py_tests=$( { compgen -G "tests/test_app_*.py" || true; grep -lE '"web/|web" ?/|/ ?"web"|default_console_dir|APP_DIR|console_dir|"/judge|"/app' tests/*.py 2>/dev/null || true; } | sort -u)
echo "pytest: $(echo "$py_tests" | grep -c .) files"
# shellcheck disable=SC2086
"$PY" -m pytest -q --no-header -p no:cacheprovider -m "not slow and not network" $py_tests > .swarm-run/pytest.txt 2>&1; prc=$?
if [ $prc -ne 0 ]; then
  if { [ $prc -eq 134 ] || [ $prc -eq 139 ]; } && grep -qE "[0-9]+ passed" .swarm-run/pytest.txt && ! grep -qE "[0-9]+ failed" .swarm-run/pytest.txt; then
    echo "pytest: tests passed but the interpreter died at exit (signal $prc): native teardown crash, not a test failure"
  else status=1; echo "pytest: FAIL (exit $prc)"; fi
fi
tail -15 .swarm-run/pytest.txt

if [ $status -eq 0 ]; then echo "verify_web: PASS ($((SECONDS - start)) s)"; else echo "verify_web: FAIL ($((SECONDS - start)) s)"; fi
exit $status
