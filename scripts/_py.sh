#!/usr/bin/env bash
# Resolve the project interpreter from any checkout or worktree. Sourced by the verify scripts.
# Order: $PYTHON, this checkout's .venv, the main checkout's .venv (worktrees link or share it), then python3.
# zsh (the agents' shell) leaves BASH_SOURCE empty: fall back to the git toplevel (Q-038/Q-039/Q-051)
_src="${BASH_SOURCE[0]:-}"
if [ -n "$_src" ]; then _here="$(cd "$(dirname "$_src")/.." && pwd)"; else _here="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"; fi
_main="$(git -C "$_here" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)/.."
if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then PY="$PYTHON"
elif [ -x "$_here/.venv/bin/python" ]; then PY="$_here/.venv/bin/python"
elif [ -x "$_main/.venv/bin/python" ]; then PY="$_main/.venv/bin/python"
else PY="$(command -v python3)"; fi
export PY

# Model weights live in the main checkout; worktrees must see the same cache (Q-019/Q-021/Q-023).
export HEARING_MODELS_DIR="${HEARING_MODELS_DIR:-$_main/models}"
mkdir -p "$HEARING_MODELS_DIR" 2>/dev/null || true
# Reinstall the editable package from the MAIN checkout when pyproject.toml changed since the last install
# (Q-011: merged tasks added deps the shared venv never got). Never run pip from a worktree: the editable
# install would re-point at the worktree path.
_stamp="$_main/.venv/.pyproject.md5"
# A failed install records its pyproject hash: later sourcings (demo.sh inside a 30 s test, every launcher) warn
# instead of re-running pip (T-205: CI's failed reinstall was retried in each demo.sh subprocess). A changed
# pyproject.toml or HEARING_REINSTALL=1 tries again.
_failed="$_main/.venv/.pyproject.failed"
_now="$(md5 -q "$_main/pyproject.toml" 2>/dev/null || md5sum "$_main/pyproject.toml" | cut -d' ' -f1)"
_deps_due() { [ "$(cat "$_stamp" 2>/dev/null)" != "$_now" ] && { [ "${HEARING_REINSTALL:-}" = 1 ] || [ "$(cat "$_failed" 2>/dev/null)" != "$_now" ]; }; }
if [ -x "$_main/.venv/bin/pip" ] && _deps_due; then
  # one installer at a time: a parallel `pip install` left torch half-installed under another task (Q-037)
  _lock="$_main/.venv/.install.lock.d"; _waited=0
  until mkdir "$_lock" 2>/dev/null; do sleep 2; _waited=$((_waited+2)); [ $_waited -ge 600 ] && { echo "WARN: install lock held >10 min; removing"; rmdir "$_lock" 2>/dev/null; }; done
  if ! _deps_due; then :   # the lock holder just installed (or failed) this pyproject.toml
  elif (cd "$_main" && "$_main/.venv/bin/pip" install -q -e ".[dev,eval,ml]" > "$_main/.venv/.install.log" 2>&1); then
    echo "$_now" > "$_stamp"; rm -f "$_failed"
  else
    echo "$_now" > "$_failed"
    echo "WARN: dependency reinstall failed (see $_main/.venv/.install.log); tests may miss packages"   # never silent (Q-045)
  fi
  rmdir "$_lock" 2>/dev/null || true
elif [ -x "$_main/.venv/bin/pip" ] && [ "$(cat "$_stamp" 2>/dev/null)" != "$_now" ]; then
  echo "WARN: dependency reinstall failed for this pyproject.toml (see $_main/.venv/.install.log); not retrying, HEARING_REINSTALL=1 retries"
fi
unset -f _deps_due; unset HEARING_REINSTALL   # one explicit retry: launchers started from here must not inherit it

# Benchmark JSON must survive the worktree that produced it (Q-111: T-024's raw results vanished with its worktree).
export HEARING_RESULTS_DIR="${HEARING_RESULTS_DIR:-$_main/eval/results}"
mkdir -p "$HEARING_RESULTS_DIR" 2>/dev/null || true

# Generated fixtures (GRID/LibriSpeech, ~2 min + network) are built once in the main checkout and shared.
export HEARING_FIXTURES_DIR="${HEARING_FIXTURES_DIR:-$_main/eval/fixtures/generated}"
mkdir -p "$HEARING_FIXTURES_DIR" 2>/dev/null || true

# One pinned Playwright: python playwright in the shared .venv, chromium installed once by scripts/setup.sh.
export HEARING_PLAYWRIGHT="${HEARING_PLAYWRIGHT:-$_main/.venv/bin/python}"
