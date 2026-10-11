#!/usr/bin/env bash
# One-command setup for a clean clone (laptops and CI). Idempotent: a second run skips pip when nothing changed.
#   bash scripts/setup.sh          # .venv + -e .[dev]
#   bash scripts/setup.sh --all    # also every optional extra declared in pyproject.toml (read with tomllib)
#   bash scripts/setup.sh --force  # reinstall even if nothing changed (or into a linked worktree .venv)
set -euo pipefail
cd "$(dirname "$0")/.."

ALL=0; FORCE=0
for arg in "$@"; do
  case "$arg" in
    --all) ALL=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "setup: unknown argument $arg" >&2; exit 2 ;;
  esac
done

os_hint() {  # os_hint <brew formula> <apt package>
  if [ "$(uname -s)" = "Darwin" ]; then echo "brew install $1"; else echo "sudo apt-get install -y $2"; fi
}

# 1. Interpreter: python3.12, else python3 >= 3.11 with a warning.
if [ ! -x .venv/bin/python ]; then
  BASE=""
  if command -v python3.12 >/dev/null 2>&1; then
    BASE=python3.12
  elif command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    BASE=python3
    echo "setup: WARNING python3.12 not found, using $(python3 -V 2>&1); the project targets 3.12 ($(os_hint python@3.12 python3.12))" >&2
  else
    echo "setup: ERROR need python3.12 (or python3 >= 3.11): $(os_hint python@3.12 python3.12)" >&2
    exit 1
  fi
  [ -L .venv ] && rm .venv  # dangling symlink left by setup_worktree.sh
  echo "setup: creating .venv with $BASE ($("$BASE" -V 2>&1))"
  "$BASE" -m venv .venv
fi
PY=.venv/bin/python
echo "setup: python $("$PY" -c 'import sys; print(sys.version.split()[0], sys.prefix)')"
[ -L .venv ] && echo "setup: note .venv is linked to $(cd .venv && pwd -P) (shared with the main checkout)"

# 2. Packages. Extras come from pyproject.toml so new extras added by other tasks are picked up by --all.
EXTRAS="dev"
if [ "$ALL" = 1 ]; then
  EXTRAS="$("$PY" -c 'import tomllib; print(",".join(tomllib.load(open("pyproject.toml", "rb"))["project"].get("optional-dependencies", {})))')"
  EXTRAS="${EXTRAS:-dev}"
fi
STAMP=.venv/.hearing-setup-stamp
WANT="$("$PY" -c 'import hashlib, sys; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())' < pyproject.toml) extras=$EXTRAS src=$(pwd -P)"
if [ "$FORCE" = 0 ] && [ -L .venv ]; then
  # An editable install from here would repoint the shared venv at this worktree; the main checkout owns it.
  echo "setup: skipping pip in a linked .venv; run scripts/setup.sh in the main checkout (or pass --force)"
elif [ "$FORCE" = 0 ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ] && "$PY" -c 'import hearing' >/dev/null 2>&1; then
  echo "setup: packages up to date (.[$EXTRAS]); use --force to reinstall"
else
  echo "setup: installing -e .[$EXTRAS]"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -e ".[$EXTRAS]"
  echo "$WANT" > "$STAMP"
fi

# 3. System libraries. ffmpeg is a warning only; libsndfile is needed by soundfile.
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg: ok ($(command -v ffmpeg))"
else
  echo "ffmpeg: MISSING (video fixtures and AV models need it): $(os_hint ffmpeg ffmpeg)"
fi
if "$PY" -c 'import soundfile' >/dev/null 2>&1; then
  echo "libsndfile: ok (soundfile $("$PY" -c 'import soundfile; print(soundfile.__libsndfile_version__)'))"
else
  echo "libsndfile: MISSING (import soundfile failed): $(os_hint libsndfile libsndfile1)"
fi

# 3b. One chromium for every worktree's scripts/screenshot.sh (HEARING_PLAYWRIGHT in scripts/_py.sh).
if "$PY" -c 'import playwright' >/dev/null 2>&1; then
  "$PY" -m playwright install chromium >/dev/null 2>&1 || echo "playwright: chromium install failed; run: $PY -m playwright install chromium"
fi

# 4. Compute device (hearing.device is written by another task; tolerate its absence).
if "$PY" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("hearing.device") is None)' >/dev/null 2>&1; then
  if out="$("$PY" -m hearing.device 2>&1)"; then
    out="$(printf '%s\n' "$out" | tail -1)"
    case "$out" in device*) echo "$out" ;; *) echo "device: $out" ;; esac
  else
    echo "device: error running python -m hearing.device ($(printf '%s\n' "$out" | tail -1))"
  fi
else
  echo "device: not available yet"
fi
echo "setup: done. Next: bash scripts/verify_fast.sh"
