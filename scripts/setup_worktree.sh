#!/usr/bin/env bash
# Called once per fresh worktree. Link heavy dependencies from the main checkout instead of reinstalling.
# Exits non-zero if the main checkout has no working .venv (run scripts/setup.sh there first).
set -euo pipefail
cd "$(dirname "$0")/.."
HERE="$(pwd -P)"
COMMON="$(git rev-parse --path-format=absolute --git-common-dir)"
MAIN="$(cd "$COMMON/.." && pwd -P)"
pyinfo() { .venv/bin/python -c 'import os, sys; print(os.path.realpath(sys.executable), sys.version.split()[0])'; }

if [ "$MAIN" = "$HERE" ]; then           # main checkout: nothing to link (never link .venv to itself)
  if [ ! -x .venv/bin/python ]; then
    echo "setup_worktree: main checkout has no working .venv; run: bash scripts/setup.sh" >&2
    exit 1
  fi
  echo "setup_worktree: main checkout, nothing to link; python: $(pyinfo)"
  exit 0
fi

if [ -d .venv ] && [ ! -L .venv ] && [ -x .venv/bin/python ]; then
  echo "setup_worktree: own .venv (not linked); python: $(pyinfo)"   # scripts/setup.sh was run in this worktree
  exit 0
fi

if [ ! -x "$MAIN/.venv/bin/python" ]; then
  echo "setup_worktree: $MAIN has no working .venv; run there: bash scripts/setup.sh" >&2
  exit 1
fi

# .gitignore's ".venv/" does not match a symlink; keep `git add -A` from committing the links
# (a committed .venv link becomes a self-link in the main checkout). info/exclude is shared by all worktrees.
EXCLUDE="$COMMON/info/exclude"
mkdir -p "$(dirname "$EXCLUDE")"
for d in frontend/node_modules web/node_modules .venv models; do
  grep -qxF "/$d" "$EXCLUDE" 2>/dev/null || echo "/$d" >> "$EXCLUDE"
  if [ -L "$d" ] && [ ! -e "$d" ]; then rm "$d"; fi   # dangling link from an earlier run
  if [ -d "$MAIN/$d" ] && [ ! -e "$d" ]; then
    mkdir -p "$(dirname "$d")" && ln -s "$MAIN/$d" "$d"
  fi
done

if ! out="$(pyinfo 2>&1)"; then
  echo "setup_worktree: .venv does not resolve to a working python: $out" >&2
  exit 1
fi
echo "setup_worktree: .venv -> $(cd .venv && pwd -P); python: $out"
