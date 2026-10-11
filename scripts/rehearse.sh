#!/usr/bin/env bash
# The Oct 10 final test (T-147): 12 steps one at a time, Enter to go on; writes results/rehearsal_<stamp>.md.
#   bash scripts/rehearse.sh [--gpu] [--caption-ref script.txt]      # the human run (opens the mic in step 2)
#   bash scripts/rehearse.sh --dry-run --no-hw --session <dir>       # print the steps, sheet from a recording
# Procedure and bars: docs/REHEARSAL.md.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/_py.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" scripts/rehearse.py "$@"
