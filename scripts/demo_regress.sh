#!/usr/bin/env bash
# Replay the demo clip in natural / focus_hq / text / synthetic against a server this script starts
# on a free port, score each recording against the clip's stems, flag regressions (exit 1).
#   bash scripts/demo_regress.sh [--fast] [--modes natural,focus_hq] [--seconds 60] [--no-lock] [--no-free]
# Arguments go straight to demo_regress.py: the manifest target is locked unless --no-lock (T-100).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/_py.sh
exec "$PY" scripts/demo_regress.py "$@"
