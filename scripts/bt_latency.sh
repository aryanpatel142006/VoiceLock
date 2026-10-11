#!/usr/bin/env bash
# Demo audio path round trip (T-144): MacBook Pro Microphone in, AirPods Pro 3 (A2DP) out.
#   bash scripts/bt_latency.sh --dry-run     # resolve devices, print the procedure, no audio
#   bash scripts/bt_latency.sh               # 3 runs x 5 clicks -> results/bt_latency_<stamp>.json, PASS/FAIL
# Options: --in NAME --out NAME --runs N --clicks N --interval S. Run by a human with the buds (opens the mic).
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/_py.sh
exec "$PY" scripts/bt_latency.py "$@"
