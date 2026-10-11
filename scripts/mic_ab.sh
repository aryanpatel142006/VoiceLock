#!/usr/bin/env bash
# iPhone-over-USB mic vs MacBook mic A/B (T-171): clicks, 3 judge takes per mic, target-absent take, dropout check,
# Shift+C, three-way listening; writes results/mic_ab_<stamp>.md and results/listening_<stamp>.md with the gate verdict.
#   bash scripts/mic_ab.sh                                   # the human run (opens the mic for the clicks)
#   bash scripts/mic_ab.sh --dry-run --no-hw --fake-devices "X iPhone Microphone,MacBook Pro Microphone,AirPods"
#   bash scripts/mic_ab.sh --from results/mic_ab_<stamp>.json   # rebuild the sheets from a run's marks
# Part B prints the server restart per mic; run it exactly (from the main checkout), the env vars matter:
#   HEARING_SESSION_LOG=results/sessions HEARING_PROFILES_DIR=$PWD/eval/fixtures/generated/profiles \
#     bash scripts/demo.sh --judge --team --mic "<label>" --restart
# Without HEARING_SESSION_LOG the server records no session; scoring then refuses (exit 3) instead of NOT MEASURED rows.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/_py.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" scripts/mic_ab.py "$@"
