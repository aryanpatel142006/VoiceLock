#!/usr/bin/env bash
# Re-enroll voice profiles from demo-mic takes (T-172): validate, write $HEARING_FIXTURES_DIR/profiles/<name>/
# (+ profiles-all), check cos(a, b) < 0.7, print the laptop and gpu_up commands (Aryan runs gpu_up).
#   bash scripts/enroll_profiles.sh aryan a1.wav a2.wav a3.wav joseph j1.wav j2.wav j3.wav
# Procedure: docs/REHEARSAL.md "Re-enroll the voice profiles".
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/_py.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" scripts/enroll_profiles.py "$@"
