#!/usr/bin/env bash
# Record one profile take through a named input device (T-172), audio only, 16 kHz mono PCM16:
#   bash scripts/record_profile.sh aryan --device "MacBook Pro Microphone" --seconds 25
#   bash scripts/record_profile.sh --list-devices
# Writes $HEARING_FIXTURES_DIR/team/<name>/profile-NN.wav (scripts/record_take.py --kind profile --no-video;
# a 3 s level check first). Opens the microphone: run it by hand, never from a test or an agent.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/_py.sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [ "${1:-}" = --list-devices ]; then exec "$PY" scripts/record_take.py --list-devices; fi
name="${1:-}"
case "$name" in ""|-*) echo "usage: $0 <name> --device \"<input label>\" [--seconds 30] [record_take.py options]" >&2; exit 1;; esac
shift
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --device) args+=(--mic "$2"); shift 2;;
    *) args+=("$1"); shift;;
  esac
done
exec "$PY" scripts/record_take.py --name "$name" --kind profile --no-video "${args[@]}"
