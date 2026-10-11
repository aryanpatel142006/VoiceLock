#!/usr/bin/env bash
# T-129: Focus HQ A/B on the A100 with voice profiles loaded, 3 runs per clip per variant.
#
#   HEARING_REMOTE_URL=ws://127.0.0.1:18765/ws/session bash scripts/t129_ab.sh [variant ...]
#
# Variants (default: all three): base (judge profile + voice profile, the T-125 setup),
# novideo (HEARING_REMOTE_VIDEO=0: no camera frames to the remote), standard (isolation standard
# instead of the judge's strict = pause_duck_asd), novideo-standard (both). The enrolled_dfn post-filter and same_bucket off
# are remote-side configs (REMOTE_PIPELINES side server / team preset pair): run this script against
# that server's URL with VARIANTS=base. Hop 0.125 is already config/tiers-remote.yaml's enrolled hop.
# w1 / w1-standard / novideo-w1-standard: the same runs, labelled for the 1.0 s window / 0.25 s hop side server
# (config/tiers-remote-w1.yaml): HEARING_REMOTE_URL=<side server> bash scripts/t129_ab.sh w1 w1-standard. The variant
# name is only the log label plus the novideo*/*standard switches; the window comes from the server.
#
# Profiles: one dir with all four, $HEARING_FIXTURES_DIR/profiles-all (override: PROFILES=<dir>):
# demo-team-easy (real voices, T-141's fixture) enrolls aryan:joseph, demo-mf-easy (GRID s1 vs s4)
# grid-s1:grid-s4 (cut from the clip's own stems, not held out). The remote enrolls its own copy by
# name, so start it with the same dir: ILAB_PROFILES=<that dir> scripts/gpu_up.sh ilab.
# Wall-clock numbers: runs every replay under `swarm-lock --exclusive`. Never becomes a baseline.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HEARING_REMOTE_URL:?set HEARING_REMOTE_URL (the A100 tunnel)}"
export HEARING_HQ_BACKEND=remote HEARING_REGRESS_NO_BASELINE=1
FIX="${HEARING_FIXTURES_DIR:-$(git rev-parse --path-format=absolute --git-common-dir)/../eval/fixtures/generated}"
DEMO="$(git rev-parse --path-format=absolute --git-common-dir)/../eval/fixtures/demo"
PROFILES="${PROFILES:-$FIX/profiles-all}"
[ -d "$PROFILES" ] || { echo "t129_ab: no profile dir $PROFILES" >&2; exit 1; }
RUNS="${RUNS:-3}"
SECONDS_PER_RUN="${SECONDS_PER_RUN:-60}"
variants=("$@")
[ ${#variants[@]} -gt 0 ] || read -r -a variants <<< "${VARIANTS:-base novideo standard}"

run_clip() {  # variant clip profiles_dir voice
  local variant=$1 clip=$2 dir=$3 voice=$4 extra=()
  case $variant in *standard) extra=(--isolation standard) ;; esac
  for i in $(seq "$RUNS"); do
    echo "== $variant $(basename "$clip") run $i/$RUNS · load $(uptime | sed 's/.*averages*: //')"
    if [ "${variant#novideo}" != "$variant" ]; then
      HEARING_REMOTE_VIDEO=0 HEARING_PROFILES_DIR="$dir" swarm-lock --exclusive -- .venv/bin/python scripts/demo_regress.py \
        --modes focus_hq --seconds "$SECONDS_PER_RUN" --no-free --profile judge --voice-profile "$voice" --clip "$clip"
    else
      HEARING_PROFILES_DIR="$dir" swarm-lock --exclusive -- .venv/bin/python scripts/demo_regress.py \
        --modes focus_hq --seconds "$SECONDS_PER_RUN" --no-free --profile judge --voice-profile "$voice" --clip "$clip" ${extra[@]+"${extra[@]}"}
    fi
  done
}

for v in "${variants[@]}"; do
  run_clip "$v" "$FIX/t141/aryan/demo-team-easy" "$PROFILES" aryan:joseph
  run_clip "$v" "$DEMO/demo-mf-easy" "$PROFILES" grid-s1:grid-s4
done
