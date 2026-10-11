#!/usr/bin/env bash
# Minutes. verify_fast, then the slow model tests and the fast benchmark (if present).
set -uo pipefail
cd "$(dirname "$0")/.."
bash scripts/verify_fast.sh || exit 1
source scripts/_py.sh
status=0
"$PY" -m pytest -q --no-header -m "slow and not network" 2>&1 | tail -30
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || [ "$rc" -eq 5 ] || { echo "slow tests failed (pytest exit $rc)"; status=1; }   # 5 = no tests collected
if [ -f eval/bench.py ]; then
  "$PY" -m eval.bench --fast 2>&1 | tail -30 || status=1
fi
# Live replay of the demo clip through a real server on a free port (natural + focus_hq, 12 s each),
# scored against the clip's stems; exit 1 on a > 1 dB SI-SDRi drop or > 0.5 s more true dropouts
# (vs the median of the last 5 runs, tolerance widened to their spread, confirmed by a second replay).
# Load on a shared machine moves these numbers (Q-246): a flagged run is re-measured once under
# `swarm-lock --exclusive`; a regression that survives is a non-blocking WARNING with the table.
# Only a crash, a missing row or no output fails verify_full.
demo_replay() {  # $1 = log file; prints the tail, returns demo_regress's exit code
  bash scripts/demo_regress.sh --fast >"$1" 2>&1
  local rc=$?
  tail -12 "$1"
  return $rc
}
only_regressed() {  # $1 = log: exit 1 with a full table and REGRESSION lines, no crash / missing session
  grep -q '^!! REGRESSION' "$1" && grep -q '^wall ' "$1" \
    && ! grep -qE 'no session recorded|^Traceback' "$1"
}
fixture_missing() {  # $1 = log: demo_regress exited 2 only because the clip or its manifest is absent
  grep -qE 'no demo clip at|no demo manifest' "$1" && ! grep -qE '^Traceback|demo_regress crashed|no session recorded' "$1"
}
if [ -f eval/fixtures/demo/grid-two-face.mp4 ] || [ -f "$(git rev-parse --path-format=absolute --git-common-dir)/../eval/fixtures/demo/grid-two-face.mp4" ]; then
  dlog=$(mktemp "${TMPDIR:-/tmp}/verify_full-demo.XXXXXX")
  demo_replay "$dlog"; drc=$?
  if [ "$drc" -eq 1 ] && only_regressed "$dlog"; then
    echo "demo replay below baseline; re-measuring once with the machine to itself (load: $(uptime | sed 's/.*load average[s]*: //'))"
    if [ -z "${SWARM_LOCK_HELD:-}" ] && command -v swarm-lock >/dev/null 2>&1; then
      swarm-lock --exclusive --wait 900 -- bash scripts/demo_regress.sh --fast >"$dlog" 2>&1; drc=$?
      tail -12 "$dlog"
    else
      demo_replay "$dlog"; drc=$?
    fi
    if [ "$drc" -eq 1 ] && only_regressed "$dlog"; then
      echo "WARNING demo replay still below baseline after an exclusive re-measure (non-blocking, numbers above):"
      grep -E '^!! REGRESSION|^baseline' "$dlog"
      drc=0
    fi
  fi
  # Q-211: an absent clip/manifest is not a failure; since T-111 a crash or a missing session also exits 2 (Q-281)
  if [ "$drc" -eq 2 ] && fixture_missing "$dlog"; then echo "demo replay skipped: clip or manifest missing (python -m eval.make_fixtures --demo)"
  elif [ "$drc" -ne 0 ]; then echo "demo replay crashed or produced no scores (exit $drc, log $dlog)"; status=1
  else rm -f "$dlog"; fi
else
  echo "demo replay skipped: no eval/fixtures/demo/grid-two-face.mp4 (python -m eval.make_fixtures --demo)"
fi
exit $status
