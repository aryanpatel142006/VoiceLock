#!/usr/bin/env bash
# Run from any cwd. Real mode never substitutes synthetic data.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-synthetic}"
out="${2:-/tmp/T-319-reproduction}"
py="$PWD/.venv/bin/python"
mkdir -p "$out"
if [[ "$mode" == synthetic ]]; then
  data="$out/synthetic_seed309.npz"
  args=()
  [[ -f "$data" ]] || args+=(--make-synthetic)
  "$py" gpu_jobs/T-319/train_head_losses.py --data "$data" --out "$out" \
    --device cpu --steps 100 --seed 0 "${args[@]}"
elif [[ "$mode" == real ]]; then
  : "${DATA:?Set DATA to a flat split-labelled team embedding NPZ}"
  for seed in 0 1 2; do
    "$py" gpu_jobs/T-319/train_head_losses.py --data "$DATA" --out "$out/seed$seed" \
      --device "${DEVICE:-auto}" --steps "${STEPS:-2000}" --seed "$seed"
  done
else
  echo 'usage: reproduce.sh synthetic|real [output_directory]' >&2
  exit 2
fi
for run in "$out" "$out"/seed*; do
  [[ -f "$run/metrics.json" ]] || continue
  for loss in raw ap subcenter ce; do
    "$py" gpu_jobs/T-319/calibrate.py --pairs "$run/${loss}_pairs.npz" --out "$run/${loss}_calibration.json"
  done
done
