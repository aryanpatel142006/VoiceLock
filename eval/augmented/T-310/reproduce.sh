#!/usr/bin/env bash
# T-310: regenerate the augmented dataset from scratch.
# Requires: python (numpy, scipy, librosa or ffmpeg rubberband, ffmpeg libopus).
# Run from the repo root: bash eval/augmented/T-310/reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python3"
  fi
fi
"$PYTHON" eval/augmented/T-310/generate.py --out eval/augmented/T-310
echo "done. verify with: \"$PYTHON\" -m pytest tests/test_augmented_t310.py -q"
