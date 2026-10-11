#!/usr/bin/env bash
# T-307 GPU run on Rutgers iLab (~30 min on one A100). Run on the iLab login node from the synced checkout
# (scripts/gpu_up.sh keeps one at $REMOTE_DIR, default ~/hearing-remote, with its .venv):
#   ssh <netid>@ilab.cs.rutgers.edu 'cd hearing-remote && bash experiments/gpu/T-307/run_ilab.sh'
# Team set (optional; wavs are not in git): rsync -a <main>/eval/augmented/T-306/ <host>:hearing-remote/eval/augmented/T-306/
# Env: DATA=/freespace/local/$USER/librispeech  OUT=$DATA/../t307-runs  ILAB_GRES=gpu:a100:1  ILAB_TIME=01:00:00
#      SPLITS=train-clean-100 (add ,train-clean-360 for 921 speakers; ~+25 min of feature extraction)
#      STEPS=20000  SEEDS=3  REUSE_JOB=<job id>  (run as an srun --overlap step inside a running job)
set -euo pipefail
cd "$(dirname "$0")/../../.."
DATA="${DATA:-/freespace/local/$USER/librispeech}"
OUT="${OUT:-$DATA/../t307-runs}"
SPLITS="${SPLITS:-train-clean-100}"
STEPS="${STEPS:-20000}"
SEEDS="${SEEDS:-3}"
ILAB_GRES="${ILAB_GRES:-gpu:a100:1}"
ILAB_TIME="${ILAB_TIME:-01:00:00}"
mkdir -p "$DATA" "$OUT"
for split in test-clean ${SPLITS//,/ }; do
  if [ ! -d "$DATA/LibriSpeech/$split" ]; then
    echo "downloading LibriSpeech $split to $DATA"
    curl -fL "https://www.openslr.org/resources/12/$split.tar.gz" | tar xz -C "$DATA"
  fi
done
[ -x .venv/bin/python ] || { echo "no .venv here: run scripts/gpu_up.sh ilab once to bootstrap it" >&2; exit 1; }
export HEARING_MODELS_DIR="${HEARING_MODELS_DIR:-$PWD/models}"
.venv/bin/python -c "from hearing.tse import enrolled; enrolled.available() or enrolled.download_model()"
cmd=(.venv/bin/python experiments/gpu/T-307/train_ap_head.py --librispeech "$DATA/LibriSpeech"
     --train-splits "$SPLITS" --out "$OUT" --steps "$STEPS" --seeds "$SEEDS")
if [ -n "${REUSE_JOB:-}" ]; then
  srun --jobid "$REUSE_JOB" --overlap "${cmd[@]}" 2>&1 | tee "$OUT/run.log"
else
  srun --gres="$ILAB_GRES" --time="$ILAB_TIME" --job-name=t307-ap-head "${cmd[@]}" 2>&1 | tee "$OUT/run.log"
fi
