#!/usr/bin/env bash
# T-420: short-window fine-tunes (0.5 / 0.75 / 1.0 s crops) from T-387's best, as separate sbatch
# jobs on free iLab GPUs; run on ilab1 from ~/tmp (code in T-420/code). Each job trains, then
# scores its own best.pt at the live window it was trained for (hop = half the window), so
# nothing needs the laptop's ssh link afterwards. j0 is eval only: the T-387 checkpoint (and
# T-391's v3, for reference) at 0.5 / 0.75 / 1.0 s, the same-window baseline.
# Usage: bash T-420/code/submit.sh j0|j1|j2|j3|j4 [gres]   (prints the job id)
# iLab rejects --cpus-per-task ("Please do not specify the number of CPUs"), so it is not passed.
# Micro-batches hold ~2-4 s of audio (T-391 s2 ran 2 x 2 s on a 24 GB card); every optimiser
# step sees ~4-4.5 s of audio whatever the crop.
set -euo pipefail
cd ~/tmp
PY=~/hearing-remote/.venv/bin/python
T387=T-374b/run_fixed/best.pt
V3=T-391/run_v3/best.pt
DATA=T-374/data
ECAPA=~/hearing-remote/models/spkrec-ecapa-voxceleb
COMMON="--order fixed --data $DATA --sir -10 10 --steps 1000000 --val-every 400 --recipe v3 --lr 1e-5 --lr-sched cosine --init $T387 --minutes 120"
# evalcmd LABEL CKPT WINDOW_S OUT_STEM
evalcmd() { echo "$PY T-420/code/eval_sep.py --ckpt $1=$2 --data $DATA --grid $DATA/grid --ecapa $ECAPA --window-s $3 --hop-s $(echo "$3 / 2" | bc -l) --out T-420/$4.json > T-420/$4.log 2>&1"; }
time=03:30:00

case "$1" in
  j0)  # baseline: T-387 and v3 at the three windows (eval only)
    gres=${2:-gpu:a4000:1}; time=03:00:00
    wrap="$(evalcmd t387 $T387 0.5 eval_t387_w0.5); $(evalcmd t387 $T387 0.75 eval_t387_w0.75); $(evalcmd t387 $T387 1.0 eval_t387_w1.0); $(evalcmd v3 $V3 0.5 eval_v3_w0.5); $(evalcmd v3 $V3 0.75 eval_v3_w0.75)" ;;
  j1)  crop=0.5;  seed=420; batch="--batch 4 --accum 2"; gres=${2:-gpu:4500_ada:1} ;;
  j2)  crop=0.75; seed=420; batch="--batch 3 --accum 2"; gres=${2:-gpu:a4500:1} ;;
  j3)  crop=0.5;  seed=2;   batch="--batch 4 --accum 2"; gres=${2:-gpu:a4500:1} ;;
  j4)  crop=1.0;  seed=420; batch="--batch 4 --accum 1"; gres=${2:-gpu:a6000:1} ;;
  *) echo "usage: $0 j0|j1|j2|j3|j4 [gres]" >&2; exit 2 ;;
esac
if [ "$1" != j0 ]; then
  train="$PY T-420/code/train_sep.py $COMMON --crop-s $crop --seed $seed $batch --out T-420/run_$1"
  wrap="$train > T-420/run_$1.log 2>&1 && $(evalcmd "$1" "T-420/run_$1/best.pt" $crop "eval_$1")"
fi
sbatch --parsable -p unlimited --gres="$gres" --mem=48G --time="$time" -J "sh-T420-$1" \
  --chdir "$HOME/tmp" -o "$HOME/tmp/T-420/slurm-%j.out" --wrap "$wrap"
