#!/usr/bin/env bash
# T-391 (B): sbatch replicates on free iLab GPUs; run on ilab1 from ~/tmp (code in T-391/code).
# Each job trains, then evaluates its own best.pt; nothing needs the laptop's ssh link afterwards.
# Usage: bash T-391/code/submit.sh s1|s2|s3 [gres]   (prints the job id)
# iLab rejects --cpus-per-task ("Please do not specify the number of CPUs"), so it is not passed.
set -euo pipefail
cd ~/tmp
PY=~/hearing-remote/.venv/bin/python
BASE=~/hearing-remote/models/MossFormer2_SS_16K/last_best_checkpoint.pt
T374=~/tmp/T-374/run_fixed/best.pt
DATA=T-374/data
ECAPA=~/hearing-remote/models/spkrec-ecapa-voxceleb
COMMON="--order fixed --data $DATA --sir -10 10 --steps 1000000 --val-every 400"
evalcmd() { echo "$PY T-391/code/eval_sep.py --ckpt $1=T-391/run_$1/best.pt --data $DATA --grid $DATA/grid --ecapa $ECAPA --out T-391/eval_$1.json"; }

case "$1" in
  s1)  # T-387's recipe (v1, constant lr 5e-5) from base on a 24 GB card
    gres=${2:-gpu:4500_ada:1}; time=04:30:00
    train="$PY T-391/code/train_sep.py $COMMON --recipe v1 --init $BASE --out T-391/run_s1 --minutes 170 --batch 2 --accum 2 --seed 11" ;;
  s2)  # v3 seed 2 from T-374's fixed-order fine-tune (T-387 is still training when this starts)
    gres=${2:-gpu:4500_ada:1}; time=04:30:00
    train="$PY T-391/code/train_sep.py $COMMON --recipe v3 --lr 1e-5 --lr-sched cosine --init $T374 --out T-391/run_s2 --minutes 170 --batch 2 --accum 2 --seed 2" ;;
  s3)  # v3 from base, long; from base the slot order must be learnt, so lr 5e-5 cosine
    gres=${2:-gpu:a6000:1}; time=05:00:00
    train="$PY T-391/code/train_sep.py $COMMON --recipe v3 --lr 5e-5 --lr-sched cosine --init $BASE --out T-391/run_s3 --minutes 240 --batch 1 --accum 4 --seed 4" ;;
  *) echo "usage: $0 s1|s2|s3 [gres]" >&2; exit 2 ;;
esac
sbatch --parsable -p unlimited --gres="$gres" --mem=48G --time="$time" -J "sh-T391-$1" \
  --chdir "$HOME/tmp" -o "$HOME/tmp/T-391/slurm-%j.out" --wrap "$train > T-391/run_$1.log 2>&1 && $(evalcmd "$1") > T-391/eval_$1.log 2>&1"
