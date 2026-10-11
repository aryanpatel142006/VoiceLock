#!/usr/bin/env bash
# T-424: one sbatch job per student variant on a free iLab GPU; run on ilab1 from ~/tmp.
# Layout on iLab: T-424/code/T-424/*.py (this dir) and T-424/code/T-391/train_sep.py (renderer).
# Usage: bash T-424/code/T-424/submit.sh A|B|C|D [gres]   (train one variant; prints the job id)
#        bash T-424/code/T-424/submit.sh ref [gres]       (eval teacher@1.0/0.5 s + base + an early student)
#        bash T-424/code/T-424/submit.sh final DEPS [gres] (eval every run_*/best.pt after jobs DEPS=a:b:c)
# iLab rejects --cpus-per-task, so it is not passed; --steps is huge so --minutes ends training.
set -euo pipefail
cd ~/tmp
PY=~/hearing-remote/.venv/bin/python
TEACHER=~/tmp/T-374b/run_fixed/best.pt   # T-387 fixed-order fine-tune
DATA=T-374/data
CODE=T-424/code/T-424
MIN=${T424_MINUTES:-270}
COMMON="--k ${T424_K:-12} --teacher $TEACHER --data $DATA --minutes $MIN --ckpt-every 30 --val-every 1000"
BASE=~/hearing-remote/models/MossFormer2_SS_16K/last_best_checkpoint.pt
ECAPA=~/hearing-remote/models/spkrec-ecapa-voxceleb
EVAL="$PY $CODE/eval_student.py --data $DATA"
mkdir -p T-424
case "$1" in
  ref)
    sbatch --parsable -p unlimited --gres="${2:-gpu:a4000:1}" --mem=32G --time=02:00:00 -J sh-T424-ref \
      --chdir "$HOME/tmp" -o "$HOME/tmp/T-424/slurm-%j.out" \
      --wrap "$EVAL --student early=T-424/run_tcn16_nokd/best.pt --teacher $TEACHER --base $BASE --ecapa $ECAPA --out T-424/eval_ref.json > T-424/eval_ref.log 2>&1"
    exit 0 ;;
  final)
    students=""
    for n in tcn16 tcn32 tcn8s tcn16_nokd; do students="$students --student $n=T-424/run_$n/best.pt"; done
    sbatch --parsable -p unlimited --gres="${3:-gpu:a4000:1}" --mem=32G --time=02:00:00 -J sh-T424-final \
      --dependency="afterany:$2" --chdir "$HOME/tmp" -o "$HOME/tmp/T-424/slurm-%j.out" \
      --wrap "$EVAL $students --out T-424/eval_students.json > T-424/eval_students.log 2>&1"
    exit 0 ;;
  A) name=tcn16; args="--variant tcn16 --alpha 1.0 --batch 8 --crop-s 3" ;;     # 16 ms, 2.7 M
  B) name=tcn32; args="--variant tcn32 --alpha 1.0 --batch 8 --crop-s 3" ;;     # 32 ms, 2.5 M
  C) name=tcn8s; args="--variant tcn8s --alpha 1.0 --batch 8 --crop-s 3" ;;     # 8 ms, 1.1 M
  D) name=tcn16_nokd; args="--variant tcn16 --alpha 0 --batch 8 --crop-s 3" ;;  # ablation: no teacher
  *) echo "usage: $0 A|B|C|D [gres]" >&2; exit 2 ;;
esac
gres=${2:-gpu:a4000:1}
sbatch --parsable -p unlimited --gres="$gres" --mem=32G --time=05:15:00 -J "sh-T424-$name" \
  --chdir "$HOME/tmp" -o "$HOME/tmp/T-424/slurm-%j.out" \
  --wrap "$PY $CODE/distill.py $args $COMMON --out T-424/run_$name >> T-424/run_$name.log 2>&1"
