#!/usr/bin/env bash
# T-391 (A): one detached driver for the A100 in job 435775, started on ilab1 with
#   cd ~/tmp && setsid nohup bash T-391/code/chain.sh > T-391/chain.log 2>&1 < /dev/null &
# Each step runs on rlab2 through `srun --jobid=435775 --overlap` (the T-374b pattern), and the
# driver needs no ssh link. Steps run one after another; a failed step is logged and the chain
# goes on, so C4 still trains if the eval or the dump breaks.
cd ~/tmp || exit 1
PY=~/hearing-remote/.venv/bin/python
BASE=~/hearing-remote/models/MossFormer2_SS_16K/last_best_checkpoint.pt
T374=T-374/run_fixed/best.pt
T387=T-374b/run_fixed/best.pt
DATA=T-374/data
ECAPA=~/hearing-remote/models/spkrec-ecapa-voxceleb
COMMON="--order fixed --data $DATA --sir -10 10 --steps 1000000 --val-every 400 --batch 4"
step() { echo "$(date '+%F %T %Z') $*"; }
run() { srun --jobid=435775 --overlap "$@"; }

step "waiting for T-374b/chain.done (T-387 train + eval)"
# T-387 was due by about 20:30; if its chain hangs (crashed trainer never writes "done"), go at 21:45
deadline=$(date -d "2026-10-10 21:45" +%s)
while [ ! -f T-374b/chain.done ] && [ "$(date +%s)" -lt "$deadline" ]; do sleep 60; done
[ -f T-374b/chain.done ] || step "T-374b/chain.done missing at the deadline; going on"

INIT=$($PY T-391/code/pick_init.py T-374/eval_fixed.json=$T374 T-374b/eval_fixed2.json=$T387 2>> T-391/chain.log) || INIT=$T374
step "C1 v3 train 110 min from $INIT"
run $PY T-391/code/train_sep.py $COMMON --recipe v3 --lr 1e-5 --lr-sched cosine --init "$INIT" \
  --out T-391/run_v3 --minutes 110 --seed 391 > T-391/run_v3.log 2>&1
step "C1 rc=$?"

step "C2 eval base / T-374 / T-387 / v3"
run $PY T-391/code/eval_sep.py --ckpt base=$BASE --ckpt t374=$T374 --ckpt t387=$T387 --ckpt v3=T-391/run_v3/best.pt \
  --data $DATA --grid $DATA/grid --ecapa $ECAPA --out T-391/eval_all.json > T-391/eval_all.log 2>&1
step "C2 rc=$?"

step "C3 gate features (v3 separator)"
run $PY T-391/code/dump_gate_features.py --ckpt T-391/run_v3/best.pt --data $DATA --ecapa $ECAPA \
  --out T-391/gate_features.npz > T-391/gate_features.log 2>&1
step "C3 rc=$?"

step "C4 v3 seed 3 from base, 240 min"
run $PY T-391/code/train_sep.py $COMMON --recipe v3 --lr 5e-5 --lr-sched cosine --init $BASE \
  --out T-391/run_c4 --minutes 240 --seed 3 > T-391/run_c4.log 2>&1
step "C4 rc=$?"
run $PY T-391/code/eval_sep.py --ckpt c4=T-391/run_c4/best.pt --data $DATA --grid $DATA/grid --ecapa $ECAPA \
  --out T-391/eval_c4.json > T-391/eval_c4.log 2>&1
step "C4 eval rc=$?"
echo finished > T-391/chain.done
