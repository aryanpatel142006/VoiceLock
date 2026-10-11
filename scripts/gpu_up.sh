#!/usr/bin/env bash
# Bring up the remote GPU session server and tunnel it to this laptop (docs/DEPLOY.md).
#
#   scripts/gpu_up.sh ilab [--side] [--dry-run]                 # Rutgers iLab: srun GPU job via the cm-ilab socket
#   scripts/gpu_up.sh ssh USER@HOST [--ssh-port N] [--public]   # any SSH GPU box
#   scripts/gpu_up.sh runpod [--gpu "NVIDIA L4"]                # create a RunPod pod, then `ssh` into it
#   scripts/gpu_up.sh local [--public]                          # same install + server here (on the GPU box, or a rehearsal)
#   scripts/gpu_up.sh docker [--cpu]                            # build + run the Dockerfile on this machine
#   scripts/gpu_up.sh probe [URL]                               # health + RTT of the remote (JSON)
#   scripts/gpu_up.sh status                                    # what is up, and its health
#   scripts/gpu_up.sh down [--terminate]                        # stop server + tunnel (+ stop/delete the pod)
#
# Environment (all optional):
#   LOCAL_PORT=18765          laptop end of the tunnel: ws://127.0.0.1:$LOCAL_PORT/ws/session
#   REMOTE_PORT=8000         server port on the GPU box
#   REMOTE_DIR=hearing-remote  checkout on the GPU box (relative to its home; iLab: a /freespace path if quota is tight)
#   REMOTE_HF_HOME=          Hugging Face cache on the GPU box (default: $REMOTE_DIR/.hf-cache; RunPod: /workspace/hf-cache)
#   REMOTE_PIPELINES=        HEARING_PIPELINES for the remote server, e.g. 'focus_hq=<hq stage spec>'
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124   used only when the GPU box has no torch
#   SSH_OPTS=                extra ssh options, e.g. "-J netid@ilab.cs.rutgers.edu -i ~/.ssh/id_ed25519"
#   RUNPOD_API_KEY           required by `runpod` and `down --terminate` on a pod
#   RUNPOD_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   SSH_PUBKEY=~/.ssh/id_ed25519.pub   key injected into the pod
#   ilab only (docs/DEPLOY.md#rutgers-ilab):
#   ILAB_HOST=ap2772@ilab.cs.rutgers.edu   login host; reached only through the ControlMaster socket below
#   ILAB_SOCKET=~/.ssh/cm-ilab  ControlPath opened by the human (no keys/Kerberos off campus)
#   ILAB_NODE=               node for the srun job (default: the login node the socket is on, e.g. ilab3;
#                            when ILAB_GRES names a GPU type, e.g. gpu:a100:1, the first node sinfo lists
#                            with that type: the A100 is only on rlab2)
#   ILAB_TIME=02:00:00       job time limit (the partition default is 7 days; always set one)
#   ILAB_GRES=gpu:1          never add --cpus-per-task/--mem to ILAB_SRUN_ARGS: the cluster rejects them there
#                            (iLab rejects every CPU flag; a job step already sees all 96 cores, T-115.
#                            ILAB_CPUS is a no-op that only prints a warning)
#   ILAB_JOB=hearing-hq      job name (down cancels only this name/id)
#   ILAB_REUSE=              id of a running job: start the new server as an extra step of it
#                            (srun --jobid --overlap) instead of cancelling it; set REMOTE_PORT to a
#                            free port (the old server keeps its own until the job ends; T-110)
#   ILAB_SRUN_ARGS=          extra srun options, e.g. "--partition=unlimited"
#   ILAB_WAIT=300            seconds to wait for the job + /health
#   ILAB_PROFILES=           local voice-profile dir (<name>/*.wav, or T-172's <name>/current + <name>/v<N>/; gitignored, so the
#                            code sync skips it): the whole tree rsynced to $REMOTE_DIR/profiles/ and served as HEARING_PROFILES_DIR (T-129)
#   ILAB_THREADS=8           OMP/MKL threads of the job's server (torch would start one per core of the
#                            shared 96-core node and contend with every other user's job; T-075)
#   REMOTE_PORT defaults to 18765 for ilab (the login node is shared; 8000 is a common port)
#   ILAB_SIDE=1 / --side      a second job next to the live one (an A/B): its own state dir
#                            (hearing-gpu/side), job name $ILAB_JOB-side, REMOTE_PORT 18767 and a free
#                            LOCAL_PORT unless set, so it never cancels the live job or its tunnel.
#                            `ILAB_SIDE=1 scripts/gpu_up.sh status|down` acts on the side job only
#
# State (host, tunnel pid, pod id) lives in ${XDG_STATE_HOME:-~/.local/state}/hearing-gpu, never in the repo.
set -euo pipefail
cd "$(dirname "$0")/.."
# --dry-run builds text only: skip _py.sh, which may run pip install (minutes on a fresh CI checkout)
[ "${1:-}" = ilab ] && [[ " $* " == *" --dry-run "* ]] || source scripts/_py.sh

LOCAL_PORT_SET="${LOCAL_PORT:-}"
LOCAL_PORT="${LOCAL_PORT:-18765}"
REMOTE_PORT_SET="${REMOTE_PORT:-}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
REMOTE_DIR="${REMOTE_DIR:-hearing-remote}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-}"
REMOTE_PIPELINES="${REMOTE_PIPELINES:-}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
SSH_OPTS="${SSH_OPTS:-}"
ILAB_HOST="${ILAB_HOST:-ap2772@ilab.cs.rutgers.edu}"
ILAB_SOCKET="${ILAB_SOCKET:-$HOME/.ssh/cm-ilab}"
ILAB_NODE="${ILAB_NODE:-}"
ILAB_TIME="${ILAB_TIME:-02:00:00}"
ILAB_GRES="${ILAB_GRES:-gpu:1}"
ILAB_SIDE="${ILAB_SIDE:-}"
ILAB_JOB="${ILAB_JOB:-hearing-hq}"
ILAB_SRUN_ARGS="${ILAB_SRUN_ARGS:-}"
ILAB_WAIT="${ILAB_WAIT:-300}"
ILAB_THREADS="${ILAB_THREADS:-8}"
ILAB_REUSE="${ILAB_REUSE:-}"
ILAB_PROFILES="${ILAB_PROFILES:-}"
# The Focus HQ ladder on the job's GPU: enrolled (same-gender guard) -> AV-MossFormer2 2 s / 1 s ->
# gate, without the CPU DeepFilterNet tiers that made the remote fall behind real time (T-075).
HQ_PIPELINES='focus_hq=tiers(config="config/tiers-remote.yaml")'
RUNPOD_API="https://rest.runpod.io/v1"
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/hearing-gpu"
URL="ws://127.0.0.1:${LOCAL_PORT}/ws/session"

use_side() {  # an A/B job beside the live one: own state, job name, ports (T-118)
  STATE="$STATE/side"
  ILAB_JOB="${ILAB_JOB}-side"
  [ -n "$REMOTE_PORT_SET" ] || { REMOTE_PORT=18767; REMOTE_PORT_SET=1; }
  if [ -z "${LOCAL_PORT_SET:-}" ]; then
    LOCAL_PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
  fi
  URL="ws://127.0.0.1:${LOCAL_PORT}/ws/session"
}
[[ " $* " == *" --side "* ]] && ILAB_SIDE=1
[ -z "$ILAB_SIDE" ] || use_side
mkdir -p "$STATE"

die() { echo "gpu_up: $*" >&2; exit 1; }
say() { echo "gpu_up: $*" >&2; }
state_get() { [ -f "$STATE/$1" ] && cat "$STATE/$1" || true; }
# The synced tree has no .git: write the laptop's HEAD (+ -dirty) to REVISION, which /health reports.
revision() { local r; r="$(git rev-parse HEAD)"; [ -z "$(git status --porcelain --untracked-files=no)" ] || r="$r-dirty"; echo "$r"; }

# ---------------------------------------------------------------- shared

wait_health() {  # wait_health <http url> <seconds>
  local url="$1" deadline=$((SECONDS + $2))
  until curl -fsS --max-time 2 "$url" >/dev/null 2>&1; do
    [ "$SECONDS" -lt "$deadline" ] || return 1
    sleep 1
  done
}

finish() {  # print health, RTT and the laptop command
  echo "$URL" > "$STATE/url"
  curl -fsS "http://127.0.0.1:${LOCAL_PORT}/health" >&2 && echo >&2
  "$PY" -m hearing.server.remote --url "$URL" --seconds 5 || say "probe failed (see JSON above)"
  cat >&2 <<EOF

gpu_up: remote session server is up. Run the laptop server with the HQ stage remote and the fast path local:

  export HEARING_REMOTE_URL=$URL
  .venv/bin/python -m hearing serve --pipelines 'focus_hq=remote(fallback="passthrough")'

Stop it (and the pod's billing) with: scripts/gpu_up.sh down
EOF
}

close_tunnel() {
  local pid; pid="$(state_get tunnel.pid)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then kill "$pid" && say "tunnel $pid closed"; fi
  rm -f "$STATE/tunnel.pid"
}

# ---------------------------------------------------------------- ssh (iLab, RunPod, any GPU box)

bootstrap_script() {  # runs on the GPU box (bash -s): venv, install, (re)start the server, wait for /health
  cat <<'REMOTE'
set -euo pipefail
cd "$REMOTE_DIR"
PY="$(command -v python3.12 || command -v python3.11 || command -v python3 || true)"
[ -n "$PY" ] && "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || { echo "remote: need python >= 3.11 (found ${PY:-none})" >&2; exit 1; }
# --system-site-packages reuses a preinstalled CUDA torch (RunPod images ship one).
[ -x .venv/bin/python ] || "$PY" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -c 'import torch' 2>/dev/null || .venv/bin/python -m pip install -q torch torchaudio --index-url "$TORCH_INDEX"
.venv/bin/python -m pip install -q -e ".[ml]"
if [ -f serve.pid ] && kill -0 "$(cat serve.pid)" 2>/dev/null; then kill "$(cat serve.pid)"; sleep 1; fi
HF="${HF:-$PWD/.hf-cache}"
nohup env HF_HOME="$HF" HEARING_PIPELINES="$PIPELINES" \
  .venv/bin/python -m hearing serve --host "$BIND" --port "$REMOTE_PORT" > serve.log 2>&1 < /dev/null &
echo $! > serve.pid
.venv/bin/python -m hearing.device
for _ in $(seq 60); do
  .venv/bin/python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$REMOTE_PORT/health', timeout=2)" 2>/dev/null && exit 0
  kill -0 "$(cat serve.pid)" 2>/dev/null || { tail -30 serve.log >&2; exit 1; }
  sleep 1
done
echo "remote: server did not become healthy; tail of serve.log:" >&2; tail -30 serve.log >&2; exit 1
REMOTE
}

cmd_ssh() {
  local target="" ssh_port="" bind="127.0.0.1"
  while [ $# -gt 0 ]; do
    case "$1" in
      --ssh-port) ssh_port="$2"; shift 2 ;;
      --public) bind="0.0.0.0"; shift ;;  # needed for the RunPod HTTP proxy; no auth, so only for demos
      -*) die "ssh: unknown option $1" ;;
      *) target="$1"; shift ;;
    esac
  done
  [ -n "$target" ] || die "usage: gpu_up.sh ssh USER@HOST [--ssh-port N] [--public]"
  # shellcheck disable=SC2206
  local ssh=(ssh -o ServerAliveInterval=15 -o StrictHostKeyChecking=accept-new $SSH_OPTS)
  [ -n "$ssh_port" ] && ssh+=(-p "$ssh_port")

  say "syncing tracked files to $target:$REMOTE_DIR"
  "${ssh[@]}" "$target" "mkdir -p $(printf %q "$REMOTE_DIR")"
  git ls-files -z | rsync -az --from0 --files-from=- -e "${ssh[*]}" ./ "$target:$REMOTE_DIR/"
  "${ssh[@]}" "$target" "echo $(revision) > $(printf %q "$REMOTE_DIR")/REVISION"
  say "synced git rev $(revision)"

  say "installing and starting the server on $target (first run installs torch: minutes)"
  "${ssh[@]}" "$target" \
    "REMOTE_DIR=$(printf %q "$REMOTE_DIR") REMOTE_PORT=$REMOTE_PORT BIND=$bind" \
    "HF=$(printf %q "$REMOTE_HF_HOME") PIPELINES=$(printf %q "$REMOTE_PIPELINES")" \
    "TORCH_INDEX=$(printf %q "$TORCH_INDEX") bash -s" < <(bootstrap_script)

  close_tunnel
  say "tunnel 127.0.0.1:$LOCAL_PORT -> $target:127.0.0.1:$REMOTE_PORT"
  "${ssh[@]}" -f -N -o ExitOnForwardFailure=yes -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "$target"
  pgrep -n -f "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" > "$STATE/tunnel.pid" || true
  echo "$target" > "$STATE/host"
  echo "$ssh_port" > "$STATE/ssh_port"
  echo "$REMOTE_DIR" > "$STATE/remote_dir"
  wait_health "http://127.0.0.1:${LOCAL_PORT}/health" 15 || die "no /health through the tunnel"
  finish
}

# ---------------------------------------------------------------- Rutgers iLab (Slurm; GPUs only inside srun jobs)

ilab_ssh() {  # ssh through the human's ControlMaster socket; never prompts, never opens a new master
  # shellcheck disable=SC2086
  ssh -o ControlPath="$ILAB_SOCKET" -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=5 $SSH_OPTS "$@"
}

ilab_srun() {  # ilab_srun NODE -> the srun command line run on the login node; the job body is "$JOB"
  if [ -n "$ILAB_REUSE" ]; then  # an extra step in the running job's allocation (shares its GPU)
    echo "srun --jobid=$ILAB_REUSE --overlap --gres=$ILAB_GRES -w $1 -J $ILAB_JOB --unbuffered${ILAB_SRUN_ARGS:+ $ILAB_SRUN_ARGS}"' bash -c "$JOB"'
    return
  fi
  echo "srun --gres=$ILAB_GRES -w $1 --time=$ILAB_TIME -J $ILAB_JOB --unbuffered${ILAB_SRUN_ARGS:+ $ILAB_SRUN_ARGS}"' bash -c "$JOB"'
}

ilab_gpu_type() {  # ilab_gpu_type GRES -> the GPU type it names (gpu:a100:1 -> a100, gpu:a100 -> a100), else nothing
  local IFS=:; set -- $1
  case "${2:-}" in ''|*[!0-9]*) echo "${2:-}" ;; esac
}

ilab_pick_node() {  # ilab_pick_node TYPE < `sinfo -h -N -o "%N %G"` -> the first node whose GRES has that type
  awk -v t="$1" '{ n = split($2, g, ","); for (i = 1; i <= n; i++) { split(g[i], f, ":"); if (f[1] == "gpu" && f[2] == t) { print $1; exit } } }'
}

ilab_job() {  # ilab_job BIND -> the body of the srun job (runs on the GPU node, in REMOTE_DIR)
  local hf="${REMOTE_HF_HOME:-\$PWD/.hf-cache}" profiles=""
  [ -z "$ILAB_PROFILES" ] || profiles=' HEARING_PROFILES_DIR="$PWD/profiles"'
  cat <<JOB
echo "job \$SLURM_JOB_ID on \$(hostname -s), CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true
.venv/bin/python -m hearing.device
exec env HF_HOME="$hf" HEARING_MODELS_DIR="\$PWD/models" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=$ILAB_THREADS MKL_NUM_THREADS=$ILAB_THREADS$profiles HEARING_PIPELINES=$(printf %q "${REMOTE_PIPELINES:-$HQ_PIPELINES}") .venv/bin/python -m hearing serve --host $1 --port $REMOTE_PORT
JOB
}

ilab_launch() {  # ilab_launch NODE LOGIN -> bash script for the login node: (re)start the job, wait for /health, print JOBID=
  local node="$1" bind="127.0.0.1" health_host="127.0.0.1"
  # The tunnel ends on the login node: a job on another node must listen on the cluster network.
  [ "$node" = "$2" ] || { bind="0.0.0.0"; health_host="$node"; }
  cat <<LAUNCH
set -euo pipefail
cd $(printf %q "$REMOTE_DIR")
[ -x .venv/bin/python ] || { echo "remote: no .venv in \$PWD (bootstrap it once: docs/DEPLOY.md)" >&2; exit 1; }
.venv/bin/python scripts/check_deps.py --install || { echo "remote: could not install the core deps of pyproject.toml" >&2; exit 1; }
.venv/bin/python -c 'import hearing, torch, qrcode' || { echo "remote: .venv cannot import hearing/torch" >&2; exit 1; }
LAUNCH
  if [ -n "$ILAB_REUSE" ]; then
    echo "squeue -h -j $ILAB_REUSE | grep -q . || { echo 'remote: job $ILAB_REUSE is not running' >&2; exit 1; }"
  else
    cat <<LAUNCH
old="\$(squeue -h -u "\$USER" -n $ILAB_JOB -o %i | tr '\\n' ' ')"
if [ -n "\${old// /}" ]; then echo "remote: cancelling previous $ILAB_JOB job(s) \$old" >&2; scancel \$old; sleep 3; fi
LAUNCH
  fi
  local log=serve.log; [ -z "$ILAB_REUSE" ] || log="serve-$REMOTE_PORT.log"
  # /health must report the synced REVISION: a server that answers with another git_rev is an older
  # process still holding the port, never the one just started (T-137).
  cat <<LAUNCH
want="\$(cat REVISION 2>/dev/null || true)"
[ -n "\$want" ] || echo "remote: no REVISION in \$PWD; cannot check which server answers /health" >&2
health() {  # -> "<git_rev> <pid ..., started ...>" of whatever answers /health on the job's node
  curl -fsS --max-time 2 http://$health_host:$REMOTE_PORT/health 2>/dev/null | .venv/bin/python -c 'import json, sys; d = json.load(sys.stdin); print(d.get("git_rev") or "unknown", "pid %s, started %s" % (d.get("pid"), d.get("started_at")))' 2>/dev/null
}
LAUNCH
  if [ -n "$ILAB_REUSE" ]; then  # the port check must run where the step binds, not on the login node
    cat <<LAUNCH
if out="\$(health)"; then
  read -r rev info <<<"\$out"
  echo "remote: $health_host:$REMOTE_PORT already serves git_rev \$rev (\$info); a new step could not bind it and /health would answer from the old server. Set REMOTE_PORT to a free port (e.g. $((REMOTE_PORT + 1)))" >&2
  exit 1
fi
LAUNCH
  fi
  if [ "$node" = "$2" ]; then
    cat <<LAUNCH
if ! .venv/bin/python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", $REMOTE_PORT))' 2>/dev/null; then
  echo "remote: port $REMOTE_PORT is taken on \$(hostname -s); set REMOTE_PORT" >&2; exit 1
fi
LAUNCH
  fi
  cat <<LAUNCH
JOB=$(printf %q "$(ilab_job "$bind")")
export JOB
nohup setsid $(ilab_srun "$node") > $log 2>&1 < /dev/null &
step=\$!
id="$ILAB_REUSE"
stale=""
mismatch() { [ -z "\$stale" ] || echo "remote: /health on $health_host:$REMOTE_PORT answers git_rev \$stale, not the synced REVISION \$want: an older server holds the port" >&2; }
for _ in \$(seq $ILAB_WAIT); do
  [ -n "\$id" ] || id="\$(squeue -h -u "\$USER" -n $ILAB_JOB -o %i | head -1)"
  if out="\$(health)"; then
    read -r rev info <<<"\$out"
    if [ -z "\$want" ] || [ "\$rev" = "\$want" ]; then echo "JOBID=\$id"; echo "REV=\$rev"; exit 0; fi
    stale="\$rev (\$info)"
  fi
  if [ -n "\$id" ] && [ -z "\$(squeue -h -j "\$id" 2>/dev/null)" ]; then
    mismatch; echo "remote: job \$id ended; tail of $log:" >&2; tail -30 $log >&2; exit 1
  fi
  if [ -n "$ILAB_REUSE" ] && ! kill -0 "\$step" 2>/dev/null; then
    mismatch; echo "remote: step in job $ILAB_REUSE ended; tail of $log:" >&2; tail -30 $log >&2; exit 1
  fi
  sleep 1
done
mismatch
echo "remote: no /health on git_rev \${want:-?} after ${ILAB_WAIT}s (job \${id:-not started}); tail of $log:" >&2; tail -30 $log >&2
if [ -n "$ILAB_REUSE" ]; then kill "\$step" 2>/dev/null || true; elif [ -n "\$id" ]; then scancel "\$id"; fi
exit 1
LAUNCH
}

cmd_ilab() {
  local dry=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) dry=1; shift ;;  # print the launch script (and its srun line) without touching iLab
      --side) shift ;;  # handled at the top (use_side): state, job name and ports of an A/B job
      *) die "ilab: unknown option $1" ;;
    esac
  done
  case " $ILAB_SRUN_ARGS $ILAB_GRES " in
    *--cpus-per-task*|*--mem*|*" -c "*) die "ilab: never pass --cpus-per-task/--mem/-c (the cluster rejects them; a step sees all 96 cores)" ;;
  esac
  [ -z "${ILAB_CPUS:-}" ] || say "WARNING: ILAB_CPUS=$ILAB_CPUS is ignored: iLab rejects every CPU flag and a job step already sees all 96 cores (T-115)"
  [ -z "$ILAB_SIDE" ] || say "side job $ILAB_JOB: state $STATE, remote port $REMOTE_PORT, local port $LOCAL_PORT (the live job and its tunnel are untouched)"
  [ -z "$ILAB_PROFILES" ] || [ -d "$ILAB_PROFILES" ] || die "ilab: ILAB_PROFILES=$ILAB_PROFILES is not a directory"
  local gpu_type; gpu_type="$(ilab_gpu_type "$ILAB_GRES")"
  [ -n "$REMOTE_PORT_SET" ] || REMOTE_PORT=18765
  local login node
  if [ -n "$dry" ]; then
    login="${ILAB_LOGIN:-ilab3}"; node="${ILAB_NODE:-$login}"
    if [ -z "$ILAB_NODE" ] && [ -n "$gpu_type" ] && [ -n "${ILAB_SINFO:-}" ]; then  # ILAB_SINFO: canned sinfo output
      node="$(ilab_pick_node "$gpu_type" <<<"$ILAB_SINFO")"; [ -n "$node" ] || die "ilab: no node has a $gpu_type GPU"
    fi
    ilab_launch "$node" "$login"
    return
  fi
  # preflight: a dead or half-open ControlMaster hangs rsync/srun later (ServerAliveInterval=30 keeps it up)
  [ -S "$ILAB_SOCKET" ] && ilab_ssh -o BatchMode=yes -o ConnectTimeout=10 "$ILAB_HOST" true 2>/dev/null \
    || die "ilab: no live ControlMaster at $ILAB_SOCKET; open it first (docs/DEPLOY.md#rutgers-ilab)"
  login="$(ilab_ssh "$ILAB_HOST" hostname -s)"
  node="${ILAB_NODE:-$login}"
  if [ -z "$ILAB_NODE" ] && [ -n "$ILAB_REUSE" ]; then  # the new step must run on the job's own node
    node="$(ilab_ssh "$ILAB_HOST" squeue -h -j "$ILAB_REUSE" -o %N)" || die "ilab: squeue failed"
    [ -n "$node" ] || die "ilab: job $ILAB_REUSE is not running; unset ILAB_REUSE"
  elif [ -z "$ILAB_NODE" ] && [ -n "$gpu_type" ]; then
    node="$(ilab_ssh "$ILAB_HOST" sinfo -h -N -o "'%N %G'" | ilab_pick_node "$gpu_type")" \
      || die "ilab: sinfo failed"
    [ -n "$node" ] || die "ilab: sinfo lists no node with a $gpu_type GPU ($ILAB_GRES); set ILAB_NODE"
    say "ILAB_GRES=$ILAB_GRES: $gpu_type is on $node"
  fi
  [ "$node" = "$login" ] || say "WARNING: job on $node, not the login node $login: the server listens on 0.0.0.0 (no auth)"

  say "syncing tracked files to $ILAB_HOST:$REMOTE_DIR (login node $login)"
  ilab_ssh "$ILAB_HOST" "mkdir -p $(printf %q "$REMOTE_DIR")"
  git ls-files -z | rsync -az --from0 --files-from=- \
    -e "ssh -o ControlPath=$ILAB_SOCKET -o ControlMaster=no -o BatchMode=yes $SSH_OPTS" ./ "$ILAB_HOST:$REMOTE_DIR/"
  local want; want="$(revision)"
  ilab_ssh "$ILAB_HOST" "echo $want > $(printf %q "$REMOTE_DIR")/REVISION"
  say "synced git rev $want"
  if [ -n "$ILAB_PROFILES" ]; then  # gitignored voice profiles; the job reads them as HEARING_PROFILES_DIR
    say "syncing voice profiles $ILAB_PROFILES -> $ILAB_HOST:$REMOTE_DIR/profiles/ ($(ls "$ILAB_PROFILES" | tr '\n' ' '))"
    rsync -az --delete -e "ssh -o ControlPath=$ILAB_SOCKET -o ControlMaster=no -o BatchMode=yes $SSH_OPTS" \
      "$ILAB_PROFILES/" "$ILAB_HOST:$REMOTE_DIR/profiles/"
  fi

  say "starting $ILAB_JOB on $node ($ILAB_GRES, --time=$ILAB_TIME), port $REMOTE_PORT"
  local out job rev
  out="$(ilab_ssh "$ILAB_HOST" bash -s < <(ilab_launch "$node" "$login"))" || die "ilab: server did not come up (see above)"
  job="$(sed -n 's/^JOBID=//p' <<<"$out" | tail -1)"
  rev="$(sed -n 's/^REV=//p' <<<"$out" | tail -1)"
  [ "$rev" = "$want" ] || die "ilab: $node:$REMOTE_PORT serves git_rev ${rev:-?}, not the synced $want"
  say "job $job is serving git_rev $rev on $node:$REMOTE_PORT"
  echo ilab > "$STATE/host"
  echo "$job" > "$STATE/ilab_job"
  echo "$REMOTE_DIR" > "$STATE/remote_dir"

  ilab_close_tunnel
  local target="127.0.0.1"; [ "$node" = "$login" ] || target="$node"
  local fwd="127.0.0.1:${LOCAL_PORT}:${target}:${REMOTE_PORT}"
  # The forward lives in the human's master (no extra process); `down` cancels exactly this spec.
  ssh -o ControlPath="$ILAB_SOCKET" -O forward -L "$fwd" "$ILAB_HOST" || die "ilab: could not forward $fwd"
  echo "$fwd" > "$STATE/ilab_forward"
  say "tunnel 127.0.0.1:$LOCAL_PORT -> $login -> $target:$REMOTE_PORT"
  wait_health "http://127.0.0.1:${LOCAL_PORT}/health" 15 || die "no /health through the tunnel"
  local served  # the tunnel's local port could be held by another forward or a local server
  served="$(curl -fsS --max-time 3 "http://127.0.0.1:${LOCAL_PORT}/health" \
    | "$PY" -c 'import json, sys; print(json.load(sys.stdin).get("git_rev") or "unknown")')" || served="?"
  [ "$served" = "$want" ] || die "ilab: /health through the tunnel (127.0.0.1:$LOCAL_PORT) reports git_rev $served, not the synced $want: another server holds LOCAL_PORT"
  finish
  say "serving git_rev $served (job $job, $node:$REMOTE_PORT via 127.0.0.1:$LOCAL_PORT)"
}

ilab_close_tunnel() {
  local fwd; fwd="$(state_get ilab_forward)"
  [ -n "$fwd" ] || return 0
  ssh -o ControlPath="$ILAB_SOCKET" -O cancel -L "$fwd" "$ILAB_HOST" 2>/dev/null && say "tunnel $fwd closed" \
    || say "could not cancel forward $fwd (master gone?)"
  rm -f "$STATE/ilab_forward"
}

ilab_down() {
  local job; job="$(state_get ilab_job)"
  # Cancel the recorded job and anything else of ours with the job name; never other jobs of the user.
  if ilab_ssh "$ILAB_HOST" "ids=\"$job \$(squeue -h -u \"\$USER\" -n $ILAB_JOB -o %i | tr '\\n' ' ')\"; \
      [ -z \"\${ids// /}\" ] || scancel \$ids; \
      for _ in \$(seq 30); do [ -z \"\$(squeue -h -u \"\$USER\" -n $ILAB_JOB)\" ] && exit 0; sleep 1; done; \
      squeue -u \"\$USER\" -n $ILAB_JOB; exit 1"; then
    say "iLab job(s) ${job:-named $ILAB_JOB} cancelled; squeue has none left"
  else
    say "could not scancel $ILAB_JOB on $ILAB_HOST, or it is still in squeue (job $job holds its GPU until --time ends)"
  fi
  ilab_close_tunnel
  rm -f "$STATE/ilab_job"
}

# ---------------------------------------------------------------- local (on the GPU box itself, or a rehearsal)

cmd_local() {  # same bootstrap without ssh; no tunnel, the server listens on REMOTE_PORT here
  local dir="${REMOTE_DIR}" bind="127.0.0.1"
  [ "${1:-}" = "--public" ] && bind="0.0.0.0"
  case "$dir" in /*) ;; *) dir="$HOME/$dir" ;; esac
  mkdir -p "$dir"
  say "syncing tracked files to $dir"
  git ls-files -z | rsync -a --from0 --files-from=- ./ "$dir/"
  REMOTE_DIR="$dir" REMOTE_PORT="$REMOTE_PORT" BIND="$bind" HF="$REMOTE_HF_HOME" \
    PIPELINES="$REMOTE_PIPELINES" TORCH_INDEX="$TORCH_INDEX" bash -s < <(bootstrap_script)
  echo local > "$STATE/host"
  echo "$dir" > "$STATE/remote_dir"
  LOCAL_PORT="$REMOTE_PORT"; URL="ws://127.0.0.1:${REMOTE_PORT}/ws/session"
  finish
}

# ---------------------------------------------------------------- RunPod

runpod_api() {  # runpod_api METHOD PATH [JSON]
  [ -n "${RUNPOD_API_KEY:-}" ] || die "RUNPOD_API_KEY is not set (RunPod > Settings > API Keys)"
  curl -fsS -X "$1" "$RUNPOD_API$2" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" ${3:+--data "$3"}
}

cmd_runpod() {
  local gpu="NVIDIA L4"
  while [ $# -gt 0 ]; do
    case "$1" in
      --gpu) gpu="$2"; shift 2 ;;
      *) die "runpod: unknown option $1" ;;
    esac
  done
  local key="${SSH_PUBKEY:-}"
  if [ -z "$key" ]; then
    for k in ~/.ssh/id_ed25519.pub ~/.ssh/id_rsa.pub; do [ -f "$k" ] && key="$k" && break; done
  fi
  [ -n "$key" ] && [ -f "$key" ] || die "no SSH public key (ssh-keygen -t ed25519, or set SSH_PUBKEY)"
  local pod; pod="$(state_get pod)"
  if [ -z "$pod" ]; then
    say "creating a RunPod pod ($gpu, $RUNPOD_IMAGE); billing starts now"
    local body
    body="$(GPU="$gpu" IMAGE="$RUNPOD_IMAGE" KEY="$(cat "$key")" "$PY" -c '
import json, os
print(json.dumps({
    "name": "hearing-gpu", "imageName": os.environ["IMAGE"], "computeType": "GPU",
    "cloudType": "SECURE", "gpuTypeIds": [os.environ["GPU"]], "gpuCount": 1,
    "containerDiskInGb": 40, "volumeInGb": 30, "volumeMountPath": "/workspace",
    "ports": ["8000/http", "22/tcp"], "env": {"PUBLIC_KEY": os.environ["KEY"]},
}))')"
    pod="$(runpod_api POST /pods "$body" | "$PY" -c 'import json, sys; print(json.load(sys.stdin)["id"])')"
    echo "$pod" > "$STATE/pod"
    say "pod $pod created"
  else
    say "reusing pod $pod (scripts/gpu_up.sh down --terminate to delete it)"
    runpod_api POST "/pods/$pod/start" >/dev/null 2>&1 || true
  fi
  local ip="" port="" deadline=$((SECONDS + 600))
  while [ -z "$ip" ] || [ -z "$port" ]; do
    [ "$SECONDS" -lt "$deadline" ] || die "pod $pod has no public SSH port after 10 min"
    sleep 5
    read -r ip port < <(runpod_api GET "/pods/$pod" | "$PY" -c '
import json, sys
p = json.load(sys.stdin)
print(p.get("publicIp") or "", ((p.get("portMappings") or {}).get("22") or ""))') || true
  done
  say "pod $pod ssh root@$ip -p $port; HTTP proxy wss://$pod-8000.proxy.runpod.net/ws/session (needs --public)"
  REMOTE_HF_HOME="${REMOTE_HF_HOME:-/workspace/hf-cache}" REMOTE_DIR="${REMOTE_DIR_POD:-/workspace/hearing-remote}" \
    cmd_ssh "root@$ip" --ssh-port "$port"
}

# ---------------------------------------------------------------- docker (this machine)

cmd_docker() {
  local gpus=(--gpus all) index="$TORCH_INDEX" tag=hearing-gpu
  if [ "${1:-}" = "--cpu" ]; then gpus=(); index=https://download.pytorch.org/whl/cpu; tag=hearing-cpu; fi
  command -v docker >/dev/null || die "docker not found"
  say "building $tag from tracked files (git archive HEAD; commit first to include changes)"
  git archive --format=tar HEAD | docker build -t "$tag" --build-arg TORCH_INDEX="$index" -
  docker rm -f hearing-gpu >/dev/null 2>&1 || true
  docker run -d --name hearing-gpu "${gpus[@]}" -p "127.0.0.1:${LOCAL_PORT}:8000" \
    -v hearing-cache:/workspace -e HEARING_PIPELINES="$REMOTE_PIPELINES" "$tag" >/dev/null
  echo "docker" > "$STATE/host"
  wait_health "http://127.0.0.1:${LOCAL_PORT}/health" 120 || { docker logs --tail 30 hearing-gpu >&2; die "container not healthy"; }
  finish
}

# ---------------------------------------------------------------- status, probe, down

current_url() { local u; u="$(state_get url)"; echo "${u:-$URL}"; }

cmd_probe() { "$PY" -m hearing.server.remote --url "${1:-$(current_url)}" --seconds "${SECONDS_PROBE:-5}"; }

cmd_status() {
  local url; url="$(current_url)"
  echo "host:   $(state_get host)"
  echo "pod:    $(state_get pod)"
  [ "$(state_get host)" != ilab ] || echo "ilab:   job $(state_get ilab_job), forward $(state_get ilab_forward)"
  echo "tunnel: $(state_get tunnel.pid)"
  echo "url:    $url"
  local health="${url/#ws:/http:}"; health="${health%/ws/session}/health"
  curl -fsS --max-time 3 "$health" && echo || echo "health: unreachable (laptop falls back to local tiers)"
}

cmd_down() {
  local host; host="$(state_get host)"
  if [ "$host" = "ilab" ]; then
    ilab_down
  elif [ "$host" = "docker" ]; then
    docker rm -f hearing-gpu >/dev/null && say "container removed"
  elif [ "$host" = "local" ]; then
    local dir; dir="$(state_get remote_dir)"
    [ -f "$dir/serve.pid" ] && kill "$(cat "$dir/serve.pid")" 2>/dev/null && say "local server stopped"
    rm -f "$dir/serve.pid"
  elif [ -n "$host" ]; then
    local port; port="$(state_get ssh_port)"
    # shellcheck disable=SC2206
    local ssh=(ssh -o ConnectTimeout=5 $SSH_OPTS)
    [ -n "$port" ] && ssh+=(-p "$port")
    local dir; dir="$(state_get remote_dir)"
    "${ssh[@]}" "$host" "cd $(printf %q "${dir:-$REMOTE_DIR}") && { [ ! -f serve.pid ] || kill \$(cat serve.pid) 2>/dev/null; rm -f serve.pid; }" \
      && say "server on $host stopped" || say "could not reach $host to stop the server"
  fi
  close_tunnel
  rm -f "$STATE/host" "$STATE/ssh_port" "$STATE/remote_dir" "$STATE/url"
  local pod; pod="$(state_get pod)"
  if [ -n "$pod" ]; then
    if [ "${1:-}" = "--terminate" ]; then
      runpod_api DELETE "/pods/$pod" >/dev/null && rm -f "$STATE/pod" && say "pod $pod deleted"
    else
      runpod_api POST "/pods/$pod/stop" >/dev/null && say "pod $pod stopped (volume kept; --terminate deletes it)"
    fi
  fi
}

case "${1:-}" in
  ilab) shift; cmd_ilab "$@" ;;
  ssh) shift; cmd_ssh "$@" ;;
  local) shift; cmd_local "$@" ;;
  runpod) shift; cmd_runpod "$@" ;;
  docker) shift; cmd_docker "$@" ;;
  probe) shift; cmd_probe "$@" ;;
  status) cmd_status ;;
  down) shift; cmd_down "$@" ;;
  *) sed -n '2,11p' "$0"; exit 2 ;;
esac
