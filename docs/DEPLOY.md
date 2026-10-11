# Deploy: the remote GPU path

The fast path (Natural: `dfn>gate>mixer`) always runs on the laptop. Focus HQ can run on a GPU box. That box runs the ordinary session server (`python -m hearing serve`), and the laptop reaches it through the `remote` stage, which is a client of the same `/ws/session` contract (docs/CONTRACTS.md#events). If the remote is down, the laptop plays its local fallback and reports the fallback tier. Nothing else changes.

```
browser/phone ──ws──► laptop: hearing serve
                        natural  = dfn>gate>mixer                (local, ≤ 100 ms)
                        focus_hq = remote(fallback="dfn>gate>mixer")
                                     │ ws (SSH tunnel 127.0.0.1:18765)
                                     ▼
                      GPU box: hearing serve   focus_hq = <HQ stage>   (Dockerfile / gpu_up.sh)
```

## Bring it up
All commands run from the repo root on laptop-a. `scripts/gpu_up.sh` syncs the **tracked** files (`git ls-files`, so no weights, fixtures or `.venv`) and creates a venv on the box (reusing a preinstalled CUDA torch through `--system-site-packages`). It installs `.[ml]`, starts `hearing serve` on the box's `127.0.0.1:8000`, opens an SSH tunnel to `127.0.0.1:18765`, and runs the probe (health + RTT). At the end it prints the laptop command.

| Where | Command | Notes |
|---|---|---|
| Rutgers iLab (Slurm) | `scripts/gpu_up.sh ilab` | Runs the server inside an `srun --gres=gpu:1` job on the login node, through the human's ControlMaster socket. See [Rutgers iLab](#rutgers-ilab) below. |
| RunPod | `RUNPOD_API_KEY=... scripts/gpu_up.sh runpod --gpu "NVIDIA L4"` | Creates a pod (image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, ports `8000/http` and `22/tcp`, 30 GB volume at `/workspace`, your `~/.ssh/id_ed25519.pub` injected), waits for SSH, then runs the `ssh` path. **Billing runs until `down`.** Also try `--gpu "NVIDIA GeForce RTX 4090"`. |
| Any SSH GPU box | `scripts/gpu_up.sh ssh user@host [--ssh-port N]` | Same as iLab. |
| Docker on a Linux GPU host | `scripts/gpu_up.sh docker` | Builds the `Dockerfile` from `git archive HEAD` (commit first) and runs it with `--gpus all` on `127.0.0.1:18765`. `--cpu` builds a CPU-only rehearsal image. |
| On the box itself / rehearsal | `scripts/gpu_up.sh local` | The same install and server without SSH, on `REMOTE_PORT`. |

### Rutgers iLab
`scripts/gpu_up.sh ilab` is the path tested on Oct 5. It differs from `ssh` because iLab GPUs are visible only inside Slurm jobs, and SSH works only through a socket that a human opened.

1. **Socket (human, once per day).** Off campus there are no keys and no Kerberos, so a human opens a ControlMaster with a password login, for example:
   `ssh -M -S ~/.ssh/cm-ilab -o ControlPersist=yes -o ServerAliveInterval=30 -fN ap2772@ilab.cs.rutgers.edu` (`ServerAliveInterval=30` keeps the socket alive through idle NAT timeouts). `gpu_up.sh ilab` first runs a preflight, `ssh -o BatchMode=yes -o ConnectTimeout=10 ... true`, through the socket and stops with the message below if it fails.
   The script never opens a master (`BatchMode=yes`, `ControlMaster=no`). If `ssh -S ~/.ssh/cm-ilab -O check ap2772@ilab.cs.rutgers.edu` fails, it stops with that message.
2. **Checkout (once).** `~/hearing-remote` already exists, with `.venv` (Python 3.12, torch 2.8.0+cu128, `clearvoice` installed `--no-deps`), `models/` and the `av_full` fixtures. `ilab` does **not** pip install anything. It rsyncs the tracked files over the checkout (edits made on iLab are overwritten), checks that `.venv` imports `hearing` and `torch`, and starts the server. If `pyproject.toml` gained a dependency, install it on iLab by hand.
3. **Job.** From the login node, `nohup setsid srun --gres=gpu:1 -w <login node> --time=$ILAB_TIME -J hearing-hq --unbuffered bash -c "$JOB"`. The job runs `python -m hearing serve --host 127.0.0.1 --port 18765` with `HEARING_PIPELINES='focus_hq=tse(candidate="av_mossformer2", window_s=2.0, hop_s=1.0, require_gpu=true)>mixer'` (the 2 s AV tier from [BAKEOFF.md](BAKEOFF.md); override it with `REMOTE_PIPELINES`). The script waits up to `ILAB_WAIT=300` s for `/health` on the login node, then records the job id.
4. **Tunnel.** `ssh -S ~/.ssh/cm-ilab -O forward -L 127.0.0.1:18765:127.0.0.1:18765`. The forward lives in the human's master, not in a new process. The laptop then reaches `ws://127.0.0.1:18765/ws/session`, and the script runs the probe.
5. **Down.** `scripts/gpu_up.sh down` runs `scancel` on the recorded job id and on any other `hearing-hq` job of yours. It waits until `squeue` shows none, then cancels the forward with `-O cancel`. Other jobs of yours (an interactive `srun --pty bash`) are never touched.

**srun constraints learned on Oct 5:**
- GPUs exist only inside a job. `nvidia-smi -L` on the login node prints `No devices found.`; `srun --gres=gpu:1 ... nvidia-smi -L` shows one RTX 4500 Ada (ilab3), `CUDA_VISIBLE_DEVICES=0`.
- **Never pass `--cpus-per-task`, `-c` or `--mem`**: the cluster rejects the job. The script refuses them in `ILAB_SRUN_ARGS`.
- **CPUs (T-115):** `squeue` shows 2 CPUs per job, but a step on the job sees all 96 (`nproc` 96, affinity 0-95). iLab rejects every CPU flag: `--cpus-per-task`, `--cpus-per-gpu` and `--ntasks-per-node` all fail with `srun.real: error: Job rejected: Please do not specify the number of CPUs. There is actually no limit.` `gpu_up.sh` passes none (a leftover `ILAB_CPUS` only prints a warning, T-118). Enrolled overruns come from the one Python session process, not the allocation. Evaluations on the job: shard them (`--scenes` / per-config runs) and run 2-3 at a time, not one per core.
- **GPU type:** `ILAB_GRES=gpu:a100:1` (a type in the GRES) without `ILAB_NODE` picks the first node `sinfo -h -N -o '%N %G'` lists with that type (the A100 is only on rlab2); the server then binds `0.0.0.0` on that node like any `ILAB_NODE` other than the login node.
- **Restart on new code without losing the GPU (T-110):** `ILAB_REUSE=<jobid> REMOTE_PORT=<free port> scripts/gpu_up.sh ilab` re-syncs the tree (and `REVISION`, which `/health` reports as `git_rev`), starts the server as an `srun --jobid --overlap` step on the job's node (from `squeue`, unless `ILAB_NODE` is set), logs to `serve-<port>.log`, never runs `scancel` and re-points the tunnel. It refuses a `REMOTE_PORT` that already answers `/health` on the job's node, and succeeds only when `/health` (on the node and through the tunnel) reports the synced `REVISION` as `git_rev`; otherwise it exits non-zero naming the stale rev (T-137). The step shares the job's CPUs; the old server keeps running until the job ends. Add `REMOTE_PIPELINES='focus_hq=…'` to run a **second config side by side** on the same job and GPU (each on its own `REMOTE_PORT`); `gpu_up.sh down` cancels the job and so stops both. Step-by-step: `docs/TESTING_GUIDE.md` §7.
- **Voice profiles on the remote (T-129):** profiles (T-125, `<name>/*.wav`) are gitignored, so the code sync never carries them. `ILAB_PROFILES=<local dir> scripts/gpu_up.sh ilab` rsyncs that dir through the ControlMaster to `$REMOTE_DIR/profiles/` (`--delete`: the remote copy mirrors it) and starts the server with `HEARING_PROFILES_DIR=$PWD/profiles`; the remote enrolls its own copy by name when the laptop sends `enroll: profile`. Unset: the remote has no profiles (a profile enroll is rejected there and only the laptop's tiers use it).
- **A second job for an A/B (T-118):** `scripts/gpu_up.sh ilab --side` (or `ILAB_SIDE=1`) keeps its own state dir (`hearing-gpu/side`), names its job `hearing-hq-side` (so it never `scancel`s the live `hearing-hq`), defaults `REMOTE_PORT` to 18767 and picks a free `LOCAL_PORT`, so the live job's tunnel stays up; it prints the side URL. `ILAB_SIDE=1 scripts/gpu_up.sh status|down` acts on the side job only.
- **Which config does the remote run?** `/health` reports `tiers_config` (short hashes of `config/tiers*.yaml` as loaded at start); `demo_regress` warns when the remote's `tiers-remote.yaml` differs from the local file.
- **Dependencies:** after the rsync the launch script runs `scripts/check_deps.py --install`, which pip-installs any core dependency of `pyproject.toml` missing from the remote `.venv` (a new `qrcode` dependency once killed the remote server at import).
- **Always pass `--time`.** The `unlimited` partition's default is **7 days**, so a forgotten job would hold a GPU for a week. The script defaults to `ILAB_TIME=02:00:00`; use `00:02:00` for a smoke test.
- **Pin the job to the login node the socket is on** (`-w ilab3`; the script reads it with `hostname -s` through the socket). The server then binds `127.0.0.1` and the forward ends on the same host. `ILAB_NODE=ilab1` works but binds `0.0.0.0` (anyone on the cluster network can connect, there is no auth) and is untested.
- **The login node is shared**, so `REMOTE_PORT` defaults to 18765 instead of 8000. The script checks that the port is free and stops if it is not. Even on `127.0.0.1`, other iLab users on ilab3 could connect to the server while it runs.
- Allocation was immediate on ilab3 (`queued and waiting` for under 1 s). `ilab` waits through a queue anyway, up to `ILAB_WAIT`.
- Run it on laptop-a: the socket exists only there. Every subcommand also takes `ILAB_HOST`, `ILAB_SOCKET`, `ILAB_JOB` and `ILAB_GRES` (see the header of `scripts/gpu_up.sh`). `scripts/gpu_up.sh ilab --dry-run` prints the login-node script without touching iLab (this is what `tests/test_gpu_up.py` checks).

Pick what the remote runs with `REMOTE_PIPELINES='focus_hq=<hq stage spec>'`. The default remote `focus_hq` is `passthrough` until the HQ stage (T-019) lands. That is enough to prove the transport and measure RTT.

Then on the laptop:

```bash
export HEARING_REMOTE_URL=ws://127.0.0.1:18765/ws/session
.venv/bin/python -m hearing serve \
  --pipelines 'natural=dfn>gate>mixer; focus_hq=remote(fallback="dfn>gate>mixer")'
```

Other commands: `scripts/gpu_up.sh status` (host, pod, tunnel, `/health`), `scripts/gpu_up.sh probe [URL]` (JSON: health, RTT p50/p95/max, lost frames, the remote's own latency breakdown and device), and `scripts/gpu_up.sh down` (stops the server and tunnel, and **stops** the pod; `down --terminate` deletes the pod and its volume). State is kept in `~/.local/state/hearing-gpu/`, never in the repo.

### Dockerfile by hand
```bash
git archive --format=tar HEAD | docker build -t hearing-gpu -          # CUDA 12.4 torch wheels
docker run --gpus all -p 127.0.0.1:8000:8000 -v hearing-cache:/workspace \
  -e HEARING_PIPELINES='focus_hq=<hq stage spec>' hearing-gpu
```
Build args: `TORCH_INDEX` (default `https://download.pytorch.org/whl/cu124`; use `.../whl/cpu` for CPU, or `""` to skip torch), `EXTRAS` (default `ml`), and `EXTRA_PIP` (extra packages for an experiment, e.g. `"onnxruntime-gpu clearvoice"`). The Hugging Face cache and `models/` live on the `/workspace` volume, so weights download once.

## The `remote` stage
`remote(url=…, fallback="passthrough", delay_ms=200, mode="focus_hq", tier="focus_hq", fallback_tier="audio-only", stall_ms=1000, connect_timeout=2, retry_s=2)`. `url` defaults to `$HEARING_REMOTE_URL`.

- **Wire.** On connect it sends `hello` (16 kHz, 10 ms frames, `client: "remote-stage"`) and `control` with `mode`. After that it sends one `audio in` frame per 10 ms hop, carrying the hop's `capture_ts`. When the local `suppression` or the target lock changes, it forwards them as `control`. The remote answers each frame with an `audio out` frame with the same `seq`; the time between the two is the **RTT**.
- **Playout.** `process` never blocks on the network. It plays the remote output of the hop `delay_ms` earlier, which is a fixed jitter buffer, so `latency_ms = delay_ms`. A hop that comes back too late is filled with the local fallback output of the same delayed hop, and counted in `misses`. Choose `delay_ms` ≥ p95 RTT + the remote compute time per hop.
- **Camera.** The session forwards every camera JPEG to a playing remote stage (`hello.video: true`). `HEARING_REMOTE_VIDEO=0` on the laptop keeps frames local (`hello.video: false`, nothing sent; T-129): with a voice profile loaded the remote's Enrolled tier needs no camera, and skipping the remote's vision frees its CPU. Without a profile the remote then has no AV Focus HQ or ASD calibration.
- **Down.** If the remote is unreachable, disconnected, or has sent no audio for `stall_ms`, the stage plays the fallback **undelayed** and sets `ctx.telemetry.tier = fallback_tier`. It reconnects every `retry_s` in a background thread and goes back to remote audio on its own. Every source switch is crossfaded over one hop.
- **Reporting.** `ctx.telemetry.tier` (`focus_hq` while remote audio plays), `ctx.telemetry.latency_ms["remote_rtt"]` (p50 over the last 5 s), and `ctx.telemetry.extras["remote"]` (`stats()`: connected, RTT p50/p95/max, sent/received/dropped, connects, misses, remote latency and device, last error). Today the wire `telemetry` event carries the tier only; the session must merge `latency_ms["remote_rtt"]` into it (see Open items). Until then, use `scripts/gpu_up.sh probe`. The stage also logs RTT every 10 s at INFO on the `hearing.server.remote` logger, but `hearing serve` does not show that logger by default.
- **Lifecycle.** One remote connection per laptop session. It opens on the first hop, and when the session drops its pipeline the stage is garbage-collected and the connection closes.

## Measured
| Date | Path | RTT p50 | RTT p95 | Notes |
|---|---|---|---|---|
| 2026-10-04 | laptop-a loopback, `gpu_up.sh local` (fresh venv, CPU torch), probe 5 s | 0.66 ms | 1.24 ms | 500/500 frames, 0 lost; remote `server` 0.1 ms/frame |
| 2026-10-04 | console client → laptop server (`focus_hq=remote(delay_ms=200)`) → loopback remote | 1.25 ms (client-side, whole chain) | n/a | tier `focus_hq`; after `gpu_up.sh down`, tier `audio-only`, 150/150 frames still returned |
| 2026-10-05 | laptop-a → iLab ilab3 (RTX 4500 Ada, cuda) via `gpu_up.sh ilab`, ControlMaster forward, probe 5 s | 5.7-5.9 ms | 6.8 ms (48.4 ms on the first probe after start) | 500/500 frames, 0 lost; remote `focus_hq` = `tse(av_mossformer2@2s)>mixer`, algorithmic 3000 ms (2 s window + 1 s hop); tse compute 1.8 ms/hop because the probe sends no video (no lips, so the mix passes through) |
| pending | laptop-a → RunPod L4 over the SSH tunnel | n/a | n/a | needs the RunPod account |

## Security
The session server has no authentication. By default it binds to `127.0.0.1` on the box and is reached only through the SSH tunnel. `--public` binds `0.0.0.0` so that the RunPod HTTP proxy `wss://<pod>-8000.proxy.runpod.net/ws/session` works. Anyone with the URL can then use it, so use `--public` only for a short demo and run `down` afterwards.

## Open items
- **Video for remote AV-TSE.** The `remote` stage only sees audio. `RemoteStage.client.send_video(seq, capture_ts, jpeg)` exists, but the session does not call it yet (it counts video frames and drops them). The session or the HQ integration has to forward JPEG frames when `focus_hq` is remote.
- **RTT on the wire.** The session should merge `ctx.telemetry.latency_ms` (`remote_rtt`) into the `telemetry` event's `latency_ms`.
- **Pipeline latency.** `Pipeline.latency_ms` is fixed at build time, so telemetry `algorithmic` shows `delay_ms` even while the stage plays the undelayed fallback.
- **Remote algorithmic delay.** Resolved (T-078): the HQ window delay adds to `delay_ms`, and `RemoteStage.latency_ms` now includes it (read from the remote's telemetry `latency_ms.algorithmic`, `stats()["remote_algorithmic_ms"]`), so telemetry `total` and the A/B raw delay are honest while the remote plays. `delay_ms` itself only has to cover the p95 RTT.
- **AV-TSE on iLab has no video yet.** Remote `focus_hq` is the AV-MossFormer2 2 s tier on CUDA, but until the session forwards JPEG frames (first item), it has no lips and passes the mix through. Audio-only `enrolled` (PLAN's primary Focus HQ tier since Oct 5) needs an enrollment clip, which the remote session does not get either. Switch with `REMOTE_PIPELINES='focus_hq=tse(candidate="enrolled", ...)>mixer'` once enrollment reaches the remote.
- **Untested here.** The `ilab` subcommand ran for real (see Measured). The `runpod` and `ssh` subcommands were not run against a real host (no GPU account or SSH host yet), and the Docker image was not built (the Docker daemon was not running on laptop-a). The shared bootstrap was run for real through `gpu_up.sh local`.

### Fixtures and test timing on shared machines
- Gitignored demo fixtures (clips, stems, `av_full`) are referenced by path through `HEARING_FIXTURES_DIR` (set by `scripts/_py.sh` to the main checkout's `eval/fixtures/generated`); on iLab export it to the checkout's fixtures directory instead of copying clips into the repo.
- Timing tests scale their timeouts by `HEARING_TEST_SLOWDOWN` (default 1; set e.g. `3` on a busy machine): the cold-Kokoro `wait_idle(2.0)` waits and the tracker wall-clock assertions. `tests/conftest.py: scaled()`.
- `hearing serve --port 0` binds an ephemeral port and prints `bound port N; health: http://127.0.0.1:N/health`.
