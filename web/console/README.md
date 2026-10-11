# Debug console

Served at `/` by `python -m hearing serve` (open http://127.0.0.1:8000). No build step: plain ES modules.

| File | Role |
|---|---|
| `index.html`, `style.css` | One screen per `docs/DESIGN.md` |
| `app.js` | WebSocket session, mic/camera capture, telemetry rendering |
| `protocol.js` | Wire format from `docs/CONTRACTS.md#events` (pure, unit tested) |
| `audio-worklet.js` | Mic framing (20 ms) and the playback jitter buffer (40 ms target, 160 ms cap) |

- `?ws=ws://host:port/ws/session` points the page at another session server; the Session field does the same.
- Keys 1-6 pick a mode. Click a face box to lock the target; the target's box is green and labelled TARGET.
- The Tier badge (also on the camera view) reads `AV · gate` / `AV` while camera frames drive the session and `Audio-only` without a camera, face detector or frames. The Vision fact shows detector + ASD, applied fps, ms per frame and dropped frames.
- **Runs on: Local / GPU / Auto** (next to the Focus HQ toggle) sends `control.hq_backend`. The readout under it is what plays now (`hq_active`) and the remote RTT (`remote_rtt_ms`); it turns amber when GPU was asked for but local plays, and its tooltip is `hq_reason`. GPU needs `HEARING_REMOTE_URL` on the server.
- The Room row shows the 1 Hz `context` vector. A `recommendation` shows a toast; "Switch" sends the mode (the server logs it as an accepted choice for the preference model). In Auto the toast only reports the switch, and the Mode readout shows `Auto → <mode>`.
- The big number is mouth-to-ear latency measured in the browser once the mic runs (mic frame + server-reported lookahead/resampling/compute + socket round trip + playback buffer + `AudioContext` output latency). Before that it shows the server's own estimate.
- Use wired headphones: echo cancellation is off so the pipeline hears the raw room.
- `getUserMedia` needs a secure context: `localhost`/`127.0.0.1` works, a LAN IP needs https.

Tests:

```sh
node --test web/console/tests/*.test.mjs        # protocol, byte-compatible with hearing.server.protocol
.venv/bin/python -m hearing serve --port 8914 &
PLAYWRIGHT=<node_modules>/playwright node web/console/tests/e2e.mjs --url http://127.0.0.1:8914 --out /tmp/shots
```

The e2e check uses Chromium's fake mic and camera, injects face boxes into telemetry (vision arrives in M2), and points a page at a dead socket to check the error state.

Vision end to end (real JPEG frames, server-side detection with the weight-free marker detector):

```sh
HEARING_FACES=mock HEARING_ASD=lipmotion .venv/bin/python -m hearing serve --port 8915 --pipelines "natural=passthrough>gate" &
PLAYWRIGHT=<node_modules>/playwright node web/console/tests/vision-e2e.mjs --url http://127.0.0.1:8915 --out /tmp/vision-shots
```

It writes a fake camera clip (green marker faces, the left one's mouth opening at 4 Hz) and a matching
noise-burst mic file, then checks boxes, the automatic TARGET pick, the AV badge, and audio-only without a camera.
`HEARING_ASD=lipmotion` matters: Light-ASD (the `auto` pick when its weights are present) is a real model and
scores green rectangles arbitrarily. On a shared machine use a free port instead of 8915.

Focus HQ backend selector (two servers: a "GPU" one and the laptop pointing at it; `--gpu-pid` kills the GPU mid-run to check the fallback):

```sh
HEARING_FACES=mock HEARING_ASD=lipmotion .venv/bin/python -m hearing serve --port $GPU --pipelines 'focus_hq=passthrough' & GPID=$!
HEARING_FACES=mock HEARING_ASD=lipmotion HEARING_REMOTE_URL=ws://127.0.0.1:$GPU/ws/session HEARING_HQ_BACKEND=local \
  .venv/bin/python -m hearing serve --port $LAPTOP --pipelines 'natural=passthrough;focus_hq=gain(db=-6)' &
PLAYWRIGHT=<node_modules>/playwright node web/console/tests/hq-backend-e2e.mjs --url http://127.0.0.1:$LAPTOP --gpu-pid $GPID --out /tmp/hq-shots
```
