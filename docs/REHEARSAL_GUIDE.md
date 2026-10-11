# Rehearsal testbed guide (Saturday 10:00 AM)

Audio is captured by a **phone** (built-in mic, or the phone's browser). No RØDE mic. The page is
`/rehearsal.html` on the same server as `/judge`.

## Setup
1. Mac: `caffeinate -dims &`, then `bash scripts/demo.sh --judge --team` and wait until `/health` says `warm: ready`.
   Profiles: `HEARING_PROFILES_DIR` set as in `docs/REHEARSAL.md` step 3.
2. Phone on the same Wi-Fi. Serve with TLS (`hearing serve --tls`, `docs/DEMO.md#phones`): browsers only give a
   microphone to https pages. Install the CA from `/hearing-ca.crt` on the phone.
3. On the phone open `https://<mac-ip>:<port>/rehearsal.html` (or the Mac's own browser for a dry run).
   Tap **Open mic** and allow the microphone. Echo cancellation, noise suppression and AGC are off on purpose.
   iPhone captures at 48 kHz; the take's `take.json` lists each stream's real rate (`sample_rates`).
4. Put the phone where the listener's mic would be. Use wired headphones or AirPods for output, not the phone speaker (feedback).

## Matrix (changes apply instantly, no restart)
| Row | Choices | What it does |
|---|---|---|
| Representation | Stock ECAPA · Team projection head · Mahalanobis | **Recorded only.** The engine has no switch yet; the label lands in `take.json` and `events.jsonl`. `models/team_projection_head.pt` was not in the repo when this was written (page warns). |
| Ambiguity policy | Best-guess stream · Duck to room baseline · Pass-through mix | Stream sets `HEARING_GATED_PLAY=stream` in the server process (read every hop). Duck clears it. Pass-through sends `ab: raw`. The env var is process-wide: all sessions on that server follow it. |
| Latency ladder | Ultra / Balanced / High accuracy | Maps to the engine presets `speed`/`balanced`/`accuracy` = window/hop **0.5/0.25, 1.0/0.5, 1.5/0.75 s**. The requested 0.5/50 ms, 1.0/100 ms, 2.0/250 ms hops are not engine presets; the page shows what runs. |
| Suppression | slider 0-100 % | `control.suppression` (the T-093 dimmer curve). |

## Run a take
1. Pick a scenario (alternating turns, simultaneous overlap, rapid interruption, or freeform) and set the matrix.
2. **Start take**. The prompter starts its clock; two speakers read the A/B cards. Press **Marker** (with a note) for anything odd.
3. **Stop take**. Wait for "take saved". If it says chunks never uploaded, keep the phone on Wi-Fi and re-take.

## Where the take goes
`results/rehearsal/<UTC stamp>_<mode>_<scenario>/` on the Mac (override with `HEARING_REHEARSAL_DIR`; gitignore-safe, never commit):
`mic.wav` (raw phone input), `out.wav` (engine output as sent to the phone), `events.jsonl` (telemetry ~10 Hz, matrix changes,
card changes, markers, captions, errors; `t_ms` since take start), `take.json`.
**Not recorded:** separated stems (the engine does not send stems to clients; the server accepts any `stem_<n>` stream through
`POST /api/rehearsal/take/<id>/audio?stream=stem_0&rate=16000` once a session tap exists). Margin is `tse.margin` when present, else
`tse_similarity`; RTF is `1 - tiers.compute_headroom`; "mouth-to-ear" is the engine's declared `latency_ms.total` (browser/BT buffers not included).

## Not done / known
- Representation switch is not wired into the engine. Stems not captured. T-284 (senior takeover) was cut, so no overnight hyperparameters were folded in.
- Not run on a real phone: the page logic and server are unit tested only.

## Dry-run checklist (before the phone rehearsal)
- Open `http://localhost:<port>/rehearsal.html` in the Mac browser first. Loopback permits mic
  access without TLS. Use the phone input over USB if available; otherwise this is a browser/API
  smoke check only, not a phone capture measurement.
- With the default matrix untouched, tap **Open mic**. The engine connects only after the mic rate
  is known, then sends one `hello` and Balanced / processed / 80% controls. The server applies
  Best-guess when the page fetches its initial config. Confirm **engine: connected** and the displayed mic rate before recording.
- Record 10 seconds, speak, and stop. Confirm `mic.wav` and `out.wav` are non-empty and `take.json`
  lists the same sample rate for both. Listen with headphones and inspect the telemetry events.
- Repeat on the phone over HTTPS. Confirm mic permission, both non-empty WAV files, and audible
  output before running comparisons. Phone TLS/permission/output routing still needs this human check.

The session code (`Session._apply_hello` / `_process_audio`) resamples input to 16 kHz and output
back to the declared client rate. An offline WebSocket regression checks that a 48 kHz, 20 ms input
produces 960 output samples. The recorder therefore uses the AudioContext rate for both streams;
this checks the protocol, not real phone hardware latency.

## Troubleshooting
- **Mic blocked / no permission prompt:** use HTTPS on the phone, allow mic access in browser/site
  settings, and reload. On the Mac use `localhost`, not its LAN IP over HTTP. Do not bypass a phone
  certificate error and assume capture will work.
- **CA installation fails:** follow `docs/DEMO.md#phones`, including iOS certificate full trust,
  and check the URL uses the certificate's host/IP. If this cannot be resolved, use the iPhone as
  a Mac input over USB where supported and open the Mac loopback page. Confirm the OS default input
  is the phone; this page uses the default device. If no phone input is available, defer the phone
  rehearsal and label any Mac-input smoke test accurately.
- **Engine lost / silent output:** keep Wi-Fi connected, wait for **engine: connected**, then start
  a fresh take. Check `/health`, wired headphone routing, the telemetry error messages, and
  `out.wav`. A non-empty file can contain silence; listen before trusting a comparison.
- **Page closed or phone drops before Stop:** already-uploaded PCM remains, with `take.json`
  written from the start and updated with each upload. Browser-queued data is lost on page close.
  Events are appended and closed per batch, so abandoned takes hold no open event file handle.

## Recover an interrupted take
Stop the server first so no uploads can append during recovery. Use the recorded rates to wrap
leftover PCM as WAV (this preserves the PCM and does not invent missing chunks):

```sh
.venv/bin/python - results/rehearsal/<take-folder> <<'PY'
import json
import sys
import wave
from pathlib import Path

folder = Path(sys.argv[1])
meta = json.loads((folder / "take.json").read_text())
for pcm in folder.glob("*.pcm"):
    rate = meta["sample_rates"].get(pcm.stem)
    if rate is None:
        raise SystemExit(f"Missing rate for {pcm.name}; inspect before recovery")
    data = pcm.read_bytes()
    with wave.open(str(pcm.with_suffix(".wav")), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(data[:len(data) // 2 * 2])
    meta["samples"][pcm.stem] = len(data) // 2
meta["recovered_interrupted"] = True
(folder / "take.json").write_text(json.dumps(meta, indent=2))
PY
```

Keep `stopped: null` and `recovered_interrupted: true` when handing this take to analysis agents:
it is incomplete and must not be treated as a finished scripted trial.
