# VoiceLock

Live selective hearing. Pick one person in the room, and VoiceLock plays you that voice while it turns the rest of the room down. The goal is assistive: for people who get overwhelmed in loud places, turn down the world without disconnecting from it.

Built at HackRU Fall 2026 by Aryan Patel, Joseph Roche and Ahmad Abbadi.

## What it does

- **Focus**: you hear the selected person, with the rest of the room reduced.
- **Quiet**: the room is silenced and you get captions from the selected person only.
- **Raw / isolated A/B** so the difference is audible on the spot.
- **Listen screen**: drag a voice into the ring to choose who to follow, and a room dial to set how far the rest of the room is turned down. In Quiet, captions fade in as the selected person speaks.
- **Welcome**: a short cinematic opening before setup.
- **Optional low-delay window** (`HEARING_TSE_WINDOW_S=0.5 HEARING_TSE_HOP_S=0.25`): shrinks the enrolled tier's window from 1.0 s to 0.5 s (a 0.75 s window-plus-hop delay in the engine). Off by default, and its end-to-end delay and quality are not among the measured results below; see `docs/research/T-420_short_window.md`.
- **Optional fine-tuned team separator** (`HEARING_SEP_CKPT`, `HEARING_PICK=fixed`) for the two enrolled voices. Weights are not in this repository; see `docs/MODELS.md` and `docs/STAGE_CARD.md`.
- **Saved voice profiles** for the people you want to follow, plus a live noise filter and level control.
- Every session records its input, output and an event log, so results come from recordings.

The engine is a chain of streaming stages (`hearing/`): audio I/O and level control, noise suppression, target-speaker extraction and picking, a fusion/gating mixer, and a FastAPI/WebSocket session server. The front end (`web/app/`) is a PWA with Listen, Setup and People screens, and a phone companion. `web/console/` is the developer console.

## Run it

Requirements: macOS or Linux, Python 3.11+, `ffmpeg`, `libsndfile`. Model weights are downloaded on first use and are not in this repository.

```bash
bash scripts/setup.sh            # creates .venv and installs the package

# the demo (opens the app; phones on the same Wi-Fi get a QR code)
bash scripts/demo.sh --judge --team --lan --app
```

Without `--app` the demo opens the stage page at `/judge`. `--team` loads the two-voice preset from `config/demo-team.yaml`, which expects saved voice profiles for each person (`scripts/enroll_profiles.sh`). Voice recordings and profiles are not included in this repository. More in `docs/DEMO.md`, `docs/APP.md` and `docs/STAGE_CARD.md`.

## Measured results

Only numbers that were measured, with their conditions (from `docs/CLAIMS.md`, section 1; the full list of what is not claimed is in the same file).

- With two real male voices (Aryan and Joseph), on a MacBook Pro (M4 Pro), the selected speaker is followed correctly when they talk alone.
- When both talk at once, the wrong voice is heard 6-25% of the time. That is the hard case; blind separation of two overlapping similar voices is the limit.
- Best run with Joseph as target: 8.6-9.1% wrong-speaker time, the other voice about 21 dB down, SI-SDRi +4.6 dB.
- Output delay: about 1.5 s on the Mac, about 1 s on a GPU server. This is a delay, not real time.
- The noise filter alone adds about 13 ms of algorithmic latency (engine side; excludes Bluetooth and the Focus delay).
- On a stand-in test clip with two different voices (not our voices): +8.3 to +8.5 dB SI-SDRi, other talker 18 dB down, locked on in about 5.5-5.9 s.
- Live captions are rough: 71-83% word error on a hard test clip with a competing talker.
- No user study. The sensory-overload framing is motivation, not a measured benefit.

## Repository layout

| Path | Contents |
| --- | --- |
| `hearing/` | The voice engine and session server |
| `web/` | The app (`web/app/`), developer console, JS SDK |
| `scripts/` | Demo launcher, enrollment, checks, rehearsal and analysis tools |
| `config/` | Presets and tier settings |
| `tests/` | Python unit and integration tests; JS tests live in `web/app/tests/` |
| `eval/` | Benchmarks, scenes and evaluation code |
| `experiments/`, `gpu_jobs/`, `third_party/` | Research experiments and GPU job scripts |
| `docs/` | Architecture, contracts, models, benchmarks, claims, runbooks, research notes |

## Team

Aryan Patel, Joseph Roche, Ahmad Abbadi.
