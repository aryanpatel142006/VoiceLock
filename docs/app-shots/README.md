# App check screenshots

Captured on macOS arm64 (Apple M4 Pro) with Playwright Chromium 153.0.8010.12
against app source revision `3823362bb1045ac8cf63c9e359d5975efbb98b30` (current
main merged, Oct 10 2026) using:

```sh
.venv/bin/python scripts/app_check.py
```

All three viewport journeys passed with zero console errors, decoded manifest
icons, and service-worker control after reload. Files are named
`SCREEN-WIDTHxHEIGHT.jpg`; each is a viewport capture at DPR 1, converted from
the command's PNG with `sips -s format jpeg -s formatOptions 70` to keep the
repository small.

| State | 390×844 | 1440×900 | 1920×1080 |
|---|---|---|---|
| welcome | [image](welcome-390x844.jpg) | [image](welcome-1440x900.jpg) | [image](welcome-1920x1080.jpg) |
| setup | [image](setup-390x844.jpg) | [image](setup-1440x900.jpg) | [image](setup-1920x1080.jpg) |
| people | [image](people-390x844.jpg) | [image](people-1440x900.jpg) | [image](people-1920x1080.jpg) |
| listen-selective | [image](listen-selective-390x844.jpg) | [image](listen-selective-1440x900.jpg) | [image](listen-selective-1920x1080.jpg) |
| listen-raw | [image](listen-raw-390x844.jpg) | [image](listen-raw-1440x900.jpg) | [image](listen-raw-1920x1080.jpg) |
| listen-quiet | [image](listen-quiet-390x844.jpg) | [image](listen-quiet-1440x900.jpg) | [image](listen-quiet-1920x1080.jpg) |

Fixture: Chromium fake mic, muted output, CPU, local passthrough/gain pipelines,
mock ASR/TTS/voice/faces adapters, no saved voice profiles. Quiet uses the real
server's captions-only path; without an isolated target, no speech is invented.
Visible numbers come from that test session's telemetry or browser measurements.
They are **not latency benchmarks or evidence of voice isolation**.

These are review snapshots, not pixel-comparison baselines: runtime IDs and
measured levels can change. Regenerate using the command in
[web/app/README.md](../../web/app/README.md), then review `results.json` before
converting the new PNGs to JPG here. Real hardware and iOS installation need a human run.
