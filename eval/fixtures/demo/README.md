# Offline demo input

`.venv/bin/python -m eval.make_fixtures --demo` builds four ~30 s two-face scenes from the cached
GRID talkers (`eval/demo_scene.py`, offline once `--av --full` has run), each as
`<name>/{clip.mp4, manifest.json, stems/{mix,target,other,noise}.wav}`: 0-5 s target alone,
5-15 s alternating turns, 15-25 s overlap at SIR 0, 25-30 s target alone; babble at SNR 10 dB.
`demo-mf-easy` (male target left, female other right) is the default for `scripts/replay.py`,
`scripts/demo_regress.sh` and the console's REPLAY button; `demo-mf-hard` (SNR 5), `demo-mm`
(s1/s2) and `demo-ff` (s4/s7) are in the picker. The manifest carries the stems
(`scripts/session_score.py`) and the per-face speaking timeline (`scripts/session_score_nr.py`).

## Legacy loop


Prepare `.venv/bin/python -m eval.make_fixtures --demo-legacy` after generating the
`av_full` GRID bank once. This produces `grid-two-face.mp4`: a cached two-face
mixture looped to 12 seconds with synchronized mono 16 kHz audio and 25 fps video.
It also copies the source mixture's clean stems to `stems/{mix,target,other,noise}.wav` and
records the target face (`target`: actor, side, box) in the manifest, so
`scripts/session_score.py` can score a replayed session against them (older manifests fall
back to `source_fixture` under `$HEARING_FIXTURES_DIR/av_full`).
No source or model is downloaded by `--demo` or replay. GRID's dataset terms
still apply; the original fixture manifest is copied into the generated manifest.

MP4s and generated manifests are intentionally ignored (repo media rule).
Copy the prepared directory to the demo laptop before going offline. Replace
`grid-two-face.mp4` with the consented demo-day recording to keep the one-click
button unchanged. Other local MP4 names can be selected through POST `/replay`.

Ingested real-room clips live in `<name>/{clip.mp4, manifest.json}` (`scripts/ingest_clip.py`, recipe in docs/DEMO.md §9b); only the manifest may be committed. The console lists every clip here in the picker next to Replay.
