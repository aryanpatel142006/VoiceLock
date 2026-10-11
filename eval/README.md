# Evaluation

Everything that turns a stage into a number. Offline by default; the network is only needed for
the `full` fixture set and for downloading Whisper weights.

```
.venv/bin/python -m eval.bench --fast                         # M0 slice: passthrough on tiny, no WER (verify_full runs this)
.venv/bin/python -m eval.make_fixtures --tiny                 # offline, 16 fixtures
.venv/bin/python -m eval.make_fixtures --full                 # network: LibriSpeech test-clean, 96 fixtures
.venv/bin/python -m eval.bench --stages passthrough --stages "gain(db=-6)>delay(ms=20)" --set tiny
.venv/bin/python -m eval.bench --stages gtcrn --set full --wer whisper --limit 8
.venv/bin/python -m eval.bakeoff --set tiny --candidates identity,oracle --windows 0.5,1,2
.venv/bin/python -m eval.scene_score eval/scenes/ --oracle    # scorer sanity check on the scripted scenes
.venv/bin/python -m eval.target_eval eval/scenes/             # TargetSelector on the scripted scenes
```

## Modules

| Module | What it does |
|---|---|
| `make_fixtures.py`, `fixtures/` | Two-talker + noise mixtures with clean stems and manifests: `eval/fixtures/generated/<set>/<id>/{mix,target,interferer,noise}.wav` + `manifest.json`, and `<set>/index.json`. Layout, grid and licences in `fixtures/README.md`. |
| `metrics.py` | `si_sdr`, `si_sdri`, `pesq_wb`, `stoi`, `wer` / `wer_batch` (normalised text), `LatencyStats` (p50/p95/max/mean in ms), `rtf`, `summarize` (mean/median/n per numeric key, nan skipped). |
| `transcribe.py` | Reference transcriber for WER: `get_transcriber("mock" \| "whisper" \| "auto")`. Whisper is faster-whisper `small.en` (CPU int8 on Macs); `auto` falls back to the mock when it is not installed or cached. The mock is deterministic. |
| `scenes/`, `scene_score.py` | Scripted multi-face scenes (plain YAML) and the target-selection scorer (accuracy, switch latency); no `hearing` imports. Schema in `scenes/README.md`. |
| `target_eval.py` | Runs `hearing.vision.target.TargetSelector` over the scenes and scores it. |
| `bench.py` | The benchmark runner every stage reports through (below). |
| `bakeoff.py`, `bakeoff_fixtures.py` | Focus HQ bake-off of TSE candidates over sliding windows (below). |

## Benchmark runner (`eval/bench.py`)

```
python -m eval.bench --stages SPEC [--stages SPEC ...] [--set tiny|full] [--wer off|mock|auto|whisper]
                     [--fixtures eval/fixtures/generated] [--out eval/results] [--limit N] [--fast]
```

- `SPEC` is a registry pipeline string such as `"dfn>gate>mixer"` or `"gain(db=-6)>delay(ms=20)"`.
  Stages outside `hearing.audio.basic` are found by importing the stage packages on demand.
- `--fast` = `--stages passthrough --set tiny --wer off`. If the tiny set is missing it is generated
  offline (any run on `--set tiny` does this); `full` must be generated first.
- For every fixture and pipeline: the pipeline is built fresh, `reset()`, and fed `mix.wav` in 10 ms
  chunks with `StageContext.default()` (suppression 0.5), followed by enough zero chunks to flush
  `latency_ms`. Each `process` call is timed with `perf_counter`. The output is shifted back by
  `latency_ms` and scored against `target.wav`.
- Per-fixture row: `latency_ms` (algorithmic, as the pipeline declares it), compute `p50_ms` /
  `p95_ms` / `max_ms` per 10 ms chunk, `rtf` (compute time / audio fed), `si_sdr`, `si_sdri`
  (vs the mix), `pesq` (wide-band), `stoi`, the same three for the unprocessed mix (`*_mix`), `wer`
  (output vs the manifest transcript, nan when `--wer off`), and `extras` (the stages'
  `ctx.telemetry.extras`, e.g. whether GTCRN ran its mock backend).
- Table: one row per pipeline. Metrics are means over fixtures; compute p50/p95 are pooled over
  every chunk of every fixture. `-` means nan (e.g. WER off, PESQ on silence).
- JSON: `eval/results/<YYYYMMDD-HHMMSS>.json` (gitignored) with `device` (`hearing.device.describe()`),
  `git_sha` (`-dirty` if tracked files changed), `machine`, `python`, `fixture_set`, `rows`,
  `aggregates` (`summarize` per pipeline plus pooled `compute_ms`) and `failures`. nan is written as `null`.
- Exit status: 0 on success, 1 if any fixture failed (each failure prints its stage spec and
  fixture id), 2 if the fixture set is missing.

`--wer mock` echoes the reference, so WER is 0 by construction; it only exercises the plumbing.
Real WER needs `--wer whisper` (or `auto` with `small.en` cached).

Measured numbers go to `docs/BENCHMARKS.md` with the command that produced them.

## Focus HQ bake-off (`eval/bakeoff.py`)

Scores every target-speaker extraction (TSE) candidate in `hearing/tse/` at several sliding-window
lengths on one audio-visual fixture set.

```
.venv/bin/python -m eval.bakeoff --set tiny --candidates identity,oracle --windows 0.5,1,2 --out eval/results/
```

| Flag | Default | Meaning |
|---|---|---|
| `--set` | `tiny` | A set dir (subdirs with `manifest.json`), one fixture dir, or `tiny` (synthetic, generated into `eval/fixtures/generated/bakeoff_tiny/`, no downloads) |
| `--candidates` | `identity,oracle` | Comma list of names from `hearing.tse.candidates()`, or `all` |
| `--windows` | `0.5,1,2` | Window lengths in seconds |
| `--hop` | `0.5` | Hop as a fraction of the window |
| `--transcriber` | `auto` | `auto` (faster-whisper if installed and `small.en` is cached, else mock), `mock`, `whisper` |
| `--limit` | all | Only the first N fixtures |
| `--out` | `eval/results/` | Writes `bakeoff_<stamp>.json` |
| `--summary` | next to the JSON | Markdown summary, ranked by SI-SDRi then latency |

Per candidate x window it reports SI-SDRi vs the mixture (mean over fixtures), corpus WER of the
estimate, RTF, algorithmic latency (= window length) and compute p50/p95 per window, GPU memory
(`torch.cuda.max_memory_allocated` on CUDA, `torch.mps.current_allocated_memory` on MPS, none on
CPU) and the device. A candidate that fails to load or raises gets an `error` and stays in the table.
With the mock transcriber WER is a plumbing check only (it echoes the reference).

Fixture layout (`eval/bakeoff_fixtures.py`): `mix.wav target.wav interferer.wav noise.wav manifest.json`,
optional `lips.npy` (`T x H x W` uint8 mouth crops at 25 fps) and `video.mp4`. Lips come from `lips.npy`,
else a fixed mouth crop of `video.mp4` at the manifest's `target_box` (normalised `[x, y, w, h]`; valid
only for frontal static talkers such as GRID), else none. `p_speaking` is an oracle proxy from the clean
target's frame energy until ASD lands. `enroll` is the manifest's `enroll_wav`, else the first 3 s of the
clean target.

### Adding a candidate
Create `hearing/tse/<name>.py` with a class implementing `hearing.tse.TSECandidate` (`name`, `licence`,
`needs` ⊆ {`lips`, `p_speaking`, `enroll`}, `load()`, `extract(mix, sr, *, lips, lips_fps, p_speaking,
enroll)` returning 16 kHz audio of the same length as `mix`) and decorate it with
`@register_candidate("<name>")`. Modules under `hearing/tse/` are discovered automatically; keep heavy
imports inside `load()`. The harness passes only the inputs listed in `needs` and wraps the candidate in
`hearing.tse.SlidingWindowExtractor`, so `extract` sees one window at a time with lips and `p_speaking`
sliced on the 25 fps grid (`round(window_s * 25)` frames). ClearerVoice: its deps are in the `ml` extra;
install the package itself with `.venv/bin/pip install --no-deps clearvoice` (it pins `numpy<2`, which
conflicts with kokoro-onnx; it imports on numpy 2.x).
