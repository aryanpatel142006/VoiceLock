# Audio fixture bank

Reproducible two-talker-plus-noise mixtures with clean stems, built by
`eval/make_fixtures.py`. Everything is 16 kHz, mono, float32 in the pipeline, saved as PCM16
WAV through `hearing.audio.io`.

```
.venv/bin/python -m eval.make_fixtures --tiny                 # offline, 16 fixtures, well under 1 s
.venv/bin/python -m eval.make_fixtures --full                 # network: streams LibriSpeech, 96 fixtures
.venv/bin/python -m eval.make_fixtures --tiny --out /tmp/fx   # custom output root (tests use tmp_path)
.venv/bin/python -m eval.make_fixtures --make-tiny-source     # network: rebuild the committed tiny clips
.venv/bin/python -m eval.make_fixtures --av --tiny            # audio-visual: GRID, 16 fixtures (network on first run)
.venv/bin/python -m eval.make_fixtures --av --full            # audio-visual: GRID, 160 fixtures
```

## Layout

```
eval/fixtures/
  tiny/                      committed sources for --tiny (3 WAVs < 100 KB each + manifest.json)
  generated/                 gitignored output root (override with --out)
    <set>/index.json         every fixture of the set, paths relative to the set directory
    <set>/<id>/mix.wav       target + interferer + noise, peak 0.9
    <set>/<id>/target.wav    clean target speech (what a stage should recover)
    <set>/<id>/interferer.wav  second talker, scaled to the SIR
    <set>/<id>/noise.wav     babble or pink noise, scaled to the SNR
    <set>/<id>/manifest.json
```

Fixture ids are `<index>_snr<dB>_sir<dB>_<noise>`, e.g. `003_snr0_sir5_babble`.

## Sets

| Set | Source | Grid | Count | Needs |
|---|---|---|---|---|
| `tiny` | the two committed talkers, both orderings, committed babble | SNR {-5, 0, 5, 10} dB x SIR {0, 5} dB | 16 | nothing (offline by construction; `datasets` is never imported) |
| `full` | LibriSpeech test-clean streamed from `openslr/librispeech_asr`: 12 pairs of 4-8 s utterances from different speakers; noise alternates babble (6 other speakers) / pink (1/f-shaped Gaussian) per pair | same | 96 | network, ~1-2 min |

Both sets are deterministic for a given `--seed` (two runs are byte-identical). The seed drives
speaker/utterance selection and the pink-noise generator of the full set; the tiny set has no
random choices, the seed is only recorded.

## Mixing

All levels are RMS over the whole clip, after the interferer and noise are trimmed or looped to
the target length:

- `snr_db = 20 log10(rms(target) / rms(noise))`
- `sir_db = 20 log10(rms(target) / rms(interferer))`
- `mix = target + interferer + noise`, then peak-normalised to 0.9 with the **same gain on every
  stem**, so the saved stems still sum to the mix (within PCM16 rounding, < 1e-4).

Code: `eval/fixtures/mix.py` (`make_mixture`, `Manifest`, `write_fixture`, `load_fixture`),
`eval/fixtures/sources.py` (`TinySource`, `LibriSpeechSource`, `make_babble`, `pink_noise`).

## Manifest fields (`manifest.json`, and each entry of `index.json`)

| Field | Meaning |
|---|---|
| `id` | fixture id (directory name) |
| `set` | `tiny` or `full` |
| `sample_rate` | 16000 |
| `duration_s` | clip length in seconds |
| `snr_db`, `sir_db` | target-to-noise and target-to-interferer RMS ratios, dB |
| `target_id`, `interferer_id` | LibriSpeech utterance ids |
| `noise_type` | `babble` or `pink` |
| `transcript` | target transcript, normalised lowercase (`eval.metrics.normalize_text`) |
| `interferer_transcript` | the same for the interferer |
| `paths` | stem files; relative to the fixture directory in `manifest.json`, to the set root in `index.json` |
| `seed` | the `--seed` the set was built with |
| `source` | dataset, split, target/interferer utterance + speaker ids (+ sex when known), and the noise provenance (babble utterance ids or the pink generator) |

## Tiny source clips (`eval/fixtures/tiny/`)

| File | Content | LibriSpeech id | Speaker |
|---|---|---|---|
| `speech_a.wav` | 3.0 s, "yes my dear watson i have solved the mystery" | 1580-141084-0019 | 1580, female |
| `speech_b.wav` | 3.0 s, "a golden fortune and a happy life" | 61-70968-0002 | 61, male |
| `noise_babble.wav` | 3.0 s babble: six RMS-normalised utterances from six other speakers summed | see `manifest.json` | 6930, 2830, 1284, 2094, 7176, 1221 |

Each file is PCM16 at 16 kHz (96 044 bytes). Utterances were chosen with a native length within
0.2 s of 3.0 s and padded with zeros / trimmed at leading silence to exactly 3.0 s, so the
transcripts in `manifest.json` stay exact. Speaker sex comes from LibriSpeech `SPEAKERS.TXT` and
was cross-checked by median pitch.

## Audio-visual sets (`--av`)

For active speaker detection, target selection (M2) and AV target extraction (M3). Same stems,
mixing maths and SNR/SIR definitions as above, plus a `video.mp4` per fixture.

| Set | Pairs | Layouts | Conditions | Count | Noise |
|---|---|---|---|---|---|
| `av_tiny` | 2 GRID pairs (s1 target / s4 interferer, then s4 / s1) | `single`, `two_face` | SNR {0, 5} x SIR {0, 5} dB | 16 | committed tiny babble |
| `av_full` | 10 GRID pairs, target talker alternating s1 / s4 | `single`, `two_face` | SNR {-5, 0, 5, 10} x SIR {0, 5} dB | 160 | babble (committed clip) / pink, alternating per pair |

- **`single`**: the target's own GRID video, native 360x288 (fits inside 640x360, never
  upscaled). The interferer is audible but off-screen (`interferer_visible: false`,
  `target_side: "center"`).
- **`two_face`**: both talkers' videos centre-cropped to 320:360 and composited side by side into
  640x360; both audible. `target_side` is `left` / `right` following L, R, R, L, ... over pairs,
  so both sides occur and each talker appears on both sides in `av_full`. The interferer video is
  looped exactly like its audio when it is shorter than the target.
- Utterances are drawn with `--seed` (default 0) from each talker's 1000 clips; every utterance is
  used once. Fixture length is the target clip (GRID clips are 3.0 s = 75 frames).
- `video.mp4`: H.264 yuv420p at 25 fps with `mix.wav` muxed as AAC (16 kHz mono), so the file
  plays as the scene. Audio length is exactly `n_frames * 640` samples: frame `k` covers samples
  `[640 k, 640 (k + 1))`.
- Target audio is the GRID clip's own track (MPEG-1 layer II, 44.1 kHz) decoded to 16 kHz mono;
  it is in sync with the video by construction.

Fixture ids are `<index>_<layout>_snr<dB>_sir<dB>_<noise>`, e.g. `004_two_face_snr0_sir0_babble`.
Load one with `eval.make_fixtures.load_av_fixture(dir) -> (AVManifest, Stems)`; the audio-only
`mix.load_fixture` rejects the extra keys.

### AV manifest fields (in addition to the audio fields above)

| Field | Meaning |
|---|---|
| `layout` | `single` or `two_face` |
| `video` | `video.mp4` (also in `paths.video`) |
| `fps` | 25.0 |
| `frame_size` | `[width, height]` of the video: `[360, 288]` single, `[640, 360]` two_face |
| `n_frames` | frame count; `duration_s == n_frames / fps` |
| `faces` | list of `{"actor": "grid-s1", "role": "target" \| "interferer", "side": "left" \| "right" \| "center", "box": [x, y, w, h]}`; boxes normalised to the frame, top-left origin |
| `target_side` | `left` / `right` (two_face) or `center` (single) |
| `interferer_visible` | `true` for two_face |
| `target_intervals` | target speech spans `[[start_s, end_s], ...]` from its GRID `.align` (silence and `sp` removed, adjacent words merged) |
| `interferer_intervals` | the same for the interferer, repeated where its clip loops, clipped to the fixture |
| `box_source` | `grid_approx`: boxes are a fixed per-talker estimate (hairline to chin), measured by eye on one frame each (GRID framing is fixed per talker), not a detector output; they are good to roughly ±5 % of the frame, not per-frame tracks |
| `source` | as above with `dataset: "GRID"`, `split: "zenodo:3625687"`, talker sexes, and `layout` |

Example (`av_tiny/004_two_face_snr0_sir0_babble/manifest.json`, abridged):

```json
{
  "id": "004_two_face_snr0_sir0_babble", "set": "av_tiny", "duration_s": 3.0, "snr_db": 0.0, "sir_db": 0.0,
  "target_id": "grid-s1-prbj4n", "interferer_id": "grid-s4-lbby1p",
  "transcript": "place red by j four now", "interferer_transcript": "lay blue by y one please",
  "layout": "two_face", "video": "video.mp4", "fps": 25.0, "frame_size": [640, 360], "n_frames": 75,
  "faces": [
    {"actor": "grid-s1", "role": "target", "side": "left", "box": [0.0742, 0.24, 0.2461, 0.63]},
    {"actor": "grid-s4", "role": "interferer", "side": "right", "box": [0.6305, 0.38, 0.2109, 0.47]}
  ],
  "target_side": "left", "interferer_visible": true,
  "target_intervals": [[0.6, 2.14]], "interferer_intervals": [[0.41, 2.21]]
}
```

### GRID source and cache

`eval/fixtures/grid.py` reads the GRID corpus from Zenodo record
[3625687](https://zenodo.org/records/3625687): `alignments.zip` (20 MB, word timings) is
downloaded whole; each speaker archive (`s1.zip`, `s4.zip`, ~400-500 MB) is **not**: single `.mpg`
clips (~0.45 MB each) are read out of it with HTTP `Range` requests. Everything is cached in
`eval/fixtures/generated/_sources/grid/` (gitignored) and reused; with a warm cache `--av` is
offline. Cold `--av --tiny` on laptop-a downloads 22.4 MB in ~28 s; warm it takes ~5-11 s.
`--av --full` adds ~10 MB. Talkers: `s1` (male), `s4` (female), checked by eye.

Code: `eval/fixtures/video.py` (ffmpeg-pipe `read_video` / `write_video` / `read_audio` /
`mux_audio`, `resample_fps`, `fit_resize`, `side_by_side`, box mapping),
`eval/fixtures/grid.py` (`GridCorpus`, `iter_utterances`, `.align` parsing),
`eval/make_fixtures.py` (`make_av`, `make_av_set`, `AVManifest`, `load_av_fixture`).
ffmpeg comes from `imageio-ffmpeg` (`eval` extra) or `PATH`; OpenCV is not used.

## Licences

- **LibriSpeech** (Panayotov, Chen, Povey, Khudanpur, ICASSP 2015), used for the committed tiny
  clips and the full set: **CC BY 4.0**. Attribution: "LibriSpeech ASR corpus, OpenSLR SLR12,
  http://www.openslr.org/12".
- **GRID audio-visual sentence corpus** (Cooke, Barker, Cunningham, Shao, "An audio-visual corpus
  for speech perception and automatic speech recognition", JASA 120(5), 2006), used for the AV
  sets: **CC BY 4.0**. Attribution: "The Grid Audio-Visual Speech Corpus, Zenodo,
  doi:10.5281/zenodo.3625687". Only talkers s1 and s4 are used; no GRID file is committed.
- Pink noise is generated; no licence.
- No real recorded noise corpus (DEMAND, MUSAN) is bundled; see `docs/decisions/T-006.md`.
