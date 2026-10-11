# Architecture

The voice backend is one Python package, `hearing/`, run either as a local live loop (lowest latency) or as a WebSocket session server (browser console now, phone app at the event). Every processing step is a `Stage` (see `docs/CONTRACTS.md#schemas`), so tiers can be swapped and benchmarked with one runner.

## Components
| Path | Responsibility | Entry point |
|---|---|---|
| `hearing/types.py` | Shared dataclasses: `AudioChunk`, `VideoFrame`, `FaceTrack`, `TargetState`, `Telemetry`, `Mode` | n/a |
| `hearing/device.py` | `pick()` → `cuda` / `mps` / `cpu`; `describe()` for telemetry | n/a |
| `hearing/audio/` | Load/save, resample, 10 ms framing, ring buffers, sounddevice I/O | `hearing live`, `hearing latency-test` |
| `hearing/enhance/` | Noise suppression stages: `dfn.py` (DeepFilterNet3 via its ONNX export + onnxruntime, 10 ms streaming, strength = slider), `gtcrn.py` (published streaming code, 16 kHz) | stage registry |
| `hearing/vision/` | `faces.py` (MediaPipe detect + landmarks + tracker), `target.py` (centre/dwell/lock), `asd.py` (Light-ASD, lip-motion fallback) | stage registry |
| `hearing/fusion/` | `gate.py` (duck non-target time by slider), `mixer.py` (target + (1-s)·residual), `tiers.py` (fallback ladder) | stage registry |
| `hearing/tse/` | Focus HQ: sliding-window AV target extraction and enrolled TSE wrappers (bake-off winners) | stage registry |
| `hearing/synth/` | `asr.py` (streaming ASR adapters), `tts.py` (TTS adapters), `voice_match.py` (apparent gender/age → voice id), `pipeline.py` | stage registry |
| `hearing/context/` | `features.py` (1 Hz context vector), `prefs.py` (log, train, recommend, auto) | n/a |
| `hearing/server/` | FastAPI app: `/health`, `/ws/session`, serves `web/console/` | `hearing serve` |
| `hearing/cli.py` | `hearing live | serve | latency-test | bench` | `python -m hearing` |
| `web/console/` | Debug console: mic + camera capture, slider, modes, face boxes, live numbers | `http://localhost:8000/` |
| `eval/` | `make_fixtures.py`, `metrics.py`, `bench.py`, `bakeoff.py`; results in `eval/results/` | `python -m eval.bench` |
| `config/` | `voices.yaml` (voice buckets → ElevenLabs/local voice ids), `tiers.yaml` | n/a |

## Data flow
```
capture (mic 16k/10ms, camera 25fps)
  ├─ audio ─► [enhance: DFN3] ─► [fusion: gate] ─► [fusion: mixer] ─► out (Natural fast)
  │             └──────────────► [tse: HQ window] ─► [mixer] ─► out (Focus HQ, delayed, crossfaded)
  │                                        └──────► [synth: ASR] ─► phrases ─► [synth: TTS] ─► out (Synthetic) / captions (Text)
  └─ video ─► [vision: faces+tracks] ─► [vision: target] ─► TargetState ─► gate / tse / voice_match
                                     └► [vision: asd] ─► p(speaking) per track
every second: [context: features] ─► telemetry + prefs log
```
Messages on the wire are defined in `docs/CONTRACTS.md#events`. Every frame carries a monotonically increasing `seq` and capture timestamp so each stage can report its own latency.

## How to run
```
bash scripts/setup_worktree.sh   # once per worktree (links the main checkout's .venv)
bash scripts/verify_fast.sh      # lint + unit tests (no weights, no network), < 60 s
.venv/bin/python -m hearing live --mode natural     # local loop, wired headphones
.venv/bin/python -m hearing serve --port 8000       # session server + console
.venv/bin/python -m eval.bench --fast               # benchmark table
```

## ml
### Latency budget (Natural fast path, local Mac, wired headphones)
| Step | Budget |
|---|---|
| Audio input buffer | 10 ms |
| DFN3 algorithmic (20 ms window, 10 ms hop, 2-frame lookahead) | 40 ms |
| DFN3 compute (RTF ≈ 0.05-0.2 on CPU) | ≤ 10 ms |
| Gate + mixer | ≤ 2 ms |
| Audio output buffer | 10 ms |
| **Total target** | **≤ 100 ms (aim 70)** |
Through the browser console add ~20-40 ms (AudioWorklet + WebSocket). Bluetooth earbuds add 150-250 ms (fine for Synthetic, not for Natural).

### Fallback tiers (chosen by `fusion/tiers.py`, shown in telemetry)
| Tier | Needs | Natural output |
|---|---|---|
| 1 Focus HQ | GPU (local CUDA/MPS fast enough or remote), face track | AV-TSE target + (1-s)·residual, ~1-2 s delay |
| 2 Enrolled | 3-5 s clean enrollment of target (auto from ASD) | audio-only TSE target + residual |
| 3 Gate | camera | DFN3 + duck when target silent |
| 4 Audio-only | nothing | DFN3 with slider |

### Candidate models (decided by benchmarks, not by papers)
| Role | Primary | Fallback | Licence notes |
|---|---|---|---|
| Noise suppression | DeepFilterNet3 streaming via ONNX export (`enc`/`erb_dec`/`df_dec` with explicit state; also try `DeepFilterNet3_ll`), `atten_lim_db` knob, 48 kHz | GTCRN streaming (16 kHz, 48k params), RNNoise | MIT/Apache |
| AV target extraction | ClearerVoice `AV_MossFormer2_TSE_16K` (offline → sliding windows) | Dolphin (non-commercial ckpt), RTFS-Net (MIT) | Apache / NC |
| Audio-only TSE | WeSep (speaker-embedding TSE) | MossFormer2_SS + ECAPA match | Apache |
| Active speaker | Light-ASD (MIT, ~1M params) | lip-opening variance (MediaPipe landmarks) | MIT |
| Faces | MediaPipe detector + face landmarker | OpenCV DNN | Apache |
| Streaming ASR | ElevenLabs Scribe v2 Realtime (~150 ms) | Nemotron-speech-streaming 0.6B local, faster-whisper | paid / NVIDIA open |
| TTS | ElevenLabs Flash v2.5 / v4 Turbo WebSocket (~75-100 ms model) | Kokoro-82M / Piper local, mock | paid / Apache |
| Apparent gender/age | MiVOLO (face+body) / InsightFace genderage | audeering wav2vec2 age-gender (voice) | check / NC |
No causal audio-visual TSE with open weights exists (Oct 2026), which is why Focus HQ is windowed and delayed while the fast path is DFN3 + visual gate.

### Streaming DeepFilterNet3 (the M1 risk)
The Python `df.enhance()` API is whole-clip. Real 10 ms streaming uses the ONNX graphs exported for the Rust real-time plugin (encoder, ERB decoder, DF decoder) run with onnxruntime, carrying the GRU states between hops; STFT analysis/synthesis comes from `pyDF` (`DF` state) or our own 48 kHz STFT with the model's window and hop. Verified Oct 4 2026: the upstream repo's `models/` directory ships `DeepFilterNet3_onnx.tar.gz` (7.9 MB) and `DeepFilterNet3_ll_onnx.tar.gz` (36 MB, low-latency variant); GTCRN's `stream/` has `gtcrn_stream.py` and `onnx_models/`. The acceptance test is chunk-wise output equal to whole-file output within tolerance, plus RTF on MPS/CPU. If that fails, GTCRN's published streaming implementation is the fast path and DFN3 becomes a windowed higher-quality option.
