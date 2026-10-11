# T-303: Hugging Face candidate smoke tests

Measured **2026-10-10**, `MacBook-Air.local`, macOS 26.6.2 arm64, Python 3.14.7,
**CPU**, float32, one Torch/ORT thread. Five fresh processes ran **sequentially** (ERes2NetV2 was rerun alone to correct input-shape metadata);
the host was not reserved against other swarm workers. Torch 2.14.1, ORT 1.28.0.
This is a model-loading/inference feasibility screen for muse-c, not a separation-quality benchmark.

## Results

Load excludes imports, downloads, hashing and input preprocessing; it includes checkpoint
load and device placement/session construction. First inference is cold; p50/p95 use the
next five calls (only five observations, not production tail latency). Synchronization is
included on accelerator devices. Peak RAM is **whole-process maximum RSS in MiB**, including
imports, downloads, parameter inspection and post-inference checks, not model-only RAM/VRAM.
Torch parameter counts are exact; ONNX counts are initializer elements including buffers;
Silero's count includes both sample-rate branches, not a single active network's parameters.

| Model | Params / initializer elements | Load ms | First ms | Inference p50 / p95 ms | Peak RAM MiB | License | Verdict |
|---|---:|---:|---:|---:|---:|---|---|
| [ECAPA2](https://huggingface.co/Jenthe/ECAPA2) | 27.108 M | 110.09 | 1133.37 | 1075.85 / 1085.08 | 1133.2 | CC-BY-NC-4.0 | try: overlap/short enrollment eval |
| [ERes2NetV2 (fp32)](https://huggingface.co/csukuangfj/speaker-embedding-models) | 17.838 M | 98.69 | 193.18 | 188.85 / 189.35 | 618.8 | Apache-2.0 | try: compare to current ECAPA |
| [SkiM small causal](https://huggingface.co/lichenda/wsj0_2mix_skim_small_causal) | 7.878 M | 43.92 | 108.90 | 103.23 / 111.13 | 415.8 | CC-BY-4.0 | try: two-male-voice bake-off |
| [ConvTasNet Lyrics causal](https://huggingface.co/cadenzachallenge/ConvTasNet_Lyrics_Causal) | 10.847 M | 161.01 | 3027.69 | 3023.54 / 3025.38 | 635.1 | Apache-2.0 weights; MIT-derived inference code | skip: music domain; slow; no chunk state API |
| [Silero VAD (ONNX)](https://huggingface.co/onnx-community/silero-vad) | 0.545 M | 50.09 | 0.46 | 0.12 / 0.14 | 113.9 | MIT | try: mobile VAD, not speaker selection |

`try` means eligible for the next evaluation, not selected for the demo. The two speaker
embedders return `[1,192]`, but their coordinate systems are different: do not feed ECAPA2
or ERes2NetV2 embeddings into an extractor trained on SpeechBrain ECAPA without retraining.
No pipeline, enrollment threshold, contract or demo configuration changed.

## Inputs and output shapes

All audio is deterministic seed-303 sine components plus noise, bounded float32, generated
in memory. No files, microphone or speakers are used. Dummy signals demonstrate finite
outputs and runtime compatibility; they cannot establish EER, SI-SDRi, WER, leak,
wrong-speaker time, word retention, target absence behavior or listening quality.

| Model / script under `eval/hf_models/T-303/` | Input | Output | Timing scope / RTF |
|---|---|---|---|
| `ecapa2_smoke.py` | 2 s mono 16 kHz `[1,32000]` | `[1,192]` | waveform embedding; RTF 0.538 |
| `eres2netv2_smoke.py` | 2 s 16 kHz converted to CMN 80-bin Kaldi fbank `[1,198,80]` | `[1,192]` | ONNX only; preprocessing excluded; RTF 0.094 |
| `skim_smoke.py` | 2 s mono **8 kHz** `[1,16000]` | two `[1,16000]` waveforms | encoder + separator + decoder; RTF 0.052 |
| `causal_tasnet_smoke.py` | 2 s **44.1 kHz stereo** `[1,2,88200]` | `[1,2 sources,2 channels,88200]` | whole waveform; RTF 1.512 |
| `silero_vad_smoke.py` | 32 ms 16 kHz, 512 samples + 64 context `[1,576]`, state `[2,1,128]`, scalar rate | probability `[1,1]`, updated state `[2,1,128]` | stateful 32 ms chunk; RTF 0.0037 |

These use each model's own inference definitions/examples, adapted to dummy input:
ECAPA2's card `torch.jit.load -> model(audio)` with its documented JIT-optimizer disable;
ERes2NetV2's 3D-Speaker/sherpa ONNX fbank interface with Torchaudio Kaldi fbank rather than
installing sherpa; ESPnet's original SkiM encoder/separator/decoder and `forward_stream`
example; Cadenza's `load_model.py` local `from_pretrained`; Silero's official ONNX wrapper's
context/state/sample-rate contract. No random weights were used. All checkpoints loaded strictly.

## Why these five; claims versus evidence

- **Short-utterance candidate 1: ECAPA2.** The [authors' paper](https://arxiv.org/abs/2401.08342)
  describes training for robustness to overlap and short utterances. The [released card](https://huggingface.co/Jenthe/ECAPA2)
  supplies a TorchScript waveform embedding example. Non-commercial CC-BY-NC-4.0.
  The dummy result verifies execution only; test on Joseph/Aryan 0.5/1/2 s clips next.
- **Short-utterance candidate 2: ERes2NetV2.** The [authors' paper](https://arxiv.org/abs/2406.02167)
  targets short-duration verification (2 s and 3 s trials). This [HF ONNX export](https://huggingface.co/csukuangfj/speaker-embedding-models)
  is the Chinese 16k-common 3D-Speaker variant, **not** necessarily the paper's VoxCeleb trial checkpoint.
  License provenance: [3D-Speaker upstream](https://github.com/modelscope/3D-Speaker) Apache-2.0.
  Do not transfer paper EER to this export or English/noisy clips. Fbank execution is not
  proof of front-end parity with every upstream deployment.
- **Streaming separation candidate 1: SkiM.** The [released config](https://huggingface.co/lichenda/wsj0_2mix_skim_small_causal)
  has `causal: true`, 2 speakers, 8 kHz. Its [pinned upstream code](https://github.com/espnet/espnet/blob/3897ed8380bfb526d5c5dd2197eccbffbba7d8f8/espnet2/enh/layers/skim.py)
  exposes real stateful `forward_stream`. The script loads the actual trained modules and
  follows the upstream full-vs-frame example across 100 encoded frames, carrying state.
  Measured 100-frame streaming inference: 28.10 ms;
  max absolute full-vs-stream difference 5.72e-05.
  This proves separator-feature parity only, not a live 16 kHz Stage or stable speaker order.
- **Streaming separation candidate 2: causal ConvTasNet Lyrics.** The [checkpoint](https://huggingface.co/cadenzachallenge/ConvTasNet_Lyrics_Causal)
  uses causal convolution and channelwise normalization; the provided `load_model.py`
  works (the README example itself points to a different nonexistent model ID).
  Both future-change and growing-prefix checks are zero away from the encoder boundary,
  demonstrating causal prefix-recompute streaming feasibility. **It does not ship an
  efficient explicit-state chunk API.** It separates vocals from accompaniment, not one
  male talker from another. Included as a negative candidate, **skip** despite executing.
  Building a production streaming wrapper or validating speech transfer is out of this smoke task.
- **Mobile candidate: Silero VAD.** [Upstream](https://github.com/snakers4/silero-vad)
  explicitly targets edge/mobile use and portable ONNX deployment. The downloaded graph
  is 2.24 MB; this CPU smoke carries context and state between calls. It is speech detection,
  not ASR or speaker identification, and this is not an Android/iOS device benchmark.

The official HF benchmark-dataset listing had ASR leaderboards but no matching official
short-utterance speaker/streaming-separation leaderboard. Selection uses primary papers,
current release contracts and measured execution, not a fabricated cross-model ranking.
A search also found `penta2himajin/tse-conv-tasnet-48k`, but its live Hub API returned no
accessible repository/files; cached search-card claims were not used as execution evidence.

## Reproduce and cleanup

From the worktree root, explicitly run one script per fresh process:

```sh
.venv/bin/python eval/hf_models/T-303/ecapa2_smoke.py
.venv/bin/python eval/hf_models/T-303/eres2netv2_smoke.py
.venv/bin/python eval/hf_models/T-303/skim_smoke.py
.venv/bin/python eval/hf_models/T-303/causal_tasnet_smoke.py
.venv/bin/python eval/hf_models/T-303/silero_vad_smoke.py
.venv/bin/python -m pytest tests/test_hf_smoke_t303.py -q
bash scripts/verify_fast.sh
```

Real runs require network and existing `ml` dependencies; no new dependency was installed.
Weights, pinned upstream inference source, Hub and Xet cache/logs go into a private system
temp directory, removed on success **and exceptions**. Downloads use pinned revisions,
never `main`, and anonymous access. Processing performs no implicit downloads.
Each script emits a JSON result; `--mock` is a deterministic offline adapter/harness check,
explicitly marked mock and containing **no measured model timings or shapes**. Mock mode
was exercised for all five; it does not substitute for any table row.

[Raw results and artifact SHA-256 hashes](../../eval/hf_models/T-303/results.json)
retain the full measurements and every HF/GitHub revision. All five real runs recorded
`scratch_cleaned: true`. Only Python, Markdown and JSON are committed; no audio or weights.
The three-part organization is shared measurement/download helper + five model-specific
entry points + offline integrity tests; human findings and raw evidence are separate artifacts.
