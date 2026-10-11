# Conditioned TSE (REAL-TSE baseline) feasibility gate (T-203)

Optional enhancement track. Research only: nothing was downloaded, installed, trained or run, no audio left the machine, and no live code, registry, config or shared doc was touched. **No measurements are reported here.** Facts were read on 2026-10-09 from the pages and repositories named below.

## 1. Verdict

**Inference: NOT VERIFIED. Trial: conditional GO only if the prerequisites in §5 are met.** The code and an entry point exist. The weights are published only as a Google Drive link, which was not fetched because no authorised spare isolated environment was available. The checkpoints' licence is unstated. The baselines are trained on clean Libri2Mix-100, so a downloadable checkpoint is not evidence of compatible, useful inference on our demo mic. Recommendation: do not spend core-rehearsal time on this; run one bounded trial after the demo, and stop if executable weights/config are still missing 45 min into that trial.

## 2. Repositories and artefacts (do not conflate)

| Item | Primary location | Revision checked | Status |
|---|---|---|---|
| Baseline code (WeSep fork for the challenge) | https://github.com/REAL-TSE/wesep-real-tse | `main` @ `2a540977a348fbaa92e623210505430e2cec608d` (last push 2026-04-14) | Code present. No LICENSE file in the root listing and GitHub reports no licence (the CoreML model card says the WeSep implementation files are Apache-2.0: unverified here). |
| Challenge eval toolkit | https://github.com/REAL-TSE/REAL-TSE-Challenge | `main` @ `8bde4f147d452de43834b5c0a4efdb85811fd5dd` | MIT. Data, evaluation, submission; not needed for inference. |
| Checkpoints `spk_emb_100`, `spk_emb_causal_100`, `tfmap_context_100`, `tfmap_context_causal_100` | Google Drive `https://drive.google.com/uc?export=download&id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC` (both READMEs) | not fetched; no hash or version published | **Link present, contents NOT VERIFIED.** The challenge site says checkpoints were emailed to registered teams (realtse.challenge@gmail.com). Expected layout per folder: `avg_model.pt` + `config.yaml`. |
| Challenge description | https://real-tse.github.io/challenge ; paper arXiv:2607.15198 (CC BY-SA 4.0) | v1, 2026-07-16 | Read abstract and challenge page only, not the paper body. |
| Third-party CoreML conversion of `spk_emb_100` (noncausal) | https://huggingface.co/ferdinandl007/REAL-TSE-BSRNN-CoreML | revision not stated | Not primary; HF licence tag `other`; fixed 8 s separator. Useful only as a second-hand statement of shapes. |
| PS4 (`TaurenMountain/PS4`, BSRNN+ECAPA fine-tuned on real meetings) | https://huggingface.co/TaurenMountain/PS4 | not inspected | Alternative 1 of at most 2; causality not stated; not inspected. |

Not conflated: `Wespeaker/wesep-real-tse` on Hugging Face returned HTTP 401 (no public repo found); `wenet-e2e/wesep` main was not used. Hugging Face revision hashes for any of these: not obtained.

## 3. Documented contract (from the repo and challenge page)

- **Sample rate:** 16 kHz, mono (`resample_rate` default 16000; the extractor resamples mixture and enrollment itself).
- **Entry point (minimal inference command, from the README, unrun):**
  ```
  python evaluate.py --pretrain path/to/spk_emb_causal_100 \
    --mixture mix.wav --enroll enroll.wav --output out.wav [--device cpu|cuda]
  ```
  It calls `wesep.load_model_local(dir)` then `extract_speech(mixture, enroll)`; there is also `extract_speech_from_pcm(pcm_mix, sr, pcm_enroll, sr)` for arrays. Flags: `--vad` (VAD on the enrollment), `--no_output_norm`. Install per README: Python 3.10, torch 2.7.1 + torchaudio 2.7.1, `requirements.txt`, and WeSpeaker pinned at `8f53b6485d9f88a207bd17e7f8dba899495ec794`. Our `.venv` is Python 3.12 and shared: do not install into it (see §5).
- **Cue encoder:** WeSpeaker ECAPA-TDNN speaker embedding (192-d, Kaldi 80-d fbank), the "spk_emb" variants; `tfmap_context` variants use a time-frequency feature map instead. Separator: BSRNN. Enrollment must be at least 0.5 s; several seconds of clean solo speech recommended.
- **Normalization:** `torchaudio.load(normalize=True)` on both inputs; `output_norm=True` by default rescales the output (so the level is not preserved, which matters for our mixer/AGC). Exact scheme: read from `extractor.py`, not verified against `config.yaml` (never opened).
- **STFT:** 512-point Hann window, 128-sample hop (8 ms) per the CoreML card; the challenge page states the online baselines use a 32 ms window and 8 ms hop (24 ms algorithmic latency). These agree (512 samples = 32 ms, 128 samples = 8 ms); neither source was checked against `config.yaml`.
- **Causality/state:** the "causal" checkpoints are causal in the model, and Track 1 (online) allows at most 100 ms algorithmic latency. **Unverified: whether the repo exposes a chunked/stateful streaming API.** `evaluate.py` processes whole files; no streaming or cache API was seen in what was read (`models/` holds only `tse_bsrnn_spk.py`). Wrapping the causal model in our 10 ms Stage protocol would need a state-carrying implementation, or a windowed run like the existing `SlidingWindowExtractor`, which loses the causal benefit.
- **Training recipe:** `examples/audio` (WIP): Libri2Mix and VoxCeleb online-mixing pipelines. All four baselines were trained on Libri2Mix-100 (clean read speech, not far-field or café noise). The README says Libri2Mix training and the challenge's real-recording tracks are separate: real-recording training data is not provided.
- **Published quality (challenge README, dev set, official, not ours):** `BSRNN_EMB_CAUSAL` is the first row of the causal tables. Reproducing these on our data: NOT MEASURED.

## 4. Why embeddings have not reliably worked: current diagnosis

This section only collates what the repo's earlier evidence shows (`docs/research/ISOLATION_DIAGNOSTIC.md`, `docs/VOICE_EMBEDDINGS.md`, T-069); T-203 measured nothing. Principal case: two male voices (Aryan/Joseph), equal and unequal levels, quieter selected voice, both target roles and swapped positions. Male/female is historical comparison only. Wrong-voice silence is not target preservation: score target words kept separately.

| Mechanism | Status | Evidence / what is missing |
|---|---|---|
| Short ECAPA crop | **Demonstrated as a margin loss** | Clean target crop margin 0.42 (0.5 s), 0.55 (1 s), 0.69 (2 s), 0.77 (4 s); margin stays positive in that scene. |
| Crop contamination | **Demonstrated** | 2 s crop with other-voice share 0/10/30/50%: margin 0.69/0.50/0.40/0.26. |
| Negative-profile rejection at cos > 0.7 | **Ruled out for one pair** | Target-vs-other profile cos 0.066, so the rule did not fire. Not shown for two similar male voices. |
| Profile vs demo-mic mismatch | **Unknown** | Profile-to-stem cos 0.78 was on synthetic-room stems; needs profile and scene through the same demo mic (T-172 re-enrol). |
| Target loss before selection | **Unknown for male/male** | Needs oracle-vs-actual per window; T-069 male/male at 1 s context: accuracy .742, wrong voice 1.75 s/10 s vs candidate ceiling .930; 2 s: .804 / 1.34 s vs .951, still NO-GO. This gap (selection, not separation ceiling) is the strongest existing evidence. |
| Missing-camera strict pause decisions | **Unknown** | The offline replay passes no ASD, so pause/held/scoring effects are not exercised. |
| `same_bucket`, `profile_negative`, readiness, active tier | **Unknown** | Live in the session/wrapper, not offline; log them per window in a T-200-style run. Saved negative profiles bypass the gender guard, so same-bucket skipping cannot explain every male/male failure. |
| Mixer / AGC / deadline fallback | **Unknown** | Not isolated from separation in the evidence read; T-200's oracle diagnostic is the tool. |

**Measured intervention that could make it work (not run):** on a fixed recording set (both male voices, equal/unequal and quieter-target levels, both roles, both positions, enrolled and tested through the demo mic), per window compare oracle-selected vs actual-selected stream, logging crop duration, contamination share, cos to target and negative profile, margin, and the same_bucket/tier/pause state. Then try one change at a time: 2 s+ clean crops (T-069 shows 1 s to 2 s helps but not enough), mic-matched profile, and a conditioned extractor that takes the cue directly instead of selecting among separated streams. The last item is what this track would test; it is not shown to beat selection.

## 5. Go/no-go prerequisites for a trial

All must hold, else NO-GO:
1. A human-authorised spare isolated environment (separate venv, Python 3.10, own torch; never the shared `.venv`, never `pip install -e` from a worktree), with no automatic GPU allocation and no purchase.
2. Weights obtained from the Google Drive link or from the organisers, with SHA-256 recorded, `avg_model.pt` and `config.yaml` present for `spk_emb_causal_100` and `spk_emb_100`, in a gitignored `$HEARING_MODELS_DIR` subfolder. Licence of the checkpoints confirmed in writing (currently unstated); the repo's code licence also unconfirmed. Record both in `docs/MODELS.md` when adopting.
3. `evaluate.py` runs on a private 16 kHz mix + enrollment pair locally, offline, and the config's STFT, normalization and cue fields are read and noted.
4. Streaming/state: confirm whether the causal checkpoint can run chunked with carried state at 10 ms hops; if not, treat it as a windowed offline extractor only.
5. Scenario set per §4, scored with SI-SDRi **and** target-words-preserved, plus wrong-speaker time, against our `enrolled`/`av_mossformer2` baselines. Run wall-clock numbers under `swarm-lock --exclusive`, and not while core rehearsal runs.
6. Time-box: stop if executable weights/config are not in hand 45 min into the trial. At most one more pretrained alternative (PS4) after the first.

No training, no self-training on uncertain output, no live integration, and no production adoption from this track.

## 6. Unverified / uncertainties

- Checkpoint contents, hashes, licence and `config.yaml` fields (STFT, normalization, embedding dim, causal chunk size).
- Whether the repo has any streaming/state API; the shape of causal-state tensors.
- Behaviour on a laptop mic, on male/male overlap, at quieter-target levels; Apple MPS support for BSRNN.
- Whether WeSpeaker ECAPA embedding in this model differs from our `speechbrain/spkrec-ecapa-voxceleb` profile space (almost certainly: profiles are not interchangeable and would need re-enrolment).
