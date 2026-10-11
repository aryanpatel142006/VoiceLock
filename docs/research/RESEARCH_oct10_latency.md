# Cutting and framing the 1.5 s delay: research (Oct 10, 2026)
Scope: Focus path on the M4 Pro (MossFormer2_SS 2-stream + ECAPA pick, 1.0 s window / 0.5 s hop, threaded, AirPods Pro 3 over A2DP). Repo facts are cited by path on `origin/main` of `~/CODE/selective-hearing`; web facts by link. "Estimate" marks numbers nobody has measured.

## 1. What delay do listeners tolerate when they can see the lips?
- Lip-sync detection: ITU-R BT.1359 puts the threshold at 45 ms audio lead to 125 ms audio lag; ATSC IS-191 asks for no more than 45 ms lag ([Wikipedia summary](https://en.wikipedia.org/wiki/Audio-to-video_synchronization)).
- Audio-visual fusion: McGurk fusion holds from -30 ms to +170 ms of audio lag ([van Wassenhove, Grant & Poeppel 2007, Neuropsychologia 45(3)](https://scholar.usuhs.edu/en/publications/temporal-window-of-integration-in-auditory-visual-speech-percepti/)).
- Lip-reading benefit: with video leading, AV intelligibility (63 % vs 9-31 % audio alone) stays stable only up to 160-200 ms of audio delay ([Grant & Greenberg, AVSP 2001](https://www.isca-archive.org/avsp_2001/grant01_avsp.html)).
- At 1.5 s, the listener sees the lips move about 1.3 s before the window where lips and sound still fuse. Lips give no help and the mismatch is visible.
- Hearing aids are a different class. Delays of 20-30 ms are rated disturbing, speech production suffers above about 30 ms, and open fittings need about 5-6 ms ([Hearing Review 2010, summarizing Stone & Moore](https://hearingreview.com/2010/01/acceptable-processing-delay-in-digital-hearing-aids)). Dillon et al. found that delays as small as 10 ms affected sound quality in commercial aids (same source).
- Conversation: ITU-T G.114 calls under 150 ms one-way transparent, 150-400 ms acceptable, and above 400 ms unacceptable for planning ([G.114 2003 text](https://www.cs.columbia.edu/%7Eandreaf/new/documents/other/T-REC-G.114-200305.pdf)).
- Captioning is the class where 1.5 s fits. Ofcom's live-subtitle target was a 3 s maximum, and real broadcasts averaged 5.1-5.6 s ([Limping Chicken on the Ofcom reports](https://limpingchicken.com/2015/12/04/deaf-news-ofcom-publishes-fourth-report-into-live-subtitles-saying-delay-is-too-long/)). In 2023 Ofcom proposed an average target of 4.5 s instead ([broadcastnow](https://www.broadcastnow.co.uk/ofcom-delay-off-peak-live-shows-to-cut-subtitle-latency/5088708.article)). DHH viewers rated latency as a significant factor, comparing about 2 s against 7-12 s ([Thompson et al., arXiv 2609.11408](https://arxiv.org/abs/2609.11408), a preprint).
- The wearer's own voice: delayed auditory feedback disrupts speech most at about 200 ms ([Wikipedia: DAF](https://en.wikipedia.org/wiki/Delayed_auditory_feedback)). The enrolled pick plays only the target (`hearing/tse/enrolled.py` docstring), so the wearer's voice should not loop back, but this has not been verified with a live mic.
- **Verdict:** 1.5 s is fine for a demo if it is framed as a *focus/caption-class* assistive tool, not as a hearing aid or a real-time ear.
  - Say it on stage: "about 1.5 seconds behind, like live captions; watch the captions, not the lips."
  - Do not claim lip-sync or hearing-aid latency.
  - Today 1.5 s is engine-side only. The Bluetooth leg comes on top, and mouth-to-ear on AirPods is still blank (`docs/BENCHMARKS.md` T-148 lines).

## 2. Cutting delay in the window/hop separator
Two facts frame this section:
- Why the delay is 1.5 s: a threaded stage delays output by window + hop, giving each window one hop of real time to compute (`hearing/tse/stage.py` module doc).
- Faster hardware alone saves nothing. For sliding-window separation, latency = shift + discarded tail + RTF x window, and "the fixed cost ... shall inevitably remain" ([Yoshioka et al., arXiv 1904.06478](https://ar5iv.labs.arxiv.org/html/1904.06478)). Any speedup has to be cashed in by shrinking the hop or the compute budget.

**A. 0.5 s window / 0.25 s hop (delay 0.75 s, -0.75 s). Config only.**
- Already measured on GRID: enrolled SI-SDRi +10.84 dB at 1 s vs +10.07 dB at 0.5 s. On the Mac, compute was p50 150 / p95 234 ms per 0.5 s window, leaving only +16 ms of p95 headroom, measured at load 9-12 (`docs/BENCHMARKS.md` window table; `docs/BAKEOFF.md` §1). The A100 path already uses 0.5/0.25.
- On an idle Mac the 1 s window's p95 fell from 323 ms (loaded) to 249-257 ms (T-284). By the same ratio, the 0.5 s window would be about 185 ms p95, about 65 ms of headroom (estimate).
- Quality risk is the ECAPA pick. Whisper's WER gain shrinks from -18 % to -14 % at 0.5 s (`docs/BAKEOFF.md` l.62). Speaker embeddings degrade sharply on short segments: VAM-ECAPA reaches 8.3 % EER at 1 s ([arXiv 2609.25007](https://arxiv.org/abs/2609.25007)), and the standard ECAPA baseline is reported far worse at 1 s than at 3 s (same paper, via search snippet, not verified in full text). The real pair's wrong-speaker time is the number to watch.
- Safety net: the adaptive hop ladder widens 1.0 -> 1.5 -> 2.0 x hop on overruns (`stage.py` `HOP_STEPS`).
- Test: same batch as T-284 under `swarm-lock --exclusive`, `demo_regress --profile team --isolation standard` on `t141/{aryan,joseph}/demo-team-{easy,hard}`, both target orders, 1 s vs 0.5 s.
- Gate: 0 / N deadline misses, SI-SDRi within 0.5 dB, wrong-speaker no worse than 9.2 / 14.8-19.7 %.
- Hours 2-3. Risk medium (pick quality). Knob: a new `HEARING_ENROLLED_WINDOW=0.5` overriding the local spec in `config/tiers.yaml`, default off.

**B. Compute budget below one hop: delay = window + budget, not window + hop.**
- At p95 255 ms, a 0.32 s budget gives 1.32 s (-0.18 s). Combined with A (0.5 s + 0.2 s) it gives about 0.7 s (estimate).
- A late window already falls back to the delayed mix with a 10 ms ramp (`stage.py` doc). The cost is a leak burst on each late window, so check the `late` count and leak.
- Hours 2. Risk low-medium. Knob: `HEARING_TSE_BUDGET_MS`, default off (= hop).

**C. Causal / stateful streaming separators: post-hackathon only.**
- Causal Conv-TasNet loses 4.7 dB SI-SNRi against non-causal (10.6 vs 15.3 dB, WSJ0-2mix; [arXiv 1809.07454](https://ar5iv.labs.arxiv.org/html/1809.07454)).
- SkiM runs at under 1 ms latency with 75 % less compute than DPRNN, at similar quality ([arXiv 2201.10800](https://arxiv.org/abs/2201.10800)).
- Short-Term Memory Convolutions give 5x faster inference and 2x lower latency on a separation U-Net with no quality loss ([ICLR 2023, arXiv 2302.04331](https://arxiv.org/abs/2302.04331)).
- MossFormer2 processes the whole sequence in parallel, and its abstract says nothing about streaming ([arXiv 2312.11825](https://arxiv.org/abs/2312.11825)).
- All of these need new weights and re-validation on the male/male pair. Days of work, high risk. Not before Sunday.

**D. Half precision on MPS. Unproven.**
- No published fp16-autocast speedup on MPS was found. One forum report shows fp16 matmul *slower* than fp32 on M1 ([PyTorch forum](https://discuss.pytorch.org/t/matrix-multiplication-implementation-in-pytorch/214503)).
- PyTorch 2.14 (installed: 2.14.1) adds fp16/bf16 prefill attention kernels with 2-4x gains, validated only on M5 and for head dims 64/96/128/256 ([2.14 blog](https://pytorch.org/blog/pytorch-2-14-release-blog/)). Whether MossFormer2's gated attention hits that path is unknown.
- Test: time `.half()` on the separator only (keep ECAPA fp32) on 50 windows. Accept only if SI-SDR is within 0.3 dB and p95 is at least 20 % lower.
- Hours 2. Risk medium (numerics). Knob: `HEARING_TSE_FP16=1`, default off.

**E. torch.compile / CoreML. Skip for this demo.**
- Inductor on MPS was unsupported in 2.1 (the advice was `aot_eager`, which gives no speedup; [forum](https://discuss.pytorch.org/t/jitting-fails-on-mps-device/192079)). Its 2.14 status was not confirmed by any source found.
- CoreML/ANE: Apple claims up to 10x for transformers *rewritten* for the ANE ([ane_transformers](https://pypi.org/project/ane-transformers)). The only MossFormer2 export found is a community ONNX export of the 48 kHz *denoiser*, which needed constant folding off for parity ([HF card](https://huggingface.co/TigreGotico/audiosronnx-mossformer2)). No MossFormer2_SS CoreML export exists.
- ECAPA costs only about 18 ms p50 per call (`docs/BENCHMARKS.md` l.476-478), so converting it gains almost nothing.
- More than 1 day of work, high risk.

## 3. Bluetooth output
- Measured AirPods latency (tap-to-sound on iPad, includes touch):
  - AirPods Pro 2: 126 ms; AirPods Pro: 167 ms ([9to5Mac 2022](https://9to5mac.com/2022/11/15/airpods-pro-2-latency-test/)).
  - Older test: wired 61 ms vs AirPods 251 ms ([Coyle](https://stephencoyle.net/airpods)).
- No public AirPods Pro 3 or macOS-AAC measurement was found.
- The repo's own round trip is 64 ms, built-in mic to speakers (`docs/BENCHMARKS.md` l.198). The AirPods row is still unmeasured (`scripts/bt_latency.sh`, bar median <= 300 ms, `config/demo-team.yaml`).
- Wired output saves roughly 65-190 ms (estimate from the gaps above), but wired earbuds have no ANC. The direct room sound then reaches the ear 1.5 s before the processed copy, which defeats "turn down the world."
- Exception: USB-C AirPods Max have a wired low-latency mode ([TechRadar](https://www.techradar.com/audio/headphones/apples-airpods-max-with-usb-c-will-get-lossless-audio-in-april-but-youll-need-to-go-wired)), only if someone has a pair.
- LE Audio/LC3 at about 20-30 ms comes from secondary sources only ([rfessentials](https://rfessentials.com/resources/rf-glossary/bluetooth-audio/)). There is no evidence the Mac and AirPods use it.
- Verdict: keep the AirPods. Run `bt_latency.sh` once (10 min, a human with the buds) and quote mouth-to-ear honestly.

## 4. The 3.9 s cold start
- Cause: the model loads lazily. The first full window submits `load()` to the worker (`stage.py` `_ensure_loaded`), and windows until then skip as `loading`. The tier reads `loading` for 3.2 s, fades in, and plays from 3.83-3.87 s (`docs/decisions/T-DEMO-HEALTH.md` items 3 and 8). The first computed window is also cold (MPS kernels, enrollment embeddings) and is not timed (`stage.py` `_Window.cold`).
- Fix 1: load the separator and ECAPA at server boot, run one dummy 1 s window on MPS, and embed the saved profiles (`enrolled.py` already caches 16 embeddings).
  - Pattern to copy: `LiveEngine.warm_up()` in `hearing/live.py` (l.370, l.786).
  - Expected: Start -> isolated drops toward the 1.5 s pipeline floor plus the first compute, about 1.8-2.2 s (estimate). If A ships, the floor is 0.75 s.
  - Check first whether a new session builds a new stage (`reset()` keeps the model only within one stage object).
  - Test: `demo_regress` "first isolated" column, before and after.
  - Hours 1-2. Risk low. Knob: `HEARING_TSE_PRELOAD=1`, default off at first, then on once green.
- Fix 2 (honest state): the session already knows the reasons `loading` / `warming up` (`hearing/server/session.py` l.194). Show "Loading voice model..." on /judge until the tier is live, instead of silently playing audio-only.
- Zero-code fallback: start the session 5 s before the reveal (T-DEMO-HEALTH item 8).

## Ranked top 3
1. **Preload + warm-up at boot (§4)**: about -1.7 to -2.1 s on the cold start, 1-2 h, low risk, knob. Add the honest "Loading" label.
2. **0.5 s window / 0.25 s hop on an idle Mac (§2A)**: 1.5 -> 0.75 s, -0.8 dB on GRID. It must pass the real-pair wrong-speaker gate in one exclusive batch. 2-3 h, knob, default off until it passes.
3. **Frame the delay honestly and measure the Bluetooth leg (§1, §3)**: "about 1.5 s + Bluetooth, caption-class, watch the captions." Run `bt_latency.sh` once. Zero code. Option B (compute budget) is the fallback if A fails its gate.
