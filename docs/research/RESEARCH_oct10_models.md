# Newer models than MossFormer2 for 2 male talkers in café noise (research, Oct 10 2026)

Scope: what we could run on the M4 Pro (MPS) by Sunday, inside a ~400 ms compute budget per 0.5 s hop on a 1.0 s window.
Nobody has measured any of these on our pair yet. Every number below comes from the linked source. "Estimate" marks my own reasoning.
Repo facts: the separator is a single backend, `_TorchBackend.separate(mix)` in `hearing/tse/blind.py` L65-110 (`MODEL = "MossFormer2_SS_16K"`, L17).
A JusperLee "look2hear" model (Dolphin) is already vendored under `hearing/tse/_dolphin/`, so the same family is cheap to vendor again.

## 0. Baseline: what MossFormer2_SS_16K scores on paper
- ClearVoice's own table (SI-SNRi): LRS2-2Mix 16 kHz **15.5**, WHAM! 8 kHz **17.4**, Libri2Mix 8 kHz **16.7**. It is one model for all sets, trained on "large scale datasets including open-sourced and private data" ([ClearVoice README](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice), [model card](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)).
- Paper MossFormer2: 55.7 M params; WHAM! 18.1, WHAMR! 17.0 SI-SDRi, all **8 kHz** ([ar5iv 2312.11825](https://ar5iv.labs.arxiv.org/html/2312.11825)).
- For scale: few public models beat that on noisy 2-speaker sets, and none by a wide margin. The gap to our real failure (two male voices overlapping, far field) is much larger than any gap between models on these benchmarks.

## 1. Separators, 2024-2026, with public weights
| Model | Noisy 2-spk numbers (SI-SNRi/SI-SDRi) | Size / speed | Weights, licence | Beats MF2 on noisy? | MPS live? | Drop-in |
|---|---|---|---|---|---|---|
| **TIGER** (ICLR 2025) | EchoSet (noise + realistic reverb, 16 kHz): large 13.73, small 12.58 vs TF-GridNet 12.85. LRS2-2Mix: 15.1 / 13.9. Libri2Mix: 17.97 / 16.67 [T1] | 0.82 M params; per 1 s of 16 kHz audio: 15.27 / 7.65 GMACs, CPU 765 / 351 ms, GPU 75 / 42 ms (hardware not named in what I read) [T1] | HF `TIGER-speech` (8 blocks), `-small` (4), `-tiny` (4 blocks, 64 ch); 2 sources, 16 kHz [T2]. Apache-2.0 card, MIT LICENSE file [T2][T3] | LRS2: 15.1 < 15.5 on paper. The only model here trained on realistic noisy-reverberant 2-speaker data, and it is "best on real-world data" in the authors' own recordings [T1] | Likely: pure PyTorch, no CUDA ops in `tiger.py` [T3]. Estimate: small fits easily, large needs measuring | **2-3 h** (vendor like Dolphin, swap the backend); 1 h for an offline A/B |
| SPMamba (2024) | WHAM! 17.4 / SDRi 17.6 [S1] | 238 GMACs/s [S1] | Libri2Mix and Echo2Mix checkpoints, Apache-2.0 [S1] | Ties MF2 on WHAM! | **No**: imports `mamba_ssm` (CUDA kernels) [S2] | not by Sunday |
| TF-Locoformer (MERL 2024) | WHAMR! S 17.4, M 18.5 vs TF-GridNet 17.1; 8 kHz [L1] | 5-22.5 M params [L1] | WSJ0/Libri2Mix/WHAMR!/DNS recipes, Apache-2.0 code [L2] | Small margin, at 8 kHz | Unlikely: TF-GridNet-class, and TF-GridNet takes 2.0 s CPU per 1 s audio [T1] | offline only |
| SepReformer | - | - | Only `SepReformer_Base_WSJ0` (clean) is released [R1] | No noisy weights | - | skip |
| pyannote PixIT `speech-separation-ami-1.0` | no quality numbers on the card [P1] | - | MIT, gated; joint diarization + separation, **trained on real AMI far-field audio**, 16 kHz [P1] | unknown | offline pipeline | 1 h offline A/B only |
| Meta SAM Audio (Dec 2025) | - | diffusion transformer, flow matching [M1] | custom SAM licence [M2] | Text/visual/span prompts, no audio prompt; "struggles to distinguish similar-sounding sources" [M2] | no | skip |

## 2. Generative / diffusion refiners for separated speech
- **Diffiner** (Sony): MIT, Zenodo weights [D1]. Uses T = 200 reverse steps. Alone it *lowers* SI-SDR (Sepformer 23.0 -> 22.0) and PESQ while raising NISQA. Only blending recovers +0.1 dB [D2]. Offline only, and no help for picking the wrong speaker.
- **SoloSpeech** (2025, generative TSE): beats USEF-TSE on noisy Libri2Mix (SI-SNR 11.12 vs 10.17, WER 0.15 vs 0.17) and on real CHiME-5 MOS (2.93 vs 2.10) [SO1]. But it uses a 200.8 M extractor, 50 steps and 8 candidates [SO1]. Licence CC BY-NC(-SA) 4.0 [SO2][SO3]. Offline only.
- **Diffusion Buffer** (Interspeech 2025): online diffusion SE with 0.3-1 s latency on a GPU [DB1]. The repo lists no checkpoint or licence [DB2]. Not runnable by Sunday.
- **FlowSE** (2505.19476): checkpoint `wenetspeech4tts_Premium`, a 24 kHz vocoder folder, no licence or RTF stated [F1]. A second FlowSE (GIST) matches 60-NFE diffusion at 5 NFE [F2]. Offline only. Both are enhancers, not separators.
- **NVIDIA `se_den_sb_16k_small`**: Schrödinger-bridge denoiser, 16 kHz, CC-BY-NC-SA-4.0 [N1]. Denoising only.
- Verdict: none of these fixes the wrong-speaker windows. All are offline or too slow for a 0.5 s hop on MPS (estimate from the step counts above).

## 3. Target-speaker extraction (TSE) that takes an enrollment clip
- **PS4** (2026, arXiv 2607.08111): BSRNN + ECAPA-TDNN (192-d), 16 kHz, **fine-tuned on real far-field meetings** (AMI, CHiME-6, AISHELL-4, AliMeeting). Single `.pt` file plus a self-contained `inference.py --mix --enroll`. Apache-2.0 [PS1].
  - REAL-T leaderboard: TER 0.639 vs 0.829 for the Libri2Mix-trained BSRNN baseline, speaker similarity 0.565 vs 0.417. 2nd overall [PS2].
  - Enrollment is longer than 5 s [PS2]. Our profiles are 14-50 s (RESEARCH_improve_enroll_agc.md).
- **REAL-TSE baselines** (`spk_emb_100`, `spk_emb_causal_100`): BSRNN, Libri2Mix-100, 16 kHz, Google Drive weights, `evaluate.py --mixture --enroll` [RT1][RT2]. The online track caps algorithmic latency at 100 ms [RT2].
  - A Core ML port of `spk_emb_100` runs at **RTF 0.197 on an M3 Max** (CPU-only beats CPU+GPU; LSTMs do not run on the ANE). It takes 8 s chunks [RT3].
  - The top challenge systems "were all built on BSRNN-style backbones" [RT2].
  - Estimate: 1 s of audio at RTF ~0.2 is about 200 ms per hop, so it is plausible on CPU. Measure; a 1 s context is far shorter than the 8-30 s used in training.
- **USEF-TSE**: USEF-TFGridNet WHAM! 17.6, Libri2Mix (16 kHz max) 16.8 SI-SDRi; 15.2 M params [U1].
  - Same-gender SDR 24.41 vs 24.83 different-gender (WSJ0-2mix) [U1].
  - Weights on HF (1.8 GB), CC BY-NC 4.0 [U2][U3].
  - Earlier note: 87 % token error on real REAL-T audio (RESEARCH_isolation.md §2). TF-GridNet-class compute means offline only.
- **MeanFlow-TSE** (one step, MIT, Libri2Mix weights, "clean" config) [MF1][MF2]. **WeSep**: no pretrained models yet [W1]. **ClearVoice TSE**: audio-only is 8 kHz, and the TSE checkpoint we have is audio-visual (`AV_MossFormer2_TSE_16K`, 735 MB) [C1][C2].
- Same-gender numbers for PS4 / REAL-TSE: none published that I found [RT2].

## 4. Light enhancement for the residual noise (on the picked stream only)
- **GTCRN**: 48.2 K params, 33 MMACs/s; streaming RTF 0.07 on an i5-12400. VCTK-DEMAND SI-SDR 18.83 / PESQ 2.87; DNS3 OVRL 2.70. DNS3 and VCTK checkpoints, MIT [G1].
- **FRCRN_SE_16K** (ClearVoice, already installed: `clearvoice 0.1.2` in the app venv): DNS-2020 PESQ 3.24, SI-SDR 19.99. **MossFormerGAN_SE_16K**: PESQ 3.57 [C1]. Speed not published.
- **DeepFilterNet3**: RTF 0.19 on a single-threaded notebook CPU [DF1]. Full-band 48 kHz, so we resample [DF2]. MIT/Apache [DF2].
- Caveat (estimate): café babble is speech-like, and these denoisers are trained to keep speech. Expect less noise but no fewer wrong-speaker windows.

## 5. Ranked top 3 to A/B tonight on our real recordings (same scorer as T-141/T-326, wrong-speaker % + SI-SDRi)
1. **TIGER-speech and TIGER-speech-small instead of MossFormer2**, same 1.0 s / 0.5 s windows and the same ECAPA pick and LDA.
   - Why: trained for noise + reverb at 16 kHz, 1/68 of MF2's params, pure PyTorch. On paper it is not better than MF2 on LRS2 (15.1 vs 15.5) [T1][C1], so this is a real coin flip.
   - Cost: ~1 h offline, ~2-3 h to wire in. Also log p95 ms per hop on MPS.
2. **PS4 TSE with the saved aryan/joseph profile wavs as enrollment**, replacing separate-then-pick. Run `spk_emb_100` alongside as the control.
   - Why: the only enrollment model with weights tuned on real far-field overlap, and its BSRNN family runs at RTF ~0.2 on Apple silicon [PS1][PS2][RT3].
   - Run offline on whole takes first, then on 1 s windows. Cost: ~1-2 h offline, ~4-6 h live (new candidate, confidence and gating differ).
3. **FRCRN_SE_16K (zero install) or GTCRN on the picked stream**, before the mixer.
   - Why: cheap residual-café cleanup [C1][G1]. Measure that the target is not muted more, and expect no change in wrong-speaker %.
   - Cost: ~1 h.
- Not worth tonight: SPMamba (CUDA), TF-Locoformer, USEF-TSE, SoloSpeech, Diffiner, FlowSE, SAM Audio (reasons above).

## Sources
[T1] https://arxiv.org/html/2410.01469v1 (Tables 2-3), https://arxiv.org/html/2410.01469v3 (EchoSet, 16 kHz, real-world eval) · [T2] https://huggingface.co/JusperLee/TIGER-speech , https://huggingface.co/JusperLee/TIGER-speech-small , https://huggingface.co/JusperLee/TIGER-speech-tiny (config.json) · [T3] https://github.com/JusperLee/TIGER (LICENSE, look2hear/models/tiger.py, inference_speech.py)
[S1] https://github.com/JusperLee/SPMamba · [S2] https://raw.githubusercontent.com/JusperLee/SPMamba/main/look2hear/models/SPMamba.py (line 23)
[L1] https://arxiv.org/html/2408.03440 · [L2] https://github.com/merlresearch/tf-locoformer · [R1] https://github.com/dmlguq456/SepReformer
[P1] https://huggingface.co/pyannote/speech-separation-ami-1.0 · [M1] https://arxiv.org/abs/2512.18099 · [M2] https://the-decoder.com/meta-brings-segment-anything-to-audio-letting-editors-pull-sounds-from-video-with-a-click-or-text-prompt/ , https://huggingface.co/AEmotionStudio/sam-audio-models
[D1] https://github.com/sony/diffiner · [D2] https://arxiv.org/html/2305.05857v3 · [SO1] https://arxiv.org/html/2505.19314v3 · [SO2] https://github.com/WangHelin1997/SoloSpeech · [SO3] https://huggingface.co/OpenSound/SoloSpeech-models
[DB1] https://arxiv.org/abs/2506.02908 · [DB2] https://github.com/sp-uhh/Diffusion-Buffer · [F1] https://github.com/Honee-W/FlowSE · [F2] https://sapl.gist.ac.kr/wp-content/uploads/2024/12/FlowSE_ICASSP2025.pdf · [N1] https://huggingface.co/nvidia/se_den_sb_16k_small
[PS1] https://huggingface.co/TaurenMountain/PS4 · [PS2] https://arxiv.org/html/2607.08111 · [RT1] https://github.com/REAL-TSE/wesep-real-tse · [RT2] https://arxiv.org/html/2607.15198 · [RT3] https://huggingface.co/ferdinandl007/REAL-TSE-BSRNN-CoreML
[U1] https://arxiv.org/html/2409.02615v3 · [U2] https://github.com/ZBang/USEF-TSE · [U3] https://huggingface.co/ZBang/USEF-TSE/tree/main · [MF1] https://arxiv.org/abs/2512.18572 · [MF2] https://github.com/rikishimizu/MeanFlow-TSE · [W1] https://github.com/wenet-e2e/wesep
[C1] https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice · [C2] https://huggingface.co/alibabasglab/AV_MossFormer2_TSE_16K · [G1] https://github.com/Xiaobin-Rong/gtcrn · [DF1] https://arxiv.org/abs/2305.08227 · [DF2] https://github.com/Rikorose/DeepFilterNet
