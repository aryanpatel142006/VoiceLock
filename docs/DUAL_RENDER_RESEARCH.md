# Research plan: live separation and enhanced replay

> October 9, 2026. These are experiments for the *same consented raw take*. “Enhanced” means a measured, human-preferred version; it does not mean perfect reconstruction or real-time playback.

## Why two versions are plausible

A live path only sees the audio that has arrived and must keep delay bounded. An offline path can revisit a complete phrase or minute, use future context, and spend more computation per second of audio. That gives it **an opportunity**, not a guarantee, to improve speaker consistency or reduce artifacts. [Look Once to Hear](https://doi.org/10.1145/3613904.3642057) describes a strict real-time target-hearing path and its limited future context; [LaFurca](https://arxiv.org/abs/2001.08998) is an example of iterative, context-aware offline separation. Neither paper proves that swapping either model into this repository will improve its demo pair.

The prior Gemini-produced `eval/fixtures/demo_listen/DEMO_TEST_VIDEO.mp4` and raw/isolated WAVs are listening references, not proof of the current code path or live speed. First reconstruct their provenance if possible: raw source, saved target profile, exact model/device/settings, render duration and post-processing. Keep the old CPU “robotic but separated” result as a candidate tradeoff, not a current bug list.

## Bounded experiment matrix

| Question | Comparison | Decision evidence |
|---|---|---|
| Does more context help this pair? | Current live output versus renders with 1, 5, 15 and available full-duration seconds on the same raw input | Target leak, wrong-speaker time, intelligibility, artifacts, blind headphone preference and wall-clock render time |
| Where does improvement come from? | Global target-stem selection, stronger separation, denoise/post-filter, and combinations one change at a time | Preserve intermediate stems and level-match playback; reject a variant that sounds cleaner but lets the other speaker through |
| Can an early preview be useful? | Fully completed short excerpt versus full-take render | Time to first preview, preview/final quality gap, visible “preview” label; no claim the full minute was ready before capture finished |
| Does it generalize? | Café-only, target-first overlap, target-silent other-speaker, one-sentence side speaker, swapped positions | Fixed held-out takes, exact run manifests and human listening; stop if only the original clip improves |
| Is the live path preserved? | Live stream with recording and enhancement job enabled versus disabled | Capture-to-ear delay, underruns, dropouts, memory and CPU/GPU use; rendering must not slow what the listener hears |

One-second updates can be a **UI refresh or sealed recording segment**. They are not evidence that each second can be perfectly isolated independent of context. Measure the live processor's actual window, hop, overlap, lookahead and delay. [Online overlap-aware diarization](https://arxiv.org/abs/2109.06483) demonstrates the quality/latency tradeoff of rolling windows; use it as a research pointer, not a latency claim for this pipeline.

## Output contract for the demo UI

Store `raw`, `live`, and `enhanced` as separate versions under one run ID, with aligned timestamps and a manifest of source/device/profile/revision/settings. Raw and Live become available as completed segments are written, with full WAVs finalized at stop. Enhanced is a background job that becomes **Ready** only when its output file is complete. The UI must present `Capturing`, `Rendering`, `Ready`, or `Failed`, permit replay/download, and never replace the earlier Live file. All three playback choices need a common timeline and level matching for a fair A/B.

For the stage: the judge hears **Live Focus** through headphones. The presenter can then say, “Here is the recording you just heard. Now here is the enhanced version of the same take after extra processing.” Show the badge and measured render time. If the offline output is not perceptibly better or is late, demonstrate the verified Live result and retain the enhanced result as research.
