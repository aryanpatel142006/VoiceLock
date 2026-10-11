# Copilot pipeline audit

Compared with `docs/research/DEEP_RESEARCH_HANDOFF.md` at the current worktree
commit. Line references below are source/config line numbers from that commit;
configuration-dependent values are qualified rather than treated as universal.

| Claim | Exact file:line | Status | Brief correction |
|---|---|---|---|
| The blind separator is `alibabasglab/MossFormer2_SS_16K` and operates at 16 kHz. | `hearing/tse/blind.py:15-18,103-106` | **CONFIRMED** | The model ID and explicit 16,000-Hz validation match the handoff. |
| MossFormer2 produces two candidate waveforms, while the playable result is one selected stream with a residual/final mix. | `hearing/tse/blind.py:122-131`; `hearing/tse/stage.py:43-45` | **CONFIRMED** | The separator requires exactly two outputs; the stage emits one target and exposes delayed-mixture-minus-target residual for the downstream mixer. |
| Enrollment enters the selector, not the separator network. | `hearing/tse/enrolled.py:408-426`; `hearing/tse/blind.py:110-131` | **CONFIRMED** | `enroll` is embedded and used to score separator streams; `BlindSeparator.separate()` only receives the mixture. |
| The enrolled selector uses `speechbrain/spkrec-ecapa-voxceleb`. | `hearing/tse/enrolled.py:10-16,65-76` | **CONFIRMED** | The model ID and SpeechBrain encoder path match the handoff. |
| Model instances are lazy, process-shared, and profile pieces/cache entries are reused. | `hearing/models.py:28-50`; `hearing/tse/enrolled.py:37-38,296-304`; `hearing/tse/enroll.py:54-62,106-108` | **CONFIRMED** | The model cache is locked and keyed by model name; enrolled cache is capped at 16 entries, while saved profiles are versioned WAV pieces (2.5-s target piece size). |
| The current enrolled delay is 1-s window / 0.5-s hop / 1.5-s threaded delay. | `config/tiers.yaml:69-79`; `hearing/tse/stage.py:11-18` | **CONFIRMED** | This is the local default configuration; it is not universal. The stage documents threaded delay as window + hop, while inline/offline operation is window-only. |
| The remote enrolled configuration is 0.5-s / 0.125-s (625-ms algorithmic delay). | `config/tiers-remote.yaml:32-47`; `hearing/tse/windowed.py:192-200` | **CONFIRMED** | This is a separate remote preset; actual end-to-end delay also depends on compute, transport, and playout buffering. |
| Mixer confidence floor is `6 + 34*confidence` dB of residual attenuation. | `hearing/fusion/mixer.py:66-69,159-205` | **CONFIRMED** | The implementation interpolates 6 dB at confidence 0 to 40 dB at confidence 1, then takes the maximum of that floor and slider room gain. |
| AGC is applied at the output, after hop/pipeline processing, with a pre-AGC tap available. | `hearing/server/session.py:522-525,1285-1292`; `hearing/audio/agc.py:1-19` | **CONFIRMED** | Under the judge profile the laptop applies `Agc` after `_process_hop`; remote-stage output disables this leg to avoid double AGC. |
| A late/skipped TSE window falls back to the delayed raw mixture. | `hearing/tse/stage.py:11-18,580-596` | **CONFIRMED** | The stage keeps `mix` as the delayed input and crossfades to it when no resolved TSE output covers a position; this is not a clean interferer stem. |
| A saved negative profile bypasses the same-bucket guard. | `hearing/server/session.py:1477-1485`; `hearing/tse/enroll.py:376-381` | **CONFIRMED** | When the target and negative are fixed saved profiles, `profile_negative` clears the telemetry guard; ordinary/no-profile same-bucket cases remain separately guarded. |

