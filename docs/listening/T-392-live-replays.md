# Live-session replays: base vs fine-tuned separator (T-392, Oct 10 2026)

**Listening files only. Nothing here is a measurement.** These are real room recordings from live app sessions
(the microphone side, `in.wav`, recorded by the session tap). There are no clean stems, so no SI-SDR, leak or
wrong-speaker numbers exist for them. Judge by ear.

Each session folder holds three files, 16 kHz mono, level-matched (active speech at -26 dBFS, peak at or below -1 dBFS):

- `room.wav`: what the microphone heard (the session's `in.wav`), unprocessed.
- `base.wav`: the shipped MossFormer2_SS_16K separator, enrolled tier, target picked by ECAPA against the saved profiles.
- `finetuned.wav`: T-374's fine-tuned fixed-order checkpoint (`HEARING_SEP_CKPT=$HEARING_MODELS_DIR/T-374/fixed_best.pt`,
  `HEARING_PICK=fixed`, T-386's knobs): slot 0 = Aryan, slot 1 = Joseph; ECAPA only decides the gate and confidence.

How they were made: `eval/live_replays.py` (offline, not real time) runs the live enrolled tier's `EnrolledCandidate`
over 1.0 s windows every 0.5 s with the saved profiles in `$HEARING_FIXTURES_DIR/profiles` (target / other as listed
below), standard isolation, slider 100, mixer confidence floor as in the session, before the output AGC, no
`HEARING_GATED_FLOOR_DB`. Each arm runs in its own process. At most the first 120 s of each `in.wav` is used.
T-387's longer fine-tune is not included (that task is blocked; its checkpoint is not on this Mac).

| folder | recorded | mic | what happened live | target : other | length |
|---|---|---|---|---|---|
| `20261010T234557Z-5/` | Oct 10, 19:45 EDT (the only Oct 10 live session that recorded audio) | MacBook Pro Microphone, speakers | `/app`, judge profile, Focus HQ; profiles joseph / aryan picked back and forth at 71-100 s, A/B raw from 104 s | aryan : joseph | first 120 s of 203 s |
| `20261009T211905Z-8/` | Oct 9, 17:19 EDT | MacBook Pro Microphone | judge profile, no profile picked (auto-calibration) | aryan : joseph | first 120 s of 231 s |
| `20261009T174419Z-2/` | Oct 9, 13:44 EDT | MacBook Pro Microphone | judge profile, no profile picked | aryan : joseph | all 115 s |

Source sessions: `results/sessions/<folder name>/in.wav` in the main checkout. The other Oct 10 live sessions under
`results/sessions/` kept telemetry only (no `in.wav`), so they could not be replayed; the Oct 9 sessions fill the
three-session minimum. Which voices are present and when was not checked by ear before writing this; listen to
`room.wav` first.

What to listen for: whether `finetuned.wav` keeps Aryan while Joseph talks over him (the fixed slot should not swap
voices mid-sentence the way the ECAPA pick can), and whether either arm mutes Aryan or plays Joseph. Known
caveat from T-386: the ECAPA gate can still mute the target in the fine-tuned arm.

Reproduce: `HEARING_MODELS_DIR=... HEARING_FIXTURES_DIR=... .venv/bin/python -m eval.live_replays <session dirs> --out <dir>`.

Written to `~/Desktop/selective-hearing-finetune/live-replays/20261010-2025/README.md` on laptop-a (the WAVs stay there; audio is never committed).
