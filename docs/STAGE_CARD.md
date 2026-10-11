# STAGE CARD · Sunday demo (~3 min) · the app `/app/` (fallback `/judge`)

<!-- Source: DECISION_demo.md (Oct 9) §1, §2, §4, with the demo order revised after live listening (Q-463): noise + one locked voice first, the overlap sentence as the reveal near the end. Pre-show list: RESEARCH_improve_enroll_agc.md. Keys checked against web/app/keys.js keyAction (identical to web/console/judge.js onKey; T-347). Gate pill: web/console/gate-state.js (T-382). Fine-tuned A/B: T-386 (BENCHMARKS), T-387. Start lines checked by T-405 with --no-open --port 0. Render: .venv/bin/python scripts/stage_card_pdf.py -> docs/STAGE_CARD.pdf -->

From rehearsal 2: **N = ____ %** wrong-speaker · **X = ____ %** WER · separator **base / fine-tuned** (circle)

## Start (in `~/CODE/selective-hearing`; wait for `JUDGE · backend Mac`, ~10 s)
- **Base:** `bash scripts/demo.sh --judge --team --lan --app`
- **Fine-tuned:** prefix `HEARING_SEP_CKPT=$PWD/models/T-374/fixed_best.pt HEARING_PICK=fixed HEARING_GATED_FLOOR_DB=20` (or the morning's T-387/T-391 `.pt`)
- iPhone USB mic: add `--mic "Aryan patel's iPhone Microphone"`. Phones: `https://<laptop IP>:8000/app/` (QR), accept the cert once, Add to Home Screen; companion only.

## Before the judge puts the AirPods in
- **Acoustic Seal Test**; use the tip size that passes. ANC on.
- Set the **volume once with the judge, starting low**.
- The judge **listens, does not talk** during Focus (own voice comes back 1 s late).
- **Point the mic at the talkers, away from the wearer.**
- Say out loud: **the sound is about 1 s behind.**

## App (`/app/`)
Welcome · Setup (mic, output, meter) · People · **Listen** (stage) · About · **D** Diagnostics. Big word: Loading voice → **Isolated** (green) / **Raw** (amber). Gate pill: **SOLO** · **OVERLAP-DUCKED** · **UNSURE** (hidden in Raw and Quiet).

## Keys (Listen, click the page first; same as `/judge`)
| Key | Does |
|---|---|
| **Start** listening button | mic on (stop: same button) |
| **2** | Joseph's saved profile (**1** = Aryan) |
| **Space** | RAW (amber) ↔ SELECTIVE (green) |
| **Shift+Q** | Quiet: silence + captions |
| **Shift+F** | back to Focus |
| **D** | Diagnostics open / close |
| **Shift+R** | labelled replay (Replay badge, violet) |
| **Shift+0** | Release target (Searching); 1 or 2 picks again |

Q, F, R, 0 alone need Diagnostics open. **Keys off** banner: click the page. Never Shift+C.

## The 8 steps
1. **Setup, said out loud:** "Joseph holds an iPhone as his mic, cabled to the laptop (like a hearing-aid remote mic). Our two voices were saved earlier as profiles from a voice memo. You wear AirPods Pro 3 with noise cancelling. A speaker plays café noise." No camera.
2. **Start listening, key 2.** Joseph talks alone 5 s; the big word reaches **Isolated** ("You hear Joseph"), gate pill **SOLO**. Say: "Focus: hear the one person you choose."
3. **HEADLINE.** Café noise on (laptop-b). Joseph keeps talking. **Space** three times mid-sentence: amber **Raw** = the raw room, green **Isolated** = Focus. Say nothing between flips.
4. **Shift+Q** (Quiet). Audio goes silent; Joseph reads two sentences of his choice, captions 1-2 s behind under his name. Say: "no prompt, no script: this is what Whisper heard, X% word error at our rehearsal." Then **Shift+F** back to Focus.
5. **Reveal.** Aryan says ONE sentence over Joseph (5 s), then stops; Joseph goes on. The gate pill may show **OVERLAP-DUCKED** / **UNSURE** while both talk. Say: "When we both talk at once the wrong voice gets through N% of the time. That is the hard case for every system; the research agrees."
6. **D** (Diagnostics): backend, delay (~1.5 s Mac), RTT, "this session is being recorded". Say: "every number on our slides was measured from recordings like this one."
7. **Close:** line 1, then line 3 (below).
8. **Only if 2-5 failed:** **Shift+R**, the labelled replay (Replay badge). Say: "recorded earlier, mixed digitally, processed now." Never shown as live.

Name Call is not in the app; mention it only if asked.

## The three spoken lines
1. **Closing (§1):** "Turn down the world without disconnecting from it."
2. **Fallback b (§2):** "noise suppression, not person selection"
3. **Far-field honesty (§4):** "From one laptop mic across a room, picking one of two similar male voices did not work in our tests: 6-25% wrong while both talked. Nobody ships that; products that select a voice have the talker at the mic, and products that hear across a table use several mics and direction. So the person you want to hear wears a clip mic, and the app does the part no product does: pick who, mute the rest, caption only them."

## Fallback order (tick each once at rehearsal)
☐ **App misbehaves:** `https://localhost:8000/judge` in a new tab (same keys). Fine-tuned wrong: restart with Base.

☐ **a. Quiet captions.** Focus wrong or stuck > 10 s: press **Shift+Q**; captions carry it.

☐ **b. Natural noise suppression.** Captions bad too: open `/simple` on the same server, **Start**, key **1** (Natural), **Space** A/B. Say line 2.

☐ **c. Labelled replay.** Capture dead: **Shift+R**, Replay badge on screen, said out loud.

☐ Drill: pull a bud out mid-demo, carry on.
