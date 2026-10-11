# Demo runbook

> **Sunday (M14, updated Oct 10 by T-405):** the stage page is the app, `bash scripts/demo.sh --judge --team --lan --app` (phones: `https://<laptop IP>:8000/app/`), with `/judge` as the fallback tab on the same server. Base vs fine-tuned separator: `docs/APP.md#1b-base-vs-fine-tuned-separator-the-ab`. The 8 stage steps and keys: `docs/STAGE_CARD.md`. The sections below are the `/judge` runbook they build on (no camera now: team preset, audio only).

## Judge Demo (60 seconds) — the one configuration we present with (T-088)

**Audio path (T-143, final):** input = the laptop's own **MacBook Pro Microphone**, output = **AirPods Pro 3**
(Bluetooth A2DP) picked explicitly on `/judge` (**D** > Output). Nothing else is plugged in (§2). Bluetooth adds ~150-250 ms mouth-to-ear; the rehearsal bar is median <= 300 ms, p95 <= 350 ms.
**AirPods settings (once, with them in the ears):** Control Centre > AirPods: Noise Control = **Noise Cancellation** (not Transparency, not Adaptive), **Conversation Awareness off**, Spatial Audio **off**; System Settings > Bluetooth > AirPods (i): **Automatic Ear Detection off**. While the page records: Control Centre > Mic Mode = **Standard**. The AirPods mic is never the input: picking it switches them to HFP (phone-call audio, no noise cancelling), and `/judge` shows a red banner.

**Plug in:** laptop-a on power, AirPods Pro 3 connected to the laptop, webcam facing the
person we isolate, a second person 1-2 m to the side. Same Wi-Fi is not needed: the laptop page is local.

**One command** (from the repo root):

```
bash scripts/demo.sh --judge
```

It stops our old server on :8000 (by PID, only if it is `hearing serve`), starts a fresh one with the models
pre-warmed, probes the iLab A100 (`scripts/gpu_up.sh probe`; used only when `/health` is ok **and** RTT p95
< 60 ms, else the Mac ladder), prints **one line** (`JUDGE · backend … · expected delay … · rev …`) and opens
`/judge` once the models are warm. The page sends one control (`{"profile": "judge"}`): Focus HQ, suppression 100, strict isolation,
sticky attention, `synth_feed` auto, Balanced delay, **automatic calibration**. Nothing to choose.

**On screen:** the huge **RAW | SELECTIVE HEARING** switch, the pill, live captions. Press **Start**, then
follow the pill: *Look at the person* → *Ask them to say one sentence* → *Calibrating 1/2 s* →
*Isolated — now let the other person talk* (about 5 s after the face locks: 1 s of them talking alone, a 2 s
calibration, 1.5 s to the first enrolled windows; measured 5.0-5.2 s on the replay clip). Keys: **Space** = A/B, **C** = calibrate by hand (backup),
**R** = replay the clip (REPLAY badge), **D** = diagnostics drawer (backend, tier, delay, RTT, SI-SDRi claim,
one line per fallback).

**The four sentences:**
1. "Hearing aids make everything louder. We make one person louder: the one you are looking at."
2. "Say one sentence for me... that's all it needs to learn your voice." (pill turns green: *Isolated*)
3. "Now you talk too." — press **Space** a few times mid-sentence: amber is the raw room, green is selective hearing.
4. "It runs on this laptop; every number you see in the drawer is measured, and if the camera or the GPU drops,
   it keeps going on its own."

**Fallbacks (no dialogs, one drawer line each):** GPU lost → Mac ladder (crossfade); camera lost → Natural,
pill *camera lost — noise suppression only*; phone / Wi-Fi lost → nothing changes on the laptop; live capture
misbehaves → **R** (replay clip). Similar voices: the pill adds *similar voices — limited isolation* on the Mac
(*using lips* on the GPU); pick a mixed-gender pair.

**Natural in one line (T-103):** plain `dfn_ll` (DeepFilterNet3-LL noise suppression) with the slider at **80** (the judge
default). It removes the noise and turns the room down by the slider; it does not separate the other talker, that is
Focus HQ's job. It needs no camera, so camera lost = the same Natural. The gate chain (`dfn_ll>gate>mixer`) scored worse
on demo-mf-easy (BENCHMARKS.md, Live replay, Natural rows) and stays selectable as `natural=natural_gate`.

**Rehearsal self-test:** `bash scripts/demo.sh --judge --check` first runs the enrolled tier's compute-headroom check
(`bash scripts/demo.sh --check` alone, T-089: FAIL below 30 %, close other apps), then prints PASS/FAIL per PLAN.md success criterion
with the measured numbers (judge flow on the replay clip, `demo_regress --fast`, `hearing latency-test` on the
demo devices (MacBook Pro Microphone -> AirPods, "Bluetooth path": median <= 300 ms, p95 <= 350 ms; hold one AirPod
at the laptop's mic while the clicks play), GPU probe, 5 s webcam lock, and the device rows: demo mic in use, mic
is not a Bluetooth (HFP) mic, output is not the laptop speakers, output is the demo output (AirPods)). The latency-test and webcam steps open devices: run it from a
logged-in Terminal, never from an agent.

### AirPods path: what breaks and what to do (T-148)

One Bluetooth audio sink per Mac: laptop-a's is the AirPods. The noise speaker (§A "Noise kit") is paired with **laptop-b or a
phone** and plays from there, never from laptop-a. The banners below are the red page banner on `/judge` (`web/console/devices.js`).

| Failure | What you see | Do this (in order) | Say |
|---|---|---|---|
| AirPods disconnect (case opened, battery, out of range) | red banner "Demo output not found (AirPods…)" or "Audio plays through the laptop speakers … feed back"; the judge hears nothing in the buds | press **Space** to RAW only if it squeals; Bluetooth menu > AirPods Pro 3 > Connect; **D** > Output = AirPods (or the banner's picker / retry); the banner clears; re-check Noise Cancellation is on | "The earbuds dropped; reconnecting, the engine kept running." |
| A phone steals the buds (AirPods auto-switch to an iPhone on the same Apple ID) | audio stops in the buds, the phone shows the AirPods; the page shows the "not found" or speakers banner | **Bluetooth off on every teammate phone** near the demo (before the judge arrives); then reconnect as above. Once: System Settings > Bluetooth > AirPods (i) > Connect to This Mac = **When Last Connected to This Mac** | |
| HFP banner (the AirPods mic became the input) | red banner "Mic is AirPods…: a Bluetooth mic switches the AirPods to HFP"; phone-call sound, no noise cancelling | **D** > Microphone = **MacBook Pro Microphone**; macOS Sound > Input = MacBook Pro Microphone; the AirPods go back to A2DP within a few seconds; the banner clears | "That was the earbud mic; we use the laptop's." |
| Noise speaker will not connect to laptop-a | it pairs, the AirPods drop | do not pair it with laptop-a; play the noise kit from laptop-b (`scripts/demo_noise.py --device …` there) or from a phone's music app | |

Measured (laptop-a, 2026-10-06, replay of `demo-mf-easy`, Mac ladder): see [BENCHMARKS.md](BENCHMARKS.md) and
`docs/decisions/T-088.md`. Live webcam/mic timing on laptop-a: (not measured: needs a human in front of the
camera; record Start → Isolated, delay and A/B at the rehearsal).

---

# Appendix: full runbook

How to stand the voice engine up for the HackRU F26 demo (Busch Student Center, Oct 10-11), rehearse it, reset it between runs, and keep it alive when a piece fails. **Demo-day sections are A-D below; setup, start-up, reset and backups are §1-§10.** The numbers to quote are in [BENCHMARKS.md](BENCHMARKS.md#slide-safe-one-line-claims). The 90-second pitch is in [HACKATHON_PLAN.md §9](HACKATHON_PLAN.md#9-demo-script-90-seconds).

Status (2026-10-06, `main` at T-071): everything is on `main` except the phone PWA (event day). Session server and console, vision in the session (MediaPipe → tracker → lock, Light-ASD with `lipmotion` fallback), the local live loop, the DFN3 fast path (`dfn_ll`, `gate`, `mixer`), the Focus HQ ladder, Synthetic/Text, context recommendations and the remote GPU path. M6 added: the **Raw | Selective hearing** A/B toggle (T-064), the acquisition pill (T-063), the **Delay** preset Speed/Balanced/Accuracy (T-065), **Replay clip** + REPLAY badge (T-066), **Glance/Normal/Sticky** attention (T-067, full console only), per-tier delay (T-062) and the association findings ([decisions/association.md](decisions/association.md), T-061/T-069). **Mac Focus HQ caveats (T-072)** are in §4 "Focus HQ on the laptop". Re-check at the rehearsal: `curl -s localhost:8000/health` lists every stage and whether it is ready.

Two pages: **`/simple`** (what a judge sees: Start, Replay clip, Raw/Selective, mode buttons, Suppression, Runs on, Delay, status line with the acquisition pill) and **`/`** (the console: adds the numbers panel, latency bar, Focus HQ ladder, Attention presets, Release lock, voice override, recommendation toast). Both use the same names below.

### iPhone over USB as the target mic: the A/B (T-171, Sat Oct 10 14:00-16:00 UTC)
Joseph holds an iPhone cabled to the Mac by **USB** (Continuity: macOS lists `<name> iPhone Microphone`; never Bluetooth = HFP), held still, **screen off** (unlocking pauses the audio), Mic Mode **Standard**. One command walks both mics (iPhone first, then `MacBook Pro Microphone`) and writes `results/mic_ab_<stamp>.md` + `results/listening_<stamp>.md` (blind key in `listening_<stamp>_key.md`, open it after listening):
```
bash scripts/mic_ab.sh                     # clicks 3 x 5 per mic, 3 judge takes (target 5 s / other over them 10 s / target 10 s),
                                           # target-absent take, 2 min hand-held dropout check, Shift+C per person, A/B/C listening
bash scripts/mic_ab.sh --dry-run --no-hw --fake-devices "X iPhone Microphone,MacBook Pro Microphone,AirPods"
bash scripts/demo.sh --judge --team --mic "Aryan patel's iPhone Microphone"   # server on the chosen mic (HEARING_INPUT_DEVICE)
```
Gate (printed in the sheet): round trip median <= 300 ms and p95 <= 350 ms, mic peak -48..-2 dBFS, and a lower wrong-speaker share than the MacBook mic in the same session set. Fallback: MacBook mic with the target at 40 cm. `judge_check` prints the chosen label as `PASS Demo mic label (chosen)`; an iPhone label is not Bluetooth, so no HFP banner.

## Team preset — one flag pins every demo knob (T-130)

`bash scripts/demo.sh --judge --team` (= `HEARING_DEMO_PRESET=team`) loads `config/demo-team.yaml` on top of the
judge profile; without `--team` nothing changes. Run of show with the preset:

1. **Before doors (T-10 min):** `bash scripts/demo.sh --judge --check --team` (add `--no-hw` for a dry run). It adds a
   "Team preset pins" row and replays `demo-team-easy` instead of `demo-mf-easy`. Record the profiles first (TESTING_GUIDE §2a)
   and set `pair:` in `config/demo-team.yaml` to the two profile names; `verified: true` only after a listen confirmed a
   male/female pair (only then is the same-gender guard switched off).
2. **Start the engine:** `bash scripts/demo.sh --judge --team` prints `TEAM PRESET · ...` above the usual `JUDGE` line,
   which says `backend Mac (measured: beats the A100 with profiles, T-129) · expected delay ~1.5 s`: the preset pins the
   Mac and never probes the GPU, even with the tunnel up (T-181). T-129 measured with profiles loaded
   ([BENCHMARKS.md → T-129 attempt 4](BENCHMARKS.md#t-129-focus-hq-tuning-on-the-a100-with-voice-profiles-2026-10-08), rows `mac` vs `base`/`novideo`/`standard`): team-easy SI-SDRi
   **+1.55 dB** / leak **-7.4 dB** on the Mac vs the A100's best **-1.40 dB**; mf-easy **+3.10 dB** / **-14.5 dB** vs
   **-0.81 dB**; the A100 only wins delay (825-950 ms vs 1500 ms). To run Focus HQ on the A100 anyway:
   `bash scripts/demo.sh --judge --team --gpu` (or `HEARING_HQ_BACKEND=remote`) brings the GPU up, probes it and pins it
   as before; the line then says `backend iLab A100 (--gpu) · ... · expected delay ~0.95 s` (or the Mac ladder if the
   probe failed).
   Noise speaker: `HEARING_DEMO_PRESET=team .venv/bin/python scripts/demo_noise.py` starts at the pinned
   `demo_noise_level_db`.
3. **Press Start facing the teammate:** the first face attention picks (the centred one) becomes an explicit lock, held
   60 s through a lost face; the teammate's voice profile is the target from the first window, the other teammate's the
   negative. **C** still re-learns the voice.
4. **Drawer (D):** the `TEAM PRESET` row lists every pin: the HQ backend (the Mac; with `--gpu` decided **once** from a
   `/health` probe of the GPU tunnel at the first Start: remote when RTT p95 < 60 ms, else the Mac ladder, re-probed
   every 10 s), the isolation (**standard**, T-129 Q-470: on the Mac with profiles it wins mf-easy +9.06 vs +7.54 dB,
   leak -19.6 vs -11.9 dB, and ties team-easy +3.54 vs +3.55 dB; BENCHMARKS "T-129 attempt 5", rows `mac-standard` /
   `mac-strict`; the drawer's **Isolation** picker switches to strict),
   the same-bucket guard, lock/hold, the Name Call name + aliases (Ariane, Arion, Arian; strict match; quiet 10 s,
   release 180 s), the Synthetic voice (pinned to the target's bucket, overridable), Mac ladder min switch 3 s,
   Synthetic/Text feed `focus`, ASR `local` (faster-whisper, temperature 0, **no prompt**: T-139 removed the
   vocabulary prompt because Whisper put "Selective Hearing," and "Select" over real speech), and the noise level.
5. **ElevenLabs:** TTS uses it whenever `ELEVENLABS_API_KEY` is set (else Kokoro). Captions stay local unless the
   venue Wi-Fi was verified with ElevenLabs: then start with `HEARING_ASR=elevenlabs bash scripts/demo.sh --judge --team`.

Hard line: no prerecorded audio or canned captions on any live path, and no caption prompt (a `caption_prompt` key, if
ever put back, is vocabulary only, never the scripted lines). The GPU job (iLab 435775, tunnelled at `ws://127.0.0.1:18765/ws/session`) is never started or cancelled
by the preset.

## Name Call fallback — "no camera needed, say my name" (T-120)

If the camera is a risk (no light, USB flaky, judges uncomfortable with a camera), drop it and go audio-only:

1. `/judge` → tick **Name Call** (or press **N**), name field = the wearer's name (e.g. `Aryan`). Unplug the camera if you like.
2. Say: *"Now no camera at all. Whoever calls my name is the person I want to hear."*
3. A judge says "Aryan, …". Pill: **Listening for “Aryan”** → **Heard “Aryan” from a new voice — locking** → **Isolated** (about 3-4 s after the name on the Mac).
4. Another person talks over them: press Space for RAW vs SELECTIVE HEARING. Then the second person says "Aryan": the target switches to them.
5. **Release** (next to the name) or 60 s of silence from the target goes back to listening (Natural noise suppression).

Say: *"No camera needed — say my name. The engine listens for the name, takes a voice print from that sentence and keeps only that voice."*
Failure drill: the pill stays on Listening after the name → say it again louder and closer; check the drawer's log; or add the spelling Whisper hears as an alias (`Aryan, Arian`).
Guards (T-123): keep the **match** select on `normal` (default; ignores "Ryan"-like names, 0 false triggers in the 5 min + "Ryan" sets). Someone named Ryan in the room under `loose` → put `Ryan` in the **not:** field. `strict` only for a name Whisper spells right (it writes "Ariane" for "Aryan" about half the time).

## A. Four-minute judge script

Setup before the judge arrives: server up (§4), `/simple` open on the laptop in Chrome, the AirPods (output picked on the page) for the judge, a teammate (the target) standing 1 m from the camera, the backup speaker playing music + chatter at a fixed volume (§8), the console `/` open in a second tab for the diagnostics step. **Pair rule (association.md): the target and the second talker must differ in gender.** Keep **Runs on = Local** and **Delay = Balanced**. **Tick Strict** (next to the pill, T-085): fewer seconds of the other person, slightly more short dropouts.

| Time | Click / do | Say |
|---|---|---|
| 0:00 | Press **Start**. Allow mic and camera. Pick **Focus HQ** (mode buttons; Natural has no voice enroller, so the pill could not pass `Target locked` there). Suppression slider at 0. | "Hearing aids make everything louder. We make one person louder: the one you face." |
| 0:15 | **Calibrate (T-085).** Ask the teammate to say one sentence alone, and press **Calibrate voice (3 s)** next to the pill (key **C** on the console) as they start. The button counts down `Listening… 2.1 s`; the pill reads `calibrating 1.2/3.0 s`, then the button shows **Calibrated ✓** (≤ 3.5 s from the press). If it says **Too quiet, try again**, press it again while they talk. The calibrated voice is kept until **Clear** or a deliberate switch to another face. | "Say one sentence for me... That's all it needs to learn your voice." |
| 0:20 | (If Calibrate was skipped or failed: teammate speaks **alone for about 3 s**, the automatic path.) Watch the pill beside the tier name on `/simple`: **Searching → Candidate → Acquiring voice → Target locked → Isolated**. If boxes show but nothing locks, tap the target's face. | "The camera finds who you are looking at, then learns their voice. That takes about three seconds; the pill shows each step." |
| 0:50 | Second person starts talking over the target, music on. Drag **Suppression** 0 → 100 slowly. | "Zero is the room as it is; a hundred is only them. You choose how much of the world to keep." |
| 1:20 | **The A/B moment.** Press **Space** (or the **Raw** button): the room comes back. Press **Space** again: **Selective hearing**. Flip 2-3 times mid-sentence, no talking between flips. Judge keeps the headphones on. | Only before the first flip: "Same moment in the sentence, same microphone." After that, silence; let the judge hear it. Say which side is live: Raw is amber. |
| 2:00 | You are already on **Focus HQ**. Delay stays **Balanced**. Point at the pill and the tier name: on the Mac it started on the gate (≈ 13 ms) and has stepped **up** to enrolled (1.5 s behind) now that the voice is learned. Once the pill reads `Isolated`, click **Speed**, wait for the short dip (2-4 s: the old setting keeps playing until the new one has isolated the voice), then **Accuracy** once. The pill must stay `Isolated` (T-078: the voice clip is kept; if it ever goes back to `Acquiring voice`, skip the clicks from then on). | "Focus HQ cleans the voice more but plays about 1.5 s behind; Speed trades quality for about 0.75 s, Accuracy the other way. It learns the voice once and keeps it, whatever setting or mode you pick; the brief dip is it jumping to the new delay instead of playing two moments on top of each other." |
| 2:40 | Click **Synthetic**, teammate says one sentence; then **Text**. | "For someone who cannot hear the voice at all, we re-speak the words in a clear voice, or show captions." |
| 3:10 | Switch to the `/` tab. Point at: big latency number and the latency bar, **Tier**, **Focus HQ ladder** (each tier's delay), **Vision**, **Underruns / dropped**. | "Nothing is hidden: this is the latency of every stage, the tier that is playing, and the glitch counters. These are measured numbers (see D)." |
| 3:30 | Optional, if the judge is curious: **Attention** Glance / Normal / Sticky; teammate looks down for 1.5 s. | "Glance follows fast, Sticky ignores glances. Normal keeps the target through a look-down: 0.985 vs 0.875 (BENCHMARKS.md, Attention presets, `g_look_down`)." |
| 3:45 | Press **Replay clip** only if live capture misbehaved, otherwise skip. Close with the claim sheet (D). | "It runs on a laptop, no cloud needed for the fast path." |

If the judge asks "does it really pick the right person?": quote D, association row, and say plainly that it **drops out** sometimes but rarely plays the wrong person (association.md).

### Noise kit: café babble and a second voice on cue (T-122)
A small Bluetooth speaker beside the target plays the background, paired with **laptop-b or a phone** (laptop-a's one Bluetooth sink is the AirPods, T-148). Never the judge's headphones: the script refuses the engine's output (`--engine-output`, else the system default output).
```
.venv/bin/python scripts/demo_noise.py --list
.venv/bin/python scripts/demo_noise.py --device "JBL" --calibrate     # pink noise; +/- until a phone SPL app reads ~65 dB(A) at the listener; note the printed level
.venv/bin/python scripts/demo_noise.py --device "JBL" --level -20     # keys: 1 café babble, 2 second talker, 3 both, +/- level, Space pause, q quit
```
Judge script cue: "Turn on the café noise" (press `1`), "now a second voice" (press `3`), "now press Space for RAW" (the Raw/Selective button in the page, not the noise kit's Space: click the page first so the key goes there; the noise keeps looping). Keep the target and the second voice different in gender. Rehearse once with the same `--level`.

## B. Ten-minute rehearsal checklist (morning of Oct 10, at the venue)

Write every number in the right-hand column on paper; they replace the engine-only numbers on the slide.

**Final test (Oct 10):** `bash scripts/rehearse.sh` runs the 12-step sheet in [REHEARSAL.md](REHEARSAL.md) and writes `results/rehearsal_<stamp>.md` (PASS/FAIL per bar, NOT MEASURED for anything the run did not record).

| Min | Check | Pass / write down |
|---|---|---|
| 0-1 | `caffeinate -dims` running; AirPods Pro 3 connected (Noise Cancellation on, Transparency/Adaptive/Conversation Awareness/Spatial/ear detection off), macOS input = MacBook Pro Microphone; Zoom/Teams closed | the AirPods are the output, never the input |
| 1-2 | Start the server (§4). `curl -s http://127.0.0.1:8000/health` | `"ok": true`; `dfn_ll`, `gtcrn`, `gate`, `mixer`, `tiers`, `tse`, `synth`, `silence` `ready` |
| 2-4 | `.venv/bin/python -m hearing latency-test --pipeline dfn_ll --in "MacBook Pro Microphone" --out AirPods --sr 48000 --clicks 10`, one AirPod held at the laptop's mic | **Bluetooth mouth-to-ear ms (median, p95)**: bar median <= 300, p95 <= 350. Last measured on built-in mic/speakers: 64 ms passthrough (T-013) |
| 4-5 | Open `/`; Start; speak; read the big latency number and **Underruns / dropped** | **console latency ms** (browser path) and glitch count; the target is 100 ms (aim 70) |
| 5-6 | In the venue light: face the camera square-on at 1 m, speak alone ~3 s; the pill must reach **Isolated**; tap the face if it stalls at Candidate; if boxes never appear, add light or move away from a window behind the target | **seconds to Isolated** (association.md: ~3.7 s mean in tests) |
| 6-7 | Click **Replay clip** on `/simple`: REPLAY badge shows, face boxes appear, you hear audio. **Stop replay**, press Start again. If it says the clip is missing, run §6 "If the room fails" preparation | clip plays with real processing; note the file name |
| 7-8 | Raw ↔ Selective: Space twice with music playing | no click on the flip (not tested with real ears, T-064); same moment in the sentence |
| 8-9 | Click **Focus HQ**, set Delay to each of Speed / Balanced / Accuracy; read the ladder on `/` | **tier that plays and its delay ms**; tier must step up from gate to enrolled once the pill reads `Isolated`; on the Mac the ladder starts on the gate (≈ 13 ms algorithmic) and steps **up** to enrolled (1.5 s at Balanced) once the voice is enrolled, see §4 |
| 9-10 | Only if the GPU is in the demo: `scripts/gpu_up.sh status` then `probe`. Pin **Runs on** accordingly (§4). Otherwise pin **Local**. Start the backup clips paused at 0:00 (§8) | RTT p50 < 20 ms to pin GPU; else Local |

## C. Failure drills

Run each once at the rehearsal. Say the "Say" line out loud: the tier indicator tells the truth, so name what is playing.

| Drill | Trigger | Do | Say |
|---|---|---|---|
| **Wi-Fi down** | disconnect Wi-Fi | The fast path and local Focus HQ never need it. Set **Runs on = Local** if the GPU was pinned, set ElevenLabs aside (`HEARING_TTS=local`), then press **Replay clip** on `/simple`: the prerecorded two-person clip runs through the real pipeline with the same controls and a **REPLAY** badge (§6 "If the room fails" for preparation). Never claim it is live; the badge is on screen | "The venue network is gone, so this is a prerecorded two-person clip through the same engine." |
| **Camera fails** | no boxes, pill stuck on **Searching**, or the process dies (§6 table) | Click **Natural**. After 1 s without frames the session is audio-only: noise suppression still works, no target lock. Raise Suppression to ~60. Restart with `HEARING_FACES=none bash scripts/demo.sh --restart` if the server dies | "Without the camera it degrades to noise suppression; the target needs eyes." |
| **Same-gender judges** | the target and the other talker are both male or both female | Association collapses for the Enrolled tier on same-gender pairs: accuracy **0.742 (M/M)** and **0.844 (F/F)** vs **0.945 (M/F)**, other person audible 1.75 / 1.22 s per 10 s (association.md, T-069 results table, `final` rows). In Focus HQ the ladder skips Enrolled by itself when both faces get the same voice group (T-070, "Same-gender pairs" in §4): check that the Enrolled row reads `same-gender pair: enrolled skipped` (the AV tier is not built on the Mac, so the gate plays there; AV plays only on the GPU box). If the row does not say so (the face estimator called one of them `neutral`), restart with `bash scripts/demo.sh --restart -- --pipelines "natural=natural_gate"` and switch to **Natural** (the gate chain: ducks by lips, never swaps people; the default Natural has no gate since T-103) at Suppression ~70; or use **Replay clip** (cross-gender GRID pair) | "Two similar voices are hard for voice-matching; the camera-based path doesn't care." |
| **Codex / GPU unavailable** | `gpu_up.sh probe` fails, `remote.connected=false`, ladder stuck below tier 1 | Set **Runs on = Local**. The ladder falls back by itself; the demo is the Mac (Local) path. What Local cannot do on the Mac (no AV tier) is in §4 "Focus HQ on the laptop" | "The GPU is the luxury path; the laptop does the demo." |
| Target drops out for a second | Pill goes **Reacquiring** | Do nothing: expected (association.md: expect dropouts, not the wrong voice); teammate keeps talking | "It lets go rather than amplify the wrong person." |
| A/B sounds identical | Raw and Selective play the same | Suppression is 0 or mode is Quiet; set slider ≥ 70 and Natural | |

Every other single-component failure (server, models, mic, ElevenLabs, ASR, CoreAudio) is in §6.

## D. What the numbers mean (one page for judges)

Condition always travels with the number. All rows from [BENCHMARKS.md](BENCHMARKS.md) unless noted; host Apple M4 Pro laptop (Aryans-MacBook-Pro.local).

| Term | Plain meaning | Our number (source row) |
|---|---|---|
| **SI-SDRi** (dB) | How much cleaner the voice is than the input, scale-free. +3 dB is audible, +10 dB is a big cleanup. | Noise only, `dfn_ll`: **+4.06 dB** ("Every mode and tier", `natural / audio-only / dfn_ll / noise only`). With a competing talker a denoiser only gets +1.73 dB (`dfn_ll`, mix). Enrolled `tse:enrolled@2`: **+11.78 dB** (clean enrollment). Focus HQ `tse:av_mossformer2@2`, GRID with lips: **+13.75 dB**; at 1 s windows only +5.26 dB (`av_mossformer2@1`) |
| **WER** | Share of words a transcriber gets wrong from the output; lower is better, above 1 means more errors than words (it also writes the other talker). | `av_mossformer2@2` cuts WER from **1.406 (raw AV mix) to 0.781**. `blind_assoc@2`: 0.552 to 0.175 (LibriSpeech, perfect speaking signal). Captions from the raw mix **0.588** vs 0.730 from `dfn_ll`-cleaned audio (Text rows) |
| **Latency, decomposed** | Mouth-to-ear = input buffer 10 ms + algorithm + compute + output buffer 10 ms; the console bar shows each part. | Natural: 13 ms algorithmic + **0.71 ms p95 compute per 10 ms chunk**, RTF 0.060 (`dfn_ll`, noise only row). Budget: target 100 ms, aim 70 (ARCHITECTURE.md). Browser adds ~20-40 ms; Bluetooth 150-250 ms. AirPods total: **the number written at rehearsal, §B** (not measured yet: needs the AirPods on laptop-a). Focus HQ is delayed by design: enrolled window + hop = 0.75 / 1.5 / 2.25 s for Speed / Balanced / Accuracy (DESIGN.md item 6; BENCHMARKS.md T-065 table: 750, 1500, 2250 ms window + hop on MPS, p95 headroom +99 / +260 / +402 ms) |
| **Association accuracy** | Of 1 s windows, how often the played stream is the person the wearer chose; "wrong-target s/10 s" is how long the other person is audible. Bar: ≥ 0.95 and < 0.5 s. | M/F pair, SNR 5 dB babble, `final`: **0.945 accuracy, 0.20 s wrong / 10 s**, 19 / 22 targets lock in 3.7 s mean (association.md, verdict table; **not at the 0.95 bar**). Same-gender: **0.742 (M/M), 0.844 (F/F)**, 1.75 / 1.22 s wrong (T-069 table). Errors are mostly dropouts, not the wrong voice. 2 s windows reach 1.000 / 0.08 s on M/F but add 1 s delay. Whole Focus HQ ladder with the same-gender guard (T-070, association.md addendum; 1 s spans of any tier, not comparable window-for-window): M/M wrong-target **1.27 → 0.87 s / 10 s**, acc 0.867 → 0.872, at +0.36 s mean delay; M/F unchanged (0.960 / 0.41 s) |
| **Attention** | How quickly it follows a new person vs holds the old one. | `normal` 0.987 overall, 0 false switches; `glance` 0.972, 2; `sticky` 0.957, 0 (BENCHMARKS.md, Attention presets) |

Honest limits to say if asked: scenes are GRID read speech with an oracle lock, Enrolled in the bench used clean enrollment, same-gender pairs are still below the bar (the guard cuts M/M wrong-target audio by a third but does not reach 0.95), and the bench host for most rows is laptop-a, not the venue. No real-room DNSMOS row yet (§9).

---

## 1. Hardware kit

| Item | Why | Check |
|---|---|---|
| Laptop-a (M4 Pro, 24 GB) + charger | runs the engine (fast path local, never needs Wi-Fi) | `caffeinate -dims` running; the lid stays open |
| **AirPods Pro 3** (A2DP output only) | what the judge hears in every mode; ANC blocks the leaking room | connected to the laptop; picked under **D** > Output on `/judge`; Noise Cancellation on, Transparency/Adaptive/Conversation Awareness/Spatial Audio/Automatic Ear Detection off; the blue Bluetooth note in the drawer is info only |
| **MacBook Pro Microphone** (built-in) | the only demo mic; the AirPods mic would switch them to HFP | picked under **D** > Microphone; Mic Mode Standard; the level meter sits in the green (peak -48 to -2 dBFS, T-145) while the target talks; no red "Bluetooth mic (HFP)" banner |
| External mic, wired headphones | not used (decision Oct 9: built-in mic + AirPods only) | leave them in the bag |
| Phone with the PWA (or the laptop browser console) | camera + mic + UI | on the same network as the laptop, or a USB/hotspot link |
| Phone hotspot | GPU path when venue Wi-Fi is bad | laptop joins it in < 30 s |
| Small Bluetooth speaker + backup clips on the laptop | plays the "noisy room" (music + chatter) on cue, from laptop-b or a phone (one Bluetooth sink per Mac: laptop-a's is the AirPods) | volume preset; clip list in §8 |
| USB-C hub, HDMI adapter, extension lead | the slide + the console on the projector | tested on the venue projector at setup |
| Earmuffs (optional) | blocks the leaking real room for a judge trying it | |

Overnight the laptop only had its built-in mic and speakers, so **every latency number for Natural on the AirPods must be measured at the rehearsal** (see §3). Until then quote the engine-side numbers only.

## 2. One-time setup (day before)

```bash
bash scripts/setup.sh --all                         # .venv + every extra; prints the device (mps)
export HEARING_MODELS_DIR="$PWD/models"             # weights stay out of git
# weights (each command is idempotent and pins a SHA-256 or revision)
.venv/bin/python -c "from hearing.enhance import dfn_graphs as d; d.provision('dfn'); d.provision('dfn_ll')"  # DFN3 ONNX, 8 + 36 MB
.venv/bin/python -c "import hearing.enhance.gtcrn as g; print(g.download_model())"        # GTCRN, 0.5 MB
.venv/bin/python -m hearing.vision.faces --download                                      # MediaPipe, 4 MB
.venv/bin/python -c "import hearing.synth.apparent_face as a; print(a.download_model())"  # InsightFace genderage (NC)
.venv/bin/python -c "import hearing.tse.blind as b, hearing.tse.enrolled as e; b.download_model(); e.download_model()"  # MossFormer2 640 MB + ECAPA 85 MB
.venv/bin/python -m eval.dnsmos --download                                                # DNSMOS, 1.4 MB
# Kokoro (local TTS fallback): kokoro-v1.0.onnx (fp32, 326 MB; 3x faster than int8 on the Mac) + voices-v1.0.bin into models/kokoro/ (docs/MODELS.md)
# Whisper caches itself on first use (base.en for live ASR, small.en for eval WER)
.venv/bin/python -m eval.make_fixtures --full        # network once: 96 LibriSpeech mixtures for the bench
.venv/bin/python -m eval.make_fixtures --av --full   # network once: 160 GRID audio-visual mixtures (Focus HQ rows)
echo 'ELEVENLABS_API_KEY=...' >> .env                # optional; without it Synthetic uses Kokoro, then the mock
```

Then `bash scripts/verify_fast.sh` (must be green) and `.venv/bin/python -m eval.bench --full` (the whole 96 + 160 fixture bank is ~1.5-2 h on laptop-a under load; `--limit 32 --tse-limit 16 --wer-limit 16` took 21 min on Oct 5; refreshes BENCHMARKS.md with this machine's numbers; `--full --from-json eval/results/<stamp>.json` re-renders the doc from a finished run).

## 3. Before every session (T-30 min)

1. `caffeinate -dims` in its own terminal (the machine must not sleep).
2. Connect the AirPods Pro 3 (settings as in the Judge Demo section above); set macOS input = MacBook Pro Microphone. Close Zoom/Teams (they grab the audio device). In the page, pick **Microphone** = MacBook Pro Microphone and **Output** = the AirPods in the device pickers and check the level meter reads "good" while the target talks; the choice is remembered per browser. A red banner means the AirPods mic (HFP) or another output is in use.
3. Start the engine (§4) and open `http://127.0.0.1:8000/` in Chrome. Allow mic + camera.
4. `curl -s http://127.0.0.1:8000/health` → `"ok": true` and `dfn_ll`, `gtcrn`, `gate`, `mixer`, `tiers`, `tse`, `synth`, `silence` all `ready`. Show your face to the camera: the console draws a box and TARGET LOCKED after the dwell.
5. Measure the real round trip on the demo devices: `.venv/bin/python -m hearing latency-test --pipeline dfn_ll --in "MacBook Pro Microphone" --out AirPods --sr 48000 --clicks 10` with one AirPod held at the laptop's mic (clicks through the AirPods into the built-in mic; median of N; `judge_check` runs the same). Overnight on laptop-a's built-in mic → speakers, passthrough measured 64 ms at 16 kHz / 5 ms blocks (T-013). Then in the console speak and watch the big mouth-to-ear number. Write both down (AirPods, Natural). Bar: median <= 300 ms, p95 <= 350 ms; above that, re-connect the AirPods and close other Bluetooth audio.
6. Slider 0 → 100 once: the room should fade like a dimmer, never jump or click (the slider asks for 3 dB per 10 points, -27 dB at 90, silence at 100, every change ramped over 30 ms). The voice stays. Measured on the demo clip in Natural, the output drops -5.5 / -9.5 / -12 / -14 dB at 25 / 50 / 75 / 100: noise follows the curve, but other voices only go down about 10-12 dB (DFN keeps speech), so to remove the other talker use Focus HQ (BENCHMARKS.md, slider sweep). Press 3 (Synthetic): you should hear the TTS voice after each phrase; captions appear.
7. GPU path (only if Focus HQ is in the demo): `scripts/gpu_up.sh status` then `scripts/gpu_up.sh probe` (RTT p50 should be < 100 ms on a good link).
8. Have the backup clips (§8) open in a player, paused at 0:00.

## 4. Start the engine

**Local loop (lowest latency, no browser):** `.venv/bin/python -m hearing live --mode natural --pipeline "dfn_ll>gate" --suppression 80 --in "MacBook Pro Microphone" --out AirPods`. Keys switch modes; `--control-port 0` prints an HTTP port for the slider. `--dry-run` prints the resolved pipeline and devices without opening audio.

**Session server + console (what the phone/website talk to).** `hearing serve` imports every stage package itself (missing weights show as `unavailable` in `/health`, never a crash):

```bash
export HEARING_MODELS_DIR="$PWD/models"
bash scripts/demo.sh --restart --simple
```

The launcher finds the listener by `PORT` (default 8000), validates it is `hearing serve`,
sends TERM and waits for exit, then verifies `/health` belongs to the new PID and
has a start time after launch. Unrelated listeners or a server that fails to exit are
reported as conflicts; they are never silently reused or force-killed. It prints the
PID, UTC `started_at`, package `version` and startup `git_rev`; the same identity appears
small at the bottom of both pages and in `session_report.py` headers. Check it before
measuring. `git_rev` is the startup commit, so commit changes before recording evidence
(it does not identify uncommitted edits). Use `PORT=8001` for a second demo instance,
`--no-open` for automation, and arguments after `--` to forward server options.
Logs are in `${TMPDIR:-/tmp}/hearing-serve-$PORT.log`.


Focus HQ placement: `--hq-backend {auto,local,remote}` (default `$HEARING_HQ_BACKEND` or `auto`) and `--remote-url http://<gpu-host>:8000` (default `$HEARING_REMOTE_URL`); `remote`/`auto` without a URL play local.

Direct `hearing serve` and the launcher both play `natural=dfn_ll` (the server default, T-103; the launcher no longer passes `--pipelines`). The gate chain is one name away: `HEARING_PIPELINES="natural=natural_gate"` (= `dfn_ll>gate>mixer`) or `bash scripts/demo.sh -- --pipelines "natural=natural_gate"`. Do **not** name `synthetic=`/`text=` there: an explicit Synthetic/Text pipeline is used as given and switches the `synth_feed` selection off (T-079). Left out, the ASR hears the raw room with one talker (the `mixture` feed; a configured Natural chain no longer feeds it, because DeepFilterNet in front of Whisper gave worse captions, WER 0.73 vs 0.59, and twice the phrase latency, 3100 vs 1390 ms, BENCHMARKS.md) and the Focus HQ ladder's output once two people talk for 2 s (`synth_feed=auto`, CONTRACTS.md). `HEARING_PIPELINES` sets the same thing for direct `hearing serve`; to override the launcher use `-- --pipelines "..."`.

| Mode (console key) | Pipeline | Notes / remote variant |
|---|---|---|
| Natural (1) | `dfn_ll` (DeepFilterNet3-LL, 13 ms, slider = attenuation). Option `natural_gate` = `dfn_ll>gate>mixer` (the gate ducks the room while the locked target's ASD says silent, bypasses with no face track). Fallback `gtcrn` | |
| Focus HQ (2) | `tiers` (`config/tiers.yaml`): Enrolled first (1 s windows / 0.5 s hop on the Mac, 1.5 s behind at Balanced, once ~2 s of the target's voice is collected, and unless the two faces are a same-gender pair, see below), the cold-start AV-MossFormer2 tier (2 s / 1 s, 3 s behind) while enrolling or for a same-gender pair, then `dfn_ll>gate>mixer`, then `dfn_ll`; each tier plays at its own delay (T-062) and the console ladder shows them | AV-MossFormer2 needs 2 s windows: **+13.75 dB SI-SDRi at 2 s vs +5.26 dB at 1 s** (16 GRID clips, BENCHMARKS.md). The **Delay** preset (Speed/Balanced/Accuracy) rewrites the Enrolled window/hop; the AV tier keeps 2/1 s. Point the tier at `remote(...)` (`docs/DEPLOY.md`) when the laptop cannot keep up. **Mac caveats: §4 "Focus HQ on the laptop"** |
| Synthetic (3) | `passthrough>synth(speak=true)` (`tiers>synth(...)` while 2+ talk, `synth_feed=auto`): ElevenLabs if `ELEVENLABS_API_KEY`, else Whisper base.en + Kokoro. Kokoro fp32 measured 0.43 s p50 TTS time to the first re-voiced sample (was 1.3 s with int8), 0.8 s from ASR phrase end including the 0.4 s end-of-phrase wait | the raw stream feeds the ASR with one talker, the ladder with two (see above) |
| Text (4) | `passthrough>synth(speak=false)`: captions only | |
| Quiet (5) | `silence` | |
| Auto (6) | the context model picks one of the above | |

Useful env: `HEARING_FACES` = `auto|mediapipe|mock|none`, `HEARING_FACES_DELEGATE` = `cpu|gpu` (default `cpu`; **leave it unset on stage**: MediaPipe's Metal path aborted the whole server when the camera started, T-060), `HEARING_ASD` = `auto|light_asd|lipmotion`, `HEARING_ASR` / `HEARING_TTS` = `mock|local|elevenlabs|auto`, `HEARING_PREFS_LOG=/tmp/prefs.jsonl` (the preference log, see §5), `HEARING_VOICE_MATCH`, `HEARING_REMOTE_URL`.

Phones on the venue Wi-Fi: `bash scripts/demo.sh --restart --lan` (https + wss, QR code); see [Phones](#phones) below.

### Phones

A phone can only open the mic/camera from a **secure context**: `http://<laptop-ip>:8000` fails on every
phone (only `localhost` is exempt). So phones use https:

```bash
bash scripts/demo.sh --restart --lan --simple     # = hearing serve --host 0.0.0.0 --tls + the demo pipelines
```

It prints the phone URLs and a QR code (also `hearing qr --port 8000`, and the page
`https://<laptop-ip>:8000/connect` shows the same QR with links). URL scheme:

| What | URL |
|---|---|
| test page | `https://<laptop-ip>:8000/simple` (or `https://<laptop-name>.local:8000/simple`, the Mac's mDNS name, where the network passes mDNS) |
| debug console | `https://<laptop-ip>:8000/` |
| WebSocket | `wss://<laptop-ip>:8000/ws/session` (pages derive it from `location`; override with `?ws=`) |
| QR + links | `https://<laptop-ip>:8000/connect` |
| certificate | `https://<laptop-ip>:8000/hearing-ca.crt` |
| SDK | `https://<laptop-ip>:8000/sdk/hearing-client.js` (`web/sdk/README.md`) |

**Certificate.** `--tls` makes a self-signed certificate (its own CA, RSA 2048, 397 days) for every LAN
IPv4 of the laptop, the laptop's own `.local` name (`<laptop-name>.local`, the only one macOS advertises over mDNS), `hearing.local` (in the certificate only; nothing advertises it), `localhost` and `127.0.0.1`, cached in `~/.hearing/tls`
(`$HEARING_TLS_DIR`). It is regenerated only when a new IP shows up (another network) or it nears expiry,
and keeps the old IPs, so each phone trusts it **once** (repeat after a regeneration). Own certificate:
`hearing serve --cert cert.pem --key key.pem`.

One-time trust step per phone (do it the day before, on the venue Wi-Fi or a hotspot):
- **iPhone (Safari)**: open `https://<laptop-ip>:8000/hearing-ca.crt` in **Safari** → Allow → Settings →
  *Profile Downloaded* → Install (passcode) → Settings → General → About → **Certificate Trust Settings** →
  turn on *hearing.local (selective hearing, self-signed)*. Without the trust step Safari lets you "visit
  this website" but then refuses the `wss://` socket, so the page says "Can't reach the engine".
- **Android (Chrome)**: download `/hearing-ca.crt` → Settings → Security (and privacy) → More security
  settings → Encryption & credentials → Install a certificate → **CA certificate** → Install anyway → pick
  the file. Quick alternative: on Chrome's warning tap *Advanced → Proceed*; mic and wss then work for
  this session.
- **mkcert alternative** (a real local CA, same trust step but one CA for every cert):
  `brew install mkcert && mkcert -install && mkcert -cert-file ~/.hearing/mkcert.pem -key-file ~/.hearing/mkcert-key.pem <laptop-ip> hearing.local localhost 127.0.0.1`,
  then `bash scripts/demo.sh --restart --lan -- --cert ~/.hearing/mkcert.pem --key ~/.hearing/mkcert-key.pem`
  and install `"$(mkcert -CAROOT)/rootCA.pem"` on the phones as above.

Phone constraints (handled in `web/console/simple.js`, `app.js` and `web/sdk/hearing-client.js`):
- **iOS Safari**: the AudioContext is created and resumed inside the Start tap (iOS keeps it suspended
  otherwise), at the native rate (48 kHz); `hello.sample_rate` says so and the server resamples to 16 kHz and
  back (`hearing.server.resample`, tested 48 k/44.1 k ↔ 16 k). AudioWorklet needs **iOS 14.5+**. No
  background audio: the page resumes when shown again; keep the screen awake and Safari in front.
- **Android Chrome**: echo cancellation, noise suppression and AGC are off in the constraints (the
  engine wants the raw room); the context runs at 16 kHz.
- Wear earphones on the phone too: the phone speaker feeds back into its mic with echo cancellation off.
- Venue Wi-Fi with client isolation (phones cannot reach the laptop): use the laptop's own hotspot or a
  phone hotspot the laptop joins; the certificate picks up the new IP on the next start (trust once more).
- Checked headless by `.venv/bin/python scripts/phone_check.py --webkit` (Playwright iPhone 14 + Pixel 7
  emulation over https with the self-signed cert, fake mic: secure context, wss, Start in the tap,
  constraints, worklet frames out and processed audio back). Real devices are a rehearsal item.

### Raw ↔ Selective hearing (the judge A/B)
The big **Raw | Selective hearing** toggle (both `/` and `/simple`; **Space** anywhere on either page) switches what the headphones play while the scene keeps running. Raw is the live mic delayed by the engine's declared latency (`latency_ms.total`), so A and B are time-aligned, and the switch is a 100 ms crossfade (no click, no jump). Audio keeps flowing to the engine in both states, so the target lock and enrollment never reset. The state is sent as `control {ab: "raw"|"processed"}` and echoed in telemetry (`ab`).
Script for the judge: (1) set the scene (target talking, a second voice and music in the room); start on **Selective hearing**. (2) "This is what you hear with it." Press Space: **Raw** (the room as is, same moment in the sentence). (3) Press Space back: the target comes forward again. Do it two or three times mid-sentence; do not explain between flips. Say which side is live (the active half is filled; Raw is amber).
Not tested with real ears: the automated checks drive the worklet directly (`node --test web/console/tests/*.test.mjs`: delay alignment, crossfade has no step above the tone's own slope, the engine ring does not back up) and click/Space the toggle in headless Chromium (`simple-e2e.mjs`). Listen once at the rehearsal for any click on the flip.

### Focus HQ backend on stage
The console's "Runs on: Local / GPU / Auto" selector (control `hq_backend`) picks where Focus HQ runs. `auto` is the default and switches to the iLab GPU only while its `/health` is green and RTT < 150 ms. **On stage, pin it**: `Local` when the venue network is unknown, `GPU` only after `scripts/gpu_up.sh probe` shows RTT < 20 ms from the venue (campus network does; a hotspot adds 30-60 ms and jitter). A pinned backend cannot change tiers mid-sentence. Bring the GPU up before the slot with `scripts/gpu_up.sh ilab` (needs the human's iLab SSH session, see DEPLOY.md) and take it down with `scripts/gpu_up.sh down`.

#### GPU bring-up that worked on 2026-10-06 (T-075, iLab A100)

The A100s sit on `rlab2` only (`sinfo -N -o "%N %P %G %t"`: `rlab2 unlimited gpu:a100:4`); the
login node behind the socket was `ilab1` (A4500s). Pass the node, or srun lands on the login node:

```bash
# human first: open ~/.ssh/cm-ilab (docs/DEPLOY.md#rutgers-ilab); check it answers in < 5 s
ssh -o ControlPath=~/.ssh/cm-ilab -o BatchMode=yes ap2772@ilab.cs.rutgers.edu hostname
ILAB_GRES=gpu:a100:1 ILAB_TIME=04:00:00 ILAB_NODE=rlab2 ILAB_WAIT=400 scripts/gpu_up.sh ilab
# fallback when the A100s are busy: ILAB_GRES=gpu:4500_ada:1 ILAB_NODE=ilab3 (or rlab1)
scripts/gpu_up.sh probe          # RTT p50 5.9 ms / p95 6.9 ms / max 20 ms, 500/500, from campus
export HEARING_REMOTE_URL=ws://127.0.0.1:18765/ws/session
.venv/bin/python -m hearing serve --hq-backend auto    # or remote; console "Runs on"
```

- Job name `hearing-hq`, one job id per bring-up (`432518` today; `squeue -u $USER -n hearing-hq`);
  the server prints `NVIDIA A100-PCIE-40GB` in `~/hearing-remote/serve.log`. A rerun of `gpu_up.sh ilab`
  scancels the previous `hearing-hq` job itself.
- **Is the GPU on current code? (T-110)** `curl -s localhost:18765/health` shows `git_rev`: the commit
  `gpu_up.sh` synced (it writes `~/hearing-remote/REVISION`, `-dirty` if the laptop had uncommitted
  changes; `gpu_up: synced git rev ...` in its output). `unknown` = a server started before T-110,
  i.e. older code and an older `config/tiers-remote.yaml` (on Oct 7 that one played enrolled at
  1 s / 0.5 s = 1.7 s instead of 0.95 s). **After every pull, restart the remote without losing the
  job:** `ILAB_REUSE=<job id> REMOTE_PORT=<free port, e.g. 18766> ILAB_NODE=rlab2 ILAB_GRES=gpu:a100:1 scripts/gpu_up.sh ilab`
  starts the new server as an extra step of the running job (`srun --jobid --overlap`, log
  `serve-<port>.log`) and re-points the laptop's tunnel at it; the old server idles until the job ends.
  Use a new port each time.
- The remote venv (`~/hearing-remote/.venv`, torch 2.8.0+cu128) does not follow `pyproject.toml`: the
  first start today died with `No module named 'qrcode'` (a core dep added on main). Fix:
  `ssh ... 'cd hearing-remote && .venv/bin/pip install "qrcode>=7.4" "cryptography>=42"'`, rerun.
- Replay through it (`hq_backend=remote`, `grid-two-face.mp4`, suppression 100): the AV tier
  (`focus_hq`, 2 s / 1 s) plays from t≈7 s, remote RTT p50 6.5 ms, playout 225 ms on the fast tiers and
  3.2 s on AV. Enrolled did not get ready within the 12 s clip (BENCHMARKS.md, T-075).
- Same-gender pairs on the A100 (offline `eval/association_eval.py --candidate av_mossformer2`, 20 scene
  runs per pair): guard + AV tier is GO, M/M 0.970 acc / 0.47 wrong-target s per 10 s, F/F 0.965 / 0.31;
  the AV tier leaks the other talker while the target is silent (docs/decisions/association.md).

### Focus HQ on the iLab A100 — what the judge gains (T-108)
`bash scripts/demo.sh --judge` probes the tunnel and prints `JUDGE · backend iLab A100 (RTT p95 7 ms) · expected delay ~0.95 s ...` when `/health` is ok and RTT p95 < 60 ms (before T-108 the probe crashed and the judge always said Mac ladder). Measured on `demo-mf-easy`, 30 s, target locked, three runs ([BENCHMARKS.md → Focus HQ through the iLab A100](BENCHMARKS.md#focus-hq-through-the-ilab-a100-t-108)):
- **Delay 950 ms instead of 1500 ms**: the A100 runs the enrolled tier on 0.5 s windows every 0.25 s, plus the 200 ms playout buffer. The judge hears the target about half a second sooner.
- **The delay the judge experiences (T-086): Mac local 1.5 s, iLab A100 0.95 s.** Since T-086 the A100 ladder (`config/tiers-remote.yaml`) runs enrolled on 0.5 s windows every 0.125 s (+10.68 dB SI-SDRi on 16 clips, the lowest-delay setting that keeps +10 dB): 0.825 s on paper, **0.95 s measured** in all three live replays because the stage plays one hop late under the live load, and **+5.7 to +6.9 dB** on `demo-mf-easy` (old 0.5 s / 0.25 s: +4.1 to +6.3 dB at 0.95-1.08 s). Tunnel RTT p50 6 ms / p95 19 ms, inside the 200 ms buffer. **Restart the GPU server once (`scripts/gpu_up.sh ilab`) after pulling T-086**: the laptop now forwards Balanced as is, and a server still on the old `tiers-remote.yaml` would run enrolled at 1 s / 0.5 s (1.7 s). Numbers: [BENCHMARKS.md → GPU demo preset](BENCHMARKS.md#gpu-demo-preset-on-the-ilab-a100-t-086-2026-10-06).
- **Judge card numbers on current code (T-110, 3 locked 60 s runs on `demo-mf-easy`, git `682f5d3`):** delay **0.95 s** measured in every run (`latency_ms.total` 973 ms; the A100 enrolled hop widens 0.125 → 0.25 s live because the job's 2 CPUs bound the remote session, so 0.83 s is the paper number), other talker **15.5-18.1 dB** down once Enrolled plays (Mac 18.8 dB), SI-SDRi **+4.4 to +9.4 dB** in the enrolled phase (+2.6 to +5.8 dB over the whole minute), **0.00-0.06 s** true dropouts, Isolated at **13.7-16.0 s**, no tier switch after Enrolled. The first T-086 GPU run said 1.7 s and -10 dB because the remote server was still T-075's: check `git_rev` in `/health` (see "Is the GPU on current code?" above). Numbers: [BENCHMARKS.md → Focus HQ through the iLab A100 on current code](BENCHMARKS.md#focus-hq-through-the-ilab-a100-on-current-code-t-110-2026-10-07).
- **Judge card numbers after T-113 (3 locked 60 s A100 runs on `demo-mf-easy`, remote ladder promotes at most every 3 s instead of 10 s):** Isolated at **7-9 s** (was 13.7-16 s), other talker **10.7-15.0 dB** down over the whole minute (was 5.3-9.5), SI-SDRi **+5.1 to +5.9 dB**, DNSMOS OVRL **2.50-2.66** (2.6-2.8 once Enrolled plays; the seconds before it play the room), **0.00-0.06 s** true dropouts, delay **0.95 s**. DNSMOS stays under 2.9 because the GPU box cannot also run DeepFilterNet live (it pushed Enrolled into overrun and the 3.2 s AV tier in T-113's runs). **Restart the GPU server after pulling T-113** (`scripts/gpu_up.sh ilab`; `git_rev` in `/health`) so it reads the new `config/tiers-remote.yaml`. Numbers: [BENCHMARKS.md → A100 Focus HQ quality](BENCHMARKS.md#a100-focus-hq-quality-where-the-remote-path-loses-vs-the-mac-t-113-2026-10-07).
- **Isolated sooner, cleaner**: remote Enrolled plays 4.4-6.6 s into the clip (Mac: 5-7 s), SI-SDRi +3.4 to +6.6 dB (Mac +1.0 / -0.5 dB in the same session), other talker 8-32 dB down.
- **Same-gender pairs**: the remote ladder also has the AV tier (AV-MossFormer2 on lips, CUDA only), which the Mac cannot run; when the same-gender guard skips Enrolled, the GPU plays AV (3.2 s behind) and the Mac plays the gate. This path was not re-measured here (no same-gender run in T-108).
- Keep the SLURM job alive for the slot (`scripts/gpu_up.sh status`); if it expires, `ILAB_GRES=gpu:a100:1 ILAB_TIME=04:00:00 scripts/gpu_up.sh ilab`, then `bash scripts/demo.sh --judge` again.

### Focus HQ on the laptop (M4 Pro, no GPU) — what to expect (T-072)
Measured with `scripts/replay.py eval/fixtures/demo/grid-two-face.mp4 --mode focus_hq --suppression 100` against `hearing serve` (session taps under `HEARING_SESSION_LOG`, read with `scripts/session_report.py`; numbers in [BENCHMARKS.md → T-072](BENCHMARKS.md#t-072-focus-hq-live-replay-on-the-m4-pro-2026-10-06)).
- **Ladder on the Mac: enrolled → gate → audio-only.** The AV tier (AV-MossFormer2, 2 s window / 1 s hop) is built on CUDA hosts only (`devices: [cuda]` in `config/tiers.yaml`); on MPS it overran its hop and starved vision. It still runs on the GPU box through `hq_backend=remote`. Telemetry lists it as `unavailable: needs cuda, this host is mps`.
- **Until the voice is enrolled: audio-only, then gate once a face is selected (1.6-4.8 s in), both `latency_ms.algorithmic` 13 ms, `total` ≈ 36-40 ms.** The 3000 ms seen on the first live run came from a `hearing serve` started before T-062 merged (its telemetry had no per-tier `delay_ms`): **run `bash scripts/demo.sh --restart` after every pull.**
- **Once the voice is enrolled: enrolled tier, 1500 ms** (balanced preset; 750 ms on Speed). The pill reads `Acquiring voice · enrolling 1.1/2.0 s` meanwhile, then `Isolated`.
- **Time to enrolled depends on clean single-talker speech.** The demo clip has two GRID talkers with 0.58 overlap, so the enroller (target `p_speaking` ≥ 0.7, every other face < 0.7) collects only ~1 s per 12 s pass: Isolated came at **26.8 s** with `--loop`. In the room: have the target speak alone for 2-3 s first; a clip-only demo must use `loop=true` (or the button) and wait ~30 s.
- **`target.locked` stays false unless the user taps a face** (it is the explicit pin). The automatic visual lock is the acquisition state: an off-centre face that is the only speaking candidate is selected after the dwell (`Acquiring voice` at 0.7-3.6 s in every replay). Face the target to make it the most central face; tap it to pin it.
- **Vision**: 25-26 fps, 0.7-4 % frames dropped while the gate plays (was 15-16 fps, 58 % dropped with the AV tier running). While the enrolled tier separates (~240-330 ms of every 0.5 s on MPS) the ASD skips frames it cannot get the GPU for within 20 ms (`vision.asd_skipped`) instead of stalling the camera. `hearing serve` preloads the models at start (`--no-warm` to skip); wait for `hearing: models warmed` in the log before the first session.

### Focus HQ on the Mac: no more flapping (T-095)
Numbers: [BENCHMARKS.md → Focus HQ over 60 s on the Mac](BENCHMARKS.md#focus-hq-over-60-s-on-the-mac-ladder-stability-t-095-2026-10-06).
- **What the judge used to hear:** under load the enrolled tier overran, the ladder dropped to the gate at the first overrun and came back a second later: 7-8 tier switches per minute, each a 1.5 s jump (replay or skip), and the gate at full suppression cut the target's words whenever ASD missed them (true dropouts 9-12 s per minute, SI-SDRi -3.0 dB).
- **Now (`config/tiers.yaml`):** a better tier is taken only after 3 s of being ready, at most one move up per 10 s, and an enrolled tier that overruns **keeps playing** for up to 10 s (its skipped windows fall back to the room for that half second). It still leaves at once if it really breaks (crash, model or enrollment lost, same-gender pair). Expect exactly **2 switches**: audio-only → gate as the face is picked, gate → enrolled when the voice is enrolled.
- **No holes in the target's voice:** both the gate and the enrolled tier duck the rest of the room by at most 20 dB at slider 100 (`gate(hold_ms=1000,max_duck_db=20)`, `mixer(max_duck_db=20)`), so when ASD or the TSE misses a word the judge hears it quieter, not cut. Latest 60 s replays of the demo clip (`demo-mf-easy`, target locked, with T-089's threaded enrolled tier) at load 4-21: SI-SDRi **+4.7 / +6.4 / +3.0 / +5.4 dB** over the minute, **+9.8 / +11.3 / +4.7 / +11.1 dB** once enrolled plays, true dropouts **0.00-0.22 s**, **2** switches each. Re-run after merging T-102/T-107 at load 4.5-7.7: **+5.1 / +5.4 / +4.6 dB**, **+9.6 / +9.6 / +9.7 dB** once enrolled plays, **0.00 / 0.00 / 0.20 s** dropouts, 2 switches, Isolated at 14.4-15.5 s. After merging T-103/T-108 (load 5.3): **+6.1 dB**, **+11.6 dB** once enrolled plays, 0.00 s dropouts, 2 switches, Isolated at 13.9 s. At load ~20 the enrolled hop widened to 1 s (2 s behind live) and it still held without a switch. Before T-089 at load 4.6-8 it was +4.7 / +4.5 / +5.0 dB; on the legacy `grid-two-face` loop at load 8-11, +2.9 / +2.8 / +2.1 dB with 0.08 s of dropouts.
- **Before the voice is enrolled** the gate holds open 1 s after the target's last speaking frame (no DFN in this tier). It sounds like the room turned down when the target pauses, not like isolation.
- **Time to isolation:** the pill reads `Isolated` **13.4-14.4 s** into the demo clip with T-092's enrollment merged (target ≤ 15 s; it was 18.3 s before that, 31-36 s on the legacy loop). In the room, have the target talk alone for 2-3 s first, and start the demo before the judge listens.
- **The one way it still goes wrong is the target, not the ladder.** In one replay at load 20, vision delivered no frames for ~0.4 s, the target lock dropped and the other face was picked (-1.4 dB, 5 switches: each one was a hard demotion that followed the target change). If the face boxes vanish, re-tap the target. Keep the laptop free of other heavy work.
- **Enrolled stays the Mac default.** Step (5) of T-095 (gate-only Focus HQ, Enrolled opt-in) did not apply: enrolled was the best tier in every measured run, and the gate was the one making Focus HQ worse than the room.
- **No "never worse than the gate" guard.** It was built, but its proxy (output energy while ASD says the target speaks vs while it is silent) favours the gate by construction because the gate ducks on the same ASD, so it was removed rather than shipped off. The bounded ducking above is what keeps Focus HQ from being worse than the room.
- **Quit the other heavy processes on the demo laptop** (other agents, `swarm run`, browsers with video): an earlier run at load 21 had 3.4 s of enrolled overruns. No run with load < 4 was measured (closest: 4.1-5.7, +4.7 dB).

### Same-gender pairs (T-070)
Enrolled voice matching swaps two talkers of the same apparent gender: it plays the other person for 1.2-1.8 s of every 10 s, against 0.2 s for a man and a woman (T-069, `docs/decisions/association.md`). So the Focus HQ ladder skips Enrolled by itself while the target and another face seen in the last 3 s get the same voice group (male/female) from the face estimator. It plays AV Focus HQ when the target's lips are in view and the AV tier is built (CUDA only since T-072, so the GPU box through `hq_backend=remote`), else the gate: **on the Mac alone a same-gender pair plays the gate**. **What the judge sees**: in the console's tier list the Enrolled row reads `same-gender pair: enrolled skipped` and AV (GPU) or the gate (Mac) is `playing`. On the GPU the output is about 1.5 s further behind (AV plays 3 s behind live, Enrolled 1.5 s); on the Mac the gate plays live (≈ 40 ms) but only ducks the room while the target is silent. The switch back to Enrolled waits 1 s after the second face stops matching, and every switch is crossfaded.
- **Best demo: a man and a woman.** Enrolled plays, 1.5 s behind, and the guard stays off.
- **Same-gender pair**: on the GPU backend keep the target's mouth in clear view of the camera so AV can play. Without lips the ladder lands on the gate, which ducks the room only while the target is silent, so both voices pass while both talk.
- Before the slot, check the `voice` chip on both faces. On the GRID test faces the estimator called a female pair `neutral` or `male` in some frames, and an unknown bucket never triggers the guard. A voice override changes only the TTS voice, not the ladder.
- "Strict" pause preset (ducks the Enrolled output while the target's ASD says silent: −50 % wrong-person audio on M/F at the cost of short dropouts): the commented pipeline line in `config/tiers.yaml`. It is off by default. A/B it at the rehearsal only if the Enrolled tier leaks the other voice during the target's pauses.

### ElevenLabs drop-in checklist (T-094)
Without a key Synthetic speaks with Kokoro (local fp32, 0.2-0.25 s to the first word on the Mac for a 1-3 word head: a fixed
~0.55 s per call plus ~0.5 s per second of audio, measured loaded). With a key it uses Flash v2.5
(~100-250 ms to the first audio). Two minutes:
1. Put the key in `.env` in the checkout (`ELEVENLABS_API_KEY=sk_...`, gitignored; `python -m hearing` loads it at start, `hearing.cli.load_dotenv`, and a variable already set in the shell wins). Never commit it.
2. Voices need no setup: `config/voices.yaml` already maps the 7 buckets to premade voices (ids in `docs/DESIGN.md#synthetic-voices`). If one id 404s, re-map it in that file by name.
3. Verify: `.venv/bin/python -m hearing synth-test --tts elevenlabs --out /tmp/synth-check` renders one sentence per bucket, prints first-audio ms, writes wavs, and exits 1 with the reason on a bad key or no network. Expect `first` under ~400 ms per bucket (the first call includes connect).
4. Start the server as usual; `telemetry.latency_ms.synth_phrase_p50` is ASR phrase end to first sample (phrase end includes the 0.4 s end-of-phrase pause; measured with Kokoro fp32 + local Whisper on the demo clip: ~1.4 s, mock-ASR bench 0.8 s, old int8 1.7 s; ElevenLabs is an estimate, ~0.6-0.8 s, until run with a key).
5. Rehearse without a key: `HEARING_TTS=elevenlabs-mock` (or `synth-test --tts elevenlabs-mock`) runs the real adapter against a fake socket with Flash-like timing (~110 ms).
A failing key falls back to Kokoro per phrase; nothing else to do on stage.

## 5. Reset between runs

| What | How |
|---|---|
| Session state (target lock, synth queue, captions) | reload the console page (a new WebSocket = a fresh `Session`) |
| The whole engine | `bash scripts/demo.sh --restart` (§4); use `bash scripts/demo.sh --stop` to shut it down |
| Preference model (so the "you usually pick Synthetic here" recommendation is reproducible) | the seed model is built in (`hearing.context.prefs.prior_model()`, a simulated user); your choices are layered on top from `HEARING_PREFS_LOG`. At the rehearsal, make the choices for the demo scenes once with `HEARING_PREFS_LOG=~/demo-backup/prefs_rehearsed.jsonl`; before each run `bash scripts/demo.sh --stop`, `cp ~/demo-backup/prefs_rehearsed.jsonl /tmp/prefs.jsonl`, restart with `HEARING_PREFS_LOG=/tmp/prefs.jsonl bash scripts/demo.sh --restart`. Without the env var the model learns in memory and forgets at restart |
| Voice override | it is per session: reload the page |
| GPU box | leave it up during the event; `scripts/gpu_up.sh down` at the end (**RunPod bills until `down`**) |
| Audio devices | System Settings → Sound: output = AirPods Pro 3, input = MacBook Pro Microphone (never the AirPods mic); re-check after any Bluetooth device connects |
| Room speaker | stop the backup clip, reset to 0:00 |

## 6. Fallbacks per failure

Every row keeps the demo going; the tier indicator in the console says which tier is live, so say it out loud.

| Failure | Symptom | Do this (in order) | Say |
|---|---|---|---|
| Server will not start | traceback at launch | `bash scripts/verify_fast.sh`; then `bash scripts/demo.sh --restart -- --pipelines "natural=passthrough"` to get the console up; last resort the backup video | |
| `dfn_ll` / `gtcrn` unavailable | `/health` lists it `unavailable` | check `HEARING_MODELS_DIR` is exported and `ls $HEARING_MODELS_DIR/dfn/dfn_ll/ $HEARING_MODELS_DIR/gtcrn/`; re-run the provision lines in §2; DFN3 broken → `--pipelines "natural=gtcrn>gate>mixer"` | |
| Browser path too slow for Natural | console number > 300 ms on the AirPods | run Natural through `hearing live` on the laptop (no browser, no WebSocket) and keep the console for the visuals | "this is the on-device path" |
| Natural sounds laggy / echoey | latency > 300 ms, voice doubles | AirPods: Noise Cancellation on, Transparency/Adaptive off; mic = MacBook Pro Microphone (a red HFP banner means the AirPods mic is the input); close other audio apps; reload the page | "Bluetooth adds about 200 ms" |
| Mic dead in the browser | no input level | Chrome site settings → allow mic; use `127.0.0.1` not the LAN IP; restart Chrome | |
| Camera / face tracking fails | no face boxes, TARGET LOCKED never appears | after 1 s without frames the session is audio-only (tier 4), which still suppresses noise; light the face, face the camera square-on, lock by clicking the box if boxes show; `HEARING_ASD=lipmotion` if Light-ASD is slow (> 40 ms/frame in telemetry) | "without the camera it degrades to noise suppression" |
| Server dies when the camera starts | process exits with `F0000 gpu_buffer_storage_cv_pixel_buffer.cc ... Check failed` or `Error creating pixel buffer: -6662` | MediaPipe ran on Metal: `unset HEARING_FACES_DELEGATE` (the default is the CPU delegate, ≈ 9 ms/frame) and run `bash scripts/demo.sh --restart`; the log line `faces: mediapipe on the cpu delegate` confirms it; still dying → `HEARING_FACES=none` (audio-only, tier 4) | "the camera is off; noise suppression still runs" |
| Focus HQ stuck below tier 1 | tier shows `gate` / `audio-only` | needs a locked target with lips in view for 1 s; check `tse` is `ready` in `/health` (not `mock`); on the remote variant, `remote.connected=false` means the GPU box or Wi-Fi is gone: the ladder falls back by itself, join the phone hotspot, `scripts/gpu_up.sh probe` | "Focus HQ needs the lips and a GPU; the fast path is always there" |
| ElevenLabs fails (no key, credits, network) | Synthetic silent or local voice | the synth chain falls back to Kokoro (local) automatically; `HEARING_TTS=local` to force; Text mode (4) still shows captions | "this is the local voice" |
| Whisper slow / ASR silent | captions lag > 2 s | `HEARING_ASR=elevenlabs` if online; else Text mode with shorter phrases; last resort the Synthetic backup clip | |
| Overlapping speakers garble Natural | B bleeds through | expected on the fast path: switch to Focus HQ (2), else say it and play the Focus HQ backup clip | "the fast path can't separate overlapping voices; Focus HQ does, 1.5 s behind" |
| Recommendation does not fire | no toast | reset the prefs log (§5); trigger the second scene again; skip to Auto (6) manually | |
| Laptop audio device grabbed / CoreAudio error | no sound at all | reconnect the AirPods (Bluetooth menu), re-pick them under **D** > Output, reload; `bash scripts/demo.sh --restart` | |
| Everything | | play the backup video of every mode (§8) | |

### If the room fails

Before going offline, prepare the clip once:

```bash
.venv/bin/python -m eval.make_fixtures --av --full  # only if the GRID bank is absent; needs network
.venv/bin/python -m eval.make_fixtures --demo       # offline: cached two-face GRID -> 12 s MP4
```

The output is `eval/fixtures/demo/grid-two-face.mp4`; copy it with the local model
weights to the demo laptop. Media is ignored by git, so a fresh checkout needs
this preparation. In a worker worktree with the bank only on main, use
`--demo --out /absolute/path/to/main/eval/fixtures/generated`, then copy the
resulting `main/eval/fixtures/demo/` to the worktree's `eval/fixtures/demo/`.
This is a repeated GRID fixture for demo insurance; replace that filename with a
consented 10–15 s two-person recording before demo day.

Run the session server from §4 with cached models, **Local** HQ backend, local
ASR/TTS and no ElevenLabs key when fully offline. On `/simple` or `/`, press
**Replay clip**. It stops live capture, creates playback without mic/camera
permissions, loops the prerecorded input through the page's existing engine
session, displays the decoded frames with face boxes and a **REPLAY** badge,
and plays processed audio. Keep using the same modes, slider, face lock, captions
and Raw/Selective A/B. **Stop replay** cancels the feeder; press Start to resume
live capture. Missing files show a preparation error in the existing banner.
No clip or model download happens on button press. Decoding needs a local
`ffmpeg` on PATH or the bundled binary from the `eval` extra (`imageio-ffmpeg`). Real faces and separation
still require the same installed backends/cached weights as live mode; missing
backends keep their usual reported fallback, never fabricated face telemetry.

For transport-only inspection (no microphone/speaker):

```bash
HEARING_SESSION_LOG=/tmp/replay-taps bash scripts/demo.sh --restart --no-open
.venv/bin/python scripts/replay.py eval/fixtures/demo/grid-two-face.mp4 --loop
```

The script opens its own ordinary WebSocket, sends hello/control then paced 20 ms
PCM and 25 fps JPEG, and prints server text events. The tap creates one session directory under `/tmp/replay-taps`, containing
`events.jsonl` (including faces, target and tier), `in.wav`, `out.wav`, and sampled
JPEGs. Inspect it with `.venv/bin/python scripts/session_report.py /tmp/replay-taps/<session-directory>`
during rehearsal with real cached backends. It does not play sound itself. For same-page viewing use
the Replay button. Browser checks are Python pytest tests marked `slow` (Python Playwright in `.venv`, like `tests/test_phone_check.py`); there is no node Playwright. Pattern: a local prepared clip and mock Text ASR test server, `getUserMedia` denied, then check both pages' preview, output, captions and stop.
The old `web/console/tests/*.mjs` scripts need node modules that are not installed; do not rely on them. The device-free DOM check against the same test server checks replay on both pages, then starts and stops
live capture with stubbed media devices to verify camera initialization and
track cleanup. It does not verify browser rendering or audible playback.

## 7. Measured numbers to quote

Section D is the judge sheet. For the AirPods demo path use [BENCHMARKS.md → Slide-safe claims for the AirPods demo path](BENCHMARKS.md#slide-safe-claims-for-the-airpods-demo-path-t-148): engine-side lines now, mouth-to-ear lines once the rehearsal sheet (§B) has the numbers. For the rest use only the lines under [BENCHMARKS.md → Slide-safe one-line claims](BENCHMARKS.md#slide-safe-one-line-claims); each one carries its condition (oracle speaking detector, clean enrollment, which laptop). Re-run `python -m eval.bench --full` on the demo laptop the day before so the claims come from that machine. Quote the AirPods mouth-to-ear number you measured in §3 for Natural, not the engine-only number.

## 8. Backup clips

Record or render these the day before and keep them in `~/demo-backup/` (never in git: `*.wav`/`*.mp4` are ignored).

| Clip | How | Used when |
|---|---|---|
| `room_raw.wav`: two teammates talking + music, 20 s, phone mic | record it at the venue (§9) | step 1 of the script, and the input for the renders below |
| `room_natural_s1.wav` | `.venv/bin/python -m eval.bench --render ~/demo-backup/room_raw.wav --stages dfn_ll --suppression 1 --render-out ~/demo-backup/room_natural_s1.wav` | Natural fails live |
| `room_natural_s05.wav` | same with `--suppression 0.5` | to play the slider half-way |
| `room_focus_hq.wav` | Focus HQ needs the lip video, so `--render` (audio only) cannot make it. Record it live: Focus HQ (2) in the console while a teammate faces the camera, capture the output with QuickTime + Loopback/BlackHole. Offline alternative: an AV-set example, `eval/fixtures/generated/av_full/<id>_two_face_*/mix.wav` vs the `av_mossformer2` output (`eval.bench --full` scores it; lips come from `video.mp4` at the manifest's `faces[role=target]` box) | Focus HQ fails or no GPU |
| `room_text.json` | `--render ... --stages "dfn_ll>synth(asr='local',speak=false)"` prints the captions in its JSON line | Text/Synthetic fail |
| Full-set examples | `eval/fixtures/generated/full/<id>/{mix,target}.wav` plus a render of `mix.wav` through `dfn_ll`: before/after with a clean reference | the numbers slide ("hear the difference") |
| Screen recording of every mode | QuickTime screen + audio (Loopback/BlackHole) of a full rehearsal run | the live demo dies |

## 9. Real phone clips for DNSMOS

DNSMOS scores real recordings without a clean reference, which is the only honest quality number for the venue room.

1. Record 5-10 clips of 10-30 s each on the phone at the venue (or any noisy room): one person facing the phone at ~1 m, with background chatter/music; vary the noise. Get a verbal OK from everyone audible.
2. Copy them to `eval/fixtures/real/` (gitignored; any rate, mono or stereo WAV/FLAC/OGG; convert m4a with `ffmpeg -i in.m4a -ac 1 -ar 16000 out.wav`).
3. `.venv/bin/python -m eval.dnsmos --stages dfn_ll --stages gtcrn` (quick table: SIG/BAK/OVRL/P808, raw vs processed), or `python -m eval.bench --full`, which adds a real-clip table and a claim to BENCHMARKS.md.

## 9b. Real two-person clips for REPLAY and scoring (T-090)

The judge demo is real voices in a real room, so record the replay clip the same way. **Recipe** (15-30 s, one take per pair):

1. Phone in **landscape**, on a stand or held still at chest height, **~1 m** from both people; both faces fully visible, lit from the front (no window behind them), sitting side by side ~0.5 m apart.
2. Decide who is the target (the one the wearer faces) and note their side as seen **in the video** (left or right of the frame).
3. Each person first **says their name** (consent on tape: "I'm Joseph and it's fine to use this clip").
4. Then **10 s target alone** (the other stays silent and still, mouth closed), **10 s other alone**, then **10 s both talking at once** (normal conversation, no shouting). Add room noise (music, chatter) if you want a noisy version; record a quiet one too.
5. AirDrop the `.mov`/`.mp4` to the laptop (any resolution, rate or rotation; ingest normalises it).

Ingest and replay:

```
.venv/bin/python scripts/ingest_clip.py ~/Downloads/IMG_1234.MOV --target left --name joseph-aryan
#   -> eval/fixtures/demo/joseph-aryan/{clip.mp4 (gitignored), manifest.json}
#      640x360 @ 25 fps H.264, mono 16 kHz AAC, speech level -18.5 dBFS (the GRID clip's)
#      prints each face's side and ASD speaking time; check the target is the right person
```

The console's clip picker (next to **Replay clip**) lists every clip in `eval/fixtures/demo/` (`<name>.mp4` and `<name>/clip.mp4`); pick one and press Replay. From a terminal, `scripts/replay.py eval/fixtures/demo/joseph-aryan --lock-target` locks the manifest's target face (oracle) instead of letting the target selector choose.

Score a recorded session (no clean stems needed; the ASD speaking timeline in the manifest stands in for them):

```
HEARING_SESSION_LOG=results/sessions .venv/bin/python -m hearing serve --port 8000   # records in.wav/out.wav/events
.venv/bin/python scripts/replay.py eval/fixtures/demo/joseph-aryan --lock-target --quiet --suppression 100
.venv/bin/python scripts/session_score_nr.py --clip eval/fixtures/demo/joseph-aryan   # newest session
```

It prints DNSMOS SIG/BAK/OVRL in -> out, **other-talker leak** (output/input energy where only the other face speaks, dB, lower is better), **target preserved** (same where only the target speaks, 0 dB = untouched), **dropouts** inside target-only speech, **time to Isolated** and the tier timeline. If the mic was on before Replay was pressed, pass `--offset-s <seconds of audio before the clip>`.

Measured on laptop-a with a synthesised stand-in (two av_full GRID talkers side by side, left alone / right alone / both, 12 s, `python -m eval.real_clip --synth`), one pass, s = 100, `--lock-target`, server cold: Natural DNSMOS OVRL 3.00 -> 3.03, leak -0.1 dB, target preserved -0.5 dB, 0 dropouts, locked-visual 0.2 s, never Isolated (tiers audio-only -> av); Focus HQ OVRL 3.00 -> 2.96, leak -5.0 dB, preserved -2.0 dB, 2 dropout frames (longest 20 ms), Focus HQ did not start playing within the 12 s (tiers audio-only -> gate). Re-run on the real clip with a warm server and `--loop` before quoting anything.

## 10. After the demo

`scripts/gpu_up.sh down` (stops billing), stop `caffeinate`, delete `eval/fixtures/real/` recordings of other people if they did not agree to keep them.
