# Design (debug console only)

The product UI (phone app, website) is built at the event from `docs/HACKATHON_PLAN.md`. This repo has one UI: the debug console in `web/console/`, a test bench for the voice engine.

## Principles
- Show the truth: live latency per stage, active tier, target confidence, errors with the tier we degraded to.
- One screen, no navigation. Large controls usable during a live demo.
- Never show apparent gender/age as a fact: show "voice: adult male (from face + voice) · change" with an override menu including "neutral".

## Layout
1. Camera view with face boxes; the target box highlighted with its p(speaking).
2. Mode buttons: Natural · Focus HQ · Synthetic · Text · Quiet · Auto.
3. Suppression slider 0-100 (Natural / Focus HQ).
4. Numbers strip: total latency, per-stage latency, tier, device.
5. Captions panel (Synthetic / Text) and the current voice chip.
6. Delay preset (Speed · Balanced · Accuracy) next to the Focus HQ toggle; also on `/simple`. Independent of mode and Local/GPU selection; telemetry `latency_preset` is echoed back. Enrolled window/hop: 0.5/0.25, 1.0/0.5, 1.5/0.75 s. These imply 0.75/1.5/2.25 s of algorithmic delay when compute keeps up, before device/transport buffers. The cold-start AV tier keeps its 2/1 s window/hop. Changing preset builds a fresh pipeline, so enrollment may need to refill. The GPU session receives the same choice, but its client's fixed playout buffer can limit the audible delay reduction.
7. Recommendation toast when the preference model suggests a mode. In Auto the toast is
   informational ("OK", no Accept): it appears when Auto enters and on every Auto switch, with
   the reason (`rule: 2 talkers`, or the model's reason). Auto goes to Focus HQ as soon as two
   people talk and back to Natural 5 s after; the learned model overrides that only when it is
   confident (T-076).

## Tokens
- font.ui: system-ui; font.mono: ui-monospace
- color.bg: #0f1216; color.panel: #171b21; color.text: #e8ecf1; color.accent: #4f8cff; color.target: #2ecc71; color.warn: #f5a524; color.error: #ef4444
- radius.card: 10px

## Synthetic / Text feed (T-074)
Synthetic and Text read the person you chose, not the room. The ASR is fed by `synth_feed`
(`docs/CONTRACTS.md`): with one talker it hears the raw room (`mixture`: fastest, and
DeepFilterNet in front of the ASR measured no better), and once two or more people have talked
for 2 s the Focus HQ ladder's output (whichever tier plays). Switching feed never drops a
caption or a queued phrase. Explicit `synthetic=`/`text=` pipelines on the serve line turn the
feed selection off. The captions panel shows the same words that are spoken. The UI needs no
control for it (default `auto`); telemetry carries `synth_feed` for the debug console. Because
overlapping speech has no pauses, the phrases are cut every 3 s at most, so the voice follows
the speaker about 1.5-2.5 s behind and in short pieces instead of one long delayed sentence.

## Synthetic voices (T-094)
The voice chip shows the bucket, never a fact about the person; the override menu lists every row
below plus "neutral". `config/voices.yaml` is the source; the bucket changes on the **next phrase**
(the phrase being spoken finishes in its old voice, no cut). Kokoro has no child or senior voices:
"young" and "senior" are a different timbre plus a speed multiplier. DNSMOS is from
`eval.dnsmos` on "Meet me by the fountain at noon, and bring your umbrella." (16 kHz).

| Bucket | Kokoro voice @ speed (DNSMOS OVRL) | ElevenLabs premade (Flash v2.5) |
|---|---|---|
| neutral | `af_river` @ 1.0 (3.44) | River `SAz9YHcvj6GT2YYXdXww` |
| female_young | `af_nova` @ 1.05 (3.37) | Jessica `cgSgspJ2msm6clMCkdW9` |
| female_adult | `af_heart` @ 1.0 (3.32) | Sarah `EXAVITQu4vr4xnSDxMaL` |
| female_senior | `bf_emma` @ 0.9 (3.22) | Matilda `XrExE9yKIg1WjnnlVkGX` |
| male_young | `am_puck` @ 1.05 (3.43) | Liam `TX3LPaxmHKxFdv7VOQHJ` |
| male_adult | `am_michael` @ 1.0 (3.41) | Brian `nPczCjzI2devNBz1zQrb` |
| male_senior | `bm_fable` @ 0.9 (3.31) | Bill `pqHfZKP75CvOlQylNhV4` |

Written judgment (no one listened live; DNSMOS and distinct accent/pitch only): all seven score
OVRL 3.2-3.4, within noise of each other. Replaced `af_bella` (3.27), `bf_alice` (3.22 with the
lowest background score), `bm_george` (3.03, worst of 21 tried) and added speed for young/senior.
`af_nicole` (1.74) and `am_eric` (3.11) were rejected. Samples: `synth-test --out results/T-094-voices`
(one wav per bucket, named `<bucket>.wav`; the 21 candidates were rendered as `<voice>.wav` in the same gitignored folder, which is not in this worktree, so regenerate with that command). **A human listen test is still pending**; DNSMOS (3.2-3.4 for all seven) does not separate the voices beyond accent and pitch.
Captions and spoken text pass through `clean_caption` (Whisper `". . . ."` phrases are dropped).

## Suppression slider (T-093)
The slider is a dimmer for the room, not a gate: equal slider steps are equal loudness steps.
`hearing.fusion.mixer.room_atten_db(s)` is the one curve; DFN's `atten_lim_db`, the mixer's
residual gain and the gate's duck depth all follow it (`room_gain = 10^(-dB/20)`).
- 0 → 90: `30 · s` dB, i.e. 3 dB per 10 points (0 is an exact bypass; no jump at 1 %).
- 90 → 100: soft knee, `-20·log10(0.5·(1 + cos(π·(s-0.9)/0.1)))` dB more, fading the last
  -27 dB of room to silence at 100 ("only the target"; DFN unlimited, residual 0).
- Every change ramps over 30 ms (`SLIDER_RAMP_MS`): the mixer per sample, DFN's mix-back per
  10 ms hop (then crossfaded by its synthesis window), the gate with its 20/250 ms envelope.
  Tests: `tests/test_mixer.py::test_mixer_slider_jump_is_click_free`,
  `tests/test_dfn.py::test_slider_jump_during_tone_is_click_free` (0 → 100 mid-tone, no
  sample step above -40 dBFS).
- The old mapping was `6 + 44·s` dB for DFN (a 6 dB jump at 1 %) and `1 - s` for the mixer/gate
  (-6 dB at 50, -20 dB at 90: almost all the change in the top 20 %).

Room attenuation vs slider (dB, one `#` per dB, 40 = silence):
```
 slider | room dB | 
   0 |   -0.0 | 
  10 |   -3.0 | ###
  20 |   -6.0 | ######
  30 |   -9.0 | #########
  40 |  -12.0 | ############
  50 |  -15.0 | ###############
  60 |  -18.0 | ##################
  70 |  -21.0 | #####################
  80 |  -24.0 | ########################
  90 |  -27.0 | ###########################
 100 |   -inf | ########################################
```
Measured room level vs this curve, SI-SDRi and DNSMOS per slider position: `docs/BENCHMARKS.md#suppression-slider-sweep-t-093`.

## Calibrate voice + Strict (T-085)
Next to the acquisition pill on `/simple` and the console: a **Calibrate voice (3 s)** button
(key **C** on the console), a **Clear** button shown once calibrated, and a **Strict** checkbox
(`web/console/calibrate.js`, shared). The button sends `control.enroll {action:"start",seconds:3}`
and shows the server's countdown (`Listening… 1.8 s`, from `telemetry.enrollment`), then
`Calibrated ✓ (again)` in the target colour, or `Too quiet, try again` in amber when the window had
under 1 s of speech. Strict sends `control.isolation` and is remembered in `localStorage`
(`hearing.isolation`) and re-sent on reconnect. The judge script uses both (`docs/DEMO.md`).

## Attention: acquisition tiebreak (T-101)
`attention.tie_margin` (default 0.15 of frame width, 0-0.5; `0` turns it off) only matters while no target is held. When two or more visible faces are within `tie_margin` of the most central one's cost, the selector does not take the more central by a hair: it looks at each face's mean `p_speaking` over the last 1.5 s and picks the eligible face that leads the runner-up by at least 0.2 (speaking alone beats silent). With no lead (both talking, or both silent) centre-of-view decides, as before. A real tie waits for evidence: at least 0.5 s, and up to 1.5 s while someone has not clearly been talking (mean above 0.5); when every tied face is clearly talking it decides at once. Explicit locks, switching away from a held target, dwell, hold and hysteresis are unchanged. Measured on demo-mf-easy free selection: 2 of 3 runs chose the manifest's target (95 %, 96 %, 29 %); before 1 of 3.

## Audio devices (T-121)
`/judge` (D drawer) and `/simple` mount `web/console/devices.js` (styles `devices.css`): Microphone and Output pickers (`enumerateDevices`; the output through `AudioContext.setSinkId` where supported, else greyed out), remembered in `localStorage` (`hearing.mic`, `hearing.sink`, deviceIds; the mic is an `ideal` constraint so an unplugged one falls back to the default). The meter is the peak of the frames already sent to the engine, in dBFS with a 24 dB/s release: **too quiet** below -40, **good**, **clipping** at or above -2. A Bluetooth badge shows when the output label matches AirPods / Bluetooth / BT (150-250 ms extra). Another mic while running restarts capture.

## Judge: saved person, Release, Keys off (T-177)
Under the RAW | SELECTIVE HEARING switch on `/judge`: a **Choose saved person** heading over two big
buttons (the first two saved voice profiles from `telemetry.enrollment.profiles`, labelled `1 Aryan`,
`2 Joseph`; disabled as `Person 1/2` until a profile is listed) and a **Release** button. The buttons send
exactly what keys **1**/**2** send (`control.enroll {action:"profile", name}`); Release (and **Shift+0**, or 0
with the drawer open, guarded like C/R/N) sends `control.enroll {action:"clear"}`, the server turns
auto-calibration off until 1 or 2, and the pill reads **Searching**. A line under the pill always names the
target (`Target: Aryan`, `Target: none (Release) · press 1 or 2`). When the window loses focus or a text
field has it (where the page ignores keys), an amber banner **Keys off: click the page** stays at the top
until focus returns (window `blur`/`focus`, `focusin`/`focusout`). There is no Calibrate button on the
judge page; live calibration stays **Shift+C**, off the stage. Pure logic: `web/console/judge-targets.js`.

## Audio-only judge page (T-168)
The decided demo has no camera. Under the team preset (`config/demo-team.yaml` `audio_only: true`, shown in the drawer's TEAM PRESET line as "audio only (no camera)") or with `/judge?audio=1`, Start opens the microphone only: no `getUserMedia` video, `hello.video` is `false`, no video frame is sent, and the camera peek is hidden (`body.audio-only`). Start waits up to 1.5 s for the first telemetry so it knows the preset before opening anything; a camera opened before the preset was known is closed and the hello re-sent. "No camera" is not "Camera lost": the pill never shows the camera-lost fallback. The session also drops camera frames under the preset (a replay clip's too; `telemetry.preset.video_dropped` counts them), so neither the Mac nor the GPU runs vision or the AV `focus_hq` tier. The console page (`/`) keeps its camera path. Logic: `web/console/audio-only.js`.

## App (`/app/`) polish rules (T-356)
The rules the final pass applied to the M14 app (`web/app/`) from T-365's punch list (`docs/UX_REVIEW.md`). The app's full spec is `web/app/DESIGN_SPEC.md`; these are the rules a later change must keep.
- **Colour is a promise about the sound.** A pressed control is a neutral fill (16 % white, white outline). Amber appears only while the room plays live (Raw). Green appears only while the chosen person is isolated and Selective plays: on Listen's Selective and person segments and on People's card (`data-playing`). People says "Following now." only then; otherwise "Chosen. Plays when you start." or "Chosen. Finding them." Setup's level meter is white when good, never green. No glow `box-shadow`, radial halos, striped fills or washes on the shell or other screens' backgrounds; Listen's A/B wash (T-347) is the one wash, because it is the A/B state itself, and it goes neutral in Quiet (nothing plays): there the word, the hear line and the caption speaker are white too.
- **Nothing measured is shown before it is measured.** `store.truthStrip()` returns `delayMs`/`tier` as `null` until audio flows (`started` or a replay), so Listen prints "—" and Setup prints "The measured delay shows here once sound flows." `personState()` has no `current` until audio flows, so no person is pressed at Idle. The steady RTT moves in 10 ms steps, so 0 prints as "<10 ms".
- **Engine words never reach the stage.** The name slot (`.ls-who`) holds a person's display name or nothing. `plainSentence()` maps the pill: an overrun reads "Catching up: the Mac fell behind, so you hear the cleaned room for a moment.", loading reads "Loading Aryan's saved voice…", and Isolated has no sentence, because the word and the name already say it. Under Raw the sentence is "Space to hear Aryan alone." Quiet's pause line is "Captions wait until Aryan is isolated." A trailing Whisper `//` is stripped. After Stop, captions dim with "From the last session."; Release clears them.
- **The stage strip is Backend · Delay now · Round trip · Tier.** "Voice match" (`tse.confidence`, formerly "Confidence"), the session id and recording live in Diagnostics.
- **The state word fills its column**: `min(var(--text-display), 100cqi / (chars × 0.5))` at Archivo's display width (72 %), so "Idle" reaches about 200 px at 1440 and "Reconnecting" still fits 320 px. scene.js is the one field; the CSS halo hides under `html.has-scene`.
- **Keys appear once**, in a legend line under the rail on keyboard screens (`hover: hover`, fine pointer, wider than 760 px), not as a chip in every button. Touch screens show what the pressed mode does as a line under Mode, because a `title` tooltip never shows on touch.
- **Problems sit in flow, never over the state.** Listen moves the shared device banner (`[data-dev-banner]`, web/console/devices.js) into `.ls-top` above the word and restyles it in app.css: the raised surface, a 3 px red rule, the words in red, and 44 px app controls. /judge keeps its own red bar, untouched.
- **No dead ends.** The nav lists only screens the server has (`_files.json`). The Diagnostics button and **D** exist only when `screens/diag.js` does. The sheet's Close sits in a sticky flex header. There is no `backdrop-filter` on the sheet or the captions; captions are type on the stage under a hairline, and the newest line blur-fades in.
- **Labels are sentence case** in the UI face at 14–15 px with Archivo's wide stretch. No tracked uppercase mono eyebrows; mono is only for live numbers.
- **The shell warms the screen modules** with `requestIdleCallback` after the first paint, so the first cut to each screen has no long import frame (T-379). `app.sw` records the service-worker registration outcome (`ok`, `untrusted` for an SSL error, `failed` or `unsupported`) for a certificate-trust UI.
- **Another controller can switch the session.** When the phone companion or /judge changes `mode`/`ab`, the page adopts telemetry's value once its own last control is 1.5 s old (`CONTROL_SETTLE_MS`), so the highlight follows what plays.
- **One name, said once; touch has no keys.** "Target: Aryan" hides while the name is under the word, and it is set in the UI face, not mono. Under Raw the sentence says "Space to hear Aryan alone." with a fine pointer, "Tap Selective to hear Aryan alone." on touch, and "Selective on the laptop plays Aryan alone." on the companion.
- **The gate pill (T-382) is an outline, not a fill.** Under the name, only while Listen shows Isolated, Selective plays and the mode is not Quiet: SOLO in Focus green, OVERLAP-DUCKED in warn yellow (never amber: amber is Raw), UNSURE in bad red. The words and the mapping are T-382's (`web/console/gate-state.js`, shared with /judge); the app only restyles them with its tokens.
- **The phone companion (T-351) keeps the same rules.** No radial wash; a pressed Focus or person button is the neutral fill and turns green only while the mirrored stage is live, isolated, Selective and Focus; the name under the word shows only while the stage plays that person (not under Raw); labels are sentence case; the link line says "Following the stage" without the session id. The laptop's Phone (QR) toggle is hidden at 600 px and below, where the device in hand is already the phone.
- **Check:** `scripts/app_check.py <origin> [--live] [--shots DIR]` walks every screen at 390/768/1440/1920 and, with `--live`, every Listen state with a fake microphone, plus the phone companion (`/app/?watch=<session>`) while the stage is isolated and under Raw. It fails on console errors, sideways scroll, placeholders in the nav, green while idle (or green on the companion while the stage is not isolated), and engine words on stage.
