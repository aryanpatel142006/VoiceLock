# Selective Hearing app: design system and screen spec (M14, T-344)

The source for how `/app/` looks, moves and tells the truth. Tokens: `web/app/tokens.css`. Static comps
(no JS, sample values): `web/app/comp/*.html`, screenshots in `web/app/comp/shots/`. Product shape and tech
choices: `web/app/DESIGN_SPEC.md` (a)-(c). `/judge` and `/simple` are not touched by anything here.

## 1. Idea

A dark stage with one light on it. The frame is washed **green** when you hear the chosen person and
**amber** when you hear the raw room, so anyone 5 m away can tell the A/B state without reading. The
largest thing on screen is always the answer to "what am I hearing?": the person's name, "The room", or
the captions. One object moves, the sound field, and only when the microphone level or the engine's
confidence moves. Nothing is animated to look alive.

Borrowed, per the plan's Inspiration table: Why Zero's press-and-hold Start and one-scene-at-a-time
progression; Kalkbrenner's numbered loader (ours steps through real telemetry) and number ticker; Gionatan
Nese's type-led black-and-white layout and hold-then-cut pacing; Pendragon Cycle's dark film framing and
up-front sound disclosure; EDOLUS's single reactive centrepiece.

## 2. Tokens (summary; values in `tokens.css`)

| Group | Tokens | Rule |
|---|---|---|
| Stage | `--color-bg #060708`, `--color-bg-raise`, `--color-line`, `--color-fg #f3f4f1`, `--color-dim` (8.6:1), `--color-faint` (4.3:1, hairlines and disabled only) | Text contrast >= 7:1 |
| A/B | `--color-raw #ffb23e` amber + `-deep`/`-wash`; `--color-focus #4bf08e` green + `-deep`/`-wash` | Amber means "you hear the room", green "you hear the person". Never decoration, never a warning, never a selection outline |
| Honesty | `--color-replay` violet, `--color-warn` yellow (not amber), `--color-bad` red, `--color-idle` grey | Each non-live state owns a colour no live state uses |
| Type | `--font-display`/`--font-ui` Archivo variable (wdth 62-125, wght 100-900), `--font-mono` JetBrains Mono variable | Width axis carries hierarchy: `--stretch-display 72%` for giant words, `100%` UI, `118%` names on chips |
| Scale | `--text-display` 72 px @390 / 256 px @1920, `--text-hero` 52/136, `--text-caption` 24/64, `--text-title` 28/56, `--text-lead`, `--text-number` | State word >= 96 px at 1920 (it is 256) |
| Numbers | `--numeric: tabular-nums slashed-zero` | Every live number in mono, tabular |
| Space | `--space-1..10` (4 px base), `--gutter` 20 px @390 / 80 px @1920, `--safe-*` with `env(safe-area-inset-*)`, `--tap 48px`, `--tap-stage 72px` | |
| Shape | `--radius-pill 4px` (status labels, chips), `--radius-control 4px` (buttons, segmented groups, cards), `--radius-sheet 10px` (sheets); only the hold-to-start disc and dots are round | One instrument-like shape (T-388: a pill on every control read as a template) |
| Motion | `--ease-film`, `--ease-out`, `--ease-in`, `--ease-spring` (CSS `linear()`), `--dur-instant 80`, `--dur-wash 140`, `--dur-quick 180`, `--dur-ui 260`, `--dur-cut 520`, `--dur-hold 240`, `--dur-hold-to-start 900`, `--motion-distance 24px`, `--motion-blur 10px` | |
| Reduced motion | `@media (prefers-reduced-motion: reduce)`: cuts 200 ms linear cross-fade, hold 0, no travel, no blur, `--scene-animate: 0`, no grain | |
| Scene contract | `--scene-release-db-per-s 24`, `--scene-floor-db -60`, `--scene-ceiling-db -6`, `--scene-dpr-max 2`, `--scene-budget-ms 4`, `--scene-animate`, `--scene-quiet-dim 0.18` | Unitless numbers read by `scene.js` with `getComputedStyle` |

The shell's placeholder names (T-345 `app.css`) are used as-is, so `tokens.css` overrides them when linked
after `app.css`. `body .devices` re-points the shared device picker's colours (meter "good" is white, not
green).

## 3. Layout system

- **One frame per screen**, full viewport (`100dvh`), three rows: top bar (pill or wordmark left, truth
  strip right), open middle (the sound field lives here, behind content), bottom block (the word and the
  controls). Content is **left-aligned** on the gutter; nothing is centred except the field.
- **1920 x 1080** (stage laptop on the projector): the word bottom-left, controls a 440 px column
  bottom-right, field centred at 64 % x / 42 % y, `min(78vh, 62vw)` across. Truth strip top-right.
- **390 x 844** (iPhone 14, PWA): one column. Pill, truth strip wraps under it, field centred at
  50 % / 30 % and 120 vw across (bleeds off both edges), word, then full-width controls (60 px segmented
  buttons, person cards in a 2-up grid, Release full width). Keyboard hints hidden (`pointer: coarse` or
  <= 720 px).
- Between 720 and 1100 px the 1920 layout applies with fluid type; nothing reflows except the Phone
  hand-off panel, which needs >= 1100 px.
- Safe areas: every edge uses `--safe-*`, so the PWA in `black-translucent` status bar never puts text
  under the notch or home indicator.

## 4. Universal honesty rules

1. Every number on screen is a telemetry field or a measured level. Before the first telemetry it reads
   "—" (an em dash), never 0, never a typical value.
2. The colour wash shows the **audible** A/B state as the session reports it (`telemetry.ab`, after the
   page's `abDelayMs` alignment), not the key the user just pressed. A pressed key gets an 80 ms
   acknowledgement on the control; the wash follows the state.
3. The sound field moves only with the real mic peak (24 dB/s release, like `devices.js`) and
   `tse.confidence`. Mic off, stalled or replaying: it does not pretend (see each state).
4. The status pill uses `judgePill`/`micPill` from `/console/judge-pill.js` and the 500 ms `PillHysteresis`,
   so it says exactly what `/judge` says. Copy may be shortened for the 390 px pill, never changed in meaning.
5. Delay is stated once, plainly, wherever audio starts: "About 1.5 s behind, like live captions." The
   truth strip shows the measured delay next to it.
6. REPLAY is never shown as live: violet badge, violet frame rule, and the word "Replay" in the pill.
7. "Recording" shows whenever the session is recorded (`session.recording`), with a red dot.
8. Comps carry a visible "Comp, sample values" tag. The app never shows that tag, and never ships a
   hard-coded sample value.

## 5. Screens and states

Each screen is one state of the shell's state machine (hash `#welcome`, `#setup`, `#people`, `#listen`,
`#diag`, `#about`, `#phone`). Per state: what it looks like at 390 and 1920, what drives it, and its
honesty rule.

### 5.1 Welcome (`#welcome`) — comp `comp/welcome.html`
Hero line "Turn down the world without disconnecting from it." in condensed display, the press-and-hold
**Hold to start** disc, three facts ("Pick one voice. The room goes quiet around it." / "About 1.5 s
behind, like live captions." / install hint), the wordmark top-left.

| | 390 | 1920 |
|---|---|---|
| Layout | Field top (30 % y), hero 52 px four lines, disc 140 px left, facts under it | Hero 136 px bottom-left, disc 168 px beside the facts, field right of centre |

| State | Look | Driven by | Honesty |
|---|---|---|---|
| idle (default) | Field drawn flat in `--color-idle`, label "idle, mic off" in the centre | no stream | The field is flat until the mic streams; no breathing, no pulse |
| holding | The ring fills clockwise over `--dur-hold-to-start`; release before full empties it in `--dur-quick` | pointer down time | Fill is pointer time, not a fake loader. The AudioContext is created inside the completing press |
| install hint | Shown only when not `display-mode: standalone`; iOS text "Share, then Add to Home Screen" | media query | Hidden once installed |
| unlock cut | On full ring: hold `--dur-hold`, then View-Transition cut to Setup (first run) or Listen (devices remembered) | Start success | If `getUserMedia` fails the cut does not happen; `micProblem()` text appears under the disc in `--color-bad` |

Keyboard: Enter or Space held on the focused disc fills it the same way.

### 5.2 Setup (`#setup`)
Microphone and Output pickers (`/console/devices.js`), a horizontal level meter, the checks list, the
honest delay line, Continue.

| | 390 | 1920 |
|---|---|---|
| Layout | Title, pickers stacked full width, meter, checks as a list of rows, Continue pinned at the bottom | Two columns: pickers + meter left (560 px), checks right; delay line under both; Continue bottom-left |

| State | Look | Driven by | Honesty |
|---|---|---|---|
| checking | Each check row shows "Checking" in `--color-dim` | pending promise | No tick until the check resolved |
| check passed / failed | Tick in `--color-fg` / cross and one-line fix in `--color-bad`. Rows: secure context, audio worklet loaded, engine reachable (`/health`), backend (Mac or GPU from `/health`) | the check itself | A failed check says how to fix it ("Open over https: run demo.sh --lan") |
| meter: too quiet / good / clipping | Bar in `--color-warn` below -40 dBFS, `--color-fg` good, `--color-bad` at >= -2 dBFS; peak value in mono | frames actually sent (devices.js) | Measured on sent frames, 24 dB/s release |
| warnings | Bluetooth delay (150-250 ms extra), HFP mic, wrong output: yellow rows from `deviceWarnings` | device labels | Shown, never hidden to look clean |

### 5.3 People (`#people`)
Saved profiles from `telemetry.enrollment.profiles` as big cards, Release, and Calibrate 3 s.

| | 390 | 1920 |
|---|---|---|
| Layout | Cards 2-up (name in expanded width, key number), Release full width, Calibrate as a quiet text button under a rule | Cards in a row, 320 px each, name at `--text-title`; Release beside; Calibrate bottom-left, separated |

| State | Look | Driven by | Honesty |
|---|---|---|---|
| no profiles yet | Cards disabled, labelled "Person 1", "Person 2", line "No saved voices on this server" | `profiles` empty | Never invent names |
| loading voice | Chosen card shows "Loading voice" (the start sequence, section 6) | `acquisition` "loading saved voice" | Steps only advance on their events |
| chosen | Card outlined in `--color-fg` (neutral, not green) | `enroll profile` echoed | Outline appears on the echo, not on tap |
| searching (Release) | All cards plain, line "Target: none. Press 1 or 2" | `enroll clear` | Same text as `/judge` |
| calibrating | "Listening 1.8 s" countdown, then "Calibrated" or "Too quiet, try again" in `--color-warn` | `telemetry.enrollment` | Off the stage path: not shown on Listen |

### 5.4 Listen (`#listen`) — comps `comp/listen-focus.html`, `listen-raw.html`, `listen-quiet.html`
The stage screen. Top: status pill (left), truth strip (right: Backend, Delay, RTT, Session/recording,
REPLAY when replaying). Bottom-left: "Hearing" + the word. Bottom-right: RAW | Selective segmented
switch, Focus | Quiet, person cards 1/2, Release. Keys exactly as `/judge`: Space, Shift+F, Shift+Q, 1, 2,
Shift+0, D, Shift+R.

| | 390 | 1920 |
|---|---|---|
| Layout | Pill, truth strip wrapped under, field top, "Hearing" 28 px + word 72 px, controls full width below | Word 256 px bottom-left (>= 96 px, legible at 5 m), controls 440 px column bottom-right, field 64 %/42 % |

| State | Look | Driven by | Honesty |
|---|---|---|---|
| Focus, isolated | Green wash; word = person's name in green; pill "Isolated" + "Joseph's saved voice"; field green, rings sharpen with confidence | `ab: processed`, `acquisition` locked, `tse.confidence` | Name shows only once the pill says Isolated; before that the word is the pill state ("Loading") |
| RAW | Amber wash; word "The room" in amber; pill "Raw room", sub "Joseph still isolated, Space to switch" | `ab: raw` | The pill keeps the engine's real state in the sub line; the word says what you hear |
| Quiet | Wash off, field dimmed to 0.18; captions take the frame (newest white, older dim, oldest faint), meta "Captions for Joseph, 1 to 2 s behind"; A/B switch hidden | `mode: quiet`, `caption` messages | Captions only from `caption` messages; interim text in `--color-dim`, finals in `--color-fg`. No caption, no text: the area says "Waiting for Joseph to speak" |
| idle (before Start) | Grey, flat field labelled "idle"; pill "Press Start"; word "Ready"; controls disabled | `micPill` idle | No wash before audio flows |
| loading voice | Field grey with the start sequence line (section 6, step 03 lit); word "Loading"; pill "Loading voice — Joseph" | `acquisition` acquiring "loading saved voice" | A step lights only when its event happened |
| searching (Release) | Wash off, field neutral white; word "Searching"; target line "Target: none. Press 1 or 2" | `enroll clear` | Same copy as `/judge` |
| voice paused / fallback | Wash stays green but the word becomes "Noise only", pill "Voice profile paused — reason" | `judgePill` fallback | Never keep the name up while isolation is off |
| similar voices | Pill note "similar voices — limited isolation" under the pill | `tiers.same_bucket` | Shown verbatim from `judgePill.note` |
| nomic (mic not streaming) | Field freezes flat, rule in `--color-bad` across the frame top, red banner "Mic not streaming — check the mic, then press Start again"; wash drops to none | `micPill` nomic (no frame for 2 s) | The page never looks live without frames |
| reconnecting | Everything desaturates to grey (`filter: saturate(0)` on the frame, not the banner), pill "Reconnecting to the engine…", numbers keep their last value in `--color-faint` with "stale" | socket closed, backoff | Stale numbers are marked stale |
| REPLAY | Violet badge "Replay" top-centre, violet 2 px frame rule, pill prefix "Replay"; field drawn from the replay clip's level in violet outlines | `replay` state | Never shown as live; badge cannot be hidden while replaying |
| keys off | Yellow banner top "Keys off: click the page" | window blur / text focus | Same as `/judge` |
| mic problem | Red banner with `micProblem()` text; Start returns | getUserMedia error | Says what to do |

A/B flip: the wash cross-fades in `--dur-wash` (140 ms) once `telemetry.ab` changes; the word swaps with a
blur-fade (`--motion-blur` to 0) in `--dur-quick`. Reduced motion: plain cross-fade.

### 5.5 Diagnostics sheet (`#diag`, key D)
A bottom sheet over Listen (Listen stays visible above it at 1920; full screen at 390).

| | 390 | 1920 |
|---|---|---|
| Layout | Full-height sheet, rows of label + mono value, close at top-right | 720 px sheet from the right edge, Listen dimmed 40 % behind |

Rows: backend (Mac / GPU + `hq_active`, `hq_reason`), tier (`tiers.active` + reason), delay EMA with
"Steady" when stable, RTT, confidence meter (a bar, value in mono), preset pins (latency preset, isolation),
server rev warning, fallback log (newest first, time + `degraded_to`), scene frame time (ms, p95). Honesty:
every row "—" until its field arrives; the frame time is measured by `scene.js`, not estimated.

### 5.6 About (`#about`)
Pitch line, then the measured claims from `CLAIMS.md` section 1 only, each followed by its condition
in `--color-dim`; the far-field honesty line verbatim.

| | 390 | 1920 |
|---|---|---|
| Layout | One column, 32 ch measure, claim number in display 56 px, condition under it | Two-column grid of claims at a 60 ch measure, pitch in hero size above |

Honesty: no number from CLAIMS sections 2-3; every number carries its condition on the same card.

### 5.7 Phone (`#phone`, `/app/?watch=<id>`) — comp `comp/phone.html`
Companion: mirrors the stage session's state word, pill and captions, with four buttons (Focus, Quiet,
person, Release) forwarded into it. The laptop side shows the hand-off panel with the QR from
`/connect?page=app`.

| | 390 | 1920 |
|---|---|---|
| Layout | Pill + "Mirroring the stage laptop, this phone is not a mic" badge, field top, "Hearing" + name, captions card, 2 x 2 buttons (72 px) | Laptop view: the companion column left (520 px) and the hand-off panel right (420 px: title, QR, one paragraph, the URL in mono) |

| State | Look | Driven by | Honesty |
|---|---|---|---|
| companion, live | As Listen at 390, wash follows the stage's `ab` | `/ws/watch` mirror | The badge always says it is not a mic and not the output |
| waiting for stage | Grey, word "Waiting", line "No stage session yet. Start it on the laptop." | no session | No mirrored values until they arrive |
| stage ended / socket lost | Grey, word "Disconnected", Retry | watch socket closed | Last captions stay but dimmed as stale |
| button forwarded | Button acknowledges in 80 ms; pressed state only after the mirrored telemetry echoes | control echo | The phone never shows a state the stage has not reported |
| Solo (labelled extra) | Own mic and session, header "Solo: this phone listens", wake lock requested | SDK path | Refused while a judge session is live, with that reason in the text |

## 6. Motion

- **Screen cuts**: View Transitions API, `--dur-cut` cross-dissolve with a hold of `--dur-hold` first.
  Shared elements: `view-transition-name: pill` on the status pill and `person` on the chosen person's chip,
  so they glide while the rest dissolves. Without the API: instant swap plus a `--dur-ui` opacity fade.
- **Choreography**: Motion (vendored ESM). One entrance per screen: the word blur-fades in
  (`--motion-blur` to 0, `--motion-distance` up), controls follow with `stagger(0.04)`. Nothing loops.
- **Start sequence** (the Kalkbrenner loader, honest): four numbered steps, each lit by a real event and
  never by a timer: `01 Mic streaming` (first audio frame sent), `02 Engine connected` (first telemetry),
  `03 Loading voice` (`acquisition` "loading saved voice: …"), `04 Isolated` (pill isolated). A step that has
  not happened stays `--color-faint`; a step that fails turns `--color-bad` with its reason. The numbers are a
  real sequence, which is the only reason they are numbered.
- **Number ticker**: delay and RTT tick from the previous measured value to the new one in `--dur-quick`;
  they never animate from 0 or past the measured value.
- **Sound field** (Canvas 2D, `scene.js`): concentric rings and a core whose radius and light follow the
  mic peak (dB mapped from `--scene-floor-db` to `--scene-ceiling-db`, 24 dB/s release); ring sharpness
  follows `tse.confidence`. One rAF loop, DPR capped at 2, draw <= 4 ms, transform/opacity in the DOM
  around it. Reduced motion: one static frame per telemetry tick.
- **Reduced motion** (`prefers-reduced-motion: reduce`): cross-fades only, no travel, no blur, static
  field, hold-to-start fill still shown (it is feedback for a held press, not decoration).

## 7. Type rules

- Display (name, state word, hero): Archivo, `font-stretch: 72%`, weight 640, leading 0.86, tracking
  -0.035em, `text-wrap: balance`. Sentence case always; no all-caps labels.
- UI: Archivo 100 % width, weight 440 body / 600 strong. Names on chips: 118 % width.
- Numbers: JetBrains Mono, `font-variant-numeric: tabular-nums slashed-zero`.
- Captions: Archivo 100 %, `--text-caption`, leading 1.18, measure 24 ch at 1920.
- Fonts load with `font-display: swap`; `<link rel="preload">` for Archivo only (90 KB). Total font weight
  130 KB, well inside the 300 KB first-load budget.

## 8. Accessibility

Contrast >= 7:1 for all text (faint is hairline/disabled only). Focus ring: 2 px `--color-fg`, 3 px
offset, on every control. Touch targets >= 48 px, stage controls 72 px. The pill is `role="status"`,
captions `aria-live="polite"`. Colour is never the only signal: the word ("Joseph" / "The room" / "Quiet")
and the pressed segment say the same thing as the wash. `prefers-contrast: more` flattens washes and lifts
the faint colour.

## 9. Comps and screenshots

| Comp | File | Shots |
|---|---|---|
| Welcome (idle, hold at 35 %) | `comp/welcome.html` | `shots/welcome-390.jpg`, `shots/welcome-1920.jpg` |
| Listen, Focus (Joseph isolated) | `comp/listen-focus.html` | `shots/listen-focus-390.jpg`, `-1920.jpg` |
| Listen, RAW | `comp/listen-raw.html` | `shots/listen-raw-390.jpg`, `-1920.jpg` |
| Listen, Quiet (captions) | `comp/listen-quiet.html` | `shots/listen-quiet-390.jpg`, `-1920.jpg` |
| Phone companion + laptop hand-off | `comp/phone.html` | `shots/phone-390.jpg`, `-1920.jpg` |

Comps are pure HTML + `tokens.css` + `comp/comp.css`, no JS. The sound field is drawn with CSS at one
sample level (`--level`, `--conf` inline), the app draws it on canvas. The QR area is a labelled
placeholder; the app fills it from `/connect`. Open them through any HTTP server (the font preload needs
an http origin), e.g. the app's own `/app/comp/welcome.html` once T-345's route is on main.

Screenshots were taken with Playwright Chromium at 390 x 844 and 1920 x 1080 (DPR 1) from a local
ephemeral-port HTTP server rooted at `web/app/`: no horizontal or vertical overflow, both fonts loaded, zero
console errors or failed requests on all ten.
