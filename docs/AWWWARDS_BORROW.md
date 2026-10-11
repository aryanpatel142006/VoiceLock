# Awwwards borrow list (round 2, Oct 10 2026)

Adds to the five sites in `docs/FRONTEND_PLAN.md` "Inspiration" (Why Zero, Paul Kalkbrenner, Gionatan Nese '26,
The Pendragon Cycle, EDOLUS). Same direction as `web/app/DESIGN_SPEC.md`: dark stage, one light, type-led, one
reactive centrepiece, every number and motion driven by live telemetry. Stack stays vanilla ES modules + CSS,
vendored Motion (`web/app/vendor/motion.js`, wrapped by `web/app/motion.js`), View Transitions, Canvas 2D.

"Opened" = the live site was loaded in a browser on Oct 10 2026 and its fonts, easings or screens were read off the
page. "Page" = only the awwwards page (description, element list, tags) was read; patterns from those are taken
from the named element, not guessed. WebGL/3D content is skipped. Only the 2D idea is borrowed.

## 1. Sites

| # | Site (studio) | URL | awwwards page | Award, date | Checked | Borrowable pattern(s) | Our screen |
|---|---|---|---|---|---|---|---|
| 1 | Byotōne (L'Étude) | https://byotone.com | /sites/byotone | Nominee, Sep 24 2026 | Opened | (a) Sound gate with two doors: "Enter with sound" (pill, green mono) and "Enter quietly" (small text link). (b) The centrepiece is a ring of grain/dots around the copy, not a solid shape. (c) Type trio: uppercase display mono (Montiac Mono, 28-78 px), grotesk body (General Sans), 12 px Space Mono labels at 0.1em tracking; easings `cubic-bezier(.165,.84,.44,1)` 0.3-0.7 s for UI, `(.33,1,.68,1)` 1.2 s for reveals. (d) "Audio testimonials with word tracking". | Welcome, Listen, Quiet captions |
| 2 | Árstraumur (davidlilja) | https://arstraumur.music | /sites/arstraumur-electronic-music | Nominee, Sep 3 2026 | Opened | Loader on a black screen that says what it is actually doing ("LOADING 87%", "READING THE STAR CATALOGUE") on a 1 px progress line, plus a quiet "SKIP TO THE MUSIC" link at the bottom. Visuals use real data (live planet positions). | Setup, Listen "loading voice" |
| 3 | Tink-on (Zui Chen) | https://tinkon.zui.ooo | /sites/tink-on | Honorable Mention, Aug 21 2026 | Opened | Built in vanilla JS. Hardware-instrument framing. Unfilled pads read "EMPTY" on a hatched fill, unknown values read "—", and labels are tiny uppercase mono with middle dots ("SFX·11L", "shape·speed·ink"). One tabular readout ("120 BPM") sits between − / + steppers. | People, truth strip, Diagnostics |
| 4 | twks (twks) | https://twks.ch/en | /sites/twks-1 | SOTD + Dev, Oct 7 2026 | Opened | One grotesk family for everything (GT Standard 400-600). Display at 53-60 px with line-height 1.0 and slightly negative tracking. UI transitions are short (150-200 ms, `cubic-bezier(.4,0,.2,1)`). | Global type, About |
| 5 | Matt Jinn (Williams Alamu) | https://mattjinn.com | /sites/matt-jinn | Honorable Mention, Sep 14 2026 | Opened | Preloader is the wordmark alone, pale, dead centre on an empty frame, followed by "Loader to Home" (awwwards element). | App shell boot, Welcome |
| 6 | MENSCH (MENSCH) | https://www.mensch.club | /sites/mensch | Honorable Mention, Aug 18 2026 | Page | Built in vanilla JS. "Press M": one key plays the music and the page reacts. A "Human View vs Agent View" toggle shows the same page in two renderings. The footer wordmark letters push aside on hover. | Listen (RAW/Selective, keys) |
| 7 | penguin.music (Manan Dua) | https://penguin.music | /sites/penguin-music | Honorable Mention, Sep 5 2026 | Page | An audio-reactive scene where light follows the live track. A "low performance mode" swaps to a simpler (pixel-art) rendering as a designed state, and a now-playing indicator is shown. | Listen scene, Diagnostics |
| 8 | VIZZ (preziotte) | https://vizz.fm | /sites/vizz | Nominee, Sep 19 2026 | Page | "Real-time music visualizer… Nothing to download, no sign-up." The picture is the signal, and the entry promise fits in one line. | Welcome copy, scene contract |
| 9 | Cerebrium (Louis Paquet et al.) | https://cerebrium.ai | /sites/cerebrium | SOTD + Dev, Sep 10 2026 | Page | A "Transparent Pricing Calculator": a number that shows its own working. Topic is real-time voice-agent infrastructure, close to ours. | Truth strip, Diagnostics |
| 10 | L.I.S.A. (Locomotive, 60fps) | https://lisa.locomotive.ca/en | /sites/l-i-s-a | SOTD + Dev, Sep 16 2026 | Page | Black/white only. The conversational interface and microcopy carry the site. Responsive score 8.6 (its highest). | Listen status line, Phone |
| 11 | LPAS (Studio Naam) | https://lpas.com | /sites/lpas | SOTD + Dev, Oct 10 2026 | Page | Homepage loader, then a full-page menu, then an "in-depth information" modal over the page. | Phone, People |
| 12 | Paradigm Festival (Bureau Breekijzer) | https://www.paradigmfestival.com | /sites/paradigm-festival | Honorable Mention, Sep 9 2026 | Page | Two colours (#1A1F0A olive-black, #F54109 signal orange). An artist modal per line-up entry. A timetable built mobile-first. | People (person sheet), Phone |
| 13 | Mosby's Files (Tubik) | https://www.mosbyfiles.com | /sites/mosbys-files | SOTD + Dev + Typography Honors, Aug 13 2026 | Page | "CSS-only skeuomorphic folder system where navigation is part of the concept": tabs that are the content. | Diagnostics drawer, About |
| 14 | Ai in Design Report 2026 (++hellohello, SEB®) | https://stateofaidesign.com | /sites/ai-in-design-report-2026 | SOTD + Dev, Aug 26 2026 | Page | Editorial charts that sit inside the prose, with a chapter header animation per section. | About (measured claims) |
| 15 | Sleep Well Creative (Victor Work) | https://sleep-well-creatives.com | /sites/sleep-well-creative | SOTD + Dev, Jan 28 2026 | Page | A wellbeing site with two colours (#0042AF / #FEF1D0) and "science-based insights, friendly design". Calm tone with facts in plain words. | About copy tone |
| 16 | Odyssée (NIGHTJAR) | https://www.odysseeclinic.com.au | /sites/odyssee | SOTD + Dev, Oct 8 2026 | Page | A clinic (care) brand with tags Clean, Minimal, Transitions, Microinteractions. Restraint reads as trust. Responsive 7.8. | Global motion restraint |
| 17 | Warm & Fuzzy (Neutral Studio) | https://www.warmnfuzzy.tv | /sites/warm-fuzzy | SOTD + Dev, Sep 12 2026 | Page | "Idle Animation" is a designed element of its own. Ours is the honest inverse: idle is designed but still. | Welcome/Listen idle |
| 18 | Squarespace Foundations (Resn) | https://brand.squarespace.com | /sites/squarespace-foundations | SOTD + Dev, Sep 1 2026 | Page | Built with plain HTML5 + JS. "Guiding principles" are presented as numbered editorial rules. | About (honesty rules) |

## 2. Top 12 patterns, ranked by impact for judges

Durations below use the tokens in `web/app/tokens.css`. Motion's vanilla `animate()` takes seconds; `motion.js`
`tokenS()` converts. Under `prefers-reduced-motion: reduce` the tokens already shorten cuts to 200 ms linear and set
`--scene-animate: 0`. Each item says what changes on top of that.

**1. Two doors in: "Hold to listen" / "Listen quietly" (Byotōne gate) → Welcome.**
It maps our Focus/Quiet product split onto the first screen, so a judge learns the product before it starts. The
primary door is the existing hold-to-start ring (Why Zero, `--dur-hold-to-start` 900 ms) and enters Focus. Below
it, a 12 px uppercase mono text button, "Listen quietly · captions only", enters Quiet. Both create the
AudioContext inside their own gesture handler (the iOS rule), then call `cut()` to `#listen` with `mode` set before
the first frame, so the Listen screen never flashes the wrong wash. The secondary door is a real `<button>`, 48 px
tall even though it looks like text. Reduced motion: the ring fill becomes a 3-step opacity change (0/50/100 %) and
the hold length stays the same, because the hold is the safety, not the decoration.

**2. A loader that says what it is really doing (Árstraumur) → Listen "loading voice", Setup checks.**
Replace any spinner with one 12 px mono line taken from telemetry: "LOADING VOICE · JOSEPH", then "MATCHING ·
CONFIDENCE 0.42" (live `tse.confidence`). Under it, a 1 px line whose `scaleX` follows a real fraction only when one
exists (profile frames received / needed). Otherwise the line is absent; it never fakes progress. A quiet bottom link,
"Hear the room meanwhile", flips to RAW (honest and useful). Code: `store.on('telemetry', t => line.textContent =
…)`, and `animate(bar, {scaleX: f}, {duration: 0.18, ease: [.33,1,.68,1]})` with `transform-origin: left`.
Reduced motion: the bar jumps (duration 0), and the text still updates.

**3. Grain ring centrepiece (Byotōne) → Listen sound field (`web/app/scene.js`).**
Draw about 500 dots on a ring, with jitter from a fixed seeded table (no `Math.random` per frame), in one `fillRect`
pass. The ring radius follows the real mic peak (24 dB/s release, as now). Dot sharpness and density follow
`tse.confidence`. In Selective, dots outside the target's arc fade to `--scene-quiet-dim`, a literal picture of
"turn down the world". In RAW, all dots stay at full alpha in amber. It is cheaper than gradients and reads as
sound, not as a loading spinner. Budget: one path, DPR ≤ 2, ≤ 4 ms (already measured in Diagnostics). Reduced
motion: a static ring whose alpha updates at most 4 times a second, with no radius motion.

**4. One page, two views (MENSCH Human/Agent View, "Press M") → Listen RAW/Selective.**
The A/B flip keeps the layout identical and swaps only the light: the wash colour, the scene palette and the state
word ("The room" / "Joseph"). Wrap the word swap in `document.startViewTransition` with `view-transition-name:
state-word` and a 140 ms (`--dur-wash`) cross-fade. The wash itself is a CSS custom-property transition on
`--wash`. Show the key on screen as a tiny keycap ("SPACE · hear the room"), as MENSCH announces "Press M". The
wash follows `telemetry.ab`, not the keypress (DESIGN_SPEC §4.2), and the keycap gets the 80 ms `ack()`. Reduced
motion: no transition on the word, colour change only.

**5. Words light up as they are said (Byotōne word-tracked testimonials) → Quiet captions.**
Each word that arrives in a `caption` message is appended as a `<span>` at 40 % opacity and animated to 100 % over
180 ms (`--dur-quick`, `ease: [.165,.84,.44,1]`). Older lines step down to `--color-dim` as new ones arrive, and
there are at most 3 lines (the oldest leaves with a 120 ms fade). Words that have not arrived are never shown in
advance and never predicted. The caption area is an `aria-live="polite"` region. Reduced motion: words appear at
full opacity with no fade.

**6. Honest empty states (Tink-on "EMPTY" pads and "—") → People, truth strip.**
An unsaved profile slot is a card with a 45° hatch (`repeating-linear-gradient` in `--color-line`) and "EMPTY ·
CALIBRATE 3 S". A number with no telemetry yet reads "—" (already the rule). The style turns the rule into a visible
design feature that judges notice. No motion is needed, so reduced motion is the same.

**7. A number that shows its working (Cerebrium's transparent calculator) → truth strip and Diagnostics.**
Tap or focus the delay readout ("1.48 s") to unfold a two-line breakdown from telemetry: "buffer 0.40 · model 0.62
· output 0.46". Components that are not measured show "—", and the sum is only shown when every part is measured.
It answers the judge's first question ("why the delay?") with live numbers. Use a `<details>` element, with height
animated by Motion from 0 to `auto` over 260 ms (`--dur-ui`). Reduced motion: the `<details>` opens without
animation.

**8. A designed light mode for slow devices (penguin.music low-performance mode) → scene, Diagnostics.**
If the measured scene draw time exceeds 4 ms for 30 consecutive frames, `scene.js` drops to "light scene" (a single
ring stroke, no dots), and the Diagnostics sheet says "Light scene · draw 6.1 ms". The drop is designed and
labelled rather than a silent stutter, and it protects the phone budget. Reduced motion forces the light scene too.

**9. Mono micro-labels with middle dots (Tink-on "SFX·11L", Byotōne "Alpha • 8.0 Hz") → truth strip, pill.**
One label style everywhere: `--font-mono`, 12 px (11 px at 390 px), uppercase, `letter-spacing: .1em`,
`font-variant-numeric: tabular-nums slashed-zero`, items joined with " · ". Example: "MAC · 1.48 S BEHIND · RTT 32 MS
· ● REC". The style is consistent, quiet and clearly instrument-like. No motion, except `tick()` on numbers (which
reduced motion disables).

**10. A wordmark-only boot (Matt Jinn loader) → app shell to Welcome.**
While fonts, the worklet and `/health` load, show only "Selective Hearing" in the display face at 40 % white,
centred. When they resolve, run `cut()` to Welcome (a 520 ms film cut), and the wordmark becomes the Welcome
wordmark through a shared `view-transition-name: wordmark`. If loading takes more than 3 s, a mono line appears
under it saying which check is pending (pattern 2). Reduced motion: a 200 ms cross-fade.

**11. One family, tight display (twks) → global type.**
Keep to Archivo alone for display and UI, plus JetBrains Mono for numbers; add no third family. Display words use
`line-height: .92`, `letter-spacing: -0.015em` and `font-stretch: 72%`. UI hover and press use 150-200 ms. The
two-family discipline is what makes the frame look designed rather than assembled. No motion component.

**12. A full-height person sheet on the phone (LPAS full-page menu, Paradigm artist modal) → Phone companion, People.**
On the phone, tapping the person chip opens a `<dialog>` sheet, 100 dvh tall, with the two people as 50 %-height
targets (name at `--text-hero`), and a Release button at the bottom inside the safe area. It slides up 24 px
(`--motion-distance`) and fades in over 260 ms with `--ease-out`, and Escape or swipe-down closes it. Big targets
also work for a nervous judge holding the phone. Reduced motion: fade only, 160 ms.

## 3. Avoid: five tells that make a site look AI-generated

1. **Fake life.** Breathing blobs, looping waveforms, ticking counters or "listening…" pulses that run with no
   signal behind them. Every honest site above (Byotōne, Árstraumur, VIZZ) is driven by something real, and ours
   must be too (DESIGN_SPEC §4).
2. **Purple-to-blue gradients and glass cards.** Gradient text, `backdrop-filter` frosted panels on everything,
   and neon glow borders. None of the 18 sites uses them; they use two colours plus black or white.
3. **The template hero.** A centred headline, a subhead, two buttons, then three icon cards ("Fast · Private ·
   Smart") with sparkle or emoji icons. Our frame is left-aligned on the gutter, with one object and one word.
4. **Uniform scroll choreography.** Every block doing the same 20 px fade-up with a 0.1 s stagger, the same
   rounded-2xl radius everywhere, and the same 300 ms ease on every element. Pick per-purpose durations (the
   tokens) and let most things not move.
5. **Generic copy and invented numbers.** Lines like "Seamless", "Unlock", "Revolutionize your hearing" or
   "AI-powered", and round made-up stats ("99 % accurate"). Use plain sentences and only CLAIMS.md section 1
   numbers with their conditions ("about 1.5 s behind, like live captions").
