// Welcome screen (M14, T-366; cold open T-429): the hero line, the press-and-hold Start disc, the honest delay line,
// the iPhone install hint, and the film's cold open in front of them. Spec: web/app/DESIGN_SPEC.md#51-welcome-welcome,
// storyboard and frame timings: docs/ux-review-shots/WELCOME_WOW.md.
//
// Hold to start (borrowed from Why Zero): the ring fills with pointer or key time, never a timer of its own, over
// --dur-hold-to-start (900 ms, 600 ms under reduced motion). Mouse, touch and Space/Enter held on the disc (or on
// the page when nothing else has focus) all drive the same pure state machine, `holdReduce` below, tested by
// web/app/tests/welcome.test.mjs. When the ring is full the session starts inside the user activation that the press
// opened, so the AudioContext is created in the gesture (iOS). If the browser says that activation has lapsed
// (`navigator.userActivation.isActive` false) the disc asks to be let go and starts on the release instead, which is
// itself a gesture. Only a mic that really opened cuts to Setup (or Listen when a mic choice is already saved); a
// failed getUserMedia keeps the screen and shows micProblem()'s line under the disc.
//
// The cold open (T-429, once per page load, about 6 s, skippable by any press or key, never in the way of the disc):
//   0.0 s  black frame;
//   0.3 s  one hairline of light grows out of the centre (a flat signal: silence);
//   1.2 s  the line breaks into 720 grains that scatter over the frame and drift: the room's noise as chaos. Nothing
//          is listening yet, so the field says so ("ambient, not the mic") and the drift is decoration, not data;
//   2.4 s  the hero line arrives word by word through a mask (kinetic type), a beat before "without";
//   4.4 s  the disc, the facts and the chrome blur-fade up; the drift settles to a still constellation by 6 s.
// From then on the field is still (DESIGN_SPEC 4: idle is designed but still). It moves again only for a cause:
//   - the hold: as the ring fills the chaos is pulled into one clean ring around the disc, the product's promise in
//     one gesture (progress-driven: real press time, never a timer). Letting go early springs it back (underdamped);
//   - the real mic: once the session streams, the ring's wave and the grains' agitation follow the store's "level"
//     events (the peak of each frame sent) with the meter's 24 dB/s release, and the label prints the dBFS.
// Depth: the pointer (or the device's tilt) offsets the grain bands by their depth and the type by a few px, on
// transforms only. Film grain and vignette are CSS; the grain tile only steps while something is happening.
// The Enter curtain (T-480, before the cold open, once per page load): a black frame, "This starts in a loud room. Turn
// your sound on." and two doors, Enter with sound / Enter silently (Enter or Space = with sound, Esc = silently). That
// click is the gesture the autoplay rules want: it creates the AudioContext and the cold open starts with its score
// (web/app/welcome-audio.js: sub hum and a thin line, the cafe murmur swelling with the grains, a tick per word, the
// hold turning the room down to one voice, a chord when the ring completes, silence when the mic opens). The choice is
// the Sound on/off toggle's too (localStorage hearing.welcome.sound). Enter/Space on the focused disc goes straight
// to the hold (the curtain steps aside with the saved choice), so a scripted or practised "focus Start, hold Enter"
// still works. Cinema on top of the field: letterbox bars that open at the promise, a slow push-in, an anamorphic
// flare on the hairline, film grain whose opacity follows the room's level.
// Reduced motion: no cold open (everything is on screen at once, a 200 ms fade), no drift, no parallax, no grain
// stepping; the hold still fills the ring and the field redraws at most 4 times a second with no easing.

import { MIC_KEY, loadChoice } from "../../console/devices.js";
import { seeded } from "../scene.js";
import { sharedSession } from "./setup.js";
import { haptic } from "../delight.js";
import { expectedDelayCopy, fetchHealth } from "../store.js";
import { installHint as platformInstallHint, platformOf, touchDevice } from "../platform.js";
import { ROOM_BED, holdMix, makeLiveSound, prepareWelcomeSound, roomLevelAt } from "../welcome-audio.js";

export const title = "Welcome";

export const HOLD_MS = 900; // --dur-hold-to-start
export const HOLD_MS_REDUCED = 600; // --dur-hold-to-start under prefers-reduced-motion
export const CUT_HOLD_MS = 240; // --dur-hold: stillness before the cut
export const LEVEL_FLOOR_DB = -60; // --scene-floor-db
export const LEVEL_CEILING_DB = -6; // --scene-ceiling-db
export const LEVEL_RELEASE_DB_PER_S = 24; // the meter's release (web/console/devices.js)
export const HOLD_KEYS = new Set([" ", "Enter"]);
export const SOUND_KEY = "hearing.welcome.sound"; // localStorage: "1" when the hold sound was turned on

/** The cold open's shot list (ms from mount). `end` is when the field goes still and the chrome is back. */
export const COLD_OPEN = Object.freeze({
  line: 300, // the hairline starts growing
  scatter: 1200, // grains leave the line
  scattered: 2800, // every grain is home
  words: 2400, // first word
  wordGap: 110, // stagger between words
  beat: 240, // extra pause before "without"
  wordMs: 720, // one word's rise
  ui: 4400, // disc, facts, chrome
  settle: 4600, // drift starts to calm
  end: 6000, // still
});
/** T-458: on a phone the cold open runs at this share of its length (the UI arrives at ~2.6 s, not 4.4 s). */
export const TOUCH_COLD_SCALE = 0.6;

/** The cold open's shot list with every time (and the word gap and beat) scaled by `k`; word rise kept. */
export function scalePlan(plan = COLD_OPEN, k = 1) {
  if (k === 1) return plan;
  return Object.freeze(Object.fromEntries(Object.entries(plan).map(([key, v]) => [key, key === "wordMs" ? v : Math.round(v * k)])));
}
export const GRAINS = 720; // three depth bands of 240
export const BANDS = [0.38, 0.66, 1]; // depth of each band: far bands are smaller, dimmer and move less
export const REDUCED_MIN_MS = 250; // reduced motion: at most 4 static redraws a second
export const MAX_SIDE_PX = 1300; // the field's backing store is capped (390x844 at DPR 3 drew at 1.9 and, with the shared
// scene canvas redrawing behind it, missed every other vsync on a 4x-throttled CPU; 1.54 holds 60 fps, T-429)
export const LABEL_MIN_MS = 100; // the field's label (a DOM text change, so a layout) updates at most 10 times a second

// ---------------------------------------------------------------- pure state machine

/**
 * Phases: idle (nothing pressed) -> holding (ring filling) -> full (ring complete; the caller starts the session now,
 * or arms it) -> armed (waiting for the release to start) -> starting (mic opening) -> done (cut) | failed (retry).
 * A release or cancel while holding drops back to idle and empties the ring.
 */
export function holdInit() {
  return { phase: "idle", source: null, t0: 0, progress: 0, error: "" };
}

/**
 * Next hold state for `ev`. Events: {type:"press", source, at} · {type:"tick", at, durationMs} ·
 * {type:"release", source} · {type:"cancel"} · {type:"arm"} · {type:"start"} · {type:"started", ok, error} ·
 * {type:"activate"} (a screen reader or switch clicked the disc: no pointer to hold, so the ring is full at once).
 * `source` names the input that pressed ("pointer:<id>" or "key:<key>") so a second finger or another key cannot
 * end someone else's hold. Unknown or out-of-phase events return the same object.
 */
export function holdReduce(s, ev) {
  switch (ev.type) {
    case "press":
      if (s.phase !== "idle" && s.phase !== "failed") return s;
      return { ...s, phase: "holding", source: ev.source, t0: ev.at, progress: 0, error: "" };
    case "tick": {
      if (s.phase !== "holding") return s;
      const dur = ev.durationMs > 0 ? ev.durationMs : HOLD_MS;
      const progress = Math.min(1, Math.max(0, (ev.at - s.t0) / dur));
      return progress >= 1 ? { ...s, phase: "full", progress: 1 } : { ...s, progress };
    }
    case "release":
      if (ev.source !== s.source) return s;
      if (s.phase === "holding") return { ...s, phase: "idle", source: null, progress: 0 };
      if (s.phase === "armed") return { ...s, phase: "starting", source: null };
      if (s.phase === "full") return { ...s, source: null }; // let go in the same frame the ring filled
      return s;
    case "activate":
      if (s.phase !== "idle" && s.phase !== "failed") return s;
      return { ...s, phase: "full", source: "activate", progress: 1, error: "" };
    case "cancel":
      if (s.phase === "holding" || s.phase === "armed") return { ...s, phase: "idle", source: null, progress: 0 };
      return s;
    case "arm":
      return s.phase === "full" ? { ...s, phase: "armed" } : s;
    case "start":
      return s.phase === "full" ? { ...s, phase: "starting" } : s;
    case "started":
      if (s.phase !== "starting") return s;
      return ev.ok
        ? { ...s, phase: "done", source: null }
        : { ...s, phase: "failed", source: null, progress: 0, error: ev.error || "Microphone unavailable. Hold to try again." };
    default:
      return s;
  }
}

/** Whether a full ring may start now: true unless the browser reports the press's user activation has lapsed. */
export function activationLive(nav = globalThis.navigator) {
  const ua = nav?.userActivation;
  return ua ? ua.isActive !== false : true;
}

/**
 * Where a successful Start cuts to: Listen when a microphone choice is already saved, else Setup. On a phone
 * (`touch`) always Listen (T-458): Start resolves once the mic streams, and Setup's checklist is the laptop's.
 */
export function startTarget(storage = globalThis.localStorage, { touch = false } = {}) {
  if (touch) return "listen";
  try { return loadChoice(MIC_KEY, storage) ? "listen" : "setup"; } catch { return "setup"; }
}

/** A press let go before this share of the ring is a tap: the disc nudges and says to keep holding (T-458). */
export const TAP_NUDGE_BELOW = 0.3;
export const NUDGE_MS = 1500;
export const NUDGE_COPY = ["Keep holding", "about 1 s"];

/** True when a release from `prev` (holding at `prev.progress`) was a quick tap, not an early let-go. */
export function wasTap(prev, next) {
  return prev?.phase === "holding" && next?.phase === "idle" && prev.progress < TAP_NUDGE_BELOW;
}

/** 0..1 light of the field for a mic peak in dBFS (the scene contract's floor/ceiling); null/NaN = 0. */
export function levelLight(db) {
  if (!Number.isFinite(db)) return 0;
  return Math.max(0, Math.min(1, (db - LEVEL_FLOOR_DB) / (LEVEL_CEILING_DB - LEVEL_FLOOR_DB)));
}

/** The level `dtS` seconds after the last measured peak `db`, released at the meter's 24 dB/s; null/NaN stays null. */
export function releasedDb(db, dtS) {
  if (!Number.isFinite(db)) return null;
  return Math.max(LEVEL_FLOOR_DB, db - LEVEL_RELEASE_DB_PER_S * Math.max(0, dtS));
}

/**
 * The install hint, or null when the app already runs installed (display-mode standalone / iOS home screen) or the
 * platform is not iOS (T-458: Android gets the beforeinstallprompt button instead, never iPhone steps).
 */
export function installHint({ standalone, ios = false, platform = ios ? "ios" : "other" }) {
  return platformInstallHint({ standalone, platform });
}

/** micProblem()'s lines say "press Start" (the /judge button); here Start is held. */
export function holdWording(text) {
  return String(text || "").replace(/\bpress Start\b/gi, (m) => (m[0] === "P" ? "Hold Start" : "hold Start"));
}

/** The disc's two lines for a phase. */
export function discCopy(phase) {
  switch (phase) {
    case "holding": return ["Keep holding", "opens the mic"];
    case "full": case "starting": return ["Opening the mic", "allow it if asked"];
    case "armed": return ["Let go to start", "opens the mic"];
    case "done": return ["Mic open", "cutting to the next step"];
    case "failed": return ["Hold to retry", "opens the mic"];
    default: return ["Hold to start", "opens the mic"];
  }
}

// ---------------------------------------------------------------- pure field maths

/** The hero line split into the words the mask reveals one by one; the beat lands before "without". */
export function heroWords(text = "Turn down the world without disconnecting from it.") {
  return text.split(/\s+/).filter(Boolean);
}

/** Start of word `i`'s rise (ms from mount): the stagger, plus one beat once before "without" (index `beatAt`). */
export function wordDelay(i, beatAt = 4, plan = COLD_OPEN) {
  return plan.words + i * plan.wordGap + (i >= beatAt ? plan.beat : 0);
}

/**
 * The grains' chaos homes, seeded so the field is the same on every load. Unit viewport coordinates; each grain has
 * a depth band (0..2), a drift phase, a drift rate and a size. Bands are interleaved so one band never clumps.
 */
export function grainTable(n = GRAINS, seed = 0x7a11) {
  const rnd = seeded(seed);
  const t = { x: new Float32Array(n), y: new Float32Array(n), band: new Uint8Array(n), phase: new Float32Array(n),
    rate: new Float32Array(n), size: new Float32Array(n), order: new Float32Array(n) };
  for (let i = 0; i < n; i++) {
    t.x[i] = rnd();
    t.y[i] = rnd();
    t.band[i] = i % BANDS.length;
    t.phase[i] = rnd() * Math.PI * 2;
    t.rate[i] = 0.4 + rnd() * 0.9; // drift cycles per second
    t.size[i] = 0.7 + rnd() * 1.1; // px at DPR 1, scaled by depth
    t.order[i] = rnd(); // when it leaves the line (0 first)
  }
  return t;
}

/** Smoothstep ease for the collapse: slow to start, decisive at the end; monotonic 0..1, clamped. */
export function collapseEase(p) {
  const x = Math.max(0, Math.min(1, p));
  return x * x * (3 - 2 * x);
}

/** How far the hairline has grown (0..1) and how far each grain is from line to home (0..1) at `tMs`. */
export function introAt(tMs, plan = COLD_OPEN) {
  const line = Math.max(0, Math.min(1, (tMs - plan.line) / (plan.scatter - plan.line)));
  const scatter = Math.max(0, Math.min(1, (tMs - plan.scatter) / (plan.scattered - plan.scatter)));
  return { line: 1 - (1 - line) ** 3, scatter };
}

/** The drift's amplitude (0..1) during and after the cold open: full while it plays, settled to 0 by `end`. */
export function ambientAt(tMs, plan = COLD_OPEN) {
  if (tMs < plan.settle) return 1;
  return 1 - Math.max(0, Math.min(1, (tMs - plan.settle) / (plan.end - plan.settle)));
}

/**
 * The field's collapse target for a hold phase: the ring's progress while holding, 1 once the ring is full (the
 * voice stays clear while the mic opens and through the cut), 0 at rest or after a failure.
 */
export function collapseTarget(phase, progress) {
  if (phase === "holding") return collapseEase(progress);
  if (phase === "full" || phase === "armed" || phase === "starting" || phase === "done") return 1;
  return 0;
}

/**
 * One step of an underdamped spring (semi-implicit Euler, substeps of at most 4 ms): the field springs back to
 * chaos when the hold is let go, with a small overshoot. `omega` rad/s, `zeta` < 1 bounces. Mutates `st` = {x, v}.
 */
export function bounceStep(st, target, dtS, { omega = 16, zeta = 0.5 } = {}) {
  let left = Math.max(0, Math.min(0.1, dtS));
  while (left > 0) {
    const h = Math.min(0.004, left);
    const a = -omega * omega * (st.x - target) - 2 * zeta * omega * st.v;
    st.v += a * h;
    st.x += st.v * h;
    left -= h;
  }
  if (Math.abs(st.x - target) < 5e-4 && Math.abs(st.v) < 5e-3) { st.x = target; st.v = 0; }
  return st;
}

/** The field's label (honesty: what drives the motion right now). */
export function fieldLabel({ cold, live, replaying, phase, db }) {
  if (replaying) return "replay, not the mic";
  if (live) return Number.isFinite(db) ? `mic live, ${Math.round(db)} dBFS` : "mic live, waiting for frames";
  if (cold) return "ambient, not the mic";
  if (phase === "holding" || phase === "armed") return "holding, mic off";
  if (phase === "full" || phase === "starting") return "opening the mic";
  return "still, mic off";
}

export const PUSH_IN = 0.08; // the camera's push-in on the field over the cold open (scale 1 -> 1.08)
export const HOLD_PUSH = 0.05; // and a little further, toward the disc, as the ring fills
export const GRAIN_MIN = 0.035; // film grain opacity with the room silent
export const GRAIN_MAX = 0.11; // and with the murmur at its peak

/** The camera's push-in 0..1 at `tMs`: a slow smoothstep from the hairline to the settle; 1 after. */
export function pushAt(tMs, plan = COLD_OPEN) {
  const x = Math.max(0, Math.min(1, (tMs - plan.line) / (plan.settle - plan.line)));
  return x * x * (3 - 2 * x);
}

/** The anamorphic flare's strength 0..1: grows with the hairline, brightest as it reaches the edges, gone as the grains leave. */
export function flareAt(tMs, plan = COLD_OPEN) {
  if (tMs <= plan.line) return 0;
  if (tMs <= plan.scatter) return ((tMs - plan.line) / (plan.scatter - plan.line)) ** 2;
  return Math.max(0, 1 - (tMs - plan.scatter) / (0.6 * (plan.scattered - plan.scatter))) ** 2;
}

/** Film grain opacity for the room's level 0..1 (the murmur's level, welcome-audio.js roomLevelAt). */
export function grainOpacity(level) {
  const x = Number.isFinite(level) ? Math.max(0, Math.min(1, level)) : 0;
  return GRAIN_MIN + (GRAIN_MAX - GRAIN_MIN) * x;
}

/**
 * What a key does while the Enter curtain is up: "bypass" (Enter/Space on the focused disc: the curtain steps aside
 * and the hold starts), "sound" / "silent" (enter), "trap" (Tab stays inside the curtain), "swallow" (nothing behind
 * the curtain reacts), "pass" (modified keys: browser shortcuts).
 */
export function curtainKey({ key, repeat = false, mod = false, onDisc = false, onQuiet = false }) {
  if (mod) return "pass";
  if (HOLD_KEYS.has(key) && onDisc) return "bypass";
  if (key === "Tab") return "trap";
  if (repeat) return "swallow";
  if (key === "Escape") return "silent";
  if (HOLD_KEYS.has(key)) return onQuiet ? "silent" : "sound";
  return "swallow";
}

// ---------------------------------------------------------------- screen

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

function svgIcon(d) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 20 20");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("class", "wl-ico");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", d);
  svg.append(p);
  return svg;
}
const ICON_SPEAKER = "M3 7.5h3.2L10.5 4v12L6.2 12.5H3zM13 7.2a3.6 3.6 0 0 1 0 5.6M15.2 5a6.6 6.6 0 0 1 0 10";

/** A 128 px film-grain tile, generated once (no asset to fetch, no network; unseeded: grain has no identity). */
function grainTile() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  if (!g) return "";
  const img = g.createImageData(128, 128);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = 128 + Math.round((Math.random() - 0.5) * 255);
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return `url(${c.toDataURL("image/png")})`;
}

/** The flare's sprite: a round white glow, drawn stretched (anamorphic) along the hairline. */
function flareSprite() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const x = c.getContext("2d");
  if (!x) return null;
  const gr = x.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, "rgba(255,255,255,0.95)");
  gr.addColorStop(0.1, "rgba(255,255,255,0.42)");
  gr.addColorStop(0.4, "rgba(255,255,255,0.1)");
  gr.addColorStop(1, "rgba(255,255,255,0)");
  x.fillStyle = gr;
  x.fillRect(0, 0, 128, 128);
  return c;
}

let coldOpenPlayed = false; // once per page load

export function mount(root, { app, signal }) {
  const session = sharedSession(app);
  const store = session.store;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const standalone = matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  const platform = platformOf();
  const touch = touchDevice();
  const fine = matchMedia("(hover: hover) and (pointer: fine)").matches;
  // the stage laptop needs no install line (T-365 P2-1); iPhone steps on iPhone only, Android gets a button (T-458)
  const hint = platform === "ios" || touch ? installHint({ standalone, platform }) : null;
  // the Enter curtain: once per page load, reduced motion too, and only when Welcome is the page's first screen with no
  // session playing (the brand link mid-demo used to raise the curtain and the loud-room soundtrack over a live session)
  const curtainUp = !coldOpenPlayed && !document.hidden && !app?.previous && !store.state.started && !store.state.replaying;
  const playCold = curtainUp && !reduced.matches;
  const coldK = touch ? TOUCH_COLD_SCALE : 1;
  const plan = scalePlan(COLD_OPEN, coldK);
  coldOpenPlayed = true;

  const label = h("span", { class: "wl-disc-label" });
  const sub = h("span", { class: "wl-disc-sub" });
  const ripple = h("span", { class: "wl-ripple", "aria-hidden": "true" });
  const disc = h("button", { type: "button", class: "wl-disc", id: "start", "data-phase": "idle", "aria-describedby": "wl-at wl-facts wl-problem" },
    h("span", { class: "wl-ring", "aria-hidden": "true" }), ripple, label, sub);
  const problem = h("p", { class: "wl-problem", id: "wl-problem", role: "alert" });
  const status = h("p", { class: "wl-sr", role: "status" });
  const fieldLabelEl = h("span", { class: "wl-field-label", "aria-hidden": "true" });
  const canvas = h("canvas", { class: "wl-field", "aria-hidden": "true" });
  const grain = h("div", { class: "wl-grain", "aria-hidden": "true" });
  const vignette = h("div", { class: "wl-vignette", "aria-hidden": "true" });
  // letterbox (2.39:1 on wide screens): closed through the black frame and the chaos, opening at the promise
  const bars = playCold ? [h("div", { class: "wl-bar wl-bar-top", "aria-hidden": "true" }), h("div", { class: "wl-bar wl-bar-bottom", "aria-hidden": "true" })] : [];
  let soundOn = false;
  try { soundOn = localStorage.getItem(SOUND_KEY) === "1"; } catch { /* private mode */ }
  const soundBtn = h("button", { type: "button", class: "wl-sound", "aria-pressed": String(soundOn) }, svgIcon(ICON_SPEAKER),
    h("span", { class: "wl-sound-text", text: soundOn ? "Sound on" : "Sound off" }));
  // T-458: on a phone the skip sits in thumb reach (welcome.css) and says that any tap skips
  const skipBtn = h("button", { type: "button", class: "wl-skip", text: touch ? "Tap to skip" : "Skip intro" });
  // Android Chrome: an Install button once the browser offers the prompt (never shown on iOS, never on a laptop)
  const installBtn = h("button", { type: "button", class: "wl-install-btn", hidden: true, text: "Install the app" });

  const hero = h("h1", { id: "wl-title", class: "wl-hero" });
  const words = heroWords();
  const wordStartsMs = words.map((_, i) => wordDelay(i, 4, plan));
  words.forEach((w, i) => {
    const outer = h("span", { class: "wl-w", style: `--wl-d: ${wordStartsMs[i]}ms` }, h("span", { class: "wl-wi", text: w }));
    hero.append(outer, i < words.length - 1 ? " " : "");
  });

  // the Enter curtain: one line, two doors. Not a <dialog>.showModal(): that makes the disc inert, and "focus Start,
  // hold Enter" (scripts/app_check.py, a presenter's habit) must keep working; Tab is trapped by hand instead.
  const enterBtn = h("button", { type: "button", class: "wl-enter" }, svgIcon(ICON_SPEAKER), h("span", { text: "Enter with sound" }));
  const quietBtn = h("button", { type: "button", class: "wl-enter-quiet", text: "Enter silently" });
  const curtain = curtainUp ? h("div", { class: "wl-curtain", role: "dialog", "aria-modal": "true", "aria-labelledby": "wl-curtain-title" },
    h("div", { class: "wl-curtain-in" },
      h("span", { class: "wl-curtain-line", "aria-hidden": "true" }),
      h("p", { id: "wl-curtain-title", class: "wl-curtain-title", text: "This starts in a loud room. Turn your sound on." }),
      h("div", { class: "wl-curtain-acts" }, enterBtn, quietBtn),
      fine ? h("p", { class: "wl-curtain-key" }, h("kbd", { text: "Enter" }), " with sound, ", h("kbd", { text: "Esc" }), " silently") : null)) : null;

  root.classList.add("screen-welcome");
  const section = h("section", { class: "wl", "data-phase": "idle", "data-cold": playCold ? "wait" : "done", "aria-labelledby": "wl-title",
    style: `--wl-k: ${coldK}` },
    canvas, vignette, ...bars, grain, fieldLabelEl,
    h("div", { class: "wl-main" },
      hero,
      h("div", { class: "wl-row" },
        h("div", { class: "wl-start" }, disc,
          h("p", { class: "wl-key" }, "or hold ", h("kbd", { text: "Space" })),
          h("p", { class: "wl-at", id: "wl-at", text: "Screen reader or switch: press Enter." })),
        h("div", { class: "wl-facts", id: "wl-facts" },
          h("p", { text: "The voice closest to you comes through. Everything else goes quiet." }),
          h("p", {}, h("strong", { class: "wl-delay", text: expectedDelayCopy(null) }), ", like live captions."),
          hint ? h("p", { class: "wl-install", text: hint }) : null,
          installBtn,
          problem)),
      status),
    soundBtn, // outside .wl-main: that block translates with the pointer, which would anchor a fixed child to it
    playCold ? skipBtn : null,
    curtain);
  root.append(section);
  if (curtain) enterBtn.focus({ preventScroll: true });
  // T-463: the delay line is the server's expected delay (/health tse_window, T-419), never a fixed "1.5 s"
  fetchHealth().then((health) => {
    const el = section.querySelector(".wl-delay");
    if (el && !signal.aborted) el.textContent = expectedDelayCopy(health);
  });
  grain.style.backgroundImage = grainTile();
  if (curtainUp) document.body.dataset.wlCold = "1"; // the chrome is out of the frame behind the curtain and the cold open

  // ---- hold state machine wiring
  let s = holdInit();
  let raf = 0;
  let cutTimer = 0;
  let target = "setup";
  let sound = null;
  const duration = () => {
    const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--dur-hold-to-start"));
    return Number.isFinite(v) && v > 0 ? v : (reduced.matches ? HOLD_MS_REDUCED : HOLD_MS);
  };

  const render = (prev) => {
    disc.style.setProperty("--wl-p", s.progress.toFixed(3));
    if (sound && s.phase === "holding") sound.hold(s.progress);
    if (prev && prev.phase === s.phase) return;
    disc.dataset.phase = s.phase;
    section.dataset.phase = s.phase;
    const [a, b] = nudgeTimer && s.phase === "idle" ? NUDGE_COPY : discCopy(s.phase);
    label.textContent = a;
    sub.textContent = b;
    disc.setAttribute("aria-busy", String(s.phase === "starting" || s.phase === "full"));
    problem.textContent = s.phase === "failed" ? s.error : "";
    status.textContent = s.phase === "holding" ? "Holding. Keep holding to start." : s.phase === "armed" ? "Let go to start."
      : s.phase === "starting" ? "Opening the microphone." : s.phase === "done" ? "Microphone open." : "";
    if (sound && prev) {
      if (s.phase === "holding" && prev.phase !== "holding") { sound.press(); sound.hold(s.progress); }
      else if ((s.phase === "idle" && prev.phase === "holding") || (s.phase === "failed" && prev.phase !== "failed")) sound.release();
      else if (s.phase === "full") sound.resolve();
      else if (s.phase === "done") { sound.silence(); sound.close(400); sound = null; } // the mic is open: silence
    }
    if (s.phase === "full" && !reduced.matches && typeof ripple.animate === "function") {
      ripple.animate([{ scale: 1, opacity: 0.7 }, { scale: 1.9, opacity: 0 }], { duration: 800, easing: "cubic-bezier(0.16, 1, 0.3, 1)" });
    }
    field.phase(s.phase, s.progress);
  };

  const dispatch = (ev) => {
    const prev = s;
    s = holdReduce(s, ev);
    if (s === prev) return;
    render(prev);
    if (s.phase === "holding" && !raf) raf = requestAnimationFrame(tick);
    if (s.phase !== "holding" && raf) { cancelAnimationFrame(raf); raf = 0; }
    // T-427 haptics (Android; a no-op elsewhere): a tick when the press lands, a double beat when the hold completes
    if (s.phase === "holding" && prev.phase !== "holding") haptic("press", { reduced: reduced.matches });
    if (wasTap(prev, s)) nudge();
    if (s.phase === "full") haptic("start", { reduced: reduced.matches });
    // a click from a screen reader or switch is itself the user activation: start now, there is no release to wait for
    if (s.phase === "full") dispatch({ type: activationLive() || s.source === "activate" ? "start" : "arm" });
    else if (s.phase === "starting" && prev.phase !== "starting") begin();
    else if (s.phase === "done" && prev.phase !== "done") cut();
  };

  function tick(now) {
    raf = 0;
    dispatch({ type: "tick", at: now, durationMs: duration() });
    if (s.phase === "holding" && !raf) raf = requestAnimationFrame(tick);
    else if (s.phase === "holding") field.phase(s.phase, s.progress);
  }

  function begin() {
    app.emit("start");
    // synchronous inside this handler: session.start() creates the AudioContext before its first await
    target = startTarget(globalThis.localStorage, { touch });
    let p;
    try { p = session.start(); } catch (err) { p = Promise.reject(err); }
    Promise.resolve(p).then((ok) => {
      if (signal.aborted) return;
      if (ok) dispatch({ type: "started", ok: true });
      else dispatch({ type: "started", ok: false, error: holdWording(store.state.mic.banner) });
    }, (err) => {
      if (!signal.aborted) dispatch({ type: "started", ok: false, error: String(err?.message || err) });
    });
  }

  // T-458: a quick tap gets an answer: a small shake, "Keep holding, about 1 s" for 1.5 s and a short buzz
  let nudgeTimer = 0;
  function nudge() {
    clearTimeout(nudgeTimer);
    disc.classList.remove("wl-nudge");
    void disc.offsetWidth; // restart the shake on a second tap
    if (!reduced.matches) disc.classList.add("wl-nudge");
    nudgeTimer = setTimeout(() => { nudgeTimer = 0; disc.classList.remove("wl-nudge"); render({ phase: "" }); }, NUDGE_MS);
    render({ phase: "" });
    status.textContent = "Keep holding the disc for about 1 second to start.";
    haptic("nudge", { reduced: reduced.matches });
  }

  function cut() {
    cutTimer = setTimeout(() => { if (!signal.aborted) app.go(target); }, reduced.matches ? 0 : CUT_HOLD_MS);
  }

  const press = (source, e) => {
    if (s.phase !== "idle" && s.phase !== "failed") return false;
    skipCold();
    ensureSound(false); // inside the gesture: a hold with sound left on (or straight through the curtain) still sounds
    app.emit("start-gesture"); // the shell's contract: synchronous inside the pointerdown / keydown
    dispatch({ type: "press", source, at: e.timeStamp || performance.now() });
    return true;
  };

  disc.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (!press(`pointer:${e.pointerId}`, e)) return;
    e.preventDefault(); // no text selection or drag image while held
    try { disc.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
  }, { signal });
  const pointerEnd = (e) => dispatch({ type: "release", source: `pointer:${e.pointerId}` });
  disc.addEventListener("pointerup", pointerEnd, { signal });
  disc.addEventListener("pointercancel", () => dispatch({ type: "cancel" }), { signal });
  disc.addEventListener("lostpointercapture", (e) => { if (s.source === `pointer:${e.pointerId}`) dispatch({ type: "cancel" }); }, { signal });
  // a mouse click is not a hold; a click with no pointer behind it (detail 0: a screen reader, a switch, voice control)
  // cannot hold, so it starts exactly as a completed hold does
  disc.addEventListener("click", (e) => {
    e.preventDefault();
    if (e.detail !== 0 || (s.phase !== "idle" && s.phase !== "failed")) return;
    skipCold();
    ensureSound(false);
    app.emit("start-gesture");
    dispatch({ type: "activate" });
  }, { signal });
  disc.addEventListener("contextmenu", (e) => e.preventDefault(), { signal }); // long-press menu on touch

  // keys: on the disc, or on the page when nothing focusable has focus (the judge can just hold Space)
  const keyTarget = (t) => t === disc || t === document.body || t === document.documentElement || t?.id === "stage" || root.contains(t) && !t.closest("a, input, select, textarea, button:not(#start)");
  addEventListener("keydown", (e) => { // the curtain's keys: capture, so nothing behind it reacts first
    if (!curtainOpen) return;
    const act = curtainKey({ key: e.key, repeat: e.repeat, mod: e.metaKey || e.ctrlKey || e.altKey, onDisc: e.target === disc, onQuiet: e.target === quietBtn });
    if (act === "pass") return;
    if (act === "bypass") { enter(soundOn, { bypass: true }); return; } // the disc's own handler presses next
    e.preventDefault();
    e.stopPropagation();
    if (act === "trap") (document.activeElement === enterBtn ? quietBtn : enterBtn).focus();
    else if (act === "sound" || act === "silent") enter(act === "sound");
  }, { signal, capture: true });
  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!HOLD_KEYS.has(e.key)) { if (!e.repeat && e.key !== "Tab" && e.key !== "Shift") skipCold(); return; }
    if (!keyTarget(e.target)) return;
    e.preventDefault(); // Space must not scroll, Enter must not click
    if (e.repeat) return;
    press(`key:${e.key}`, e);
  }, { signal });
  addEventListener("keyup", (e) => {
    if (!HOLD_KEYS.has(e.key)) return;
    if (s.source === `key:${e.key}`) e.preventDefault();
    dispatch({ type: "release", source: `key:${e.key}` });
  }, { signal });
  addEventListener("blur", () => dispatch({ type: "cancel" }), { signal });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { dispatch({ type: "cancel" }); skipCold(); sound?.mute(true); } else if (soundOn) sound?.mute(false);
  }, { signal });

  // ---- sound: on/off is one choice (the curtain's door or the toggle), remembered; the engine exists only while on
  function setSound(on) {
    soundOn = on;
    soundBtn.setAttribute("aria-pressed", String(soundOn));
    soundBtn.querySelector(".wl-sound-text").textContent = soundOn ? "Sound on" : "Sound off";
    try { localStorage.setItem(SOUND_KEY, soundOn ? "1" : "0"); } catch { /* private mode */ }
    if (!soundOn && sound) { sound.close(); sound = null; }
  }
  /** Creates and starts the soundtrack; call only inside a gesture (click, keydown, pointerdown). */
  function ensureSound(cold) {
    if (!soundOn || sound || s.phase === "done") return;
    sound = makeLiveSound({ plan, wordStartsMs });
    sound?.start({ cold });
  }
  soundBtn.addEventListener("click", () => { setSound(!soundOn); ensureSound(false); }, { signal });

  // ---- the curtain: the click (or Enter/Space/Esc) chooses sound, creates the context and starts the film
  let curtainOpen = Boolean(curtain);
  if (curtain) { // Perf: the soundtrack's sampled envelopes render in idle slices while the person reads the curtain
    const idle = window.requestIdleCallback ? (f) => window.requestIdleCallback(f, { timeout: 400 }) : (f) => setTimeout(f, 30);
    prepareWelcomeSound({ idle: (f) => idle(() => { if (!signal.aborted) f(); }) });
  }
  function enter(withSound, { bypass = false } = {}) {
    if (!curtainOpen) return;
    curtainOpen = false;
    curtain.dataset.state = "out";
    setTimeout(() => curtain.remove(), reduced.matches ? 0 : 800);
    setSound(withSound);
    if (bypass || !playCold) { // straight to the still end state (reduced motion, or a hold on the disc)
      ensureSound(false);
      if (!playCold) delete document.body.dataset.wlCold;
      if (!bypass) disc.focus({ preventScroll: true });
      return;
    }
    coldT0 = performance.now();
    section.dataset.cold = "play"; // the CSS timeline (word rises, letterbox, UI arrival) starts now
    field.begin(coldT0);
    ensureSound(true);
    addEventListener("pointerdown", (e) => {
      if (e.target !== disc && !disc.contains(e.target) && !soundBtn.contains(e.target)) skipCold();
    }, { signal, capture: true });
    setTimeout(() => { if (!signal.aborted && !coldDone) { skipCold(); } }, plan.end);
    disc.focus({ preventScroll: true });
  }
  enterBtn.addEventListener("click", () => enter(true), { signal });
  quietBtn.addEventListener("click", () => enter(false), { signal });

  // ---- the cold open: CSS owns the type and the UI (delays from COLD_OPEN); the canvas owns the light
  let coldDone = !playCold;
  let coldT0 = performance.now();
  function skipCold() {
    if (coldDone || curtainOpen) return;
    coldDone = true;
    for (const bar of bars) { // the letterbox opens from wherever it is, then leaves the DOM
      const from = getComputedStyle(bar).translate;
      const to = bar.classList.contains("wl-bar-top") ? "0 -101%" : "0 101%";
      if (typeof bar.animate === "function" && !reduced.matches) {
        bar.animate([{ translate: from === "none" ? "0 0" : from }, { translate: to }], { duration: 360, easing: "cubic-bezier(.2, .7, .1, 1)", fill: "forwards" })
          .finished.then(() => bar.remove(), () => bar.remove());
      } else bar.remove();
    }
    section.dataset.cold = "done";
    delete document.body.dataset.wlCold;
    skipBtn.remove();
    field.cold(false);
    sound?.skip();
  }
  skipBtn.addEventListener("click", skipCold, { signal });
  let installPrompt = null;
  addEventListener("beforeinstallprompt", (e) => {
    e.preventDefault(); // our button asks instead of Chrome's mini-infobar
    installPrompt = e;
    installBtn.hidden = standalone;
  }, { signal });
  addEventListener("appinstalled", () => { installBtn.hidden = true; installPrompt = null; }, { signal });
  installBtn.addEventListener("click", async () => {
    if (!installPrompt) return;
    const p = installPrompt;
    installPrompt = null;
    installBtn.hidden = true;
    try { await p.prompt(); } catch { /* dismissed or already used */ }
  }, { signal });

  // ---- the field: Canvas 2D, 720 grains in three depth bands; still unless something drives it
  const field = makeField({ canvas, disc, section, grain, labelEl: fieldLabelEl, store, reduced, coldT0, playCold, waiting: playCold, fine, signal, plan });
  field.cold(playCold);
  field.phase("idle", 0);

  render(null);

  return () => {
    delete document.body.dataset.wlCold;
    curtainOpen = false;
    field.destroy();
    if (sound) { sound.close(); sound = null; }
    cancelAnimationFrame(raf);
    clearTimeout(cutTimer);
    clearTimeout(nudgeTimer);
  };
}

/** The grain field. Draws only when something drives it: the cold open, the hold, the spring, the mic, the pointer. */
function makeField({ canvas, disc, section, grain, labelEl, store, reduced, coldT0, playCold, waiting = false, fine, signal, plan = COLD_OPEN }) {
  const g = canvas.getContext("2d", { alpha: false, desynchronized: true });
  const grains = grainTable();
  const n = grains.x.length;
  const flare = flareSprite();
  const rt = {
    t0: coldT0, waiting, lastGrain: "",
    w: 1, h: 1, dpr: 1, raf: 0, lastT: 0, cold: playCold, phase: "idle", progress: 0,
    collapse: { x: 0, v: 0 }, target: 0, springing: false,
    px: 0, py: 0, // pointer / tilt, -1..1
    db: null, dbAt: 0, shownDb: null, live: false, lastC: "", labelAt: -Infinity, replaying: false, lastLabel: "", lastStatic: -Infinity, staticTimer: 0,
    discX: 0.5, discY: 0.5, discR: 84, discAt: -Infinity,
    colors: { bg: "#060708", fg: [243, 244, 241], idle: [125, 131, 139] },
  };
  const readColors = () => {
    const cs = getComputedStyle(document.documentElement);
    const bg = cs.getPropertyValue("--color-bg").trim();
    if (bg) rt.colors.bg = bg;
  };

  function resize() {
    const cw = canvas.clientWidth || innerWidth;
    const ch = canvas.clientHeight || innerHeight;
    const side = Math.max(cw, ch, 1);
    rt.dpr = Math.min(devicePixelRatio || 1, 2, Math.max(1, MAX_SIDE_PX / side)); // a full-frame canvas: cap the raster
    const w = Math.max(1, Math.round(cw * rt.dpr));
    const hh = Math.max(1, Math.round(ch * rt.dpr));
    if (w !== canvas.width || hh !== canvas.height) { canvas.width = w; canvas.height = hh; }
    rt.w = w;
    rt.h = hh;
    rt.discAt = -Infinity;
    readColors();
    schedule();
  }

  function discRect(now) {
    if (now - rt.discAt < 250) return;
    rt.discAt = now;
    const r = disc.getBoundingClientRect();
    if (r.width > 0) {
      rt.discX = (r.left + r.width / 2) / (canvas.clientWidth || innerWidth);
      rt.discY = (r.top + r.height / 2) / (canvas.clientHeight || innerHeight);
      rt.discR = r.width / 2;
    }
  }

  function draw(now) {
    rt.raf = 0;
    const dt = rt.lastT ? Math.min(0.1, (now - rt.lastT) / 1000) : 0;
    rt.lastT = now;
    const t = now - rt.t0;
    const w = rt.w;
    const hgt = rt.h;
    const dpr = rt.dpr;
    if (rt.waiting) { // behind the Enter curtain: the black frame, nothing else
      g.fillStyle = rt.colors.bg;
      g.fillRect(0, 0, w, hgt);
      return;
    }

    // what drives the frame
    if (reduced.matches) { rt.collapse.x = rt.target; rt.collapse.v = 0; rt.springing = false; }
    else if (rt.phase === "holding") { rt.collapse.x = rt.target; rt.collapse.v = 0; rt.springing = false; }
    else if (rt.collapse.x !== rt.target || rt.collapse.v !== 0) {
      bounceStep(rt.collapse, rt.target, dt, rt.target === 0 ? { omega: 15, zeta: 0.42 } : { omega: 24, zeta: 1 });
      rt.springing = rt.collapse.x !== rt.target || rt.collapse.v !== 0;
    }
    const c = rt.collapse.x;
    const intro = rt.cold ? introAt(t, plan) : { line: 1, scatter: 1 };
    const ambient = rt.cold && !reduced.matches ? ambientAt(t, plan) : 0;
    const db = rt.live ? releasedDb(rt.db, (now - rt.dbAt) / 1000) : null;
    rt.shownDb = db; // the label prints the same released value the light shows
    const level = rt.live ? levelLight(db) : 0;
    const agitation = Math.max(ambient, level * 0.8);
    if (c > 0 || rt.phase !== "idle") discRect(now);

    g.fillStyle = rt.colors.bg;
    g.fillRect(0, 0, w, hgt);

    const cx = rt.discX * w;
    const cy = rt.discY * hgt;
    const ringR = (rt.discR + 26) * dpr;
    const ts = now / 1000;
    const wave = 0.06 + level * 0.9; // the clear voice's wave: the real level once the mic streams
    const ce = c; // collapse (already eased while holding)
    // the camera: a slow push-in over the cold open, then a little further toward the disc as the ring fills
    const zoom = 1 + PUSH_IN * (rt.cold && !reduced.matches ? pushAt(t, plan) : 1) + HOLD_PUSH * ce;
    const zx = w / 2 + (cx - w / 2) * ce;
    const zy = hgt / 2 + (cy - hgt / 2) * ce;
    for (let b = 0; b < BANDS.length; b++) {
      const depth = BANDS[b];
      const [r, gg, bb] = rt.colors.idle;
      const [fr, fg, fb] = rt.colors.fg;
      const mix = ce * (0.55 + 0.45 * depth);
      const cr = Math.round(r + (fr - r) * mix);
      const cg = Math.round(gg + (fg - gg) * mix);
      const cb = Math.round(bb + (fb - bb) * mix);
      const alpha = (0.34 + 0.56 * depth) * (0.35 + 0.65 * intro.line) * (1 - ce * 0.15) + ce * 0.2;
      g.fillStyle = `rgba(${cr},${cg},${cb},${Math.min(1, alpha).toFixed(3)})`;
      const px = rt.px * 14 * depth * dpr;
      const py = rt.py * 10 * depth * dpr;
      const size = Math.max(1.2, (1 + 1.4 * depth) * dpr) * (1 + ce * 0.4);
      const driftA = (6 + 10 * depth) * dpr * agitation;
      g.beginPath();
      for (let i = b; i < n; i += BANDS.length) {
        // home in chaos, with drift and parallax
        let hx = grains.x[i] * w + px + Math.sin(ts * grains.rate[i] * 2 + grains.phase[i]) * driftA;
        let hy = grains.y[i] * hgt + py + Math.cos(ts * grains.rate[i] * 1.7 + grains.phase[i] * 1.3) * driftA;
        if (intro.scatter < 1) {
          // the hairline: grains sit on the centre line until their turn to leave
          const lx = grains.x[i] * w;
          const half = Math.abs(grains.x[i] - 0.5) * 2;
          if (half > intro.line) continue; // not lit yet: the line is still growing
          const start = grains.order[i] * 0.6;
          const k = Math.max(0, Math.min(1, (intro.scatter - start) / 0.4));
          const e = 1 - (1 - k) ** 3;
          hx = lx + (hx - lx) * e;
          hy = hgt * 0.5 + (hy - hgt * 0.5) * e;
        }
        hx = zx + (hx - zx) * zoom;
        hy = zy + (hy - zy) * zoom;
        let x = hx;
        let y = hy;
        if (ce > 0) {
          // the one clear voice: a ring around the disc, its wave the real level
          const th = (i / n) * Math.PI * 2 * 3 + depth; // three turns keep bands apart
          const rr = ringR + (depth - 0.66) * 14 * dpr + Math.sin(th * 6 + ts * 5) * wave * 10 * dpr;
          const tx = cx + Math.cos(th) * rr;
          const ty = cy + Math.sin(th) * rr;
          x = hx + (tx - hx) * ce;
          y = hy + (ty - hy) * ce;
        }
        g.rect(x, y, size, size);
      }
      g.fill();
    }
    if (intro.line < 1 && intro.scatter === 0) {
      // the hairline itself, while the grains are still on it
      g.fillStyle = `rgba(243,244,241,${(0.5 * intro.line).toFixed(3)})`;
      const len = intro.line * w;
      g.fillRect((w - len) / 2, hgt / 2 - 0.5 * dpr, len, Math.max(1, dpr));
    }
    const fl = rt.cold && !reduced.matches ? flareAt(t, plan) : 0;
    if (fl > 0.002 && flare) {
      // anamorphic flare: one round glow stretched wide along the line, and a hot core; additive, white only
      const len = Math.max(intro.line, 0.2) * w * 1.35;
      const fh = (46 + 40 * fl) * dpr;
      g.globalCompositeOperation = "lighter";
      g.globalAlpha = 0.5 * fl;
      g.drawImage(flare, (w - len) / 2, hgt / 2 - fh / 2, len, fh);
      g.globalAlpha = 0.7 * fl;
      g.drawImage(flare, w / 2 - 34 * dpr, hgt / 2 - 34 * dpr, 68 * dpr, 68 * dpr);
      g.globalAlpha = 1;
      g.globalCompositeOperation = "source-over";
    }
    // film grain follows the room: the murmur's level (the same breakpoints the sound uses), turned down by the hold
    const room = (rt.cold && !reduced.matches ? roomLevelAt(t, plan) : ROOM_BED) * holdMix(ce).room;
    const gr = grainOpacity(room).toFixed(3);
    if (gr !== rt.lastGrain) { rt.lastGrain = gr; grain.style.opacity = gr; }
    const cText = ce.toFixed(3);
    if (cText !== rt.lastC) { rt.lastC = cText; section.style.setProperty("--wl-c", cText); } // a style recalc only on change
    if (now - rt.labelAt >= LABEL_MIN_MS) { rt.labelAt = now; label(); } // the dBFS readout at 10 Hz, not per frame

    const keep = (rt.cold && t < plan.end) || rt.phase === "holding" || rt.springing || (rt.live && now - rt.dbAt < 1200);
    if (keep && !reduced.matches) rt.raf = requestAnimationFrame(draw);
    else rt.lastT = 0;
  }

  function label() {
    const text = fieldLabel({ cold: rt.cold, live: rt.live, replaying: rt.replaying, phase: rt.phase, db: rt.live ? rt.shownDb : null });
    if (text !== rt.lastLabel) { rt.lastLabel = text; labelEl.textContent = text; }
  }

  function schedule() {
    if (rt.raf) return;
    if (reduced.matches) {
      const now = performance.now();
      const wait = REDUCED_MIN_MS - (now - rt.lastStatic);
      if (wait > 0) { if (!rt.staticTimer) rt.staticTimer = setTimeout(() => { rt.staticTimer = 0; schedule(); }, wait); return; }
      rt.lastStatic = now;
    }
    rt.raf = requestAnimationFrame(draw);
  }

  // pointer / tilt parallax: one frame per move, coalesced by rAF; fine pointers only (touch is the hold)
  if (fine && !reduced.matches) {
    addEventListener("pointermove", (e) => {
      rt.px = (e.clientX / innerWidth) * 2 - 1;
      rt.py = (e.clientY / innerHeight) * 2 - 1;
      schedule(); // the field canvas reads rt.px/py on its next frame; the type holds still (no style recalc per move)
    }, { signal, passive: true });
  }
  if (!reduced.matches && "DeviceOrientationEvent" in globalThis && !fine) {
    addEventListener("deviceorientation", (e) => { // fires without a permission prompt where the platform allows it
      if (e.gamma == null || e.beta == null) return;
      rt.px = Math.max(-1, Math.min(1, e.gamma / 30));
      rt.py = Math.max(-1, Math.min(1, (e.beta - 40) / 30));
      schedule();
    }, { signal, passive: true });
  }

  const offLevel = store.on("level", ({ db }) => {
    rt.db = db;
    rt.dbAt = performance.now();
    rt.live = store.state.started && !store.state.replaying;
    schedule();
  });
  const offChange = store.on("change", ({ keys }) => {
    if (!keys.includes("started") && !keys.includes("replaying")) return;
    rt.live = store.state.started && !store.state.replaying;
    rt.replaying = Boolean(store.state.replaying);
    if (!store.state.started) rt.db = null;
    rt.labelAt = -Infinity;
    schedule();
  });
  addEventListener("resize", resize, { signal });
  resize();

  return {
    cold(on) { rt.cold = Boolean(on); if (!on) rt.waiting = false; rt.labelAt = -Infinity; schedule(); },
    begin(t0) { rt.waiting = false; rt.t0 = t0; rt.cold = true; rt.labelAt = -Infinity; schedule(); },
    phase(phase, progress) {
      rt.phase = phase;
      rt.progress = progress;
      rt.target = collapseTarget(phase, progress);
      rt.labelAt = -Infinity;
      schedule();
    },
    destroy() {
      offLevel();
      offChange();
      cancelAnimationFrame(rt.raf);
      clearTimeout(rt.staticTimer);
    },
  };
}
