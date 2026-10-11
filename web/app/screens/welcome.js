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
// Sound: off until the "Sound on" button is pressed (autoplay rules: that press is the gesture, and its tiny
// AudioContext is separate from the session's); the hold then plays the collapse: a band of noise narrowing into one
// tone, cut when the ring completes.
// Reduced motion: no cold open (everything is on screen at once, a 200 ms fade), no drift, no parallax, no grain
// stepping; the hold still fills the ring and the field redraws at most 4 times a second with no easing.

import { MIC_KEY, loadChoice } from "../../console/devices.js";
import { seeded } from "../scene.js";
import { sharedSession } from "./setup.js";
import { haptic } from "../delight.js";

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
 * {type:"release", source} · {type:"cancel"} · {type:"arm"} · {type:"start"} · {type:"started", ok, error}.
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

/** Where a successful Start cuts to: Listen when a microphone choice is already saved, else Setup. */
export function startTarget(storage = globalThis.localStorage) {
  try { return loadChoice(MIC_KEY, storage) ? "listen" : "setup"; } catch { return "setup"; }
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

/** The install hint, or null when the app already runs installed (display-mode standalone / iOS home screen). */
export function installHint({ standalone, ios }) {
  if (standalone) return null;
  return ios ? "For full screen: tap Share, then Add to Home Screen." : "On iPhone: Share, then Add to Home Screen.";
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

/** The hold sound's parameters at ring progress `p`: noise narrowing (Hz, gain) into one tone (Hz, gain). */
export function holdSoundAt(p) {
  const x = Math.max(0, Math.min(1, p));
  return { noiseHz: 140 + 3800 * (1 - x) ** 2, noiseGain: 0.09 * (1 - x), toneHz: 196 + 98 * x, toneGain: 0.05 * x * x };
}

// ---------------------------------------------------------------- hold sound (opt-in, own tiny AudioContext)

/** Created inside a user gesture. Nothing is audible until start(). */
function makeHoldSound(Ctx = globalThis.AudioContext || globalThis.webkitAudioContext) {
  const ctx = new Ctx();
  const noiseBuf = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
  const data = noiseBuf.getChannelData(0);
  const rnd = seeded(0x5afe);
  for (let i = 0; i < data.length; i++) data[i] = rnd() * 2 - 1;
  let nodes = null;
  const stop = (afterS) => {
    const n = nodes;
    if (!n) return;
    nodes = null;
    const at = ctx.currentTime + afterS;
    n.noise.stop(at);
    n.tone.stop(at);
    n.fifth.stop(at);
  };
  return {
    ctx,
    start() {
      if (ctx.state === "suspended") ctx.resume().catch(() => undefined);
      stop(0);
      const t = ctx.currentTime;
      const noise = ctx.createBufferSource();
      noise.buffer = noiseBuf;
      noise.loop = true;
      const filter = ctx.createBiquadFilter();
      filter.type = "lowpass";
      filter.Q.value = 0.8;
      const noiseGain = ctx.createGain();
      noiseGain.gain.value = 0;
      noise.connect(filter).connect(noiseGain).connect(ctx.destination);
      const tone = ctx.createOscillator();
      tone.type = "sine";
      const toneGain = ctx.createGain();
      toneGain.gain.value = 0;
      tone.connect(toneGain).connect(ctx.destination);
      const fifth = ctx.createOscillator(); // joins only at the resolve
      fifth.type = "sine";
      const fifthGain = ctx.createGain();
      fifthGain.gain.value = 0;
      fifth.connect(fifthGain).connect(ctx.destination);
      noise.start(t);
      tone.start(t);
      fifth.start(t);
      nodes = { noise, filter, noiseGain, tone, toneGain, fifth, fifthGain };
      this.set(0);
    },
    set(p) {
      const n = nodes;
      if (!n) return;
      const { noiseHz, noiseGain, toneHz, toneGain } = holdSoundAt(p);
      const t = ctx.currentTime;
      n.filter.frequency.setTargetAtTime(noiseHz, t, 0.03);
      n.noiseGain.gain.setTargetAtTime(noiseGain, t, 0.03);
      n.tone.frequency.setTargetAtTime(toneHz, t, 0.03);
      n.toneGain.gain.setTargetAtTime(toneGain, t, 0.03);
    },
    release() { // let go early: the room comes back for a moment, then fades
      const n = nodes;
      if (!n) return;
      const t = ctx.currentTime;
      n.filter.frequency.setTargetAtTime(2400, t, 0.08);
      n.noiseGain.gain.setTargetAtTime(0.05, t, 0.05);
      n.noiseGain.gain.setTargetAtTime(0, t + 0.15, 0.1);
      n.toneGain.gain.setTargetAtTime(0, t, 0.05);
      stop(0.7);
    },
    resolve() { // the ring completed: the noise is cut, one clear tone rings out with its fifth
      const n = nodes;
      if (!n) return;
      const t = ctx.currentTime;
      n.noiseGain.gain.setTargetAtTime(0, t, 0.02);
      n.tone.frequency.setTargetAtTime(294, t, 0.02);
      n.toneGain.gain.setTargetAtTime(0.07, t, 0.02);
      n.toneGain.gain.setTargetAtTime(0, t + 0.12, 0.25);
      n.fifth.frequency.setValueAtTime(441, t);
      n.fifthGain.gain.setTargetAtTime(0.035, t + 0.04, 0.02);
      n.fifthGain.gain.setTargetAtTime(0, t + 0.16, 0.3);
      stop(1.4);
    },
    close() { stop(0); ctx.close().catch(() => undefined); },
  };
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

let coldOpenPlayed = false; // once per page load

export function mount(root, { app, signal }) {
  const session = sharedSession(app);
  const store = session.store;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const standalone = matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  const ios = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const fine = matchMedia("(hover: hover) and (pointer: fine)").matches;
  // the stage laptop needs no install line (T-365 P2-1): phones and tablets only
  const hint = ios || matchMedia("(pointer: coarse)").matches ? installHint({ standalone, ios }) : null;
  const playCold = !coldOpenPlayed && !reduced.matches && !document.hidden;
  coldOpenPlayed = true;

  const label = h("span", { class: "wl-disc-label" });
  const sub = h("span", { class: "wl-disc-sub" });
  const ripple = h("span", { class: "wl-ripple", "aria-hidden": "true" });
  const disc = h("button", { type: "button", class: "wl-disc", id: "start", "data-phase": "idle", "aria-describedby": "wl-facts wl-problem" },
    h("span", { class: "wl-ring", "aria-hidden": "true" }), ripple, label, sub);
  const problem = h("p", { class: "wl-problem", id: "wl-problem", role: "alert" });
  const status = h("p", { class: "wl-sr", role: "status" });
  const fieldLabelEl = h("span", { class: "wl-field-label", "aria-hidden": "true" });
  const canvas = h("canvas", { class: "wl-field", "aria-hidden": "true" });
  const grain = h("div", { class: "wl-grain", "aria-hidden": "true" });
  const vignette = h("div", { class: "wl-vignette", "aria-hidden": "true" });
  let soundOn = false;
  try { soundOn = localStorage.getItem(SOUND_KEY) === "1"; } catch { /* private mode */ }
  const soundBtn = h("button", { type: "button", class: "wl-sound", "aria-pressed": String(soundOn) }, svgIcon(ICON_SPEAKER),
    h("span", { class: "wl-sound-text", text: soundOn ? "Sound on" : "Sound off" }));
  const skipBtn = h("button", { type: "button", class: "wl-skip", text: "Skip intro" });

  const hero = h("h1", { id: "wl-title", class: "wl-hero" });
  const words = heroWords();
  words.forEach((w, i) => {
    const outer = h("span", { class: "wl-w", style: `--wl-d: ${wordDelay(i)}ms` }, h("span", { class: "wl-wi", text: w }));
    hero.append(outer, i < words.length - 1 ? " " : "");
  });

  root.classList.add("screen-welcome");
  const section = h("section", { class: "wl", "data-phase": "idle", "data-cold": playCold ? "play" : "done", "aria-labelledby": "wl-title" },
    canvas, vignette, grain, fieldLabelEl,
    h("div", { class: "wl-main" },
      hero,
      h("div", { class: "wl-row" },
        h("div", { class: "wl-start" }, disc,
          h("p", { class: "wl-key" }, "or hold ", h("kbd", { text: "Space" }))),
        h("div", { class: "wl-facts", id: "wl-facts" },
          h("p", { text: "Pick one voice. The room goes quiet around it." }),
          h("p", {}, h("strong", { text: "About 1.5 s behind" }), ", like live captions."),
          hint ? h("p", { class: "wl-install", text: hint }) : null,
          problem)),
      status),
    soundBtn, // outside .wl-main: that block translates with the pointer, which would anchor a fixed child to it
    playCold ? skipBtn : null);
  root.append(section);
  grain.style.backgroundImage = grainTile();
  if (playCold) document.body.dataset.wlCold = "1";

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
    if (sound && s.phase === "holding") sound.set(s.progress);
    if (prev && prev.phase === s.phase) return;
    disc.dataset.phase = s.phase;
    section.dataset.phase = s.phase;
    const [a, b] = discCopy(s.phase);
    label.textContent = a;
    sub.textContent = b;
    disc.setAttribute("aria-busy", String(s.phase === "starting" || s.phase === "full"));
    problem.textContent = s.phase === "failed" ? s.error : "";
    status.textContent = s.phase === "holding" ? "Holding. Keep holding to start." : s.phase === "armed" ? "Let go to start."
      : s.phase === "starting" ? "Opening the microphone." : s.phase === "done" ? "Microphone open." : "";
    if (sound) {
      if (s.phase === "holding" && prev?.phase !== "holding") sound.start();
      else if (s.phase === "idle" && prev?.phase === "holding") sound.release();
      else if (s.phase === "full") sound.resolve();
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
    if (s.phase === "full") haptic("start", { reduced: reduced.matches });
    if (s.phase === "full") dispatch({ type: activationLive() ? "start" : "arm" });
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
    target = startTarget();
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

  function cut() {
    cutTimer = setTimeout(() => { if (!signal.aborted) app.go(target); }, reduced.matches ? 0 : CUT_HOLD_MS);
  }

  const press = (source, e) => {
    if (s.phase !== "idle" && s.phase !== "failed") return false;
    skipCold();
    if (soundOn && !sound) { try { sound = makeHoldSound(); } catch { sound = null; } } // inside the gesture
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
  disc.addEventListener("click", (e) => e.preventDefault(), { signal }); // a click is not a hold
  disc.addEventListener("contextmenu", (e) => e.preventDefault(), { signal }); // long-press menu on touch

  // keys: on the disc, or on the page when nothing focusable has focus (the judge can just hold Space)
  const keyTarget = (t) => t === disc || t === document.body || t === document.documentElement || t?.id === "stage" || root.contains(t) && !t.closest("a, input, select, textarea, button:not(#start)");
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
  document.addEventListener("visibilitychange", () => { if (document.hidden) { dispatch({ type: "cancel" }); skipCold(); } }, { signal });

  // ---- sound on/off: the press is the gesture that may create the context (nothing plays until a hold)
  soundBtn.addEventListener("click", () => {
    soundOn = !soundOn;
    soundBtn.setAttribute("aria-pressed", String(soundOn));
    soundBtn.querySelector(".wl-sound-text").textContent = soundOn ? "Sound on" : "Sound off";
    try { localStorage.setItem(SOUND_KEY, soundOn ? "1" : "0"); } catch { /* private mode */ }
    if (soundOn && !sound) { try { sound = makeHoldSound(); } catch { sound = null; } }
    if (!soundOn && sound) { sound.close(); sound = null; }
  }, { signal });

  // ---- the cold open: CSS owns the type and the UI (delays from COLD_OPEN); the canvas owns the light
  let coldDone = !playCold;
  const coldT0 = performance.now();
  function skipCold() {
    if (coldDone) return;
    coldDone = true;
    section.dataset.cold = "done";
    delete document.body.dataset.wlCold;
    skipBtn.remove();
    field.cold(false);
  }
  skipBtn.addEventListener("click", skipCold, { signal });
  if (playCold) {
    addEventListener("pointerdown", (e) => { if (e.target !== disc && !disc.contains(e.target)) skipCold(); }, { signal, capture: true });
    setTimeout(() => { if (!signal.aborted && !coldDone) { skipCold(); } }, COLD_OPEN.end);
  }

  // ---- the field: Canvas 2D, 720 grains in three depth bands; still unless something drives it
  const field = makeField({ canvas, disc, section, labelEl: fieldLabelEl, store, reduced, coldT0, playCold, fine, signal });
  field.cold(playCold);
  field.phase("idle", 0);

  render(null);

  return () => {
    delete document.body.dataset.wlCold;
    field.destroy();
    if (sound) { sound.close(); sound = null; }
    cancelAnimationFrame(raf);
    clearTimeout(cutTimer);
  };
}

/** The grain field. Draws only when something drives it: the cold open, the hold, the spring, the mic, the pointer. */
function makeField({ canvas, disc, section, labelEl, store, reduced, coldT0, playCold, fine, signal }) {
  const g = canvas.getContext("2d", { alpha: false, desynchronized: true });
  const grains = grainTable();
  const n = grains.x.length;
  const rt = {
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
    const t = now - coldT0;
    const w = rt.w;
    const hgt = rt.h;
    const dpr = rt.dpr;

    // what drives the frame
    if (reduced.matches) { rt.collapse.x = rt.target; rt.collapse.v = 0; rt.springing = false; }
    else if (rt.phase === "holding") { rt.collapse.x = rt.target; rt.collapse.v = 0; rt.springing = false; }
    else if (rt.collapse.x !== rt.target || rt.collapse.v !== 0) {
      bounceStep(rt.collapse, rt.target, dt, rt.target === 0 ? { omega: 15, zeta: 0.42 } : { omega: 24, zeta: 1 });
      rt.springing = rt.collapse.x !== rt.target || rt.collapse.v !== 0;
    }
    const c = rt.collapse.x;
    const intro = rt.cold ? introAt(t) : { line: 1, scatter: 1 };
    const ambient = rt.cold && !reduced.matches ? ambientAt(t) : 0;
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
    const cText = ce.toFixed(3);
    if (cText !== rt.lastC) { rt.lastC = cText; section.style.setProperty("--wl-c", cText); } // a style recalc only on change
    if (now - rt.labelAt >= LABEL_MIN_MS) { rt.labelAt = now; label(); } // the dBFS readout at 10 Hz, not per frame

    const keep = (rt.cold && t < COLD_OPEN.end) || rt.phase === "holding" || rt.springing || (rt.live && now - rt.dbAt < 1200);
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
      section.style.setProperty("--wl-px", rt.px.toFixed(3));
      section.style.setProperty("--wl-py", rt.py.toFixed(3));
      schedule();
    }, { signal, passive: true });
  }
  if (!reduced.matches && "DeviceOrientationEvent" in globalThis && !fine) {
    addEventListener("deviceorientation", (e) => { // fires without a permission prompt where the platform allows it
      if (e.gamma == null || e.beta == null) return;
      rt.px = Math.max(-1, Math.min(1, e.gamma / 30));
      rt.py = Math.max(-1, Math.min(1, (e.beta - 40) / 30));
      section.style.setProperty("--wl-px", rt.px.toFixed(3));
      section.style.setProperty("--wl-py", rt.py.toFixed(3));
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
    cold(on) { rt.cold = Boolean(on); rt.labelAt = -Infinity; schedule(); },
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
