// Sound scene (M14, T-349; design pass C, T-390): the one moving object of /app, a Canvas 2D field behind the stage.
// Spec: web/app/DESIGN_SPEC.md sections 4 (honesty) and 6 (motion); tokens: web/app/tokens.css "Sound scene contract";
// the borrowed patterns: docs/AWWWARDS_BORROW.md 3 (grain ring) and 8 (light scene).
//
// Two layers of light, back to front:
//   - the room: ~540 grains on a wide ring in three depth bands (far bands smaller and dimmer). Their light and
//     spread follow the real mic peak. When isolation engages (Selective audible) the room recedes: it is pulled
//     back in depth, far bands more than near ones, and dims, by how sure the engine is (tse.confidence);
//   - the target voice: a calm form at the centre (a wide pool of light, a focus ring, a bright core). Its light
//     follows the stream the engine plays (store "playback", the peak of each audio-out frame; T-403: it falls in
//     the target's pauses, which agc.level_db did not), and the focus ring is
//     a lens pull on tse.confidence: a sharp single ring when sure, a soft doubled ring when not. The gate state
//     (tse.gated / soft_gated, as the SOLO / OVERLAP-DUCKED / UNSURE pill) dims the form.
// Inputs, all real:
//   - the mic peak (store "level", the frames actually sent, or the replay clip's frames while replaying), with the
//     meter's 24 dB/s release; when the level events stop, the drawn level keeps releasing to the floor;
//   - the output peak (store "playback", the engine's audio-out frames as they are queued to play, same release);
//   - telemetry: `ab` (amber RAW / green Focus), `tse` (confidence, gated, soft_gated), Quiet (dim),
//     REPLAY (violet, dashed focus ring), Release (neutral white, the form an empty outline);
//   - the pill: flat and grey before Start (`micPill` idle) and when the mic stops streaming (`micPill` nomic).
// Idle is designed but still: nothing breathes, pulses, drifts or loops on a timer. A ripple travels out only after
// a real level onset. Every change of look eases with a critically damped spring (no overshoot), interruptible.
//
// Draw time is measured per drawn frame (performance.now() around the draw) and so is the rAF interval; both are
// kept for the last 10 s and exposed as `scene.stats()`, `hearingApp.scene.stats()` and an `app.emit("scene", stats)`
// once a second, for Diagnostics (which reads `p95Ms` / `lastMs` / `frames`). If 30 drawn frames in a row exceed the
// draw budget the scene drops to the light scene (one ring stroke for the room, no grains), reported as `light`.
// Reduced motion (`--scene-animate: 0`): no rAF loop and no radius motion; the field is redrawn statically, at most
// 4 times a second, with light (alpha) only; no ripples; looks switch without easing.
//
// The DOM: <div class="scene"><canvas></canvas><span class="scene-label"></span></div>, fixed behind the stage,
// placed by motion.css (per screen and width). It has its own view-transition-name, so screen cuts never freeze it.

import { store as appStore } from "./store.js";
import { gateState } from "../console/gate-state.js";

export const WINDOW_MS = 10_000; // stats window (the acceptance's "over 10 s")
export const FRAME_BUDGET_MS = 16.7; // one 60 Hz frame
const MAX_SIDE_PX = 1400; // backing-store cap: soft light needs no more, and raster cost grows with pixels (2 missed
// vsyncs in 10 s at 1886 px, none at 1400 px; Playwright 1920x1080 DPR 2, T-349)
export const GRAINS = 540; // the room's grains: three depth bands of 180, one path per band
const MAX_RIPPLES = 6;
const RIPPLE_S = 1.4;
const ONSET_DB = 6; // a ripple starts when the peak jumps this far above the released level
const MIN_DRAW_MS = 1000 / 60 - 2; // draw at most at 60 Hz: a 120 Hz display (ProMotion) would double the raster cost for
// level frames that arrive every 20 ms anyway (T-379)
const COLOR_TAU_S = 0.05; // colour follows a look change within ~140 ms (--dur-wash)
export const SPRING_S = { look: 0.26, recede: 0.52, conf: 0.3, energy: 0.22 }; // critically damped, settle times (s):
// --dur-ui for a look, the film cut (--dur-cut 520 ms) for the room receding, about 3 telemetry ticks for confidence
export const LIGHT_AFTER = 30; // consecutive over-budget draws before the light scene (AWWWARDS_BORROW pattern 8)
export const REDUCED_MIN_MS = 250; // reduced motion: at most 4 static redraws a second

/** Defaults when tokens.css is not linked (tests, an old cache). Same numbers as the tokens. */
export const DEFAULTS = {
  releaseDbPerS: 24, floorDb: -60, ceilingDb: -6, dprMax: 2, budgetMs: 4, animate: 1, quietDim: 0.18,
  colors: { focus: "#4bf08e", raw: "#ffb23e", replay: "#b9a6ff", idle: "#3a3f45", faint: "#6c7279", fg: "#f3f4f1",
    bad: "#ff7266" },
};

/**
 * Pure: the field's look for an app state (store.state). Flat looks draw the floor level whatever the mic does.
 *   idle      before Start (micPill idle)                 flat, grey, label
 *   nomic     the mic stopped streaming (micPill nomic)  flat, grey, red label
 *   replay    the server replays a clip                   violet dashed outlines, the clip's level
 *   lost      socket reconnecting                         grey (desaturated), the mic's level
 *   quiet     Quiet mode                                  dimmed to --scene-quiet-dim, neutral
 *   raw       RAW is audible (telemetry.ab)               amber, whatever the target state (you hear the room)
 *   searching Release, or no target yet (pill "look")     neutral white
 *   loading   the saved voice is loading (pill loading)   grey, the mic's level (DESIGN_SPEC "loading voice")
 *   focus     Selective is audible                        green, rings sharpen with tse.confidence
 */
export function sceneLook(s) {
  if (!s) return "idle";
  if (s.replaying) return "replay";
  if (!s.started || s.pill?.state === "nomic") return s.pill?.state === "nomic" ? "nomic" : "idle";
  if (s.connection !== "open") return "lost";
  if (s.mode === "quiet") return "quiet";
  if ((s.telemetry?.ab || s.ab) === "raw") return "raw";
  if (s.released || s.pill?.state === "look") return "searching";
  if (s.pill?.state === "loading") return "loading";
  return "focus";
}

const FLAT = new Set(["idle", "nomic"]);
const LABELS = { idle: "idle · mic off", nomic: "mic not streaming", replay: "replay · clip level", lost: "reconnecting",
  loading: "loading voice" };

/**
 * Pure: colour keys, opacity and flags for a look. `room` / `form` are the colour keys of the room's grains and the
 * target's form; `formWeight` is how present the form is (RAW: a ghost, you hear the room; Release: an empty outline).
 */
export function lookStyle(look, cfg = DEFAULTS) {
  const flat = FLAT.has(look);
  const color = { focus: "focus", raw: "raw", replay: "replay", searching: "fg", quiet: "fg", loading: "faint",
    lost: "faint" }[look] || "idle";
  const room = { focus: "fg", raw: "raw", replay: "replay", searching: "fg", quiet: "fg", loading: "faint",
    lost: "faint" }[look] || "idle";
  const formWeight = { focus: 1, replay: 0.8, quiet: 0.6, loading: 0.45, lost: 0.4, raw: 0.22, searching: 0.35 }[look]
    ?? 0.3;
  const alpha = look === "quiet" ? cfg.quietDim : flat ? 0.75 : look === "lost" ? 0.55 : look === "loading" ? 0.8 : 1;
  return { look, flat, color, room, form: color, formWeight, filled: !flat && look !== "searching", alpha,
    dashed: look === "replay", label: LABELS[look] || "", bad: look === "nomic" };
}

/**
 * Pure: how far the room recedes (0 = in front, full light; 1 = pulled all the way back). Only while the chosen
 * voice is what plays: in Focus by the engine's confidence (unknown counts as the floor), a little while it looks
 * for someone. RAW, Quiet, idle and the rest keep the room where it is.
 */
export function recedeTarget(look, confidence) {
  if (look === "focus") {
    const c = typeof confidence === "number" && Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence)) : 0;
    return 0.45 + 0.55 * c;
  }
  if (look === "searching" || look === "loading") return 0.15;
  return 0;
}

/** Pure: the form's light under the gate state (console/gate-state.js): SOLO full, OVERLAP-DUCKED half, UNSURE low. */
export function gateWeight(telemetry) {
  const g = gateState(telemetry);
  return g === "ducked" ? 0.55 : g === "unsure" ? 0.3 : 1;
}

/**
 * Pure: the level (dBFS) of what the engine plays, `dtS` seconds after the last measured audio-out peak `db`
 * (store "playback", the meter's release), or null before any output arrived. It drops when the chosen person
 * stops talking; telemetry agc.level_db holds through pauses, so the form no longer reads it.
 */
export function streamDb(db, dtS, cfg = DEFAULTS) {
  return typeof db === "number" && Number.isFinite(db) ? releasedDb(db, dtS, cfg) : null;
}

/**
 * Pure: one step of a critically damped spring towards `target` (closed form, stable for any dt, never overshoots
 * from rest). `settleS` is the time to settle within 1 %. Mutates and returns `st` = {x, v}.
 */
export function springStep(st, target, dtS, settleS) {
  if (!(settleS > 0) || !(dtS > 0)) { if (!(settleS > 0)) { st.x = target; st.v = 0; } return st; }
  const w = 6.64 / settleS; // e^(-wT)(1 + wT) = 0.01
  const x0 = st.x - target;
  const k = st.v + w * x0;
  const e = Math.exp(-w * dtS);
  st.x = target + (x0 + k * dtS) * e;
  st.v = (k - w * (x0 + k * dtS)) * e;
  if (Math.abs(st.x - target) < 1e-4 && Math.abs(st.v) < 1e-3) { st.x = target; st.v = 0; }
  return st;
}

/** Pure: a seeded PRNG (mulberry32), so the grains sit in the same place on every load and every frame. */
export function seeded(seed = 0x5eed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Pure: the room's grain table, built once. Bands 0 (near) / 1 / 2 (far) are contiguous thirds. Per grain: angle
 * (cos, sin), radius as a fraction of the outer ring (a soft band around 0.74), and a signed spread weight (how far
 * it moves out or in with the level).
 */
export function grainTable(n = GRAINS, seed = 0x5eed) {
  const rnd = seeded(seed);
  const cos = new Float32Array(n), sin = new Float32Array(n), rho = new Float32Array(n), spread = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = rnd() * Math.PI * 2;
    cos[i] = Math.cos(a);
    sin[i] = Math.sin(a);
    const gauss = (rnd() + rnd() + rnd() - 1.5) / 1.5; // -1..1, peaked at 0
    rho[i] = Math.max(0.3, Math.min(1, 0.74 + 0.26 * gauss));
    spread[i] = rnd() * 2 - 1;
  }
  return { n, band: Math.floor(n / 3), cos, sin, rho, spread };
}

/** Pure: the drawn level in dB `dtS` seconds after a peak of `db` (the meter's release), floored. */
export function releasedDb(db, dtS, cfg = DEFAULTS) {
  if (typeof db !== "number" || !Number.isFinite(db)) return cfg.floorDb;
  return Math.max(cfg.floorDb, db - cfg.releaseDbPerS * Math.max(0, dtS));
}

/** Pure: dBFS -> 0..1 between the scene floor and ceiling. */
export function levelUnit(db, cfg = DEFAULTS) {
  const u = (db - cfg.floorDb) / (cfg.ceilingDb - cfg.floorDb);
  return u > 0 ? (u < 1 ? u : 1) : 0;
}

/** Draw-time and frame-interval statistics over a sliding window (ms). Pure, injectable clock. */
export class FrameStats {
  constructor(windowMs = WINDOW_MS) {
    this.windowMs = windowMs;
    this.draws = []; // [atMs, drawMs]
    this.intervals = []; // [atMs, intervalMs]
    this.lastFrameAt = null;
    this.totalDraws = 0;
  }

  draw(atMs, ms) {
    this.draws.push([atMs, ms]);
    this.totalDraws++;
    this.trim(atMs);
  }

  frame(atMs) {
    if (this.lastFrameAt !== null) this.intervals.push([atMs, atMs - this.lastFrameAt]);
    this.lastFrameAt = atMs;
    this.trim(atMs);
  }

  /** A gap in the loop that is not a frame (tab hidden, loop stopped): do not count it as a long frame. */
  pause() { this.lastFrameAt = null; }

  trim(nowMs) {
    const cut = nowMs - this.windowMs;
    while (this.draws.length && this.draws[0][0] < cut) this.draws.shift();
    while (this.intervals.length && this.intervals[0][0] < cut) this.intervals.shift();
  }

  summary(budgetMs = DEFAULTS.budgetMs) {
    const d = this.draws.map((x) => x[1]);
    const iv = this.intervals.map((x) => x[1]);
    return {
      windowS: this.windowMs / 1000,
      draws: d.length,
      totalDraws: this.totalDraws,
      drawLastMs: d.length ? round(d[d.length - 1]) : null,
      drawMeanMs: d.length ? round(d.reduce((a, b) => a + b, 0) / d.length) : null,
      drawP95Ms: d.length ? round(pct(d, 0.95)) : null,
      drawMaxMs: d.length ? round(Math.max(...d)) : null,
      overBudget: d.filter((x) => x > budgetMs).length,
      budgetMs,
      frames: iv.length,
      intervalP95Ms: iv.length ? round(pct(iv, 0.95)) : null,
      intervalMaxMs: iv.length ? round(Math.max(...iv)) : null,
      longFrames: iv.filter((x) => x > FRAME_BUDGET_MS * 1.5).length, // a missed vsync at 60 Hz
    };
  }
}

function pct(values, q) {
  const s = values.slice().sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

function round(x) { return Math.round(x * 1000) / 1000; }

/** Parse "#rgb", "#rrggbb" or "rgb(r g b)" / "rgb(r, g, b)" into [r, g, b]. */
export function parseColor(text, fallback = [128, 128, 128]) {
  const s = String(text || "").trim();
  let m = s.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (m) {
    const h = m[1].length === 3 ? m[1].replace(/./g, "$&$&") : m[1];
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  }
  m = s.match(/^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i);
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : fallback;
}

/** Read the scene contract from the tokens (computed style), falling back to DEFAULTS. */
export function readConfig(el) {
  const cs = getComputedStyle(el);
  const num = (name, d) => {
    const v = parseFloat(cs.getPropertyValue(name));
    return Number.isFinite(v) ? v : d;
  };
  const col = (name, d) => parseColor(cs.getPropertyValue(name), parseColor(d));
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  return {
    releaseDbPerS: num("--scene-release-db-per-s", DEFAULTS.releaseDbPerS),
    floorDb: num("--scene-floor-db", DEFAULTS.floorDb),
    ceilingDb: num("--scene-ceiling-db", DEFAULTS.ceilingDb),
    dprMax: num("--scene-dpr-max", DEFAULTS.dprMax),
    budgetMs: num("--scene-budget-ms", DEFAULTS.budgetMs),
    animate: reduce ? 0 : num("--scene-animate", DEFAULTS.animate),
    quietDim: num("--scene-quiet-dim", DEFAULTS.quietDim),
    colors: Object.fromEntries(Object.entries(DEFAULTS.colors).map(([k, d]) => [k, col(`--color-${k}`, d)])),
  };
}

let instance = null;

/**
 * Mount the scene once (idempotent: later calls return the same instance). motion.js calls it at boot; a screen
 * may call it too. Returns {element, canvas, stats(), look(), redraw(), destroy()}.
 */
export function mountScene({ store = appStore, app = globalThis.hearingApp, host = document.body } = {}) {
  if (instance) return instance;
  const root = document.createElement("div");
  root.className = "scene";
  root.setAttribute("aria-hidden", "true");
  const canvas = document.createElement("canvas");
  const label = document.createElement("span");
  label.className = "scene-label";
  root.append(canvas, label);
  host.prepend(root);
  document.documentElement.classList.add("has-scene");
  const g = canvas.getContext("2d", { alpha: true, desynchronized: true });

  let cfg = readConfig(root);
  const stats = new FrameStats();
  const grains = grainTable();
  const rt = {
    w: 0, h: 0, dpr: 1, raf: 0, dirty: true,
    peakDb: cfg.floorDb, peakAt: 0, lastDrawnDb: cfg.floorDb, playDb: null, playAt: 0,
    room: null, form: null, alpha: { x: 0, v: 0 }, weight: { x: 0, v: 0 }, gate: { x: 1, v: 0 },
    recede: { x: 0, v: 0 }, conf: { x: 0, v: 0 }, energy: { x: 0, v: 0 },
    ripples: [], lastT: 0, light: false, overRun: 0, lastStaticAt: -Infinity, staticTimer: 0,
    sig: new Float64Array(10), // what the last drawn frame showed: an unchanged field is not redrawn
  };

  function resize() {
    cfg = readConfig(root);
    const side = Math.max(canvas.clientWidth, canvas.clientHeight, 1);
    rt.dpr = Math.min(globalThis.devicePixelRatio || 1, cfg.dprMax, Math.max(1, MAX_SIDE_PX / side));
    const w = Math.max(1, Math.round(canvas.clientWidth * rt.dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * rt.dpr));
    if (w !== canvas.width || h !== canvas.height) {
      canvas.width = w;
      canvas.height = h;
    }
    rt.w = w;
    rt.h = h;
    rt.dirty = true;
    schedule(true);
  }

  // ---- inputs (all real) ----
  const offs = [
    store.on("level", ({ db }) => {
      const now = performance.now();
      const released = releasedDb(rt.peakDb, (now - rt.peakAt) / 1000, cfg);
      const style = lookStyle(sceneLook(store.state), cfg);
      if (cfg.animate && !rt.light && !style.flat && db - released >= ONSET_DB && levelUnit(db, cfg) > 0.15) {
        rt.ripples.push({ t0: now, u: levelUnit(db, cfg) });
        if (rt.ripples.length > MAX_RIPPLES) rt.ripples.shift();
      }
      rt.peakDb = Math.max(db, released);
      rt.peakAt = now;
      schedule();
    }),
    store.on("playback", ({ db }) => {
      const now = performance.now();
      rt.playDb = rt.playDb === null ? db : Math.max(db, releasedDb(rt.playDb, (now - rt.playAt) / 1000, cfg));
      rt.playAt = now;
      schedule();
    }),
    store.on("telemetry", () => { rt.dirty = true; schedule(); }),
    store.on("change", ({ keys }) => {
      if (keys.some((k) => k === "pill" || k === "mode" || k === "replaying" || k === "started" || k === "connection"
        || k === "released" || k === "ab" || k === "truth")) {
        rt.dirty = true;
        schedule(true);
      }
    }),
  ];

  const css = (c, a) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a > 0 ? (a < 1 ? a.toFixed(3) : 1) : 0})`;
  function easeColor(cur, target, k) {
    if (!cur) return Float32Array.from(target);
    for (let i = 0; i < 3; i++) cur[i] += (target[i] - cur[i]) * k;
    return cur;
  }

  // ---- draw ----
  function draw(now) {
    const t0 = performance.now();
    const s = store.state;
    const look = sceneLook(s);
    const style = lookStyle(look, cfg);
    const dt = rt.lastT ? Math.min(0.1, (now - rt.lastT) / 1000) : 0;
    rt.lastT = now;
    const instant = !cfg.animate || dt === 0;
    const tel = s.telemetry;
    const conf = typeof s.truth?.confidence === "number" && !style.flat ? Math.max(0, Math.min(1, s.truth.confidence)) : 0;

    // ease every look change (springs: interruptible, no overshoot); reduced motion jumps
    const kc = instant ? 1 : 1 - Math.exp(-dt / COLOR_TAU_S);
    rt.room = easeColor(rt.room, cfg.colors[style.room], kc);
    rt.form = easeColor(rt.form, cfg.colors[style.form], kc);
    const step = (st, target, settle) => springStep(st, target, dt, instant ? 0 : settle);
    step(rt.alpha, style.alpha, SPRING_S.look);
    step(rt.weight, style.formWeight, SPRING_S.look);
    step(rt.gate, look === "focus" ? gateWeight(tel) : 1, SPRING_S.look);
    step(rt.recede, recedeTarget(look, s.truth?.confidence), SPRING_S.recede);
    step(rt.conf, conf, SPRING_S.conf);
    const sDb = streamDb(rt.playDb, (now - rt.playAt) / 1000, cfg);
    const db = style.flat ? cfg.floorDb : releasedDb(rt.peakDb, (now - rt.peakAt) / 1000, cfg);
    const u = levelUnit(db, cfg);
    // the form's light: the stream the engine plays while the chosen voice is audible; otherwise the room's level
    // reaches it only as faintly as formWeight lets it (RAW: you hear the room, not the person)
    step(rt.energy, style.flat ? 0 : look === "focus" && sDb !== null ? levelUnit(sDb, cfg) : u, SPRING_S.energy);
    rt.lastDrawnDb = db;
    if (style.flat || !cfg.animate || rt.light) rt.ripples.length = 0;
    while (rt.ripples.length && now - rt.ripples[0].t0 >= RIPPLE_S * 1000) rt.ripples.shift();

    // nothing changed since the last frame: skip the draw (an unchanged field does not move)
    const sig = rt.sig;
    let changed = rt.dirty || rt.ripples.length > 0;
    const put = (i, v) => { if (sig[i] !== v) { changed = true; sig[i] = v; } };
    put(0, Math.round(u * 512));
    put(1, Math.round(rt.alpha.x * 400));
    put(2, Math.round(rt.weight.x * 400));
    put(3, Math.round(rt.gate.x * 400));
    put(4, Math.round(rt.recede.x * 400));
    put(5, Math.round(rt.conf.x * 400));
    put(6, Math.round(rt.energy.x * 400));
    put(7, (rt.room[0] + rt.room[1] * 7 + rt.form[2] * 13 + rt.form[0] * 17) | 0);
    put(8, rt.w + rt.h * 1e5);
    put(9, rt.light ? 1 : 0);
    if (!changed) return false;
    rt.dirty = false;
    if (label.textContent !== style.label) label.textContent = style.label;
    label.classList.toggle("is-bad", style.bad);
    label.dataset.look = look;
    root.dataset.look = look;
    root.dataset.light = rt.light ? "1" : "";

    const { w, h, dpr } = rt;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.globalCompositeOperation = "source-over";
    g.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const R = Math.min(w, h) * 0.46; // outer ring; the canvas box leaves room for the halo (motion.css)
    const A = rt.alpha.x;
    const rec = rt.recede.x;
    const moves = cfg.animate ? 1 : 0; // reduced motion: light only, no radius motion
    const lvl = style.flat ? 0 : u;

    // 1. the room's own light: a wide, faint band under the grains; it goes out as the room recedes
    if (!style.flat) {
      const a = A * (0.05 + 0.13 * lvl) * (1 - 0.8 * rec);
      if (a > 0.002) {
        const grad = g.createRadialGradient(cx, cy, R * 0.32, cx, cy, R * 1.04);
        grad.addColorStop(0, css(rt.room, 0));
        grad.addColorStop(0.5, css(rt.room, a));
        grad.addColorStop(1, css(rt.room, 0));
        g.fillStyle = grad;
        g.fillRect(cx - R * 1.04, cy - R * 1.04, R * 2.08, R * 2.08);
      }
    }

    // 2. the room: three depth bands, far first. Receding pulls each band back (far ones more) and dims it
    const swell = 1 + 0.05 * lvl * moves;
    if (rt.light) {
      g.lineWidth = dpr * 1.5;
      g.strokeStyle = css(rt.room, A * (0.25 + 0.55 * lvl) * (1 - 0.7 * rec));
      g.beginPath();
      g.arc(cx, cy, R * 0.74 * swell * (1 - 0.12 * rec), 0, Math.PI * 2);
      g.stroke();
    } else {
      const { band, cos, sin, rho, spread } = grains;
      for (let b = 2; b >= 0; b--) {
        const depth = (1 - rec * (0.08 + 0.07 * b)) * swell; // far bands recede further: parallax
        const size = dpr * (1.9 - 0.45 * b) * (1 - 0.3 * rec);
        const a = A * (style.flat ? 0.3 - 0.07 * b : (0.2 + 0.8 * lvl) * (0.9 - 0.24 * b)) * (1 - 0.72 * rec);
        if (a < 0.004) continue;
        const push = R * 0.09 * lvl * moves * (1 - 0.5 * rec);
        g.fillStyle = css(rt.room, a);
        g.beginPath();
        const end = b === 2 ? grains.n : (b + 1) * band;
        for (let i = b * band; i < end; i++) {
          const r = R * rho[i] * depth + push * spread[i];
          g.rect(cx + cos[i] * r - size / 2, cy + sin[i] * r - size / 2, size, size);
        }
        g.fill();
      }
    }

    // 3. ripples: one per real onset, travelling out through the room and fading
    for (let i = 0; i < rt.ripples.length; i++) {
      const rp = rt.ripples[i];
      const p = (now - rp.t0) / (RIPPLE_S * 1000);
      const e = 1 - (1 - p) ** 3;
      g.lineWidth = dpr * (1 + 1.5 * (1 - p));
      g.strokeStyle = css(rt.room, A * 0.45 * rp.u * (1 - p) ** 2 * (1 - 0.6 * rec));
      g.beginPath();
      g.arc(cx, cy, R * (0.22 + 0.84 * e), 0, Math.PI * 2);
      g.stroke();
    }

    // 4. the target voice: pool of light, focus ring (a lens pull on confidence), bright core
    const wF = rt.weight.x * rt.gate.x * A;
    const e = rt.energy.x;
    const sharp = rt.conf.x;
    g.globalCompositeOperation = "lighter";
    if (style.filled && wF > 0.004) {
      const pool = R * (0.34 + 0.2 * e * moves);
      const grad = g.createRadialGradient(cx, cy, 0, cx, cy, pool);
      const a = wF * (0.12 + 0.4 * e);
      grad.addColorStop(0, css(rt.form, a));
      grad.addColorStop(0.35, css(rt.form, a * 0.42));
      grad.addColorStop(1, css(rt.form, 0));
      g.fillStyle = grad;
      g.fillRect(cx - pool, cy - pool, pool * 2, pool * 2);
    }
    const ring = R * (0.2 + 0.02 * e * moves);
    g.setLineDash(style.dashed ? [3 * dpr, 7 * dpr] : []);
    g.lineWidth = dpr * (1 + 5 * (1 - sharp));
    g.strokeStyle = css(rt.form, wF * (0.22 + 0.58 * sharp));
    g.beginPath();
    g.arc(cx, cy, ring, 0, Math.PI * 2);
    g.stroke();
    if (sharp < 0.97 && !style.flat) { // the out-of-focus double: it closes onto the ring as confidence rises
      g.lineWidth = dpr * (1 + 3 * (1 - sharp));
      g.strokeStyle = css(rt.form, wF * 0.3 * (1 - sharp));
      g.beginPath();
      g.arc(cx, cy, ring * (1 + 0.16 * (1 - sharp)), 0, Math.PI * 2);
      g.stroke();
    }
    g.setLineDash([]);
    const core = R * (0.065 + 0.045 * e * moves);
    g.beginPath();
    g.arc(cx, cy, core, 0, Math.PI * 2);
    if (style.filled) {
      g.fillStyle = css(rt.form, wF * (0.5 + 0.45 * e));
      g.fill();
      if (e > 0.02) { // the hot centre: white light inside the colour, only while the stream carries sound
        const hot = g.createRadialGradient(cx, cy, 0, cx, cy, core);
        hot.addColorStop(0, css(cfg.colors.fg, wF * 0.55 * e));
        hot.addColorStop(1, css(cfg.colors.fg, 0));
        g.fillStyle = hot;
        g.fill();
      }
    } else {
      g.lineWidth = dpr * 1.5;
      g.strokeStyle = css(rt.form, Math.max(wF, A * 0.5) * 0.9);
      g.stroke();
    }
    g.globalCompositeOperation = "source-over";

    const ms = performance.now() - t0;
    stats.draw(now, ms);
    // a designed light scene instead of a silent stutter: sticky for this page
    rt.overRun = ms > cfg.budgetMs ? rt.overRun + 1 : 0;
    if (!rt.light && rt.overRun >= LIGHT_AFTER) { rt.light = true; rt.dirty = true; app?.emit?.("scene-light", { drawMs: ms }); }
    return true;
  }

  // one rAF loop while animating; it stops by itself once nothing moves (a settled field draws nothing)
  let idleFrames = 0;
  let lastTick = -Infinity;
  function loop(now) {
    rt.raf = 0;
    if (now - lastTick < MIN_DRAW_MS) { rt.raf = requestAnimationFrame(loop); return; } // 120 Hz: every other vsync
    lastTick = now;
    stats.frame(now);
    const drew = draw(now);
    const settled = !drew && !rt.ripples.length && releasedDb(rt.peakDb, (now - rt.peakAt) / 1000, cfg) <= cfg.floorDb
      && (streamDb(rt.playDb, (now - rt.playAt) / 1000, cfg) ?? cfg.floorDb) <= cfg.floorDb;
    idleFrames = settled || !drew ? idleFrames + 1 : 0;
    if (idleFrames > 30) { stats.pause(); rt.lastT = 0; return; }
    rt.raf = requestAnimationFrame(loop);
  }

  function staticDraw() {
    rt.staticTimer = 0;
    rt.lastStaticAt = performance.now();
    rt.lastT = 0;
    draw(rt.lastStaticAt);
  }

  function schedule(now = false) {
    if (!cfg.animate) {
      // reduced motion: static frames only, light (alpha) changes at most 4 times a second; a look change at once
      if (document.hidden) return;
      if (now) { clearTimeout(rt.staticTimer); rt.dirty = true; staticDraw(); return; }
      if (!rt.staticTimer) {
        rt.staticTimer = setTimeout(staticDraw, Math.max(0, REDUCED_MIN_MS - (performance.now() - rt.lastStaticAt)));
      }
      return;
    }
    if (!rt.raf && !document.hidden) { idleFrames = 0; rt.raf = requestAnimationFrame(loop); }
  }

  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  const mq = matchMedia("(prefers-reduced-motion: reduce)");
  const onMq = () => { cancelAnimationFrame(rt.raf); rt.raf = 0; resize(); };
  mq.addEventListener("change", onMq);
  // devicePixelRatio changes (window moved to another display, zoom) do not resize the box: watch the ratio itself
  let dprMq = null;
  const onDpr = () => { resize(); watchDpr(); };
  function watchDpr() {
    dprMq?.removeEventListener("change", onDpr);
    dprMq = matchMedia(`(resolution: ${globalThis.devicePixelRatio || 1}dppx)`);
    dprMq.addEventListener("change", onDpr);
  }
  watchDpr();
  const onVis = () => {
    if (document.hidden) { cancelAnimationFrame(rt.raf); rt.raf = 0; clearTimeout(rt.staticTimer); rt.staticTimer = 0; stats.pause(); }
    else { rt.lastT = 0; rt.dirty = true; schedule(!cfg.animate); }
  };
  document.addEventListener("visibilitychange", onVis);
  const emitter = setInterval(() => app?.emit?.("scene", api.stats()), 1000);
  resize();

  const api = {
    element: root,
    canvas,
    /** Diagnostics: draw time (ms, last/mean/p95/max over 10 s), rAF interval p95/max, long frames, budget. */
    stats: () => {
      const sum = stats.summary(cfg.budgetMs);
      // p95Ms / lastMs: the names Diagnostics reads (screens/diag.js sceneInfo)
      return { ...sum, p95Ms: sum.drawP95Ms, lastMs: sum.drawLastMs, look: sceneLook(store.state),
        animate: Boolean(cfg.animate), light: rt.light, dpr: rt.dpr, width: rt.w, height: rt.h,
        levelDb: round(rt.lastDrawnDb), recede: round(rt.recede.x) };
    },
    look: () => sceneLook(store.state),
    redraw: () => { rt.dirty = true; schedule(true); },
    destroy() {
      offs.forEach((off) => off());
      ro.disconnect();
      mq.removeEventListener("change", onMq);
      dprMq?.removeEventListener("change", onDpr);
      clearTimeout(rt.staticTimer);
      document.removeEventListener("visibilitychange", onVis);
      clearInterval(emitter);
      cancelAnimationFrame(rt.raf);
      root.remove();
      document.documentElement.classList.remove("has-scene");
      if (app?.scene === api) delete app.scene;
      instance = null;
    },
  };
  if (app) app.scene = api;
  instance = api;
  return api;
}

/** The mounted scene, or null. */
export function scene() { return instance; }
