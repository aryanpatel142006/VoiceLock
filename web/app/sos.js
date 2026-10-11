// SOS (M14, T-460): one press cuts the room and plays calm noise; hold 1.5 s to come back.
// The remote is a stand-in for a band on the wrist with one button. No medical claim is made anywhere here.
//
// The sound the listener hears comes from this browser (session.js browserAudio: worklet -> master gain ->
// destination), so SOS mutes on the client: session.setOutputGain(0, 150) ramps the master gain to silence with no
// network round trip. The engine keeps running untouched; it only hears {sos:true|false} as a control message so it
// can echo telemetry.sos for the phone companion and the session log.
//
// Triggers (each fires once per press; e.repeat is ignored): the big SOS button shown on Listen, Shift+S, and the
// paired band key from localStorage `hearing.sos` = {code, key, label, sound: brown|pink|white} (T-461's pair
// screen writes it). While the pair screen (#sos) is open, keys do nothing here.
// Return: hold 1.5 s (pointer anywhere on the calm frame, or Space/Enter/Shift+S/the band key held), the same
// press-and-hold pattern as Welcome's Start. Noise out and room back over 600 ms.
//
// Diagnostics: window.hearingSos.last.pressToSilenceMs is performance.now() from the press event to the moment the
// AudioContext clock passes the end of the 150 ms ramp (plus outputLatencyMs, the device's own buffer, reported
// apart). The pure half (band match, keys, hold, noise buffer, contrast) is exported for node --test.

import { STORAGE_KEY, loadSos, matchesBand } from "./screens/sos.js";

export const SOS_KEY = STORAGE_KEY;
export const CUT_MS = 150; // room -> silence
export const NOISE_IN_MS = 400; // calm noise fade-in
export const RETURN_MS = 600; // noise out, room back
export const HOLD_MS = 1500; // hold to come back
export const NOISE_LOOP_S = 5;
export const NOISE_LOWPASS_HZ = 500;
export const NOISE_DBFS = -30; // RMS of the looped buffer
const SEAM_S = 0.05; // the loop's tail is cross-faded into its head so the 5 s seam does not click
const HISTORY = 20;

// ---------------------------------------------------------------- pure helpers

/** The paired band from localStorage (T-461's loadSos); before pairing code/key/label are null, so only Shift+S and
 * the button trigger. */
export function loadBand(storage = globalThis.localStorage) {
  return loadSos(storage);
}

export { matchesBand }; // T-461's: by e.code, or by e.key when the band has no code

/** Shift+S, without Cmd/Ctrl/Alt (Cmd+Shift+S stays the browser's). */
export function isShiftS(e) {
  return Boolean(e.shiftKey) && !e.metaKey && !e.ctrlKey && !e.altKey && (e.code === "KeyS" || String(e.key).toLowerCase() === "s");
}

/**
 * What a keydown means for SOS, or null. `on`: SOS is showing. `e`: {key, code, shiftKey, metaKey, ctrlKey, altKey,
 * repeat, inField}; `pairing`: the pair screen is open. Returns "press" (turn SOS on), "hold" (start the hold to come
 * back), "swallow" (SOS is on: keep the key from the stage, so Space cannot flip A/B under the calm frame) or null.
 */
export function sosKey(e, { on, band, pairing = false }) {
  if (pairing || e.inField) return null;
  const trigger = isShiftS(e) || matchesBand(band, e);
  if (!on) return trigger && !e.repeat ? "press" : trigger ? "swallow" : null;
  if (e.key === "Tab" || e.metaKey || e.ctrlKey) return null;
  if (e.repeat) return "swallow";
  return trigger || e.key === " " || e.key === "Enter" ? "hold" : "swallow";
}

/**
 * The hold to come back, a pure state machine like Welcome's holdReduce. Phases: off -> on (calm frame) ->
 * holding (ring filling) -> off when full. `source` names the input that is holding ("pointer:<id>" / "key:<code>")
 * so another finger or key cannot end someone else's hold. Events: {type:"press"} (SOS on) ·
 * {type:"hold", source, at} · {type:"tick", at, durationMs} · {type:"release", source} · {type:"reset"}.
 */
export function sosInit() {
  return { phase: "off", source: null, t0: 0, progress: 0 };
}

export function sosReduce(s, ev) {
  switch (ev.type) {
    case "press":
      return s.phase === "off" ? { ...s, phase: "on", source: null, progress: 0 } : s;
    case "hold":
      return s.phase === "on" ? { ...s, phase: "holding", source: ev.source, t0: ev.at, progress: 0 } : s;
    case "tick": {
      if (s.phase !== "holding") return s;
      const progress = Math.min(1, Math.max(0, (ev.at - s.t0) / (ev.durationMs > 0 ? ev.durationMs : HOLD_MS)));
      return progress >= 1 ? { ...s, phase: "off", source: null, progress: 0 } : { ...s, progress };
    }
    case "release":
      return s.phase === "holding" && ev.source === s.source ? { ...s, phase: "on", source: null, progress: 0 } : s;
    case "reset":
      return sosInit();
    default:
      return s;
  }
}

/** Deterministic PRNG (mulberry32) so tests and the loop are repeatable. */
export function rng(seed = 0x5eed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** RBJ low-pass biquad (Q = 1/sqrt 2) over `x` in place. */
export function lowpass(x, rate, hz = NOISE_LOWPASS_HZ) {
  const w = (2 * Math.PI * Math.min(hz, rate * 0.45)) / rate;
  const alpha = Math.sin(w) / (2 * Math.SQRT1_2);
  const cos = Math.cos(w);
  const a0 = 1 + alpha;
  const b0 = (1 - cos) / 2 / a0, b1 = (1 - cos) / a0, b2 = b0, a1 = (-2 * cos) / a0, a2 = (1 - alpha) / a0;
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  for (let i = 0; i < x.length; i++) {
    const y = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1; x1 = x[i]; y2 = y1; y1 = y;
    x[i] = y;
  }
  return x;
}

export function rmsDb(x) {
  let e = 0;
  for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  return 10 * Math.log10(e / Math.max(1, x.length) + 1e-20);
}

/**
 * One loop of calm noise: `kind` brown (integrated white), pink (Paul Kellet's filter) or white, low-passed at
 * 500 Hz, its tail cross-faded into its head (a seamless loop) and scaled to -30 dBFS RMS. Float32Array, `seconds` long.
 */
export function noiseSamples(kind, rate, { seconds = NOISE_LOOP_S, dbfs = NOISE_DBFS, random = rng() } = {}) {
  const n = Math.round(seconds * rate);
  const seam = Math.round(SEAM_S * rate);
  const x = new Float32Array(n + seam);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, last = 0;
  for (let i = 0; i < x.length; i++) {
    const w = random() * 2 - 1;
    if (kind === "white") x[i] = w;
    else if (kind === "pink") {
      b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.969 * b2 + w * 0.153852;
      b3 = 0.8665 * b3 + w * 0.3104856; b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898;
      x[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
      b6 = w * 0.115926;
    } else { // brown: leaky integrator so it never drifts off
      last = (last + 0.02 * w) / 1.02;
      x[i] = last;
    }
  }
  lowpass(x, rate);
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i];
  mean /= x.length;
  for (let i = 0; i < x.length; i++) x[i] -= mean;
  const out = x.slice(0, n);
  for (let i = 0; i < seam; i++) { // equal-power cross-fade: the extra tail fades out over the head fading in
    const t = (i + 0.5) / seam;
    out[i] = out[i] * Math.sin((t * Math.PI) / 2) + x[n + i] * Math.cos((t * Math.PI) / 2);
  }
  const k = 10 ** ((dbfs - rmsDb(out)) / 20);
  for (let i = 0; i < n; i++) out[i] *= k;
  return out;
}

/** WCAG contrast ratio of two #rrggbb colours. */
export function contrast(a, b) {
  const lum = (hex) => {
    const c = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
      .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  };
  const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

/** The SOS palette (also in sos.css): night blue, not red/amber/green (those are the room/person/errors). */
export const SOS_COLORS = { bg: "#0d1b33", fg: "#f3f4f1", soft: "#c9d6ec", ring: "#9cc3ff" };

/** Press -> silence record for Diagnostics. */
export function silenceRecord({ pressAt, silentAt, source, outputLatencyMs = null, audio = true }) {
  return {
    source,
    pressToSilenceMs: Math.round((silentAt - pressAt) * 10) / 10,
    outputLatencyMs: outputLatencyMs === null ? null : Math.round(outputLatencyMs * 10) / 10,
    audio, // false: nothing was playing (no session audio), so the cut is the handler's time only
    at: Date.now(),
  };
}

// ---------------------------------------------------------------- browser half

function inField(t) {
  return Boolean(t?.closest?.("input, select, textarea, [contenteditable='true']"));
}

function mount() {
  const doc = document;
  const win = window;
  const reduced = win.matchMedia("(prefers-reduced-motion: reduce)");
  const link = doc.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("./sos.css", import.meta.url).href;
  doc.head.append(link);

  const el = (tag, attrs = {}, ...kids) => {
    const n = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") n.textContent = v;
      else n.setAttribute(k, v);
    }
    n.append(...kids.filter(Boolean));
    return n;
  };

  // The trigger: one big button, shown on Listen only (sos.css keys it off body[data-screen]).
  const btn = el("button", { type: "button", class: "sos-btn", "aria-label": "SOS: cut the room now", "aria-keyshortcuts": "Shift+S" },
    el("span", { class: "sos-btn-word", text: "SOS" }),
    el("span", { class: "sos-btn-sub", text: "Cut the room" }));

  // The calm frame.
  const ring = el("span", { class: "sos-calm-ring", "aria-hidden": "true" });
  const holdLabel = el("span", { class: "sos-calm-hold-label", text: "Hold to come back" });
  const hold = el("div", { class: "sos-calm-hold", role: "button", tabindex: "0", "aria-describedby": "sos-calm-hint" }, ring, holdLabel);
  const status = el("p", { class: "sos-calm-status", role: "status", "aria-live": "polite" });
  const keyHint = el("p", { class: "sos-calm-hint", id: "sos-calm-hint" });
  const frame = el("div", { class: "sos-calm", role: "dialog", "aria-modal": "true", "aria-labelledby": "sos-calm-title", hidden: "" },
    el("div", { class: "sos-calm-body" },
      el("h1", { class: "sos-calm-title", id: "sos-calm-title", text: "You're safe." }),
      el("p", { class: "sos-calm-line", text: "The room is off. Only a soft hush plays." }),
      hold,
      keyHint,
      status));
  doc.body.append(btn, frame);

  let s = sosInit();
  let band = loadBand();
  let raf = 0;
  let noise = null; // {ctx, gain, src}
  let latched = new Set(); // keys that turned SOS on: they must come up before they can start a hold
  let opener = null;
  const log = [];
  const api = {
    get on() { return s.phase !== "off"; },
    get phase() { return s.phase; },
    last: null,
    log,
    band: () => band,
    press: (source = "api") => press(win.performance.now(), source),
  };
  win.hearingSos = api;

  const session = () => win.hearingApp?.session || null;
  const pairing = () => doc.body.dataset.screen === "sos" || win.location.hash === "#sos";

  function describeKeys() {
    const b = band.label || band.key || band.code;
    keyHint.textContent = b ? `Hold anywhere, Space, or the band button (${b}) for 1.5 seconds.`
      : "Hold anywhere, or Space, for 1.5 seconds.";
  }

  // ---- sound
  function noiseCtx() {
    if (!noise) {
      const Ctx = win.AudioContext || win.webkitAudioContext;
      if (!Ctx) return null;
      const ctx = new Ctx({ latencyHint: "interactive" });
      const gain = ctx.createGain();
      gain.gain.value = 0;
      gain.connect(ctx.destination);
      noise = { ctx, gain, src: null, kind: null, buffer: null };
    }
    noise.ctx.resume?.().catch(() => {});
    const sink = session()?._rt?.audio?.ctx?.sinkId; // the output the app plays to (devices.js picker)
    if (typeof sink === "string" && sink && noise.ctx.setSinkId && noise.ctx.sinkId !== sink) noise.ctx.setSinkId(sink).catch(() => {});
    return noise;
  }

  function noiseIn() {
    const n = noiseCtx();
    if (!n) return;
    const { ctx, gain } = n;
    if (n.kind !== band.sound || !n.buffer) {
      const buf = ctx.createBuffer(1, Math.round(NOISE_LOOP_S * ctx.sampleRate), ctx.sampleRate);
      buf.copyToChannel(noiseSamples(band.sound, ctx.sampleRate), 0);
      n.buffer = buf;
      n.kind = band.sound;
    }
    if (!n.src) {
      n.src = ctx.createBufferSource();
      n.src.buffer = n.buffer;
      n.src.loop = true;
      n.src.connect(gain);
      n.src.start();
    }
    ramp(gain.gain, ctx.currentTime, 1, NOISE_IN_MS);
  }

  function noiseOut() {
    if (!noise?.src) return;
    const { ctx, gain, src } = noise;
    ramp(gain.gain, ctx.currentTime, 0, RETURN_MS);
    noise.src = null;
    try { src.stop(ctx.currentTime + RETURN_MS / 1000 + 0.05); } catch { /* already stopped */ }
  }

  function ramp(param, t, v, ms) {
    if (param.cancelAndHoldAtTime) param.cancelAndHoldAtTime(t);
    else { param.cancelScheduledValues(t); param.setValueAtTime(param.value, t); }
    param.linearRampToValueAtTime(v, t + ms / 1000);
  }

  // ---- press / return
  function press(pressAt, source) {
    if (s.phase !== "off") return;
    band = loadBand();
    const sess = session();
    const cut = sess?.setOutputGain?.(0, CUT_MS) || null; // first: the room goes before anything is drawn
    noiseIn();
    sess?.send?.({ sos: true });
    measure(pressAt, source, cut);
    opener = doc.activeElement;
    show(true);
    dispatch({ type: "press" });
  }

  function comeBack() {
    session()?.setOutputGain?.(1, RETURN_MS);
    noiseOut();
    session()?.send?.({ sos: false });
    api.lastReturnAt = win.performance.now();
    show(false);
  }

  function measure(pressAt, source, cut) {
    const done = (silentAt, audio, ctx) => {
      const lat = ctx ? ((ctx.outputLatency || 0) + (ctx.baseLatency || 0)) * 1000 : null;
      const rec = silenceRecord({ pressAt, silentAt, source, outputLatencyMs: lat, audio });
      api.last = rec;
      log.push(rec);
      if (log.length > HISTORY) log.shift();
      console.info(`[sos] press -> silence ${rec.pressToSilenceMs} ms (${source}${audio ? "" : ", nothing playing"})`);
      win.hearingApp?.emit?.("sos", { on: true, ...rec });
    };
    if (!cut?.ctx) return done(win.performance.now(), false, null);
    const poll = () => (cut.ctx.state !== "running" || cut.ctx.currentTime >= cut.endTime
      ? done(win.performance.now(), cut.ctx.state === "running", cut.ctx) : win.setTimeout(poll, 4));
    poll();
  }

  function show(on) {
    const apply = () => {
      frame.hidden = !on;
      doc.body.dataset.sos = on ? "on" : "";
      for (const n of [doc.getElementById("chrome"), doc.getElementById("stage"), doc.getElementById("sheet"), btn]) if (n) n.inert = on;
      if (on) hold.focus({ preventScroll: true });
      else if (opener?.isConnected && opener !== doc.body) opener.focus?.({ preventScroll: true });
    };
    // reduced motion: sos.css fades the frame's opacity; otherwise a view-transition cross-fade
    if (!reduced.matches && doc.startViewTransition) {
      try { doc.startViewTransition(apply); return; } catch { /* fall through */ }
    }
    apply();
  }

  // ---- hold state
  function dispatch(ev) {
    const prev = s;
    s = sosReduce(s, ev);
    if (s === prev) return;
    frame.dataset.phase = s.phase;
    hold.style.setProperty("--sos-p", s.progress.toFixed(3));
    if (prev.phase !== s.phase) {
      holdLabel.textContent = s.phase === "holding" ? "Keep holding" : "Hold to come back";
      status.textContent = s.phase === "holding" ? "Holding. Keep holding to bring the room back." : "";
    }
    if (s.phase === "holding" && !raf) raf = win.requestAnimationFrame(tick);
    if (s.phase !== "holding" && raf) { win.cancelAnimationFrame(raf); raf = 0; }
    if (s.phase === "off" && prev.phase !== "off") comeBack();
  }

  function tick(now) {
    raf = 0;
    dispatch({ type: "tick", at: now, durationMs: HOLD_MS });
    if (s.phase === "holding" && !raf) raf = win.requestAnimationFrame(tick);
  }

  // ---- inputs
  btn.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    press(e.timeStamp || win.performance.now(), "button");
  });
  btn.addEventListener("click", (e) => { if (e.detail === 0) press(win.performance.now(), "button"); }); // keyboard activation

  // The hold listens on the window, not the frame: a mouse that has not moved since the frame appeared is hit-tested
  // against Chrome's stale hover target (<html>), so its pointerdown never reaches the frame. Capture phase, so the
  // press's own pointerdown on the button (SOS still off here) is left alone.
  win.addEventListener("pointerdown", (e) => {
    if (s.phase === "off" || e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    try { frame.setPointerCapture?.(e.pointerId); } catch { /* pointer already gone */ }
    dispatch({ type: "hold", source: `pointer:${e.pointerId}`, at: win.performance.now() });
  }, true);
  for (const t of ["pointerup", "pointercancel"]) {
    win.addEventListener(t, (e) => { if (s.phase === "holding") dispatch({ type: "release", source: `pointer:${e.pointerId}` }); }, true);
  }
  frame.addEventListener("contextmenu", (e) => e.preventDefault()); // a long touch is the hold, not a menu

  doc.addEventListener("keydown", (e) => {
    const what = sosKey({ key: e.key, code: e.code, shiftKey: e.shiftKey, metaKey: e.metaKey, ctrlKey: e.ctrlKey, altKey: e.altKey,
      repeat: e.repeat, inField: inField(e.target) }, { on: s.phase !== "off", band, pairing: pairing() });
    if (!what) return;
    e.preventDefault();
    e.stopImmediatePropagation(); // keys.js (Space = A/B) is bound on document after this module
    const id = `key:${e.code || e.key}`;
    if (what === "press") { latched.add(id); press(e.timeStamp || win.performance.now(), isShiftS(e) ? "shift+s" : "band"); }
    else if (what === "hold" && !latched.has(id)) dispatch({ type: "hold", source: id, at: win.performance.now() });
  });
  doc.addEventListener("keyup", (e) => {
    const id = `key:${e.code || e.key}`;
    latched.delete(id);
    if (e.key === "Shift") for (const k of [...latched]) if (k === "key:KeyS") latched.delete(k);
    if (s.phase === "holding") dispatch({ type: "release", source: id });
    if (s.phase !== "off" && !inField(e.target)) e.preventDefault();
  });
  win.addEventListener("blur", () => { latched = new Set(); if (s.phase === "holding") dispatch({ type: "release", source: s.source }); });
  win.addEventListener("storage", (e) => { if (e.key === SOS_KEY) { band = loadBand(); describeKeys(); } });
  win.addEventListener("hashchange", () => { band = loadBand(); describeKeys(); }); // back from the pair screen
  describeKeys();
}

if (typeof document !== "undefined" && typeof window !== "undefined") {
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount, { once: true });
}
