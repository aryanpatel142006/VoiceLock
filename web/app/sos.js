// SOS (M14, T-460): one press cuts the room and plays calm noise; hold 1.5 s to come back.
// The remote is a stand-in for a band on the wrist with one button. No medical claim is made anywhere here.
//
// The sound the listener hears comes from this browser (session.js browserAudio: worklet -> master gain ->
// destination), so SOS mutes on the client: session.setOutputGain(0, 150) ramps the master gain to silence with no
// network round trip. The engine keeps running untouched; it only hears {sos:true|false} as a control message so it
// can echo telemetry.sos for the phone companion and the session log.
//
// Triggers (each fires once per press; e.repeat is ignored): the big SOS button shown on Listen, Shift+S, and the
// paired band key from localStorage `hearing.sos` = {code, key, label, sound: brown|pink|white|music} (T-461's pair
// screen writes it). While the pair screen (#sos) is open, keys do nothing here.
// T-478: the calm frame has a Calm noise / Chill music toggle (same storage as the pair screen); flipping it while
// SOS is on cross-fades the two loops over 400 ms. Pointer and Space/Enter on the toggle work the toggle, never the hold.
// Return: hold 1.5 s (pointer anywhere on the calm frame, or Space/Enter/Shift+S/the band key held), the same
// press-and-hold pattern as Welcome's Start. Noise out and room back over 600 ms.
//
// Diagnostics: window.hearingSos.last.pressToSilenceMs is performance.now() from the press event to the moment the
// AudioContext clock passes the end of the 150 ms ramp (plus outputLatencyMs, the device's own buffer, reported
// apart). The pure half (band match, keys, hold, noise buffer, contrast) is exported for node --test.

import { KINDS, STORAGE_KEY, chooseKind, loadSos, matchesBand, soundKind } from "./screens/sos.js";

export const SOS_KEY = STORAGE_KEY;
export const CUT_MS = 150; // room -> silence
export const NOISE_IN_MS = 400; // calm noise fade-in
export const SWAP_MS = 400; // noise <-> music cross-fade while SOS is on
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
 * back), "swallow" (SOS is on: keep the key from the stage, so Space cannot flip A/B under the calm frame), "pass"
 * (a key for the focused sound toggle: its default runs, the stage never sees it) or null.
 */
export function sosKey(e, { on, band, pairing = false }) {
  if (pairing || e.inField) return null;
  const trigger = isShiftS(e) || matchesBand(band, e);
  if (!on) return trigger && !e.repeat ? "press" : trigger ? "swallow" : null;
  if (e.key === "Tab" || e.metaKey || e.ctrlKey) return null;
  // the sound toggle on the calm frame keeps Space/Enter/arrows: "pass" = its default (the button's click) runs, but
  // the stage keys (keys.js: Space = A/B, which would cancel the click) never see it
  if (e.inToggle && !trigger && (e.key === " " || e.key === "Enter" || /^Arrow/.test(e.key))) return e.repeat ? "swallow" : "pass";
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

// ---- chill music (T-478): synthesised here, no file and no network. 60 BPM, one chord per bar of 4 beats, the
// progression I-vi-IV-V in C (C, Am, F, G), so one loop is 16 s. Each chord is 4 pad voices (bass + 3 chord tones),
// every voice a sine and a triangle detuned +-5 cents, with 1.5 s raised-cosine attack/release overlapping the next
// chord. Over it, very soft pentatonic bell notes on some beats (seeded, so the loop is the same every time). The
// dry loop is written circularly (a note that runs past the end wraps into the head), then lowpass 1.5 kHz and a small
// feedback-delay reverb run over two copies and the second copy is kept: the loop is seamless by construction.
export const MUSIC_BPM = 60;
export const MUSIC_LOOP_S = 16;
export const MUSIC_LOWPASS_HZ = 1500;
const MIDI = (m) => 440 * 2 ** ((m - 69) / 12);
// [bass, chord tones] in MIDI: C2+C3 E3 G3 · A1+A2 C3 E3 · F1+F2 A2 C3 · G1+G2 B2 D3, voiced close around C3
export const MUSIC_CHORDS = [[36, 55, 60, 64], [33, 57, 60, 64], [41, 57, 60, 65], [43, 55, 59, 62]];
const PENTATONIC = [72, 74, 76, 79, 81]; // C5 D5 E5 G5 A5

function addTone(out, rate, { start, dur, freq, amp, attack, release, kind }) {
  const n = out.length;
  const i0 = Math.round(start * rate);
  const len = Math.round(dur * rate);
  const a = Math.max(1, Math.round(attack * rate));
  const r = Math.max(1, Math.round(release * rate));
  const w = (2 * Math.PI * freq) / rate;
  for (let j = 0; j < len; j++) {
    let env;
    if (kind === "bell") env = Math.min(1, j / a) * Math.exp(-j / r); // soft strike, long fade
    else env = j < a ? 0.5 - 0.5 * Math.cos((Math.PI * j) / a) : j > len - r ? 0.5 - 0.5 * Math.cos((Math.PI * (len - j)) / r) : 1;
    const p = w * j;
    const v = kind === "bell" ? Math.sin(p) + 0.12 * Math.sin(2 * p)
      : kind === "tri" ? 0.81 * (Math.sin(p) - Math.sin(3 * p) / 9) : Math.sin(p);
    out[(i0 + j) % n] += amp * env * v;
  }
}

/**
 * One 16 s loop of soft ambient music at `rate`, Float32Array, scaled to `dbfs` RMS like the noise. Deterministic:
 * the melody comes from `random` (seeded PRNG).
 */
export function musicSamples(rate, { dbfs = NOISE_DBFS, random = rng(0x6d75) } = {}) {
  const beat = 60 / MUSIC_BPM;
  const bar = 4 * beat;
  const n = Math.round(MUSIC_LOOP_S * rate);
  const dry = new Float32Array(n);
  const fade = 1.5;
  MUSIC_CHORDS.forEach((chord, c) => {
    chord.forEach((m, v) => {
      const amp = v === 0 ? 0.55 : 0.32;
      for (const [cents, kind] of [[-5, "sine"], [5, "tri"]]) {
        addTone(dry, rate, { start: c * bar, dur: bar + fade, freq: MIDI(m) * 2 ** (cents / 1200), amp, attack: fade, release: fade, kind });
      }
    });
  });
  const beats = Math.round(MUSIC_LOOP_S / beat);
  let last = -1;
  for (let b = 0; b < beats; b++) {
    if (random() > 0.38) continue;
    let k = Math.floor(random() * PENTATONIC.length);
    if (k === last) k = (k + 2) % PENTATONIC.length; // no repeated note
    last = k;
    const at = b * beat + (random() < 0.3 ? beat / 2 : 0);
    addTone(dry, rate, { start: at, dur: 3.5, freq: MIDI(PENTATONIC[k]), amp: 0.16, attack: 0.06, release: 0.9, kind: "bell" });
  }
  // two periods through the stateful lowpass + reverb; the second is the steady loop
  const two = new Float32Array(2 * n);
  two.set(dry, 0);
  two.set(dry, n);
  lowpass(two, rate, MUSIC_LOWPASS_HZ);
  const wet = reverb(two, rate);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = 0.75 * two[n + i] + 0.35 * wet[n + i];
  const k = 10 ** ((dbfs - rmsDb(out)) / 20);
  for (let i = 0; i < n; i++) out[i] *= k;
  return out;
}

/** A light Schroeder-style reverb: 4 parallel damped feedback combs, then 2 allpasses. Tail ~2 s. */
function reverb(x, rate) {
  const y = new Float32Array(x.length);
  for (const [ms, g] of [[29.7, 0.78], [37.1, 0.76], [41.1, 0.75], [43.7, 0.74]]) {
    const d = Math.round((ms / 1000) * rate);
    const buf = new Float32Array(d);
    let lp = 0;
    for (let i = 0, p = 0; i < x.length; i++, p = (p + 1) % d) {
      const out = buf[p];
      lp = 0.6 * out + 0.4 * lp; // damping: the tail darkens
      buf[p] = x[i] + g * lp;
      y[i] += out / 4;
    }
  }
  for (const ms of [5.0, 1.7]) {
    const d = Math.round((ms / 1000) * rate);
    const buf = new Float32Array(d);
    for (let i = 0, p = 0; i < y.length; i++, p = (p + 1) % d) {
      const b = buf[p];
      const v = y[i] + 0.5 * b;
      buf[p] = v;
      y[i] = b - 0.5 * v;
    }
  }
  return y;
}

/** The loop for a `hearing.sos` sound: "music" or a noise colour. */
export function soundSamples(sound, rate) {
  return sound === "music" ? musicSamples(rate) : noiseSamples(sound, rate);
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

function inToggle(t) {
  return Boolean(t?.closest?.(".sos-calm-kinds"));
}

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
  const line = el("p", { class: "sos-calm-line" });
  const kindBtns = KINDS.map((k) => el("button", { type: "button", class: "sos-calm-kind", "data-kind": k.id, "aria-pressed": "false", text: k.name }));
  const kinds = el("div", { class: "sos-calm-kinds", role: "group", "aria-label": "Calm sound" }, ...kindBtns);
  const frame = el("div", { class: "sos-calm", role: "dialog", "aria-modal": "true", "aria-labelledby": "sos-calm-title", hidden: "" },
    el("div", { class: "sos-calm-body" },
      el("h1", { class: "sos-calm-title", id: "sos-calm-title", text: "You're safe." }),
      line,
      kinds,
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

  function describeSound() {
    const kind = soundKind(band.sound);
    for (const b of kindBtns) b.setAttribute("aria-pressed", String(b.dataset.kind === kind));
    line.textContent = kind === "music" ? "The room is off. Only soft music plays." : "The room is off. Only a soft hush plays.";
  }

  // ---- sound: master gain (fade in / out) <- one voice per playing loop (its own gain, for the noise/music swap)
  function makeCtx() { // created suspended outside a gesture (warm()); press() resumes it
    if (!noise) {
      const Ctx = win.AudioContext || win.webkitAudioContext;
      if (!Ctx) return null;
      const ctx = new Ctx({ latencyHint: "interactive" });
      const gain = ctx.createGain();
      gain.gain.value = 0;
      gain.connect(ctx.destination);
      noise = { ctx, gain, voice: null, buffers: new Map() };
    }
    return noise;
  }

  function noiseCtx() {
    if (!makeCtx()) return null;
    noise.ctx.resume?.().catch(() => {});
    const sink = session()?._rt?.audio?.ctx?.sinkId; // the output the app plays to (devices.js picker)
    if (typeof sink === "string" && sink && noise.ctx.setSinkId && noise.ctx.sinkId !== sink) noise.ctx.setSinkId(sink).catch(() => {});
    return noise;
  }

  function bufferFor(n, sound) {
    if (!n.buffers.has(sound)) { // music is ~0.25 s of synthesis at 48 kHz, so it is pre-rendered when chosen (warm())
      const x = soundSamples(sound, n.ctx.sampleRate);
      const buf = n.ctx.createBuffer(1, x.length, n.ctx.sampleRate);
      buf.copyToChannel(x, 0);
      n.buffers.set(sound, buf);
    }
    return n.buffers.get(sound);
  }

  function startVoice(n, sound, fadeMs) {
    const { ctx } = n;
    const g = ctx.createGain();
    g.gain.value = fadeMs ? 0 : 1;
    g.connect(n.gain);
    const src = ctx.createBufferSource();
    src.buffer = bufferFor(n, sound);
    src.loop = true;
    src.connect(g);
    src.start();
    if (fadeMs) ramp(g.gain, ctx.currentTime, 1, fadeMs);
    n.voice = { sound, src, gain: g };
  }

  function stopVoice(n, voice, ms) {
    if (!voice) return;
    ramp(voice.gain.gain, n.ctx.currentTime, 0, ms);
    try { voice.src.stop(n.ctx.currentTime + ms / 1000 + 0.05); } catch { /* already stopped */ }
  }

  function noiseIn() {
    const n = noiseCtx();
    if (!n) return;
    if (n.voice?.sound !== band.sound) {
      stopVoice(n, n.voice, 0);
      startVoice(n, band.sound, 0);
    }
    ramp(n.gain.gain, n.ctx.currentTime, 1, NOISE_IN_MS);
  }

  /** The toggle flipped while SOS is on: the new loop fades in as the old one fades out. */
  function swapSound() {
    if (!noise?.voice || noise.voice.sound === band.sound) return;
    const old = noise.voice;
    startVoice(noise, band.sound, SWAP_MS);
    stopVoice(noise, old, SWAP_MS);
  }

  function warm() { // render the music loop off the press path, once the page is idle
    if (band.sound !== "music" || noise?.buffers?.has("music")) return;
    const idle = win.requestIdleCallback || ((f) => win.setTimeout(f, 200));
    idle(() => { if (band.sound === "music" && makeCtx()) bufferFor(noise, "music"); });
  }

  function noiseOut() {
    if (!noise?.voice) return;
    ramp(noise.gain.gain, noise.ctx.currentTime, 0, RETURN_MS);
    stopVoice(noise, noise.voice, RETURN_MS);
    noise.voice = null;
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
    describeSound();
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
    if (inToggle(e.target)) return; // the toggle's own click
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
      repeat: e.repeat, inField: inField(e.target), inToggle: inToggle(e.target) }, { on: s.phase !== "off", band, pairing: pairing() });
    if (!what) return;
    if (what === "pass") { e.stopImmediatePropagation(); return; }
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
    if (s.phase !== "off" && inToggle(e.target)) e.stopImmediatePropagation(); // Space's click fires on keyup (keys.js would cancel it)
    else if (s.phase !== "off" && !inField(e.target)) e.preventDefault();
  });
  for (const b of kindBtns) {
    b.addEventListener("click", () => {
      band = { ...band, sound: chooseKind(win.localStorage, b.dataset.kind).sound };
      describeSound();
      warm();
      swapSound();
    });
  }
  win.addEventListener("blur", () => { latched = new Set(); if (s.phase === "holding") dispatch({ type: "release", source: s.source }); });
  const reload = () => { band = loadBand(); describeKeys(); describeSound(); warm(); };
  win.addEventListener("storage", (e) => { if (e.key === SOS_KEY) reload(); });
  win.addEventListener("hashchange", reload); // back from the pair screen
  reload();
}

if (typeof document !== "undefined" && typeof window !== "undefined") {
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount, { once: true });
}
