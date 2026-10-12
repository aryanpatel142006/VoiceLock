// The calm sounds, synthesised (no file, no network, no licence): the SOS calm loops (brown / pink / white noise and
// the chill-music pad) and Play My Sounds' gentle rain. Pure functions over Float32Array, so they run under node --test,
// and in a Worker (synth-worker.js): a loop is 5-16 s of audio, which is 50-250 ms of arithmetic at 48 kHz, far too
// long for a click handler (a 640 ms stall at 4x CPU throttle before this file existed). renderSamples() hands the
// work to the worker and caches the result for the page; prewarmSamples() renders everything in the background so the
// first press finds its loop ready. sos.js and sounds.js re-export the pure parts, so nothing that imported them moved.

export const SYNTH_RATE = 48000; // the rate the loops are rendered at when no AudioContext has said otherwise
export const NOISE_LOOP_S = 5;
export const NOISE_LOWPASS_HZ = 500;
export const NOISE_DBFS = -30; // RMS of the looped buffer
const SEAM_S = 0.05; // the loop's tail is cross-faded into its head so the 5 s seam does not click
export const RAIN_LOOP_S = 6;

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

/**
 * Gentle rain, one seamless loop: pink noise with its rumble taken out (high-passed around 300 Hz) and its top
 * softened (low-passed at 4.5 kHz), a slow swell, and sparse soft drops written circularly. -32 dBFS RMS.
 */
export function rainSamples(rate, { seconds = RAIN_LOOP_S, dbfs = -32, random = rng(0x7261) } = {}) {
  const n = Math.round(seconds * rate);
  const seam = Math.round(0.25 * rate);
  const x = new Float32Array(n + seam);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < x.length; i++) {
    const w = random() * 2 - 1;
    b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.969 * b2 + w * 0.153852;
    b3 = 0.8665 * b3 + w * 0.3104856; b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898;
    x[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.2;
    b6 = w * 0.115926;
  }
  const low = lowpass(Float32Array.from(x), rate, 300);
  for (let i = 0; i < x.length; i++) x[i] -= low[i];
  lowpass(x, rate, 4500);
  const out = x.slice(0, n);
  for (let i = 0; i < seam; i++) {
    const t = (i + 0.5) / seam;
    out[i] = out[i] * Math.sin((t * Math.PI) / 2) + x[n + i] * Math.cos((t * Math.PI) / 2);
  }
  for (let i = 0; i < n; i++) { // a slow swell whose periods divide the loop, so it stays seamless
    const t = i / n;
    out[i] *= 0.85 + 0.1 * Math.sin(2 * Math.PI * 2 * t) + 0.05 * Math.sin(2 * Math.PI * 3 * t + 1);
  }
  let r = 0;
  for (let i = 0; i < n; i++) r += out[i] * out[i];
  const bed = Math.sqrt(r / n);
  const drops = Math.round(seconds * 7);
  for (let d = 0; d < drops; d++) { // soft drops: a short decaying tone, 1.8-3.6 kHz
    const at = Math.floor(random() * n);
    const f = 1800 + random() * 1800;
    const amp = bed * (0.6 + random() * 0.9);
    const len = Math.round(0.03 * rate);
    for (let j = 0; j < len; j++) {
      const env = Math.min(1, j / (0.002 * rate)) * Math.exp(-j / (0.008 * rate));
      out[(at + j) % n] += amp * env * Math.sin((2 * Math.PI * f * j) / rate);
    }
  }
  const k = 10 ** ((dbfs - rmsDb(out)) / 20);
  for (let i = 0; i < n; i++) out[i] *= k;
  return out;
}

/** The loop for a sound id at `rate`. */
export function samplesFor(id, rate) {
  if (id === "brown" || id === "white") return noiseSamples(id, rate);
  if (id === "rain") return rainSamples(rate);
  if (id === "pad") return musicSamples(rate);
  throw new Error(`unknown sound ${id}`);
}

/** Every loop by id, synchronously (the worker's half, and the fallback when there is no Worker). "pad" is the music. */
export function samplesSync(id, rate) {
  const kind = id === "pad" ? "music" : id;
  if (kind === "brown" || kind === "pink" || kind === "white") return noiseSamples(kind, rate);
  if (kind === "music") return musicSamples(rate);
  if (kind === "rain") return rainSamples(rate);
  throw new Error(`unknown sound ${id}`);
}

// ---------------------------------------------------------------- the worker bridge and the page's cache

const cache = new Map(); // "<kind>@<rate>" -> Float32Array (rendered) | Promise<Float32Array> (rendering)
const pending = new Map(); // request id -> {resolve, reject}
let worker = null; // null: not tried · false: unavailable (no Worker, or it failed) · Worker
let seq = 0;
let rate = SYNTH_RATE;

const keyOf = (id, r) => `${id === "pad" ? "music" : id}@${r}`;

/** The output rate the page plays at, once an AudioContext has reported it (prewarm renders at this rate). */
export function knownRate() { return rate; }
export function noteRate(r) { if (Number.isFinite(r) && r > 0) rate = r; }

function getWorker() {
  if (worker !== null) return worker;
  try {
    if (typeof Worker !== "function" || typeof document === "undefined") { worker = false; return worker; }
    const w = new Worker(new URL("./synth-worker.js", import.meta.url), { type: "module" });
    w.onmessage = (e) => {
      const p = pending.get(e.data?.id);
      if (!p) return;
      pending.delete(e.data.id);
      if (e.data.error) p.reject(new Error(e.data.error));
      else p.resolve(e.data.samples);
    };
    w.onerror = () => { // the module did not load (old browser, odd server): every request falls back to the main thread
      worker = false;
      for (const p of pending.values()) p.reject(new Error("synth worker failed"));
      pending.clear();
      try { w.terminate(); } catch { /* gone */ }
    };
    worker = w;
  } catch { worker = false; }
  return worker;
}

/** True when loops render off the main thread. */
export function hasWorker() { return Boolean(getWorker()); }

/** The loop if it is already rendered (no synthesis here), else null. */
export function cachedSamples(id, r = rate) {
  const hit = cache.get(keyOf(id, r));
  return hit instanceof Float32Array ? hit : null;
}

/** The loop, rendered in the worker (or on this thread when there is none), cached for the page. */
export function renderSamples(id, r = rate) {
  const key = keyOf(id, r);
  const hit = cache.get(key);
  if (hit) return hit instanceof Promise ? hit : Promise.resolve(hit);
  const w = getWorker();
  let p;
  if (!w) {
    p = Promise.resolve().then(() => samplesSync(id, r));
  } else {
    p = new Promise((resolve, reject) => {
      const n = ++seq;
      pending.set(n, { resolve, reject });
      try { w.postMessage({ id: n, kind: id === "pad" ? "music" : id, rate: r }); } catch (err) { pending.delete(n); reject(err); }
    }).catch(() => samplesSync(id, r)); // the worker failed: once, on this thread
  }
  p = p.then((x) => { cache.set(key, x); return x; }, (err) => { cache.delete(key); throw err; });
  cache.set(key, p);
  return p;
}

/** Render `ids` in the background, one after another, when the page is idle. Nothing happens without a worker: a
 * synchronous render in an idle callback would be the stall this file exists to remove. */
export function prewarmSamples(ids, r = rate, win = globalThis) {
  if (!hasWorker()) return false;
  const idle = typeof win.requestIdleCallback === "function" ? (f) => win.requestIdleCallback(f) : (f) => win.setTimeout(f, 250);
  const todo = ids.filter((id) => !cache.has(keyOf(id, r)));
  const next = () => {
    const id = todo.shift();
    if (!id) return;
    renderSamples(id, r).catch(() => {}).then(() => idle(next));
  };
  idle(next);
  return true;
}
