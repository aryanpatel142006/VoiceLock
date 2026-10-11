// Welcome's sound design (T-480): the cold open's soundtrack and the hold's collapse, all synthesised in WebAudio
// (no audio files, no network). Storyboard: docs/ux-review-shots/WELCOME_WOW.md; a 15 s OfflineAudioContext render
// of this exact graph is docs/ux-review-shots/welcome-sound-sample.wav (web/app/tests/welcome_sound_render.py).
//
// It only ever runs after a gesture: the Enter curtain's "Enter with sound" (or the Sound toggle, or a hold when
// sound was left on) creates the AudioContext in that click, so no browser reports an autoplay block.
//
//   shot 1-2  a low sub hum fades in; a thin sine pair widens in stereo as the hairline grows, then breaks;
//   shot 3    the room rushes in (a filtered-noise swell) and a cafe murmur builds with the grain field: six babbling
//             "voices" (noise through two moving formant filters, gated by syllable envelopes), a far crowd bed, and
//             cups and cutlery (short inharmonic clinks), all panned across the stereo field;
//   shot 4    each word of the promise lands with a soft tick, left to right;
//   rest      the murmur settles to a quiet room bed (ROOM_BED of its peak);
//   hold      the room low-passes and ducks with the ring's progress while one warm, voice-like formant pad rises;
//             let go early and the room comes back; a full ring resolves on a gentle G major chord, and the mic
//             opening cuts everything to silence (so the app's real audio is never masked).
// Level: a soft-knee shaper in front of the output holds every sample under PEAK (-12 dBFS); every gain starts at 0
// and moves by ramps, every source stops only after its gain reached 0 (no clicks).
//
// Every method takes an optional `at` (context seconds), so the OfflineAudioContext render plays the same calls.

export const PEAK_DBFS = -12;
export const PEAK = 10 ** (PEAK_DBFS / 20); // 0.251
export const KNEE = 0.7 * PEAK; // the shaper is linear below this
export const ROOM_BED = 0.34; // the murmur's level at rest, as a share of its cold-open peak
export const ROOM_GAIN = 0.85; // the murmur bus at full (the voices' own levels are calibrated under it)
export const VOICE_HZ = 196; // G3: the one clear voice
export const CHORD_HZ = [196, 246.94, 293.66, 392]; // G3 B3 D4 G4: resolved
export const VOICE_GAIN = 0.42; // the voice pad at a full ring
export const TICK_LAND = 0.3; // a word "lands" this share of its rise after it starts (the rise eases out)

const clamp01 = (x) => Math.max(0, Math.min(1, x));

/** A small seeded PRNG (mulberry32) so the room sounds the same on every load and in the offline render. */
export function rng(seed = 1) {
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
 * The room's (murmur's) level 0..1 at `tMs` after the cold open starts, for a plan shaped like welcome.js COLD_OPEN:
 * silent until the scatter, swelling with the grain field to 1 when every grain is home, held through the promise,
 * settling to ROOM_BED by `end`. The audio bus and the film grain both read these breakpoints.
 */
export function roomBreakpoints(plan) {
  const mid = plan.scatter + 0.45 * (plan.scattered - plan.scatter);
  return [[plan.scatter, 0], [mid, 0.6], [plan.scattered, 1], [plan.settle, 1], [plan.end, ROOM_BED]];
}

export function roomLevelAt(tMs, plan) {
  const bp = roomBreakpoints(plan);
  if (tMs <= bp[0][0]) return 0;
  for (let i = 1; i < bp.length; i++) {
    const [t1, v1] = bp[i];
    const [t0, v0] = bp[i - 1];
    if (tMs <= t1) return v0 + (v1 - v0) * ((tMs - t0) / Math.max(1, t1 - t0));
  }
  return ROOM_BED;
}

/** Smoothstep, the same ease the field's collapse uses. */
function ease(p) {
  const x = clamp01(p);
  return x * x * (3 - 2 * x);
}

/**
 * The hold's mix at ring progress `p` (0..1): the room's low-pass cutoff (Hz) and duck (gain multiplier), and the
 * voice pad's gain. The cutoff falls exponentially (equal steps sound equal), the duck and the voice cross.
 */
export function holdMix(p) {
  const e = ease(p);
  return {
    cutoffHz: 9000 * (240 / 9000) ** e,
    room: 1 - 0.88 * e,
    voice: e ** 1.4,
  };
}

/**
 * The cold open's sound cues (seconds from the start) for a plan and the word start times (ms, welcome.js
 * wordDelay): the hum, the line's growth and break, the room's rush, one tick per word (pitch falling a little
 * across the sentence, panned left to right).
 */
export function introCues(plan, wordStartsMs = [], wordMs = plan.wordMs ?? 720) {
  const s = (ms) => ms / 1000;
  const n = wordStartsMs.length;
  return {
    humIn: s(plan.line) * 0.2,
    lineFrom: s(plan.line),
    lineTo: s(plan.scatter),
    rushFrom: s(plan.scatter) - 0.25,
    rushPeak: s(plan.scatter) + 0.55,
    end: s(plan.end),
    ticks: wordStartsMs.map((ms, i) => ({
      at: s(ms + TICK_LAND * wordMs),
      hz: 1760 - i * 40 - (i === n - 1 ? 120 : 0), // the last word lands lower: the sentence resolves
      pan: n > 1 ? -0.4 + (0.8 * i) / (n - 1) : 0,
    })),
  };
}

/** The soft-knee shaper curve: linear under KNEE, then tanh into PEAK, so no sample can exceed -12 dBFS. */
export function shaperCurve(n = 4097) {
  const c = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1;
    const a = Math.abs(x);
    const y = a <= KNEE ? a : KNEE + (PEAK - KNEE) * Math.tanh((a - KNEE) / (PEAK - KNEE));
    c[i] = Math.sign(x) * y;
  }
  return c;
}

/**
 * One babbling voice's envelopes (syllables and formant motion), seeded. Returns {amp, move} Float32Arrays of
 * `seconds * rate` samples: amp 0..1 (raised-cosine syllables in words in phrases), move -1..1 (slow formant glide).
 */
export function babble(seed, seconds, rate) {
  const r = rng(seed);
  const len = Math.round(seconds * rate);
  const amp = new Float32Array(len);
  const move = new Float32Array(len);
  let t = r() * 0.4;
  while (t < seconds - 0.3) {
    const words = 2 + Math.floor(r() * 5); // a phrase
    for (let w = 0; w < words && t < seconds - 0.3; w++) {
      const syl = 1 + Math.floor(r() * 3);
      for (let k = 0; k < syl; k++) {
        const d = 0.11 + r() * 0.17;
        const a = 0.45 + r() * 0.55;
        if (t + d > seconds - 0.05) break; // the loop seam stays silent
        const i0 = Math.round(t * rate);
        const i1 = Math.min(len, Math.round((t + d) * rate));
        for (let i = i0; i < i1; i++) amp[i] = Math.max(amp[i], a * 0.5 * (1 - Math.cos((2 * Math.PI * (i - i0)) / (i1 - i0))));
        t += d * (0.8 + r() * 0.3);
      }
      t += 0.06 + r() * 0.16; // between words
    }
    t += 0.35 + r() * 0.8; // between phrases
  }
  // formant glide: two slow sines whose periods divide the loop, so it loops without a jump
  const k1 = 2 + Math.floor(r() * 3);
  const k2 = 5 + Math.floor(r() * 4);
  const p1 = r() * 2 * Math.PI;
  const p2 = r() * 2 * Math.PI;
  for (let i = 0; i < len; i++) {
    const x = (2 * Math.PI * i) / len;
    move[i] = 0.62 * Math.sin(k1 * x + p1) + 0.38 * Math.sin(k2 * x + p2);
  }
  return { amp, move };
}

/** Sparse cups and cutlery for one side of the room: inharmonic sine pairs and short noise taps, seeded. */
export function clinks(seed, seconds, rate, count = 6) {
  const r = rng(seed);
  const len = Math.round(seconds * rate);
  const out = new Float32Array(len);
  for (let e = 0; e < count; e++) {
    const at = Math.round((0.2 + r() * (seconds - 0.6)) * rate);
    if (r() < 0.6) { // a glass or a cup
      const f1 = 2200 + r() * 2600;
      const f2 = f1 * (1.48 + r() * 0.2);
      const tau = 0.05 + r() * 0.09;
      const a = 0.25 + r() * 0.35;
      const n = Math.min(len - at, Math.round(tau * 6 * rate));
      for (let i = 0; i < n; i++) {
        const tt = i / rate;
        const env = Math.min(1, i / (0.0015 * rate)) * Math.exp(-tt / tau);
        out[at + i] += a * env * (0.7 * Math.sin(2 * Math.PI * f1 * tt) + 0.3 * Math.sin(2 * Math.PI * f2 * tt));
      }
    } else { // a fork on a plate: a short bright tap
      const tau = 0.006 + r() * 0.01;
      const a = 0.2 + r() * 0.2;
      const n = Math.min(len - at, Math.round(tau * 6 * rate));
      let prev = 0;
      for (let i = 0; i < n; i++) {
        const w = r() * 2 - 1;
        const hp = w - prev; // a first difference: bright
        prev = w;
        out[at + i] += a * hp * Math.min(1, i / (0.0005 * rate)) * Math.exp(-i / rate / tau);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------- the graph

/** Move a param to `v` from wherever it is now (cancelAndHold where the browser has it; else cancel + set). */
function retarget(param, v, at, tau) {
  if (typeof param.cancelAndHoldAtTime === "function") param.cancelAndHoldAtTime(at);
  else { const cur = param.value; param.cancelScheduledValues(at); param.setValueAtTime(cur, at); }
  param.setTargetAtTime(v, at, tau);
}

function bufferFrom(ctx, data, rate = ctx.sampleRate) {
  const b = ctx.createBuffer(1, data.length, rate);
  b.getChannelData(0).set(data);
  return b;
}

function panner(ctx, pan) {
  if (typeof ctx.createStereoPanner === "function") {
    const p = ctx.createStereoPanner();
    p.pan.value = pan;
    return p;
  }
  return ctx.createGain(); // very old WebKit: mono, still audible
}

/**
 * Builds the soundtrack on `ctx` (an AudioContext created inside a gesture, or an OfflineAudioContext). Nothing is
 * audible until start(). `plan` and `wordStartsMs` come from welcome.js (COLD_OPEN scaled, wordDelay).
 */
export function createWelcomeSound(ctx, { plan, wordStartsMs = [], seed = 0x480 } = {}) {
  const now = () => ctx.currentTime;
  const sources = [];
  const keep = (src) => { sources.push(src); return src; };

  // output: master (mute/fade) -> soft-knee shaper (the -12 dBFS ceiling) -> destination
  const master = ctx.createGain();
  master.gain.value = 0;
  const shaper = ctx.createWaveShaper();
  shaper.curve = shaperCurve();
  master.connect(shaper).connect(ctx.destination);

  const intro = ctx.createGain(); // hum, line, rush, ticks (faded out by skip)
  intro.gain.value = 1;
  intro.connect(master);

  // the room: murmur -> roomBus (cold-open swell) -> roomLP (the hold's low-pass) -> roomDuck (the hold's duck)
  const roomBus = ctx.createGain();
  roomBus.gain.value = 0;
  const roomLP = ctx.createBiquadFilter();
  roomLP.type = "lowpass";
  roomLP.frequency.value = 9000;
  roomLP.Q.value = 0.5;
  const roomDuck = ctx.createGain();
  roomDuck.gain.value = 1;
  roomBus.connect(roomLP).connect(roomDuck).connect(master);

  const r = rng(seed);
  const noiseData = new Float32Array(ctx.sampleRate * 2);
  for (let i = 0; i < noiseData.length; i++) noiseData[i] = r() * 2 - 1;
  const noise = bufferFrom(ctx, noiseData);
  const noiseSrc = (offset = 0) => {
    const s = keep(ctx.createBufferSource());
    s.buffer = noise;
    s.loop = true;
    s._offset = offset;
    return s;
  };

  // six babbling voices: noise -> F1 and F2 band-passes (moved by the glide) -> syllable gate -> pan
  const ENV_RATE = 22050; // old WebKit only takes 22.05 kHz and up
  const VOICES = [
    { f1: 520, f2: 1350, pan: -0.75, len: 6.7, lvl: 0.95 },
    { f1: 640, f2: 1150, pan: 0.6, len: 7.9, lvl: 0.85 },
    { f1: 430, f2: 1700, pan: -0.25, len: 8.3, lvl: 0.7 },
    { f1: 700, f2: 1250, pan: 0.3, len: 7.1, lvl: 0.75 },
    { f1: 380, f2: 2050, pan: -0.5, len: 9.4, lvl: 0.55 }, // further away: quieter, below
    { f1: 560, f2: 1500, pan: 0.85, len: 8.8, lvl: 0.5 },
  ];
  VOICES.forEach((v, k) => {
    const env = babble(seed + 11 * (k + 1), v.len, ENV_RATE);
    const src = noiseSrc(r() * 1.9);
    const f1 = ctx.createBiquadFilter();
    f1.type = "bandpass"; f1.frequency.value = v.f1; f1.Q.value = 3.2;
    const f2 = ctx.createBiquadFilter();
    f2.type = "bandpass"; f2.frequency.value = v.f2; f2.Q.value = 4.5;
    const gate = ctx.createGain();
    gate.gain.value = 0; // the syllable envelope (audio rate) is the whole gain
    const ampSrc = keep(ctx.createBufferSource());
    ampSrc.buffer = bufferFrom(ctx, env.amp, ENV_RATE);
    ampSrc.loop = true;
    const lvl = ctx.createGain();
    lvl.gain.value = v.lvl * 0.5;
    ampSrc.connect(lvl).connect(gate.gain);
    const glide = keep(ctx.createBufferSource());
    glide.buffer = bufferFrom(ctx, env.move, ENV_RATE);
    glide.loop = true;
    const cents = ctx.createGain();
    cents.gain.value = 420;
    glide.connect(cents);
    cents.connect(f1.detune);
    cents.connect(f2.detune);
    const mixF2 = ctx.createGain();
    mixF2.gain.value = 0.7;
    src.connect(f1).connect(gate);
    src.connect(f2).connect(mixF2).connect(gate);
    gate.connect(panner(ctx, v.pan)).connect(roomBus);
  });

  // the far crowd: two wide bands of low noise, slowly breathing
  [-0.6, 0.6].forEach((pan, k) => {
    const src = noiseSrc(0.5 + k * 0.7);
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass"; bp.frequency.value = 480 + k * 140; bp.Q.value = 0.8;
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass"; lp.frequency.value = 1700;
    const g = ctx.createGain();
    g.gain.value = 0.11;
    const lfo = keep(ctx.createOscillator());
    lfo.frequency.value = 0.21 + k * 0.08;
    const depth = ctx.createGain();
    depth.gain.value = 0.035;
    lfo.connect(depth).connect(g.gain);
    src.connect(bp).connect(lp).connect(g).connect(panner(ctx, pan)).connect(roomBus);
  });

  // cups and cutlery, left and right, looping at different lengths so they never line up
  [[-0.7, 7.3], [0.65, 9.1], [0.1, 11.7]].forEach(([pan, len], k) => {
    const s = keep(ctx.createBufferSource());
    s.buffer = bufferFrom(ctx, clinks(seed + 101 * (k + 1), len, ctx.sampleRate, k === 2 ? 3 : 5));
    s.loop = true;
    const g = ctx.createGain();
    g.gain.value = k === 2 ? 0.08 : 0.12;
    s.connect(g).connect(panner(ctx, pan)).connect(roomBus);
  });

  // the one clear voice: two saws a few cents apart and a sine under them, through "ah" formants, gentle vibrato
  const voice = ctx.createGain();
  voice.gain.value = 0;
  voice.connect(master);
  const vib = keep(ctx.createOscillator());
  vib.frequency.value = 5.1;
  const vibDepth = ctx.createGain();
  vibDepth.gain.value = 7; // cents
  vib.connect(vibDepth);
  const voiceIn = ctx.createGain();
  voiceIn.gain.value = 1;
  [[-4, "sawtooth"], [5, "sawtooth"], [0, "sine"]].forEach(([det, type]) => {
    const o = keep(ctx.createOscillator());
    o.type = type;
    o.frequency.value = VOICE_HZ;
    o.detune.value = det;
    vibDepth.connect(o.detune);
    const g = ctx.createGain();
    g.gain.value = type === "sine" ? 0.5 : 0.22;
    o.connect(g).connect(voiceIn);
  });
  const soften = ctx.createBiquadFilter();
  soften.type = "lowpass"; soften.frequency.value = 2600; soften.Q.value = 0.4;
  voiceIn.connect(soften);
  [[730, 5, 0.9], [1090, 7, 0.5], [2440, 9, 0.18]].forEach(([f, q, g0]) => {
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass"; bp.frequency.value = f; bp.Q.value = q;
    const g = ctx.createGain();
    g.gain.value = g0 * 2.2;
    soften.connect(bp).connect(g).connect(voice);
  });
  const body = ctx.createGain(); // the fundamental, unfiltered: warmth
  body.gain.value = 0.32;
  soften.connect(body).connect(voice);

  // ---- the cold-open layers (scheduled by start)
  function hum(t0, end) {
    const g = ctx.createGain();
    g.gain.value = 0;
    g.connect(intro);
    [[49, 0.3], [98, 0.5], [147, 0.2]].forEach(([f, a]) => {
      const o = ctx.createOscillator();
      o.frequency.value = f;
      const og = ctx.createGain();
      og.gain.value = a;
      o.connect(og).connect(g);
      o.start(t0);
      o.stop(end + 3);
    });
    const c = introCues(plan, wordStartsMs);
    g.gain.setValueAtTime(0, t0);
    g.gain.setTargetAtTime(0.06, t0 + c.humIn, 0.35);
    g.gain.setTargetAtTime(0.03, t0 + c.lineTo + 0.4, 0.6); // the room takes over
    g.gain.setTargetAtTime(0, t0 + c.end - 0.6, 0.4);
  }
  function line(t0) {
    const c = introCues(plan, wordStartsMs);
    const from = t0 + c.lineFrom;
    const to = t0 + c.lineTo;
    [[-1, 1568], [1, 1568 * 1.004]].forEach(([side, f]) => {
      const o = ctx.createOscillator();
      o.frequency.setValueAtTime(f, from);
      o.frequency.setValueAtTime(f, to);
      o.frequency.exponentialRampToValueAtTime(f * 0.84, to + 0.35); // the line breaks: a small fall
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, from);
      g.gain.linearRampToValueAtTime(0.022, to); // grows with the hairline
      g.gain.setTargetAtTime(0, to, 0.09);
      const p = panner(ctx, 0);
      if (p.pan) { p.pan.setValueAtTime(0, from); p.pan.linearRampToValueAtTime(0.8 * side, to); }
      o.connect(g).connect(p).connect(intro);
      o.start(from);
      o.stop(to + 1.2);
    });
  }
  function rush(t0) {
    const c = introCues(plan, wordStartsMs);
    const from = t0 + c.rushFrom;
    const src = ctx.createBufferSource();
    src.buffer = noise;
    src.loop = true;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass"; bp.Q.value = 0.9;
    bp.frequency.setValueAtTime(260, from);
    bp.frequency.exponentialRampToValueAtTime(2600, t0 + c.rushPeak);
    bp.frequency.exponentialRampToValueAtTime(900, t0 + c.rushPeak + 1.2);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, from);
    g.gain.linearRampToValueAtTime(0.07, t0 + c.rushPeak);
    g.gain.setTargetAtTime(0, t0 + c.rushPeak, 0.45);
    src.connect(bp).connect(g).connect(intro);
    src.start(from);
    src.stop(t0 + c.rushPeak + 3.5);
  }
  function ticks(t0) {
    for (const k of introCues(plan, wordStartsMs).ticks) {
      const at = t0 + k.at;
      const o = ctx.createOscillator();
      o.frequency.value = k.hz;
      const o2 = ctx.createOscillator(); // a soft body an octave down
      o2.frequency.value = k.hz / 2;
      const g = ctx.createGain();
      g.gain.setValueAtTime(0, at);
      g.gain.linearRampToValueAtTime(0.075, at + 0.002);
      g.gain.setTargetAtTime(0, at + 0.002, 0.022);
      const g2 = ctx.createGain();
      g2.gain.value = 0.6;
      const p = panner(ctx, k.pan);
      o.connect(g);
      o2.connect(g2).connect(g);
      g.connect(p).connect(intro);
      o.start(at); o2.start(at);
      o.stop(at + 0.25); o2.stop(at + 0.25);
    }
  }

  let started = false;
  let closed = false;
  let held = false; // a hold is under way
  const api = {
    ctx,
    /** Starts every source. `cold`: play the cold-open score from `at`; else fade the room bed in. */
    start({ cold = true, at = now() + 0.03 } = {}) {
      if (started) return;
      started = true;
      if (ctx.state === "suspended" && typeof ctx.resume === "function") ctx.resume().catch(() => undefined);
      for (const s of sources) s.start(at, s._offset || 0);
      master.gain.setValueAtTime(0, at);
      master.gain.setTargetAtTime(1, at, 0.02);
      if (cold && plan) {
        hum(at, at + introCues(plan, wordStartsMs).end);
        line(at);
        rush(at);
        ticks(at);
        for (const [ms, v] of roomBreakpoints(plan)) roomBus.gain.linearRampToValueAtTime(v * ROOM_GAIN, at + ms / 1000);
      } else {
        roomBus.gain.setValueAtTime(0, at);
        roomBus.gain.setTargetAtTime(ROOM_BED * ROOM_GAIN, at, 0.4);
      }
    },
    /** The cold open was skipped: the score's layers fade, the room settles to its bed. */
    skip(at = now()) {
      if (!started || closed) return;
      retarget(intro.gain, 0, at, 0.06);
      retarget(roomBus.gain, ROOM_BED * ROOM_GAIN, at, 0.25);
    },
    /** A hold began: make sure the room is there to be turned down. */
    press(at = now()) {
      if (!started || closed) return;
      held = true;
      retarget(intro.gain, 0, at, 0.06);
      if (roomBus.gain.value < 0.5 * ROOM_BED * ROOM_GAIN) retarget(roomBus.gain, ROOM_BED * ROOM_GAIN, at, 0.08);
    },
    /** The ring at progress `p`: the room low-passes and ducks, the voice rises. Call per frame. */
    hold(p, at = now()) {
      if (!started || closed || !held) return;
      const m = holdMix(p);
      roomLP.frequency.setTargetAtTime(m.cutoffHz, at, 0.03);
      roomDuck.gain.setTargetAtTime(m.room, at, 0.03);
      voice.gain.setTargetAtTime(m.voice * VOICE_GAIN, at, 0.04);
    },
    /** Let go early (or the mic failed): the room comes back, the voice goes. */
    release(at = now()) {
      if (!started || closed) return;
      held = false;
      roomLP.frequency.setTargetAtTime(9000, at, 0.12);
      roomDuck.gain.setTargetAtTime(1, at, 0.1);
      voice.gain.setTargetAtTime(0, at, 0.07);
    },
    /** The ring completed: the room is gone, the voice resolves into a chord that rings out. */
    resolve(at = now()) {
      if (!started || closed) return;
      held = false;
      roomDuck.gain.setTargetAtTime(0, at, 0.05);
      voice.gain.setTargetAtTime(0, at + 0.03, 0.08);
      CHORD_HZ.forEach((f, i) => {
        const o = ctx.createOscillator();
        o.type = i === 0 ? "triangle" : "sine";
        o.frequency.value = f;
        const g = ctx.createGain();
        const t = at + i * 0.018; // a soft strum
        g.gain.setValueAtTime(0, t);
        g.gain.setTargetAtTime(i === 0 ? 0.05 : 0.04, t, 0.012);
        g.gain.setTargetAtTime(0, t + 0.16, 0.42);
        o.connect(g).connect(panner(ctx, (i - 1.5) * 0.25)).connect(master);
        o.start(t);
        o.stop(t + 3.5);
      });
    },
    /** The mic opened: everything goes to silence quickly. */
    silence(at = now()) {
      if (!started || closed) return;
      retarget(master.gain, 0, at, 0.04);
    },
    /** Sound toggle off (or the tab hid): fade out and suspend; true again resumes. */
    mute(on, at = now()) {
      if (closed) return;
      if (on) {
        retarget(master.gain, 0, at, 0.04);
        if (typeof ctx.suspend === "function" && !("startRendering" in ctx)) setTimeout(() => { if (!closed) ctx.suspend().catch(() => undefined); }, 250);
      } else {
        if (ctx.state === "suspended") ctx.resume().catch(() => undefined);
        retarget(master.gain, 1, at, 0.1);
      }
    },
    /** Fade out and release the context (live only). */
    close(afterMs = 200) {
      if (closed) return;
      if (started) retarget(master.gain, 0, now(), 0.03);
      closed = true;
      setTimeout(() => { if (typeof ctx.close === "function") ctx.close().catch(() => undefined); }, afterMs);
    },
  };
  return api;
}

/** Creates the live context in the caller's gesture; null where WebAudio is missing or refuses. */
export function makeLiveSound(opts) {
  const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Ctx) return null;
  try {
    const ctx = new Ctx({ latencyHint: "playback" });
    return createWelcomeSound(ctx, opts);
  } catch {
    return null;
  }
}

/**
 * The opening as a recording: the cold open from 0 s, a hold let go at 60 % (`letGo`), a full hold that resolves
 * (`full`), and the mic opening (`mic`), rendered offline. Returns the AudioBuffer. Used by
 * web/app/tests/welcome_sound_render.py for docs/ux-review-shots/welcome-sound-sample.wav.
 */
export async function renderOpening({ plan, wordStartsMs, seconds = 15, sampleRate = 22050, holdMs = 900,
  letGo = 7.6, full = 10.2, mic = 11.6, Offline = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext } = {}) {
  const ctx = new Offline(2, Math.round(seconds * sampleRate), sampleRate);
  const snd = createWelcomeSound(ctx, { plan, wordStartsMs });
  snd.start({ cold: true, at: 0 });
  const holdFrom = (t0, upTo) => {
    snd.press(t0);
    for (let t = 0; t <= upTo * holdMs; t += 1000 / 60) snd.hold(Math.min(1, t / holdMs), t0 + t / 1000);
  };
  holdFrom(letGo, 0.6);
  snd.release(letGo + 0.6 * holdMs / 1000);
  holdFrom(full, 1);
  snd.resolve(full + holdMs / 1000);
  snd.silence(mic);
  return ctx.startRendering();
}
