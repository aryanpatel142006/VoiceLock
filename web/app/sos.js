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
// Comfort Companion (PR #402, comfort.js): with the `hearing.comfort` setting on, the calm frame also asks "What's too
// much right now?" and shows a plan; its buttons live in .sos-comfort, which (like the sound toggle) never starts the
// hold. The plan's sound chip calls setSound(); "Just quiet" fades the calm loop out while SOS stays on.
//
// Diagnostics: window.hearingSos.last.pressToSilenceMs is performance.now() from the press event to the moment the
// AudioContext clock passes the end of the 150 ms ramp (plus outputLatencyMs, the device's own buffer, reported
// apart). The pure half (band match, keys, hold, noise buffer, contrast) is exported for node --test.

import { KINDS, STORAGE_KEY, chooseKind, loadSos, matchesBand, soundKind } from "./screens/sos.js";
import { loadComfort, mountComfort } from "./comfort.js";
import {
  MUSIC_BPM, MUSIC_CHORDS, MUSIC_LOOP_S, MUSIC_LOWPASS_HZ, NOISE_DBFS, NOISE_LOOP_S, NOISE_LOWPASS_HZ,
  cachedSamples, hasWorker, knownRate, lowpass, musicSamples, noiseSamples, noteRate, prewarmSamples, renderSamples,
  rmsDb, rng, samplesSync, soundSamples,
} from "./synth.js";

// the pure synth lives in synth.js (so a Worker can run it); tests and sounds.js keep importing it from here
export { MUSIC_BPM, MUSIC_CHORDS, MUSIC_LOOP_S, MUSIC_LOWPASS_HZ, NOISE_DBFS, NOISE_LOOP_S, NOISE_LOWPASS_HZ,
  lowpass, musicSamples, noiseSamples, rmsDb, rng, soundSamples };

export const SOS_KEY = STORAGE_KEY;
export const CUT_MS = 150; // room -> silence
export const NOISE_IN_MS = 400; // calm noise fade-in
export const SWAP_MS = 400; // noise <-> music cross-fade while SOS is on
export const RETURN_MS = 600; // noise out, room back
export const HOLD_MS = 1500; // hold to come back
export const FADE_OUT_MS = 200; // the calm frame's fade-out (sos.css sos-fade-out); the fade-in is 220 ms
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
  // Unpaired navigation keys keep their normal browser behavior; a paired key is handled below
  if (!trigger && (/^(Arrow|Page)/.test(e.key) || e.key === "Escape")) return null;
  // a button on the calm frame (sound toggle, Comfort) keeps Space/Enter: "pass" = its default (the button's click)
  // runs, but the stage keys (keys.js: Space = A/B, which would cancel the click) never see it
  if (e.inToggle && !trigger && (e.key === " " || e.key === "Enter")) return e.repeat ? "swallow" : "pass";
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
    case "back": // a screen reader or switch pressed Hold to go back once: no pointer or key to hold
      return s.phase === "on" ? { ...s, phase: "off", source: null, progress: 0 } : s;
    case "reset":
      return sosInit();
    default:
      return s;
  }
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

// a control on the calm frame, not the hold. The call overlay sits beside .sos-comfort (comfort.js), so it is named too:
// without it a press on End was taken as the hold (pointer captured by the frame) and End's click never fired
export function inToggle(t) {
  return Boolean(t?.closest?.(".sos-calm-kinds, .sos-comfort, .cf-callscreen"));
}

function inField(t) {
  return Boolean(t?.closest?.("input, select, textarea, [contenteditable='true']"));
}

function mount() {
  const doc = document;
  const win = window;
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
  const holdLabel = el("span", { class: "sos-calm-hold-label", text: "Hold to go back" });
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
  let latched = new Set(); // one keydown per press; repeats cannot start a second action
  let opener = null;
  let quiet = false; // Comfort: this loop stays silent; Play My Sounds (sounds.js, started by the press) is the sound
  let modeBefore = null; // the engine mode before SOS put it in Quiet; restored on the way back
  const showHooks = []; // comfort.js listens for the frame coming on / going off
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

  /** The engine's own Quiet (T-149): the extracted voice stops, captions keep coming. Back to Focus on the way out. */
  function engineQuiet(on) {
    const sess = session();
    if (typeof sess?.setMode !== "function") return;
    try {
      if (on) {
        if (modeBefore === null) modeBefore = sess.store?.state?.mode || "focus_hq";
        sess.setMode("quiet");
      } else {
        sess.setMode("focus_hq");
        modeBefore = null;
      }
    } catch { /* no live session: nothing to switch */ }
  }
  const pairing = () => doc.body.dataset.screen === "sos" || win.location.hash === "#sos";

  function describeKeys() {
    const b = band.label || band.key || band.code;
    keyHint.textContent = b ? `Band (${b}): press to stop audio; hold 1.5 seconds to return to Focus.`
      : "Hold anywhere, or hold Space on the big button. With a screen reader or switch, press the Hold to go back button once.";
  }

  function describeSound() {
    const kind = soundKind(band.sound);
    for (const b of kindBtns) b.setAttribute("aria-pressed", String(b.dataset.kind === kind));
    line.textContent = quiet ? "The room is off. It's quiet."
      : kind === "music" ? "The room is off. Only soft music plays." : "The room is off. Only a soft hush plays.";
  }

  /** The comfort plan's sound: a noise colour, the music, or "silence" (this press only; the saved choice is kept). */
  function setSound(id) {
    if (id === "silence") {
      quiet = true;
      if (noise) ramp(noise.gain.gain, noise.ctx.currentTime, 0, SWAP_MS);
      describeSound();
      return;
    }
    if (id !== "music" && !["brown", "pink", "white"].includes(id)) return;
    const wasQuiet = quiet;
    quiet = false;
    band = { ...band, sound: id };
    describeSound();
    warm();
    if (s.phase === "off") return;
    swapSound();
    if (wasQuiet && noise) ramp(noise.gain.gain, noise.ctx.currentTime, 1, SWAP_MS);
  }

  // ---- sound: master gain (fade in / out) <- one voice per playing loop (its own gain, for the noise/music swap)
  function makeCtx() { // created suspended outside a gesture (warm()); press() resumes it
    if (!noise) {
      const Ctx = win.AudioContext || win.webkitAudioContext;
      if (!Ctx) return null;
      const ctx = new Ctx({ latencyHint: "interactive" });
      noteRate(ctx.sampleRate); // synth.js renders at the real output rate from here on
      const gain = ctx.createGain();
      gain.gain.value = 0;
      gain.connect(ctx.destination);
      noise = { ctx, gain, voice: null, buffers: new Map(), waiting: null };
    }
    return noise;
  }

  function setSink(n) {
    const sink = session()?._rt?.audio?.ctx?.sinkId; // the output the app plays to (devices.js picker)
    if (typeof sink === "string" && sink && n.ctx.setSinkId && n.ctx.sinkId !== sink) n.ctx.setSinkId(sink).catch(() => {});
  }

  function noiseCtx() {
    if (!makeCtx()) return null;
    noise.ctx.resume?.().catch(() => {}); // the press path: the context already exists (prime), so this is a resume only
    setSink(noise);
    return noise;
  }

  /** Perf: opening an AudioContext (the device, the audio thread) is the slow part of the first press, so it is made
   * on the first gesture anywhere on the page (Start, a key, a tap), kept suspended, and press() only resumes it.
   * The sound loops render in the worker right after, at the context's real rate. */
  function prime() {
    win.removeEventListener("pointerdown", prime, true);
    win.removeEventListener("keydown", prime, true);
    if (noise || !makeCtx()) return;
    noise.ctx.suspend?.().catch(() => {}); // made inside a gesture it would run; it plays nothing until the press
    setSink(noise);
    warm();
  }
  win.addEventListener("pointerdown", prime, true);
  win.addEventListener("keydown", prime, true);

  /** The AudioBuffer for a loop, or null when the worker has not rendered it yet (startVoice waits for it). Nothing
   * is synthesised on this thread while a worker exists: a loop is 50-250 ms of arithmetic, a visible stall. */
  function bufferFor(n, sound) {
    if (n.buffers.has(sound)) return n.buffers.get(sound);
    const rate = n.ctx.sampleRate;
    const x = cachedSamples(sound, rate) || (hasWorker() ? null : samplesSync(sound, rate));
    if (!x) return null;
    const buf = n.ctx.createBuffer(1, x.length, rate);
    buf.copyToChannel(x, 0);
    n.buffers.set(sound, buf);
    return buf;
  }

  function startVoice(n, sound, fadeMs) {
    const { ctx } = n;
    const buf = bufferFor(n, sound);
    if (!buf) { // not rendered yet (a cold cache): the worker renders it and the voice fades in when it arrives
      const token = { sound };
      n.waiting = token;
      renderSamples(sound, ctx.sampleRate).then(() => {
        if (n.waiting !== token) return;
        n.waiting = null;
        if (s.phase === "off" || quiet || band.sound !== sound || n.voice?.sound === sound) return;
        startVoice(n, sound, Math.max(fadeMs, NOISE_IN_MS));
      }, () => { if (n.waiting === token) n.waiting = null; });
      return;
    }
    n.waiting = null;
    const g = ctx.createGain();
    g.gain.value = fadeMs ? 0 : 1;
    g.connect(n.gain);
    const src = ctx.createBufferSource();
    src.buffer = buf;
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

  /** Render the loops off the press path: the chosen one first, then the rest, in the worker while the page is idle
   * (synth.js caches them for the page, so Play My Sounds finds them too). Without a worker the old path stays:
   * the music is pre-rendered once on this thread when it is the chosen sound. */
  function warm() {
    const rate = noise?.ctx.sampleRate || session()?._rt?.audio?.ctx?.sampleRate || knownRate();
    if (prewarmSamples([band.sound, "brown", "pink", "white", "music"], rate, win)) return;
    if (band.sound !== "music" || noise?.buffers?.has("music")) return;
    const idle = win.requestIdleCallback || ((f) => win.setTimeout(f, 200));
    idle(() => { if (band.sound === "music" && makeCtx()) bufferFor(noise, "music"); });
  }

  function noiseOut() {
    if (noise) noise.waiting = null;
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
    const comfortOn = loadComfort((() => { try { return win.localStorage; } catch { return null; } })()).on;
    quiet = comfortOn; // Comfort: this calm loop stays silent; show(true) below opens Play My Sounds, which starts the
    // remembered sound inside this press (comfort.js autoplaySounds). Pressing the band again stops it (sosKey "hold").
    describeSound();
    const sess = session();
    const cut = sess?.setOutputGain?.(0, CUT_MS) || null; // first: the room goes before anything is drawn
    if (!comfortOn) noiseIn();
    measure(pressAt, source, cut);
    opener = doc.activeElement;
    show(true);
    dispatch({ type: "press" });
    // Perf: the engine's Quiet control and the socket message wait for the next frame, so the calm frame paints
    // first (the room is already cut on the master gain above; the engine only stops its stream a frame later)
    win.requestAnimationFrame(() => {
      if (s.phase === "off") return; // came back within the frame
      if (comfortOn) engineQuiet(true);
      sess?.send?.({ sos: true });
    });
  }

  function comeBack() {
    session()?.setOutputGain?.(1, RETURN_MS);
    engineQuiet(false);
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

  /** The frame comes and goes on an opacity fade (sos.css, compositor only). Not a view transition: that snapshotted
   * the whole page into two full-screen textures on every open and close, a visible stall on a Retina display with
   * the engine on the same GPU. The fade-out keeps the frame drawn (inert to the pointer) until it is gone. */
  let leaveTimer = 0;
  function show(on) {
    win.clearTimeout(leaveTimer);
    leaveTimer = 0;
    if (on) {
      delete frame.dataset.leaving;
      frame.hidden = false;
    } else {
      frame.dataset.leaving = "1";
      leaveTimer = win.setTimeout(() => { leaveTimer = 0; frame.hidden = true; delete frame.dataset.leaving; }, FADE_OUT_MS);
    }
    doc.body.dataset.sos = on ? "on" : "";
    for (const n of [doc.querySelector(".skip"), doc.getElementById("chrome"), doc.getElementById("stage"), doc.getElementById("sheet"), btn]) {
      if (n) n.inert = on;
    }
    if (on) hold.focus({ preventScroll: true });
    else if (opener?.isConnected && opener !== doc.body) opener.focus?.({ preventScroll: true });
    for (const f of showHooks) f(on);
  }

  // ---- hold state
  function dispatch(ev) {
    const prev = s;
    s = sosReduce(s, ev);
    if (s === prev) return;
    frame.dataset.phase = s.phase;
    hold.style.setProperty("--sos-p", s.progress.toFixed(3));
    if (prev.phase !== s.phase) {
      holdLabel.textContent = s.phase === "holding" ? "Keep holding" : "Hold to go back";
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
  // A screen reader, switch or voice control "clicks" Hold to go back with no pointer behind it (detail 0) and cannot
  // hold, so that one press comes back. A key held on the button is the ordinary hold: heldByKey keeps it out.
  let heldByKey = false;
  hold.addEventListener("keydown", () => { heldByKey = true; });
  hold.addEventListener("keyup", () => { heldByKey = false; });
  hold.addEventListener("click", (e) => {
    if (e.detail === 0 && !heldByKey) dispatch({ type: "back" });
  });

  doc.addEventListener("keydown", (e) => {
    const what = sosKey({ key: e.key, code: e.code, shiftKey: e.shiftKey, metaKey: e.metaKey, ctrlKey: e.ctrlKey, altKey: e.altKey,
      repeat: e.repeat, inField: inField(e.target), inToggle: inToggle(e.target) }, { on: s.phase !== "off", band, pairing: pairing() });
    if (!what) return;
    if (what === "pass") { e.stopImmediatePropagation(); return; }
    e.preventDefault();
    e.stopImmediatePropagation(); // keys.js (Space = A/B) is bound on document after this module
    const id = `key:${e.code || e.key}`;
    const bandPress = matchesBand(band, e);
    if (what === "press") {
      latched.add(id);
      press(e.timeStamp || win.performance.now(), isShiftS(e) ? "shift+s" : "band");
      // A long press of the one band button also returns from a closed SOS frame.
      if (bandPress) dispatch({ type: "hold", source: id, at: win.performance.now() });
    } else if (what === "hold" && !latched.has(id)) {
      latched.add(id);
      if (bandPress) {
        win.hearingComfort?.stop?.();
        setSound("silence");
        engineQuiet(true);
      }
      dispatch({ type: "hold", source: id, at: win.performance.now() });
    }
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
  mountComfort({
    doc, win, body: frame.querySelector(".sos-calm-body"), before: kinds, setSound,
    soundNow: () => (quiet ? null : soundKind(band.sound)),
    setQuiet: (on) => { try { session()?.setMode?.(on ? "quiet" : "focus_hq"); } catch { /* no live session */ } },
    modeNow: () => session()?.store?.state?.mode || null,
    captions: () => session()?.store?.state?.captions || [],
    store: () => session()?.store || null,
    onShow: (f) => showHooks.push(f),
    focusHold: () => { if (s.phase !== "off") hold.focus({ preventScroll: true }); },
  });
}

if (typeof document !== "undefined" && typeof window !== "undefined") {
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount, { once: true });
}
