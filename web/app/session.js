// App session (M14, T-346): the /ws/session socket, mic capture and playback for /app, ported from
// web/console/judge.js so the stage page and the app share one behaviour: the judge profile on connect, the
// RAW side delayed by the engine's declared latency (abDelayMs) so A/B line up, the jitter buffer per mode
// (T-135), reconnect with backoff (0.5/1/2/4 s), the iPhone-mic dropout swap without a session restart and the
// hello device labels (T-121/T-133/T-287), the AGC floor-rule mirror on RAW (T-179) and the server's replay.
// The app is audio-only (the decided demo has no camera): hello.video is false and no video is ever opened.
//
// Writes only to the store (web/app/store.js); screens read the store and call the methods returned here.
// Everything that touches the browser is injectable (`env`), so node --test drives it with fakes.

import { MIC_START_TIMEOUT_MS, micProblem, withTimeout } from "../console/judge-pill.js";
import { RELEASE_CONTROL, personControl, savedPeople } from "../console/judge-targets.js";
import { mountDevices, peakDb } from "../console/devices.js";
import {
  KIND_AUDIO_IN, KIND_AUDIO_OUT, audioContextOptions, control, floatToPcm16, hello, packFrame, pcm16ToFloat,
  sessionUrl, unpackFrame,
} from "../console/protocol.js";
import {
  APP_MODES, abWorklet, backoffMs, jitterFor, jitterSamples, meterStep, modeFields, nextAb, store as appStore,
  addCaption,
} from "./store.js";

export const FRAME_MS = 20;
export const RENDER_MS = 100;
export const CALIBRATE_S = 3;
export const CLIENT = "app";
const FIRST_TELEMETRY_MS = 1500; // Start waits this long for the preset (its demo mic) when connected without telemetry
const WS_OPEN = 1;
const CONTROL_SETTLE_MS = 1500; // a control this page sent is in the telemetry well within this (10 Hz ticks)

/** The real browser audio path: AudioContext (inside the Start tap), the shared worklet, the mic stream. */
async function browserAudio({ playbackOnly, jitter, devices }) {
  let ctx = new AudioContext(audioContextOptions());
  ctx.resume().catch(() => {}); // inside the user gesture (iOS); WebKit rejects on a closed context (mic denied)
  let stream;
  try {
    stream = playbackOnly ? new MediaStream() : await devices.openMic();
  } catch (e) {
    ctx.close();
    throw e;
  }
  let source;
  try {
    if (!playbackOnly) source = ctx.createMediaStreamSource(stream);
  } catch { // Firefox: a 16 kHz context cannot take a 48 kHz mic
    await ctx.close();
    ctx = new AudioContext({ latencyHint: "interactive" });
    if (!playbackOnly) source = ctx.createMediaStreamSource(stream);
  }
  await ctx.audioWorklet.addModule(new URL("../console/audio-worklet.js", import.meta.url));
  const rate = ctx.sampleRate;
  const node = new AudioWorkletNode(ctx, "hearing-io", {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
    processorOptions: { frameSamples: Math.round((rate * FRAME_MS) / 1000), ...jitterSamples(jitter, rate) },
  });
  source?.connect(node);
  // T-460: one master gain in front of the speakers, so SOS (web/app/sos.js) can silence the room client-side
  const master = ctx.createGain();
  node.connect(master);
  master.connect(ctx.destination);
  return { ctx, node, stream, rate, source, master };
}

function browserEnv() {
  const g = globalThis;
  return {
    WebSocket: g.WebSocket,
    setTimeout: (...a) => g.setTimeout(...a),
    clearTimeout: (...a) => g.clearTimeout(...a),
    setInterval: (...a) => g.setInterval(...a),
    clearInterval: (...a) => g.clearInterval(...a),
    fetch: (...a) => g.fetch(...a),
    perfNow: () => g.performance.now(),
    wallNow: () => (g.performance.timeOrigin + g.performance.now()) / 1000, // capture_ts, client clock (s)
    location: g.location,
    openAudio: browserAudio,
  };
}

/**
 * Create the app session. `opts`: {store, url, devices, env}. `devices` defaults to web/console/devices.js mounted
 * into `session.devicesElement` (the Setup screen moves that element where it wants it; its red mic banner is
 * page-level, as on /judge). Call `connect()` once at load and `start()` from the Start tap.
 */
export function createSession({ store = appStore, url = null, devices = null, env: envIn = {} } = {}) {
  const env = { ...browserEnv(), ...envIn };
  const rt = {
    controlAt: -Infinity, // perfNow() of this page's last mode / A/B control (adoptRemote waits it out)
    url: url || sessionUrl(env.location || { search: "", protocol: "http:", host: "" }),
    ws: null,
    retryTimer: null,
    renderTimer: null,
    audio: null,
    seq: 0,
    frameAt: 0, // perfNow() of the last audio frame sent (or of Start finishing): the "mic not streaming" clock
    startToken: 0,
    rttMs: null,
    jitter: "focus_hq",
    agcFloorRule: false,
    levelAt: 0,
    level: -60,
    playAt: 0,
    play: -60,
    hqActive: null,
    sent: [], // every control sent (devtools and tests)
    closed: false,
    outputGain: 1, // T-460 master gain (SOS); kept across Stop/Start so a restart under SOS stays silent
  };
  const set = (patch) => store.set(patch);
  const st = () => store.state;
  const note = (text) => store.note(text);

  let devicesElement = null;
  if (!devices && typeof document !== "undefined") {
    devicesElement = document.createElement("div");
    devicesElement.className = "app-devices";
    devices = mountDevices(devicesElement, {
      onMicChange: async () => { if (rt.audio && !st().replaying) { stopMedia(); await start(); } },
      onNote: note,
      onSwap: (stream, info) => swapMic(stream, info),
    });
  }

  // ---------------------------------------------------------------- socket

  function connect() {
    rt.closed = false;
    env.clearTimeout(rt.retryTimer);
    let ws;
    set({ connection: st().attempt > 0 ? "reconnecting" : "connecting", retryInMs: null });
    tick();
    try {
      ws = new env.WebSocket(rt.url + (rt.url.includes("?") ? "&" : "?") + "replay=1");
    } catch (e) {
      lost(e?.message || String(e));
      return;
    }
    ws.binaryType = "arraybuffer";
    rt.ws = ws;
    ws.onopen = () => {
      if (rt.ws !== ws) return;
      if (st().attempt > 0) note("engine back");
      rt.frameAt = env.perfNow(); // the stall clock starts again on reconnect
      set({ connection: "open", attempt: 0, retryInMs: null });
      store.emit("connection", { connection: "open", attempt: 0, retryInMs: null });
      sendHello();
      sendProfile({ ab: st().ab, ...modeFields(st().mode) });
      tick();
    };
    ws.onmessage = (e) => (typeof e.data === "string" ? onText(e.data) : onBinary(e.data));
    ws.onclose = () => { if (rt.ws === ws) lost("socket closed"); };
  }

  function lost(why) {
    const s = st();
    if (s.connection === "open" || s.attempt === 0) note(`engine lost (${why}); retrying`);
    rt.ws = null;
    if (s.replaying) clearReplay();
    store.resetEngine();
    const delay = backoffMs(s.attempt);
    set({ connection: "reconnecting", attempt: s.attempt + 1, retryInMs: delay, telemetry: null, sessionId: null });
    store.emit("session", { sessionId: null });
    store.emit("connection", { connection: "reconnecting", attempt: s.attempt + 1, retryInMs: delay });
    tick();
    if (!rt.closed) rt.retryTimer = env.setTimeout(connect, delay);
  }

  function send(obj) {
    if (rt.ws?.readyState === WS_OPEN) rt.ws.send(JSON.stringify(obj));
  }

  function sendControl(fields) {
    const msg = control(fields);
    rt.sent.push(msg);
    if (rt.sent.length > 200) rt.sent.shift();
    send(msg);
    return msg;
  }

  /** The judge profile (first saved person, auto-calibration on); after Release the release follows it, so a
   * reconnect never re-targets on its own (T-177). */
  function sendProfile(fields = {}) {
    rt.controlAt = env.perfNow();
    const iso = st().isolation;
    // the room dial survives a reconnect: the profile resets suppression to 100, so a turned-up room is re-sent (T-427)
    const sup = st().suppression;
    sendControl({ profile: "judge", ...(iso ? { isolation: iso } : {}), ...(sup !== 100 && sup != null ? { suppression: sup } : {}), ...fields });
    if (st().released) sendControl(RELEASE_CONTROL);
  }

  /** The mic and output in use (T-121), saved devices not found (T-133). */
  function deviceLabels() {
    if (!devices) return {};
    return { input_device: devices.inputLabel()?.slice(0, 200) ?? null, output_device: devices.outputLabel()?.slice(0, 200) ?? null,
      devices_missing: devices.missing() };
  }

  function sendHello(extra = {}) {
    // extra (T-287 mic_switch) right after "type": session_report prints the first 160 characters of each hello
    send({ type: "hello", ...extra, ...hello(rt.audio ? rt.audio.rate : 16000, FRAME_MS, false), client: CLIENT, ...deviceLabels() });
  }

  function onText(text) {
    let msg;
    try { msg = JSON.parse(text); } catch { return; }
    if (msg.type === "telemetry") onTelemetry(msg);
    else if (msg.type === "caption") {
      set({ captions: addCaption(st().captions, msg) });
      store.emit("caption", msg);
    } else if (msg.type === "error") {
      note(`${msg.stage}: ${msg.message}${msg.degraded_to ? ` → ${msg.degraded_to}` : ""}`);
      store.emit("error", msg);
    } else if (msg.type === "session") {
      set({ sessionId: msg.session_id || null });
      tick();
      store.emit("session", { sessionId: msg.session_id || null });
    } else if (msg.type === "replay") {
      if (msg.active) set({ replaying: true, replayClip: msg.clip || null });
      else clearReplay();
      tick();
      store.emit("replay", { active: Boolean(msg.active), clip: msg.clip || null });
    }
  }

  function onTelemetry(t) {
    set({ telemetry: t, ...adoptRemote(t) });
    // T-287: before the next render, so a Start right after the first telemetry opens the preset's demo mic
    devices?.setExpected({ input: t.preset?.pins?.input_device || null, output: t.preset?.pins?.output_device || null });
    postAb();
    setJitter(jitterFor(t));
    watchBackend(t);
    const floorRule = !!t.agc?.floor_rule; // T-179: the server's AGC floor rule, mirrored on the RAW side
    if (rt.audio && floorRule !== rt.agcFloorRule) {
      rt.agcFloorRule = floorRule;
      rt.audio.node.port.postMessage({ type: "agc", on: true, floorRule });
    }
    tick();
    store.emit("telemetry", t);
  }

  function watchBackend(t) {
    const active = t.hq_active || null;
    if (rt.hqActive && active && active !== rt.hqActive) {
      note(active === "local" ? `GPU lost → Mac ladder (crossfade): ${t.hq_reason || ""}` : `GPU back: ${t.hq_reason || ""}`);
    }
    rt.hqActive = active;
  }

  function onBinary(buffer) {
    let frame;
    try { frame = unpackFrame(buffer); } catch { return; }
    if (st().replaying && frame.kind === KIND_AUDIO_IN) { // the replayed room: RAW side of A/B, and the meter
      const samples = pcm16ToFloat(frame.payload);
      meter(samples);
      rt.audio?.node.port.postMessage({ type: "replay-raw", samples }, [samples.buffer]);
      return;
    }
    if (frame.kind !== KIND_AUDIO_OUT) return;
    if (frame.captureTs > 0) {
      const rtt = 1000 * (env.wallNow() - frame.captureTs);
      rt.rttMs = rt.rttMs == null ? rtt : 0.9 * rt.rttMs + 0.1 * rtt;
    }
    const samples = pcm16ToFloat(frame.payload);
    playback(samples); // before the transfer below empties the buffer
    rt.audio?.node.port.postMessage({ type: "play", samples }, [samples.buffer]);
  }

  // ---------------------------------------------------------------- media

  /** Start (call it from the tap: the AudioContext is created synchronously inside it). Resolves when streaming. */
  async function start() {
    if (rt.audio && !st().replaying) return true;
    if (st().replaying) await replay(false);
    const token = ++rt.startToken;
    set({ starting: true, mic: { ...st().mic, banner: "" } });
    try {
      // T-158: a hung permission prompt or addModule must not leave Start disabled for ever
      await withTimeout(startAudio(false, token), MIC_START_TIMEOUT_MS, env.setTimeout, env.clearTimeout);
    } catch (e) {
      if (token !== rt.startToken) return false; // a newer Start owns the state
      rt.startToken++; // abandon: if it resolves later, startAudio tears it down
      stopMedia();
      note(`microphone unavailable: ${e?.message || e}`);
      set({ starting: false, mic: { ...st().mic, banner: micProblem(e) } });
      return false;
    }
    if (!st().telemetry && st().connection === "open") await firstTelemetry(FIRST_TELEMETRY_MS);
    sendHello();
    set({ starting: false, started: true });
    tick();
    return true;
  }

  async function startAudio(playbackOnly = false, token = null) {
    const audio = await env.openAudio({ playbackOnly, jitter: rt.jitter, devices });
    if (token !== null && token !== rt.startToken) { // T-158: timed out or superseded while pending
      audio.stream?.getTracks?.().forEach((t) => t.stop());
      audio.ctx?.close?.();
      throw new Error("abandoned: Start timed out or was pressed again");
    }
    audio.node.port.postMessage({ type: "agc", on: true, ...(rt.agcFloorRule ? { floorRule: true } : {}) }); // T-135 RAW levelled
    audio.node.port.onmessage = (e) => {
      if (e.data?.type !== "frame" || st().replaying) return;
      devices?.feed(e.data.samples);
      meter(e.data.samples);
      if (rt.ws?.readyState !== WS_OPEN) return;
      rt.ws.send(packFrame(KIND_AUDIO_IN, rt.seq++, env.wallNow(), floatToPcm16(e.data.samples)));
      rt.frameAt = env.perfNow();
    };
    await audio.ctx?.resume?.();
    if (audio.master && rt.outputGain !== 1) audio.master.gain.value = rt.outputGain;
    rt.audio = audio;
    rt.seq = 0;
    rt.frameAt = env.perfNow();
    if (!playbackOnly) await devices?.started(audio.stream, audio.ctx);
    postAb();
  }

  /** The level the scene and Setup meter read: peak of each frame sent (or replayed), 24 dB/s release. */
  function meter(samples) {
    const now = env.perfNow();
    const m = meterStep(rt.level, peakDb(samples), rt.levelAt ? (now - rt.levelAt) / 1000 : 0);
    rt.level = m.db;
    rt.levelAt = now;
    store.emit("level", { db: m.db, band: m.band, replay: st().replaying });
  }

  /** The engine's output as it is queued to play: peak of each audio-out frame, the meter's 24 dB/s release. It
   * falls when the chosen person stops talking (the scene's target form dims), unlike agc.level_db, which holds. */
  function playback(samples) {
    const now = env.perfNow();
    rt.play = meterStep(rt.play, peakDb(samples), rt.playAt ? (now - rt.playAt) / 1000 : 0).db;
    rt.playAt = now;
    store.emit("playback", { db: rt.play });
  }

  function firstTelemetry(ms) {
    return new Promise((resolve) => {
      const t0 = env.perfNow();
      const poll = () => (st().telemetry || env.perfNow() - t0 > ms ? resolve() : env.setTimeout(poll, 50));
      poll();
    });
  }

  /** T-287: the mic changed mid-session (the iPhone dropped, or Retry switched back): feed the running worklet
   * from the new stream without restarting the socket or session, and say so in a hello (`mic_switch`). */
  function swapMic(stream, info) {
    const a = rt.audio;
    if (!a?.source) return false;
    let source;
    try {
      source = a.ctx.createMediaStreamSource(stream);
    } catch (e) {
      note(`mic switch to ${info.to} failed: ${e?.message || e}`);
      return false;
    }
    a.source.disconnect();
    a.stream.getTracks().forEach((t) => t.stop());
    source.connect(a.node);
    Object.assign(a, { source, stream });
    rt.frameAt = env.perfNow();
    note(`mic ${info.reason}: ${info.from || "?"} → ${info.to}`);
    sendHello({ mic_switch: `${info.reason}: ${info.from || "?"} -> ${info.to}`.slice(0, 300) });
    tick();
    return true;
  }

  function stopMedia() {
    if (rt.audio) {
      rt.audio.stream?.getTracks?.().forEach((t) => t.stop());
      rt.audio.ctx?.close?.();
      rt.audio = null;
      devices?.stopped();
    }
    rt.rttMs = null;
    rt.level = -60;
    rt.levelAt = 0;
    rt.play = -60;
    rt.playAt = 0;
    store.emit("playback", { db: -60 });
    set({ started: false, starting: false });
    sendHello();
    tick();
  }

  /** Playback jitter buffer for the mode playing; the worklet applies it on the next re-buffer. */
  function setJitter(kind) {
    if (kind === rt.jitter) return;
    rt.jitter = kind;
    set({ jitter: kind });
    if (rt.audio?.rate) rt.audio.node.port.postMessage({ type: "config", ...jitterSamples(kind, rt.audio.rate) });
  }

  function postAb() {
    rt.audio?.node.port.postMessage(abWorklet(st().ab, st().telemetry));
  }

  // ---------------------------------------------------------------- replay (the server feeds this session)

  function endpoint(path) {
    const u = new URL(rt.url);
    u.protocol = u.protocol === "wss:" ? "https:" : "http:";
    u.pathname = path;
    u.search = "";
    return u;
  }

  /** The server's default replay clip (the team mix under the team preset), else the demo clip. */
  async function defaultClip() {
    try {
      const res = await env.fetch(endpoint("/replay/clips"));
      if (res.ok) return (await res.json()).default;
    } catch { /* older server */ }
    return "demo-mf-easy/clip.mp4";
  }

  /** Shift+R on /judge: the prerecorded clip through this session, behind the REPLAY badge. `on` false stops it. */
  async function replay(on = !st().replaying, clip = null) {
    const s = st();
    if (!s.sessionId || rt.ws?.readyState !== WS_OPEN) throw new Error("Connect to the server first");
    const u = endpoint("/replay");
    if (!on) {
      u.search = new URLSearchParams({ session_id: s.sessionId, action: "stop" });
      await env.fetch(u, { method: "POST" });
      return false;
    }
    stopMedia();
    set({ replaying: true });
    try {
      await startAudio(true); // playback only: never asks for the mic
      u.search = new URLSearchParams({ session_id: s.sessionId, action: "start", loop: "true", clip: clip || await defaultClip() });
      const res = await env.fetch(u, { method: "POST" });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Replay failed");
    } catch (e) {
      clearReplay();
      note(`replay: ${e?.message || e}`);
      throw e;
    }
    return true;
  }

  function clearReplay() {
    set({ replaying: false, replayClip: null });
    stopMedia();
  }

  // ---------------------------------------------------------------- controls (the screens call these)

  /**
   * Another controller (the phone companion, /judge) can switch the shared session: once this page's own last
   * control has had time to arrive (CONTROL_SETTLE_MS), the engine's `mode` / `ab` win, so the highlight never
   * lies about what plays.
   */
  function adoptRemote(t) {
    if (env.perfNow() - rt.controlAt < CONTROL_SETTLE_MS) return {};
    const patch = {};
    const mode = t.mode === "quiet" ? "quiet" : t.mode === "focus_hq" ? "focus_hq" : null;
    if (mode && mode !== st().mode) patch.mode = mode;
    if ((t.ab === "raw" || t.ab === "processed") && t.ab !== st().ab) patch.ab = t.ab;
    if (patch.mode || patch.ab) note(`switched by another controller: ${Object.values(patch).join(", ")}`);
    return patch;
  }

  function setAb(pressed) {
    const ab = nextAb(st().ab, pressed);
    rt.controlAt = env.perfNow();
    set({ ab });
    postAb();
    sendControl({ ab });
    tick();
    return ab;
  }

  function setMode(mode) {
    if (!APP_MODES.includes(mode)) throw new Error(`unknown app mode "${mode}" (${APP_MODES.join(", ")})`);
    rt.controlAt = env.perfNow();
    set({ mode });
    sendControl(modeFields(mode));
    tick();
    return mode;
  }

  /** Saved person 1 or 2 (index 0/1); false when that profile is not listed. */
  function choosePerson(i) {
    const names = savedPeople(st().telemetry);
    const msg = personControl(names, i);
    if (!msg) {
      note(`no saved voice profile ${i + 1}`);
      return false;
    }
    set({ released: false }); // the server turns auto-calibration back on with the profile
    sendControl(msg);
    note(`target: saved profile ${names[i]} (${i + 1})`);
    tick();
    return true;
  }

  function release() {
    set({ released: true, captions: [] }); // the old target's words are not the room's now (P1-3)
    sendControl(RELEASE_CONTROL);
    note("target released (searching; 1 or 2 picks a saved person)");
    tick();
  }

  /** Calibrate a new voice (off the stage path): the server's 3 s enrollment window. */
  function calibrate(seconds = CALIBRATE_S) {
    set({ released: false });
    sendControl({ enroll: { action: "start", seconds } });
    note(`calibration (${seconds} s)`);
    tick();
  }

  /** The room dial (T-427): how far the room is turned down, 0 (as it is) .. 100 (silent), the T-093 curve. */
  function setSuppression(value) {
    const suppression = Math.max(0, Math.min(100, Math.round(Number(value))));
    if (!Number.isFinite(suppression)) return st().suppression;
    rt.controlAt = env.perfNow();
    set({ suppression });
    sendControl({ suppression });
    return suppression;
  }

  function setIsolation(isolation) {
    set({ isolation });
    sendControl({ isolation });
  }

  // ---------------------------------------------------------------- render tick

  /** Recompute the derived state (pill hysteresis, truth strip): on every event and every RENDER_MS. */
  function tick() {
    const ended = rt.audio?.stream?.getAudioTracks?.().some((tr) => tr.readyState === "ended");
    const sinceFrameMs = ended ? Infinity : env.perfNow() - rt.frameAt;
    const mic = { ...st().mic };
    if (devices) {
      Object.assign(mic, { label: devices.inputLabel() ?? null, output: devices.outputLabel() ?? null, missing: devices.missing() });
    }
    mic.level = rt.audio ? rt.level : -60;
    mic.band = rt.audio ? meterStep(rt.level, -90, 0).band : "idle";
    if (JSON.stringify(mic) !== JSON.stringify(st().mic)) set({ mic });
    store.derive(env.perfNow(), { sinceFrameMs, rttMs: rt.rttMs });
  }

  function startTicking() {
    if (!rt.renderTimer) rt.renderTimer = env.setInterval(tick, RENDER_MS);
  }

  /**
   * T-460: the master output gain (0..1) ramped linearly over `rampMs` from where it is now. Client-side only: the
   * engine keeps running. Returns {ctx, endTime} (the AudioContext time the ramp ends; SOS measures press -> silence
   * against it) or null when no audio is open (the value is still kept for the next Start).
   */
  function setOutputGain(v, rampMs = 0) {
    rt.outputGain = Math.min(1, Math.max(0, Number(v) || 0));
    const a = rt.audio;
    const g = a?.master?.gain;
    if (!g || !a.ctx) return null;
    const t = a.ctx.currentTime;
    if (g.cancelAndHoldAtTime) g.cancelAndHoldAtTime(t);
    else { g.cancelScheduledValues(t); g.setValueAtTime(g.value, t); }
    const endTime = t + Math.max(0, rampMs) / 1000;
    if (rampMs > 0) g.linearRampToValueAtTime(rt.outputGain, endTime);
    else g.setValueAtTime(rt.outputGain, t);
    return { ctx: a.ctx, endTime };
  }

  /** Close the socket for good (tests, page unload). */
  function close() {
    rt.closed = true;
    env.clearTimeout(rt.retryTimer);
    env.clearInterval(rt.renderTimer);
    rt.renderTimer = null;
    stopMedia();
    const ws = rt.ws;
    rt.ws = null;
    ws?.close?.();
  }

  return {
    store,
    devices,
    devicesElement,
    connect: () => { startTicking(); connect(); },
    start,
    stop: stopMedia,
    toggle: () => (rt.audio && !st().replaying ? (stopMedia(), Promise.resolve(false)) : start()),
    setAb,
    setMode,
    choosePerson,
    release,
    calibrate,
    setSuppression,
    setIsolation,
    setOutputGain,
    get outputGain() { return rt.outputGain; },
    replay,
    retryMic: () => devices?.retryMic(),
    send: sendControl,
    tick,
    close,
    get sessionId() { return st().sessionId; },
    get recording() { return st().truth.recording; },
    /** Internals for the headless checks and devtools (window.hearingApp). */
    _rt: rt,
    _swapMic: swapMic,
  };
}
