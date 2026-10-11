// T-346: the app's session and store. Pure state (pill hysteresis, A/B, mode, person, truth strip) and the session
// driven by a fake WebSocket, a fake worklet and the real T-287 mic controller over a fake mediaDevices.
// Run: node --test web/app/tests/*.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  JITTER_MS, RETRY_MS, Store, abWorklet, addCaption, backoffMs, captionState, jitterFor, jitterSamples, meterStep,
  modeFields, nextAb, personState, rawPill, truthStrip,
} from "../store.js";
import { CLIENT, createSession } from "../session.js";
import { createMic } from "../../console/devices.js";
import { KIND_AUDIO_IN, KIND_AUDIO_OUT, floatToPcm16, packFrame, unpackFrame } from "../../console/protocol.js";

const TELEMETRY = {
  type: "telemetry", mode: "focus_hq", effective_mode: "focus_hq", device: "mps", hq_backend: "auto", hq_active: "local",
  hq_reason: "no GPU configured", latency_ms: { total: 1480 }, latency_preset: "balanced", isolation: "strict",
  acquisition: { state: "locked", since_s: 3, detail: "known voice: aryan" },
  enrollment: { source: "profile", profile: "aryan", profiles: ["aryan", "joseph"] },
  tiers: { active: "enrolled", tiers: [{ name: "enrolled", eligible: true }] }, tse: { confidence: 0.82 },
  agc: { floor_rule: false }, git_rev: "abc1234",
};

// ------------------------------------------------------------------ pure store rules

test("backoff steps 0.5, 1, 2, then 4 s for ever, as /judge", () => {
  assert.deepEqual([0, 1, 2, 3, 4, 9].map(backoffMs), [500, 1000, 2000, 4000, 4000, 4000]);
  assert.deepEqual(RETRY_MS, [500, 1000, 2000, 4000]);
});

test("jitter profile follows the mode the engine plays", () => {
  assert.equal(jitterFor(TELEMETRY), "focus_hq");
  assert.equal(jitterFor({ effective_mode: "natural" }), "natural");
  assert.equal(jitterFor(null), "focus_hq");
  assert.deepEqual(jitterSamples("focus_hq", 16000), { targetSamples: 1920, maxSamples: 3840 });
  assert.deepEqual(jitterSamples("natural", 48000), { targetSamples: 1920, maxSamples: 7680 });
  assert.equal(JITTER_MS.natural, 40);
});

test("A/B: RAW delayed by the engine's total latency; Space flips, a button sets", () => {
  assert.deepEqual(abWorklet("raw", TELEMETRY), { type: "ab", state: "raw", delayMs: 1480 });
  assert.equal(abWorklet("processed", null).delayMs, 0);
  assert.equal(nextAb("processed"), "raw");
  assert.equal(nextAb("raw"), "processed");
  assert.equal(nextAb("raw", "raw"), "raw");
});

test("mode: Focus is focus_hq, Quiet is quiet; captions header only in Quiet", () => {
  assert.deepEqual(modeFields("focus_hq"), { mode: "focus_hq" });
  assert.deepEqual(modeFields("quiet"), { mode: "quiet" });
  assert.deepEqual(modeFields("anything"), { mode: "focus_hq" });
  const q = captionState({ ...TELEMETRY, synth_feed: "mixture" }, "quiet");
  assert.equal(q.quiet, true);
  assert.equal(q.speaker, "aryan");
  assert.equal(q.degraded, true);
  assert.equal(captionState({ ...TELEMETRY, synth_feed: "mixture" }, "focus_hq").degraded, false);
  assert.equal(captionState({ ...TELEMETRY, quiet: { feed: "paused", reason: "nobody isolated" } }, "quiet").paused, "nobody isolated");
});

test("person: first two saved profiles, current target, Release", () => {
  const p = personState(TELEMETRY);
  assert.deepEqual(p.names, ["aryan", "joseph"]);
  assert.deepEqual(p.labels, ["Aryan", "Joseph"]);
  assert.equal(p.current, "aryan");
  assert.equal(p.line, "Target: Aryan");
  const r = personState(TELEMETRY, { released: true });
  assert.equal(r.current, null);
  assert.equal(r.line, "Target: none (Release) · press 1 or 2");
  assert.equal(personState(null).line, "Target: —");
  assert.equal(personState({ enrollment: { source: "calibrated", profiles: [] } }).calibrated, true);
});

test("pill: connecting, idle until Start, never live while the mic stalls, Searching after Release", () => {
  const base = { connected: true, started: true, replaying: false, sinceFrameMs: 20, released: false, telemetry: TELEMETRY };
  assert.equal(rawPill({ ...base, connected: false, attempt: 0 }).text, "Connecting to the engine…");
  assert.equal(rawPill({ ...base, connected: false, attempt: 2 }).text, "Reconnecting to the engine…");
  assert.equal(rawPill({ ...base, started: false }).state, "idle");
  assert.equal(rawPill({ ...base, sinceFrameMs: 2500 }).state, "nomic");
  assert.equal(rawPill({ ...base, sinceFrameMs: Infinity }).state, "nomic");
  assert.equal(rawPill({ ...base, released: true }).text, "Searching");
  assert.equal(rawPill(base).state, "isolated");
  assert.equal(rawPill({ ...base, replaying: true, started: false }).state, "isolated"); // replay plays without the mic
  const loading = { ...TELEMETRY, acquisition: { state: "acquiring", detail: "loading saved voice: aryan" } };
  assert.equal(rawPill({ ...base, telemetry: loading }).state, "loading");
});

test("pill hysteresis: isolated at once, any other state only after 500 ms", () => {
  const s = new Store();
  s.set({ connection: "open", started: true, telemetry: TELEMETRY });
  s.derive(0, { sinceFrameMs: 10 });
  assert.equal(s.state.pill.state, "isolated");
  const look = { ...TELEMETRY, acquisition: { state: "searching", detail: "" } };
  s.set({ telemetry: look });
  s.derive(100, { sinceFrameMs: 10 });
  assert.equal(s.state.pill.state, "isolated", "one dropped window never flickers the pill");
  s.derive(650, { sinceFrameMs: 10 });
  assert.equal(s.state.pill.state, "look");
  s.set({ telemetry: TELEMETRY });
  s.derive(660, { sinceFrameMs: 10 });
  assert.equal(s.state.pill.state, "isolated");
  assert.equal(s.state.pillChanges, 2);
});

test("truth strip: every field null until telemetry, then measured values only", () => {
  const empty = truthStrip(null);
  for (const k of ["backend", "device", "delayMs", "rttMs", "gpuRttMs", "tier", "confidence", "recording", "sessionId"]) {
    assert.equal(empty[k], null, k);
  }
  const t = truthStrip(TELEMETRY, { connected: true, sessionId: "s1", delayMs: 1500, rttMs: 30 });
  assert.equal(t.backend, "Mac");
  assert.equal(t.device, "mps");
  assert.equal(t.delayMs, 1500);
  assert.equal(t.rttMs, 30);
  assert.equal(t.tier, "Enrolled voice");
  assert.equal(t.confidence, 0.82);
  assert.equal(t.sessionId, "s1");
  assert.equal(t.recording, null, "the server did not say: unknown, never assumed");
  assert.equal(truthStrip({ ...TELEMETRY, recording: true }).recording, true);
  assert.equal(truthStrip({ ...TELEMETRY, recording: false }).recording, false);
  const gpu = truthStrip({ ...TELEMETRY, hq_active: "remote", remote_rtt_ms: 12.4 });
  assert.equal(gpu.backend, "GPU");
  assert.equal(gpu.gpuRttMs, 12);
  assert.equal(truthStrip({ ...TELEMETRY, preset: { pins: { rev_warning: "GPU on old rev" } } }).revWarning, "GPU on old rev");
});

test("store derive: delay is a 50 ms-step EMA; replay and session id reach the truth strip", () => {
  const s = new Store();
  s.set({ connection: "open", telemetry: TELEMETRY, sessionId: "abc", replaying: true });
  s.derive(0, { rttMs: 33 });
  assert.equal(s.state.truth.delayMs, 1500);
  assert.equal(s.state.truth.rttMs, 30);
  assert.equal(s.state.truth.replay, true);
  assert.equal(s.state.truth.sessionId, "abc");
  assert.equal(s.state.abDelayMs, 1480);
});

test("store events: change lists keys; unknown events throw; a throwing listener does not stop others", () => {
  const s = new Store();
  const seen = [];
  const off = s.on("change", ({ keys }) => seen.push(keys));
  const offBoom = s.on("change", () => { throw new Error("boom"); });
  const origError = console.error;
  console.error = () => {};
  try {
    s.set({ ab: "raw" });
    s.set({ ab: "raw" }); // no change, no event
  } finally { console.error = origError; }
  assert.deepEqual(seen, [["ab"]]);
  off();
  offBoom();
  assert.throws(() => s.on("nope", () => {}), /unknown store event/);
  s.note("hello");
  assert.equal(s.state.log.at(-1).text, "hello");
});

test("meter: peak with 24 dB/s release; captions keep one partial line and the last four", () => {
  assert.equal(meterStep(-20, -90, 0.5).db, -32);
  assert.equal(meterStep(-20, -10, 0.1).db, -10);
  assert.equal(meterStep(-59, -90, 1).db, -60);
  assert.equal(meterStep(-1, -1, 0).band, "clipping");
  let c = addCaption([], { text: "see", final: false });
  c = addCaption(c, { text: "see you", final: true });
  assert.deepEqual(c.map((x) => x.text), ["see you"]);
  for (const w of ["a", "b", "c", "d"]) c = addCaption(c, { text: w, final: true });
  assert.deepEqual(c.map((x) => x.text), ["a", "b", "c", "d"]);
});

// ------------------------------------------------------------------ session with fakes

class FakeWS {
  static all = [];
  constructor(url) { this.url = url; this.readyState = 0; this.sent = []; FakeWS.all.push(this); }
  send(data) { this.sent.push(data); }
  close() { this.readyState = 3; }
  open() { this.readyState = 1; this.onopen?.(); }
  drop() { this.readyState = 3; this.onclose?.(); }
  text(obj) { this.onmessage?.({ data: JSON.stringify(obj) }); }
  binary(buf) { this.onmessage?.({ data: buf }); }
  json() { return this.sent.filter((d) => typeof d === "string").map((d) => JSON.parse(d)); }
}

function fakeEnv() {
  const timers = [];
  let clock = 0;
  return {
    timers,
    advance(ms) { clock += ms; },
    env: {
      WebSocket: FakeWS,
      setTimeout: (fn, ms) => { const t = { fn, ms }; timers.push(t); return t; },
      clearTimeout: (t) => { const i = timers.indexOf(t); if (i >= 0) timers.splice(i, 1); },
      setInterval: () => 1,
      clearInterval: () => {},
      perfNow: () => clock,
      wallNow: () => 1000 + clock / 1000,
      fetch: async () => ({ ok: true, json: async () => ({ default: "team/clip.mp4" }) }),
      location: { search: "", protocol: "http:", host: "127.0.0.1:9" },
      openAudio: async ({ playbackOnly, devices }) => {
        const stream = playbackOnly ? { getTracks: () => [], getAudioTracks: () => [] } : await devices.openMic();
        const posted = [];
        const ctx = { createMediaStreamSource: (s) => ({ s, connect() {}, disconnect() {} }), close() {}, resume: async () => {} };
        return { ctx, stream, rate: 16000, source: playbackOnly ? null : ctx.createMediaStreamSource(stream),
          node: { port: { posted, postMessage(m) { posted.push(m); }, onmessage: null } } };
      },
    },
  };
}

const IPHONE = { deviceId: "iphone-id", label: "Aryan patel's iPhone Microphone" };
const MACBOOK = { deviceId: "macbook-id", label: "MacBook Pro Microphone (Built-in)" };

class FakeTrack extends EventTarget {
  constructor(dev) { super(); this.label = dev.label; this.deviceId = dev.deviceId; this.readyState = "live"; }
  getSettings() { return { deviceId: this.deviceId }; }
  stop() { this.readyState = "ended"; }
  unplug() { this.readyState = "ended"; this.dispatchEvent(new Event("ended")); }
}

function fakeMedia(devices) {
  return {
    devices: [...devices],
    async enumerateDevices() { return this.devices.map((d) => ({ kind: "audioinput", deviceId: d.deviceId, label: d.label })); },
    async getUserMedia({ audio }) {
      const want = audio.deviceId?.exact;
      const dev = want ? this.devices.find((d) => d.deviceId === want) : this.devices.find((d) => d.deviceId === MACBOOK.deviceId);
      if (!dev) throw Object.assign(new Error("gone"), { name: "OverconstrainedError" });
      const track = new FakeTrack(dev);
      return { getAudioTracks: () => [track], getTracks: () => [track], track };
    },
  };
}

/** The devices handle session.js expects (mountDevices' API) over the real T-287 mic controller. */
function fakeDevices(media, onSwap) {
  const mem = new Map();
  const storage = { getItem: (k) => mem.get(k) ?? null, setItem: (k, v) => mem.set(k, v), removeItem: (k) => mem.delete(k) };
  const mic = createMic({ media, storage, onSwap: (s, info) => onSwap()(s, info) });
  return {
    mic,
    openMic: () => mic.open(),
    inputLabel: () => mic.label(),
    outputLabel: () => (mic.label() ? "AirPods Pro" : null),
    missing: () => (mic.lost() ? ["mic"] : []),
    setExpected: ({ input }) => mic.setExpected(input),
    started: async (stream) => mic.attach(stream),
    stopped: () => mic.detach(),
    feed() {},
    retryMic: () => mic.retry(),
  };
}

function makeSession(envOverrides = {}) {
  FakeWS.all = [];
  const fe = fakeEnv();
  Object.assign(fe.env, envOverrides);
  const media = fakeMedia([IPHONE, MACBOOK]);
  let session;
  const devices = fakeDevices(media, () => session._swapMic);
  session = createSession({ store: new Store(), devices, env: fe.env });
  return { session, fe, devices, store: session.store };
}

const flush = () => new Promise((r) => setImmediate(r));

test("connect: hello labelled as the app, then the judge profile with A/B and mode", async () => {
  const { session, store } = makeSession();
  session.connect();
  const ws = FakeWS.all[0];
  assert.equal(ws.url, "ws://127.0.0.1:9/ws/session?replay=1");
  assert.equal(store.state.connection, "connecting");
  assert.equal(store.state.pill.text, "Connecting to the engine…");
  ws.open();
  const [h, p] = ws.json();
  assert.equal(h.type, "hello");
  assert.equal(h.client, CLIENT);
  assert.equal(h.video, false, "the app never sends video");
  assert.ok("input_device" in h && "output_device" in h && "devices_missing" in h);
  assert.deepEqual(p, { type: "control", profile: "judge", ab: "processed", mode: "focus_hq" });
  ws.text({ type: "session", session_id: "sid-1" });
  assert.equal(session.sessionId, "sid-1");
  assert.equal(store.state.truth.sessionId, "sid-1");
});

test("reconnect with backoff; the attempt resets once the engine is back; Release survives a reconnect", async () => {
  const { session, fe, store } = makeSession();
  session.connect();
  FakeWS.all[0].open();
  session.release();
  const delays = [];
  for (let i = 0; i < 5; i++) {
    FakeWS.all.at(-1).drop();
    assert.equal(store.state.connection, "reconnecting");
    fe.advance(600); // the pill hysteresis holds a non-isolated state 500 ms
    session.tick();
    assert.equal(store.state.pill.text, "Reconnecting to the engine…");
    const t = fe.timers.pop();
    delays.push(t.ms);
    t.fn();
  }
  assert.deepEqual(delays, [500, 1000, 2000, 4000, 4000]);
  const ws = FakeWS.all.at(-1);
  ws.open();
  assert.equal(store.state.attempt, 0);
  assert.equal(store.state.connection, "open");
  const msgs = ws.json();
  assert.equal(msgs[1].profile, "judge");
  assert.deepEqual(msgs[2], { type: "control", enroll: { action: "clear" } }, "Release re-sent after the profile");
  assert.ok(store.state.log.some((l) => l.text === "engine back"));
});

test("Start: mic frames go out as audio-in, AGC on, A/B delay and jitter follow telemetry", async () => {
  const { session, fe, store } = makeSession();
  session.connect();
  const ws = FakeWS.all[0];
  ws.open();
  ws.text(TELEMETRY);
  assert.equal(await session.start(), true);
  assert.equal(store.state.started, true);
  const port = session._rt.audio.node.port;
  assert.deepEqual(port.posted[0], { type: "agc", on: true });
  assert.deepEqual(port.posted.find((m) => m.type === "ab"), { type: "ab", state: "processed", delayMs: 1480 });
  // a worklet frame -> a binary audio-in frame with seq 0
  const levels = [];
  store.on("level", (l) => levels.push(l));
  const samples = new Float32Array(320).fill(0.25);
  port.onmessage({ data: { type: "frame", samples } });
  const bin = ws.sent.filter((d) => typeof d !== "string");
  assert.equal(bin.length, 1);
  const f = unpackFrame(bin[0]);
  assert.equal(f.kind, KIND_AUDIO_IN);
  assert.equal(f.seq, 0);
  assert.equal(f.payload.byteLength, 640);
  assert.ok(Math.abs(levels[0].db - 20 * Math.log10(0.25)) < 0.01, "level is the measured peak");
  // Space: RAW, delayed by the engine latency
  assert.equal(session.setAb(), "raw");
  assert.deepEqual(port.posted.at(-1), { type: "ab", state: "raw", delayMs: 1480 });
  assert.deepEqual(ws.json().at(-1), { type: "control", ab: "raw" });
  // the engine falls back to Natural: 40 ms jitter buffer
  ws.text({ ...TELEMETRY, effective_mode: "natural", latency_ms: { total: 120 } });
  assert.deepEqual(port.posted.filter((m) => m.type === "config").at(-1), { type: "config", targetSamples: 640, maxSamples: 2560 });
  assert.equal(port.posted.filter((m) => m.type === "ab").at(-1).delayMs, 120);
  assert.equal(store.state.jitter, "natural");
  // AGC floor rule mirrored on RAW
  ws.text({ ...TELEMETRY, agc: { floor_rule: true } });
  assert.deepEqual(port.posted.filter((m) => m.type === "agc").at(-1), { type: "agc", on: true, floorRule: true });
  // audio out -> the worklet's play queue, and the page<->engine RTT
  fe.advance(200);
  const plays = [];
  store.on("playback", (p) => plays.push(p.db));
  ws.binary(packFrame(KIND_AUDIO_OUT, 1, 1000.1, floatToPcm16(new Float32Array(320).fill(0.5))));
  assert.equal(port.posted.at(-1).type, "play");
  assert.ok(Math.abs(plays.at(-1) - -6.02) < 0.1, `playback peak ${plays.at(-1)}`); // measured before the transfer
  fe.advance(500);
  ws.binary(packFrame(KIND_AUDIO_OUT, 2, 1000.6, floatToPcm16(new Float32Array(320))));
  assert.ok(Math.abs(plays.at(-1) - -18.02) < 0.1, "the target paused: released 12 dB in 0.5 s");
  assert.ok(Math.abs(session._rt.rttMs - 100) < 1);
  // the mic stalls: the pill says so after the hysteresis hold
  fe.advance(2500);
  session.tick();
  fe.advance(600);
  session.tick();
  assert.equal(store.state.pill.state, "nomic");
});

test("Focus / Quiet, persons 1/2 and Release send the /judge controls", async () => {
  const { session, store } = makeSession();
  session.connect();
  const ws = FakeWS.all[0];
  ws.open();
  assert.equal(session.choosePerson(1), false, "no profiles listed yet");
  ws.text(TELEMETRY);
  session.setMode("quiet");
  assert.deepEqual(ws.json().at(-1), { type: "control", mode: "quiet" });
  assert.equal(store.state.caption.quiet, true);
  assert.throws(() => session.setMode("synthetic"), /unknown app mode/);
  assert.equal(session.choosePerson(1), true);
  assert.deepEqual(ws.json().at(-1), { type: "control", enroll: { action: "profile", name: "joseph" } });
  session.release();
  assert.deepEqual(ws.json().at(-1), { type: "control", enroll: { action: "clear" } });
  assert.equal(store.state.person.line, "Target: none (Release) · press 1 or 2");
  session.choosePerson(0);
  assert.equal(store.state.released, false);
  ws.text({ type: "caption", text: "see you at noon", final: true, t0: 1, t1: 2 });
  assert.equal(store.state.captions.at(-1).text, "see you at noon");
});

test("iPhone mic: Start opens the preset's mic; a dropout swaps to the MacBook mic in the same session, with a mic_switch hello", async () => {
  const { session, devices, store } = makeSession();
  session.connect();
  const ws = FakeWS.all[0];
  ws.open();
  ws.text({ ...TELEMETRY, preset: { pins: { input_device: IPHONE.label, audio_only: true } } });
  await session.start();
  assert.equal(devices.inputLabel(), IPHONE.label);
  const hello = ws.json().filter((m) => m.type === "hello").at(-1);
  assert.equal(hello.input_device, IPHONE.label);
  assert.equal(hello.output_device, "AirPods Pro");
  const before = FakeWS.all.length;
  devices.mic.label(); // live
  const track = session._rt.audio.stream.getAudioTracks()[0];
  track.unplug();
  for (let i = 0; i < 5; i++) await flush();
  assert.equal(FakeWS.all.length, before, "no reconnect: the session keeps running");
  assert.equal(devices.inputLabel(), MACBOOK.label);
  const sw = ws.json().filter((m) => m.type === "hello").at(-1);
  assert.match(sw.mic_switch, /^lost \(ended\): Aryan patel's iPhone Microphone -> MacBook Pro Microphone/);
  assert.equal(Object.keys(sw)[1], "mic_switch", "right after type, as on /judge");
  assert.deepEqual(sw.devices_missing, ["mic"]);
  assert.ok(store.state.log.some((l) => l.text.startsWith("mic lost (ended)")));
});

test("replay: playback only, the server's default clip, RAW fed from the replayed room, REPLAY in the truth strip", async () => {
  const calls = [];
  const fetch = async (u, o) => { calls.push([String(u), o?.method || "GET"]); return { ok: true, json: async () => ({ default: "team/clip.mp4" }) }; };
  const { session, store } = makeSession({ fetch });
  session.connect();
  const ws = FakeWS.all[0];
  ws.open();
  await assert.rejects(session.replay(true), /Connect to the server first/);
  ws.text({ type: "session", session_id: "sid-9" });
  await session.replay(true);
  assert.equal(calls[0][0], "http://127.0.0.1:9/replay/clips");
  assert.match(calls[1][0], /\/replay\?session_id=sid-9&action=start&loop=true&clip=team%2Fclip\.mp4/);
  assert.equal(calls[1][1], "POST");
  ws.text({ type: "replay", active: true, clip: "team/clip.mp4" });
  assert.equal(store.state.truth.replay, true);
  ws.binary(packFrame(KIND_AUDIO_IN, 0, 0, floatToPcm16(new Float32Array(160).fill(0.1))));
  assert.equal(session._rt.audio.node.port.posted.at(-1).type, "replay-raw");
  ws.text({ type: "replay", active: false });
  assert.equal(store.state.replaying, false);
  assert.equal(store.state.truth.replay, false);
});
