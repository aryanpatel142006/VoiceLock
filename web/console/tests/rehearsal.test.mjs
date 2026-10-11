// T-285: rehearsal page logic.
import assert from "node:assert/strict";
import { test } from "node:test";
import { TakeUploader, gaugeFill, promptAt, simultaneous, takeLabel, telemetryView } from "../rehearsal.js";

test("telemetryView reads speaker, confidence, latency and RTF; empty telemetry is all null", () => {
  assert.equal(telemetryView(null).rtf, null);
  const v = telemetryView({ enrollment: { profile: "Aryan" }, tse: { confidence: 0.8 }, latency_ms: { total: 412 },
    tiers: { compute_headroom: 0.7, active: "enrolled" }, tse_similarity: 0.31 });
  assert.equal(v.speaker, "Aryan");
  assert.equal(v.confidence, 0.8);
  assert.equal(v.margin, 0.31);
  assert.equal(v.latencyMs, 412);
  assert.ok(Math.abs(v.rtf - 0.3) < 1e-9);
  assert.equal(v.tier, "enrolled");
  assert.equal(telemetryView({ target: { track_id: 3, confidence: 0.5 } }).speaker, "track 3");
});

test("gaugeFill clamps and treats missing as empty", () => {
  assert.equal(gaugeFill(null), 0);
  assert.equal(gaugeFill(2), 1);
  assert.equal(gaugeFill(0.25, 0, 0.5), 0.5);
});

test("promptAt walks the cards by elapsed time and finishes; freeform never ends", () => {
  const s = { kind: "alternating", cards: [{ who: "A", seconds: 4 }, { who: "B", seconds: 3 }] };
  assert.equal(promptAt(s, 0).index, 0);
  assert.equal(promptAt(s, 3.9).left.toFixed(1), "0.1");
  assert.equal(promptAt(s, 4).index, 1);
  assert.equal(promptAt(s, 7).done, true);
  assert.equal(promptAt({ kind: "freeform", cards: [{ who: "ANY", seconds: 0 }] }, 999).done, false);
  assert.equal(promptAt({ cards: [] }, 0).done, true);
});

test("simultaneous returns the partner card of an overlap", () => {
  const s = { cards: [{ who: "A", overlap_with: "B" }, { who: "B", overlap_with: "A" }, { who: "A" }] };
  assert.deepEqual(simultaneous(s, 0).map((c) => c.who), ["B"]);
  assert.deepEqual(simultaneous(s, 2), []);
});

test("takeLabel slugs like the server", () => {
  assert.equal(takeLabel("script", "alt 1/x"), "script_alt-1-x");
  assert.equal(takeLabel("", ""), "mode_scenario");
});

test("TakeUploader batches, and keeps data when an upload fails", async () => {
  const calls = [];
  let fail = true;
  const up = new TakeUploader(async (path, body) => { calls.push([path, body]); if (fail) throw new Error("x"); }, "t1", 16000);
  up.addAudio("mic", new Float32Array([0, 0.5]));
  up.addEvent({ kind: "a" });
  assert.equal(await up.flush(), false);
  assert.equal(up.failures, 2);
  assert.equal(up.pending, 2);
  fail = false;
  assert.equal(await up.flush(), true);
  assert.match(calls.at(-1)[0] + calls.at(-2)[0], /audio\?stream=mic&rate=16000/);
  assert.equal(up.sent.mic, 2);
  assert.equal(up.sent.events, 1);
});

test("page waits for mic rate then sends one hello and the boot matrix controls", async () => {
  // Only browser/device boundaries are doubled; execute the real page init and click handler.
  const saved = new Map();
  const install = (key, value) => {
    saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  };
  const element = () => ({ children: [], dataset: {}, setAttribute() {},
    replaceChildren(...children) { this.children = children; },
    set innerHTML(value) { this.firstChild = {}; this.lastChild = {}; } });
  const elements = new Map();
  const get = (id) => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
  const sockets = [];
  class Socket {
    static OPEN = 1;
    constructor() { this.sent = []; this.readyState = 1; sockets.push(this); }
    send(data) { this.sent.push(JSON.parse(data)); }
  }
  try {
    install("document", { getElementById: get, createElement: element });
    install("Option", function (title, id) { this.value = id; });
    install("location", { protocol: "https:", host: "localhost", search: "" });
    install("WebSocket", Socket);
    install("navigator", { mediaDevices: { getUserMedia: async () => ({}) } });
    install("AudioContext", class {
      sampleRate = 48000;
      audioWorklet = { addModule: async () => {} };
      createMediaStreamSource() { return { connect() {} }; }
      async resume() {}
    });
    install("AudioWorkletNode", class { port = {}; connect() {} });
    install("fetch", async (path) => ({ ok: true, json: async () => path.endsWith("options") ? {
      representations: { ecapa: { label: "Stock ECAPA" } }, policies: { stream: { label: "Stream" } },
      ladder: { balanced: { label: "Balanced" } }, projection_head_present: true,
      scripts: [{ id: "freeform", kind: "freeform", cards: [{ who: "ANY", line: "Speak" }] }],
    } : {
      matrix: { representation: "ecapa", policy: "stream", ladder: "balanced", suppression: 0.8 },
      controls: { latency_preset: "balanced", ab: "processed", suppression: 80 }, gated_play: "stream",
    } }));
    await import(`../rehearsal.js?boot-test=${Date.now()}`);
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(sockets.length, 0, "no session at an assumed rate before mic permission");
    const opening = get("mic-open").onclick();
    await new Promise((resolve) => setImmediate(resolve));
    assert.equal(sockets.length, 1);
    sockets[0].onopen();
    await opening;
    assert.deepEqual(sockets[0].sent, [
      { type: "hello", sample_rate: 48000, frame_ms: 20, video: false, client: "rehearsal" },
      { type: "control", mode: "focus_hq", latency_preset: "balanced", ab: "processed", suppression: 80 },
    ]);
    await get("mic-open").onclick();
    assert.equal(sockets.length, 1, "opening an already open mic does not re-hello");
    assert.equal(sockets[0].sent.length, 2);
  } finally {
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  }
});
