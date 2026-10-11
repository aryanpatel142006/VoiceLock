// node --test web/console/tests/*.test.mjs
// Expected bytes come from hearing.server.protocol.pack_frame and hearing.audio.io.float_to_pcm16.
import assert from "node:assert/strict";
import test from "node:test";

import {
  KIND_AUDIO_IN, control, dbfs, floatToPcm16, hello, packFrame, pcm16ToFloat, sessionUrl,
  headroomInfo, hqBackendInfo, ladderInfo, splitLatency, tierInfo, unpackFrame,
} from "../protocol.js";

test("packFrame matches the Python header byte for byte", () => {
  const frame = packFrame(KIND_AUDIO_IN, 7, 12.5, new Uint8Array([1, 2]));
  assert.deepEqual([...new Uint8Array(frame)], [1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 41, 64, 1, 2]);
});

test("unpackFrame round-trips and rejects short frames", () => {
  const pcm = floatToPcm16(new Float32Array([0.25, -0.5]));
  const f = unpackFrame(packFrame(3, 2 ** 32 + 5, 1.75, pcm));
  assert.equal(f.kind, 3);
  assert.equal(f.seq, 5); // u32 wraps like the server's `seq & 0xFFFFFFFF`
  assert.equal(f.captureTs, 1.75);
  assert.deepEqual([...pcm16ToFloat(f.payload)], [0.25, -0.5]);
  assert.throws(() => unpackFrame(new ArrayBuffer(12)), /header/);
});

test("floatToPcm16 scales and clips like hearing.audio.io", () => {
  const bytes = new Uint8Array(floatToPcm16(new Float32Array([0, 0.5, -1, 1, 1.5, -0.25])));
  assert.deepEqual([...bytes], [0, 0, 0, 64, 0, 128, 255, 127, 255, 127, 0, 224]);
});

test("hello and control messages follow CONTRACTS", () => {
  assert.deepEqual(hello(16000, 20, true),
    { type: "hello", sample_rate: 16000, frame_ms: 20, video: true, client: "console" });
  assert.deepEqual(control({ suppression: 40 }), { type: "control", suppression: 40 });
});

test("sessionUrl honours ?ws= and follows the page's scheme", () => {
  assert.equal(sessionUrl({ search: "?ws=ws://x:1/ws/session", protocol: "http:", host: "a" }),
    "ws://x:1/ws/session");
  assert.equal(sessionUrl({ search: "", protocol: "https:", host: "h:9" }), "wss://h:9/ws/session");
  assert.equal(sessionUrl({ search: "", protocol: "http:", host: "127.0.0.1:8000" }),
    "ws://127.0.0.1:8000/ws/session");
});

test("splitLatency separates stage compute from the fixed keys", () => {
  const s = splitLatency({ dfn: 3, gate: 0.4, algorithmic: 40, resample: 0, server: 4, total: 64 });
  assert.deepEqual(s.stages, [["dfn", 3], ["gate", 0.4]]);
  assert.equal(s.algorithmic, 40);
  assert.equal(s.total, 64);
  assert.equal(splitLatency(undefined).total, null);
  const v = splitLatency({ gate: 0.4, server: 1, total: 41, vision: 8.5 });
  assert.deepEqual(v.stages, [["gate", 0.4]]);  // vision is per frame, not per hop
  assert.equal(v.vision, 8.5);
  assert.equal(splitLatency({}).vision, null);
});

test("tierInfo tells AV tiers from audio-only", () => {
  assert.deepEqual(tierInfo("audio-only"), { av: false, label: "Audio-only" });
  assert.deepEqual(tierInfo("gate"), { av: true, label: "AV · gate" });
  assert.deepEqual(tierInfo("av"), { av: true, label: "AV" });
  assert.deepEqual(tierInfo("focus_hq"), { av: true, label: "Focus HQ" });
  assert.deepEqual(tierInfo("enrolled"), { av: false, label: "Enrolled voice" });
  assert.deepEqual(tierInfo("new_tier"), { av: true, label: "AV · new tier" });
  assert.deepEqual(tierInfo(undefined), { av: null, label: "—" });
});

test("ladderInfo lists the tiers best first with the active one and the HQ model", () => {
  const info = ladderInfo({
    active: "gate", reason: "focus_hq: no lips", delay_ms: 1500,
    tiers: [
      { name: "focus_hq", eligible: false, reason: "no lips",
        health: [{ candidate: "av_mossformer2", device: "mps", rtf: 0.7, ok: false }] },
      { name: "enrolled", eligible: false, reason: "enrolling", health: [] },
      { name: "gate", eligible: true, reason: "", health: [] },
      { name: "audio-only", eligible: true, reason: "", health: [] },
    ],
  });
  assert.deepEqual(info.rows.map((r) => [r.label, r.active, r.eligible]), [
    ["Focus HQ", false, false], ["Enrolled voice", false, false], ["AV · gate", true, true],
    ["Audio-only", false, true],
  ]);
  assert.equal(info.hq.candidate, "av_mossformer2");
  assert.equal(info.delayMs, 1500);
  assert.deepEqual(ladderInfo(null), { rows: [], hq: null, delayMs: null, reason: "" });
});

test("dbfs of a full-scale square wave is 0 and silence is floored", () => {
  assert.ok(Math.abs(dbfs(new Float32Array([1, -1, 1, -1]))) < 1e-6);
  assert.equal(dbfs(new Float32Array(160)), -90);
});

test("hqBackendInfo shows where Focus HQ plays, its RTT and a GPU fallback", () => {
  assert.equal(hqBackendInfo(null).label, "—");
  const gpu = hqBackendInfo({ hq_backend: "remote", hq_active: "remote", hq_reason: "remote GPU playing", remote_rtt_ms: 23.4 });
  assert.deepEqual(gpu, {
    requested: "remote", active: "remote", fallback: false, rttMs: 23.4, label: "GPU · RTT 23 ms",
    reason: "remote GPU playing",
  });
  const down = hqBackendInfo({ hq_backend: "remote", hq_active: "local", hq_reason: "remote down", remote_rtt_ms: null });
  assert.equal(down.fallback, true);
  assert.equal(down.label, "Local");
  const auto = hqBackendInfo({ hq_backend: "auto", hq_active: "remote", remote_rtt_ms: 1.25 });
  assert.equal(auto.label, "Auto → GPU · RTT 1.3 ms");
  assert.equal(auto.fallback, false); // auto picking local is a choice, not a failure
});

test("headroomInfo shows the free share and the enrolled hop", () => {
  assert.deepEqual(headroomInfo(null), { text: "—", low: false });
  const tiers = { compute_headroom: 0.25, tiers: [{ name: "enrolled", health: [{
    candidate: "enrolled", utilisation: 0.75, compute_p50_ms: 562, compute_p95_ms: 700, hop_s: 0.75, hop_base_s: 0.5,
  }] }] };
  const info = headroomInfo(tiers);
  assert.equal(info.low, true);
  assert.match(info.text, /^25 % free · enrolled 562\/700 ms per window \(p50\/p95\) every 0.75 s \(widened from 0.5 s\)$/);
});

test("audioContextOptions: native rate on iOS (48 kHz, hello says so), 16 kHz elsewhere", async () => {
  const { audioContextOptions, isIOS } = await import("../protocol.js");
  const iphone = { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15", maxTouchPoints: 5 };
  const ipad = { userAgent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", platform: "MacIntel", maxTouchPoints: 5 };
  const pixel = { userAgent: "Mozilla/5.0 (Linux; Android 14; Pixel 7) Chrome/120 Mobile", maxTouchPoints: 5 };
  assert.ok(isIOS(iphone) && isIOS(ipad) && !isIOS(pixel) && !isIOS(undefined));
  assert.equal(audioContextOptions(iphone).sampleRate, undefined);
  assert.equal(audioContextOptions(pixel).sampleRate, 16000);
});

test("sessionUrl follows the page scheme: https -> wss for phones on the LAN", () => {
  const loc = { protocol: "https:", host: "192.168.1.20:8000", search: "" };
  assert.equal(sessionUrl(loc), "wss://192.168.1.20:8000/ws/session");
});
