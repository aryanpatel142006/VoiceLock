// node --test web/console/tests/*.test.mjs
// Drives the AudioWorklet processor directly (no browser) and records its output across an A/B switch.
import assert from "node:assert/strict";
import test from "node:test";

const SR = 16000;
globalThis.sampleRate = SR;
globalThis.AudioWorkletProcessor = class { constructor() { this.port = { postMessage() {}, onmessage: null }; } };
let Processor;
globalThis.registerProcessor = (_name, cls) => { Processor = cls; };
await import("../audio-worklet.js");

const Q = 128;
const raw = (n) => Math.sin(2 * Math.PI * 440 * n / SR) * 0.5; // the mic: a tone
const proc = (n) => 0.25 + 0 * n; // the engine output: a constant, so the two sides are distinguishable

function run({ delayMs, switchAt, seconds = 1, to = "raw" }) {
  const p = new Processor({ processorOptions: { frameSamples: 320 } });
  p.onMessage({ type: "ab", state: "processed", delayMs });
  const out = [];
  const quanta = Math.floor((seconds * SR) / Q);
  let n = 0;
  for (let q = 0; q < quanta; q++) {
    if (q * Q >= switchAt && p.rawTarget === (to === "raw" ? 0 : 1)) p.onMessage({ type: "ab", state: to });
    p.onMessage({ type: "play", samples: Float32Array.from({ length: Q }, (_, k) => proc(n + k)) });
    const input = Float32Array.from({ length: Q }, (_, k) => raw(n + k));
    const output = new Float32Array(Q);
    p.process([[input]], [[output]]);
    out.push(...output);
    n += Q;
  }
  return { out, p };
}

test("raw side is the mic delayed by the declared latency, time-aligned", () => {
  const delayMs = 50;
  const d = (delayMs * SR) / 1000;
  const { out } = run({ delayMs, switchAt: 2048, seconds: 1 });
  // well after the 100 ms crossfade the output is exactly the mic `d` samples ago
  for (let i = 8000; i < 8200; i++) assert.ok(Math.abs(out[i] - raw(i - d)) < 1e-5, `sample ${i}`);
});

test("switching crossfades over ~100 ms: no jump larger than the signal's own slope", () => {
  const { out } = run({ delayMs: 40, switchAt: 4096, seconds: 1 });
  const first = out.findIndex((v, i) => i > 4096 && Math.abs(v - 0.25) > 1e-3);
  assert.ok(first >= 4096, "output leaves the engine value only after the switch");
  let maxStep = 0;
  for (let i = 2000; i < out.length; i++) maxStep = Math.max(maxStep, Math.abs(out[i] - out[i - 1]));
  assert.ok(maxStep < 0.1, `max sample-to-sample step ${maxStep.toFixed(4)} (a hard cut would be ~0.25)`);
  // fully on raw 100 ms (+ a little) after the switch
  assert.ok(Math.abs(out[4096 + 1700] - raw(4096 + 1700 - 640)) < 1e-4);
});

test("engine ring keeps draining while raw plays", () => {
  const { p } = run({ delayMs: 20, switchAt: 1024, seconds: 0.5 });
  assert.equal(p.rawTarget, 1);
  assert.ok(p.buffered <= p.targetSamples + Q, "engine output must not back up while raw plays");
});

test("switching back to processed returns exactly to the engine output", () => {
  const p = new Processor({ processorOptions: { frameSamples: 320 } });
  p.onMessage({ type: "ab", state: "raw", delayMs: 20 });
  let last;
  for (let q = 0; q < 200; q++) {
    if (q === 50) p.onMessage({ type: "ab", state: "processed" });
    p.onMessage({ type: "play", samples: new Float32Array(Q).fill(0.25) });
    last = new Float32Array(Q);
    p.process([[new Float32Array(Q).fill(0.1)]], [[last]]);
  }
  assert.ok(last.every((v) => Math.abs(v - 0.25) < 1e-6));
});

test("replay raw input can be heard through A/B without opening a mic", () => {
  const p = new Processor({ processorOptions: { frameSamples: 320 } });
  p.onMessage({ type: "ab", state: "raw", delayMs: 0 });
  let last;
  for (let q = 0; q < 30; q++) {
    p.onMessage({ type: "replay-raw", samples: new Float32Array(Q).fill(0.4) });
    last = new Float32Array(Q);
    p.process([], [[last]]);
  }
  assert.ok(last.every((v) => Math.abs(v - .4) < 1e-6));
});

test("AGC (T-135): the raw side is levelled to -20 dBFS like the server output", () => {
  const rms = (a) => 10 * Math.log10(a.reduce((s, v) => s + v * v, 0) / a.length);
  for (const [amp, want] of [[0.5 * 10 ** (-12 / 20), -20], [0.4, -20], [0.01, -43 + 12]]) {
    const p = new Processor({ processorOptions: { frameSamples: 320 } });
    p.onMessage({ type: "agc", on: true });
    p.onMessage({ type: "ab", state: "raw", delayMs: 0 });
    const out = [];
    for (let q = 0, n = 0; q < Math.floor((3 * SR) / Q); q++, n += Q) {
      const input = Float32Array.from({ length: Q }, (_, k) => amp * 2 * raw(n + k)); // raw() is 0.5 * sine
      const output = new Float32Array(Q);
      p.process([[input]], [[output]]);
      out.push(...output);
    }
    const got = rms(out.slice(2 * SR));
    assert.ok(Math.abs(got - want) < 1.5, `amp ${amp}: ${got.toFixed(1)} dBFS, want ${want}`);
  }
  const off = new Processor({ processorOptions: { frameSamples: 320 } });
  off.onMessage({ type: "agc", on: true });
  off.onMessage({ type: "agc", on: false });
  assert.equal(off.rawAgc, null);
});
