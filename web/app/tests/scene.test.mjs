// T-390: the sound scene's pure parts: which look a state draws, how far the room recedes, the gate and stream
// inputs, the spring that eases every change, the seeded grain table and the Diagnostics stats.
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULTS, FrameStats, GRAINS, grainTable, gateWeight, levelUnit, lookStyle, recedeTarget, releasedDb, sceneLook,
  seeded, springStep, streamDb,
} from "../scene.js";

const live = { started: true, connection: "open", pill: { state: "isolated" }, mode: "focus_hq", telemetry: { ab: "processed" } };

test("looks follow the real state: idle before Start, amber RAW, green Focus, white after Release", () => {
  assert.equal(sceneLook(null), "idle");
  assert.equal(sceneLook({ started: false }), "idle");
  assert.equal(sceneLook({ ...live, pill: { state: "nomic" } }), "nomic");
  assert.equal(sceneLook(live), "focus");
  assert.equal(sceneLook({ ...live, telemetry: { ab: "raw" } }), "raw");
  assert.equal(sceneLook({ ...live, released: true }), "searching");
  assert.equal(sceneLook({ ...live, mode: "quiet" }), "quiet");
  assert.equal(sceneLook({ ...live, connection: "reconnecting" }), "lost");
  assert.equal(sceneLook({ ...live, replaying: true }), "replay");
});

test("the room recedes only while the chosen voice plays, by the engine's confidence", () => {
  assert.equal(recedeTarget("raw", 0.95), 0);
  assert.equal(recedeTarget("quiet", 0.95), 0);
  assert.equal(recedeTarget("idle", 1), 0);
  assert.equal(recedeTarget("focus", null), 0.45); // unknown confidence: the floor, never invented
  assert.ok(recedeTarget("focus", 0.9) > recedeTarget("focus", 0.35));
  assert.equal(recedeTarget("focus", 1), 1);
  assert.equal(recedeTarget("focus", 7), 1); // clamped
  assert.ok(recedeTarget("searching", 1) < recedeTarget("focus", 0));
});

test("RAW draws the room in amber and the form as a ghost; Release leaves the form an empty outline", () => {
  const raw = lookStyle("raw");
  assert.equal(raw.room, "raw");
  assert.ok(raw.formWeight < 0.3);
  const focus = lookStyle("focus");
  assert.equal(focus.room, "fg");
  assert.equal(focus.form, "focus");
  assert.equal(focus.formWeight, 1);
  assert.equal(lookStyle("searching").filled, false);
  assert.equal(lookStyle("idle").flat, true);
  assert.equal(lookStyle("idle").filled, false);
  assert.equal(lookStyle("quiet").alpha, DEFAULTS.quietDim);
  assert.equal(lookStyle("replay").dashed, true);
});

test("the gate state dims the form as the pill reads: SOLO full, OVERLAP-DUCKED half, UNSURE low", () => {
  assert.equal(gateWeight({ tse: { confidence: 0.9, gated: false, soft_gated: false } }), 1);
  assert.equal(gateWeight({ tse: { confidence: 0.9, gated: true, soft_gated: true } }), 0.55);
  assert.equal(gateWeight({ tse: { confidence: 0.2, gated: false, soft_gated: false } }), 0.3);
  assert.equal(gateWeight({ tse: { confidence: 0.9, gated: true, soft_gated: false } }), 0.3);
  assert.equal(gateWeight({}), 1); // no field: nothing to dim by
});

test("stream energy is the measured playback peak, released at 24 dB/s, or nothing before output", () => {
  assert.equal(streamDb(-23.5, 0), -23.5);
  assert.equal(streamDb(-20, 0.5), -32); // the target paused: the form dims with the meter's release
  assert.equal(streamDb(-20, 10), DEFAULTS.floorDb);
  assert.equal(streamDb(null, 0), null);
  assert.equal(streamDb(undefined, 0), null);
});

test("level: the meter's 24 dB/s release onto the floor, mapped between floor and ceiling", () => {
  assert.equal(releasedDb(-20, 0.5), -32);
  assert.equal(releasedDb(-20, 10), DEFAULTS.floorDb);
  assert.equal(releasedDb(NaN, 0), DEFAULTS.floorDb);
  assert.equal(levelUnit(-60), 0);
  assert.equal(levelUnit(-6), 1);
  assert.equal(levelUnit(-33), 0.5);
});

test("the spring settles on its target within its settle time and never overshoots from rest", () => {
  const st = { x: 0, v: 0 };
  let max = 0;
  for (let t = 0; t < 0.52; t += 1 / 60) { springStep(st, 1, 1 / 60, 0.52); max = Math.max(max, st.x); }
  assert.ok(st.x > 0.98, `x=${st.x}`);
  assert.ok(max <= 1 + 1e-9, `overshoot ${max}`);
  // frame-rate independent: one 0.2 s step lands where twelve 1/60 s steps do
  const a = springStep({ x: 0, v: 0 }, 1, 0.2, 0.52);
  const b = { x: 0, v: 0 };
  for (let i = 0; i < 12; i++) springStep(b, 1, 0.2 / 12, 0.52);
  assert.ok(Math.abs(a.x - b.x) < 1e-6);
  // reduced motion (settle 0): jumps
  assert.deepEqual(springStep({ x: 0.2, v: 3 }, 0.8, 1 / 60, 0), { x: 0.8, v: 0 });
});

test("grains are seeded: the same field on every load, three equal depth bands on a ring", () => {
  const a = grainTable();
  const b = grainTable();
  assert.equal(a.n, GRAINS);
  assert.equal(a.band * 3, GRAINS);
  assert.deepEqual(Array.from(a.rho.slice(0, 20)), Array.from(b.rho.slice(0, 20)));
  for (let i = 0; i < a.n; i++) {
    assert.ok(a.rho[i] >= 0.3 && a.rho[i] <= 1);
    assert.ok(Math.abs(a.cos[i] ** 2 + a.sin[i] ** 2 - 1) < 1e-5);
    assert.ok(a.spread[i] >= -1 && a.spread[i] <= 1);
  }
  const r = seeded(1);
  assert.notEqual(r(), r());
});

test("frame stats keep 10 s, count over-budget draws and missed vsyncs, and pauses are not long frames", () => {
  const s = new FrameStats(10_000);
  for (let i = 0; i < 60; i++) { s.frame(i * 16.7); s.draw(i * 16.7, i === 10 ? 5 : 0.2); }
  s.pause();
  s.frame(5000);
  s.frame(5016.7);
  s.frame(5050);
  const sum = s.summary(4);
  assert.equal(sum.draws, 60);
  assert.equal(sum.overBudget, 1);
  assert.equal(sum.longFrames, 1);
  assert.equal(sum.drawMaxMs, 5);
});
