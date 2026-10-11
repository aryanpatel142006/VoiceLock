// T-366: the Welcome screen's pure parts: the press-and-hold state machine (mouse, touch and Space/Enter all feed
// it), where a successful Start cuts to, the field's light and the install hint.
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  HOLD_MS, HOLD_MS_REDUCED, activationLive, discCopy, holdWording, holdInit, holdReduce, installHint, levelLight, startTarget,
} from "../screens/welcome.js";

const run = (events, s = holdInit()) => events.reduce(holdReduce, s);
const press = (source = "pointer:1", at = 0) => ({ type: "press", source, at });
const tick = (at, durationMs = HOLD_MS) => ({ type: "tick", at, durationMs });

test("the ring fills with held time and completes at the hold duration", () => {
  let s = run([press("pointer:1", 1000)]);
  assert.equal(s.phase, "holding");
  assert.equal(s.progress, 0);
  s = holdReduce(s, tick(1450));
  assert.equal(s.phase, "holding");
  assert.equal(s.progress, 0.5);
  s = holdReduce(s, tick(1000 + HOLD_MS - 1));
  assert.equal(s.phase, "holding");
  s = holdReduce(s, tick(1000 + HOLD_MS + 50));
  assert.equal(s.phase, "full");
  assert.equal(s.progress, 1);
});

test("letting go early empties the ring and needs a fresh press", () => {
  const s = run([press("key: ", 0), tick(300), { type: "release", source: "key: " }]);
  assert.equal(s.phase, "idle");
  assert.equal(s.progress, 0);
  assert.equal(holdReduce(s, tick(5000)).phase, "idle"); // a stale frame after release fills nothing
});

test("cancel (blur, pointercancel, tab hidden) drops the hold", () => {
  const s = run([press(), tick(500), { type: "cancel" }]);
  assert.deepEqual([s.phase, s.progress], ["idle", 0]);
});

test("another finger or key cannot end someone else's hold, and a second press is ignored", () => {
  let s = run([press("pointer:1", 0), tick(200)]);
  s = holdReduce(s, { type: "release", source: "pointer:2" });
  assert.equal(s.phase, "holding");
  s = holdReduce(s, { type: "release", source: "key:Enter" });
  assert.equal(s.phase, "holding");
  const again = holdReduce(s, press("key:Enter", 300));
  assert.equal(again, s); // same object: nothing changed, t0 kept
});

test("Space and Enter hold the same way as the pointer", () => {
  for (const source of ["key: ", "key:Enter", "pointer:7"]) {
    const s = run([press(source, 10), tick(10 + HOLD_MS)]);
    assert.equal(s.phase, "full", source);
  }
});

test("reduced motion only shortens the fill; the hold is still required", () => {
  let s = run([press("pointer:1", 0), tick(HOLD_MS_REDUCED / 2, HOLD_MS_REDUCED)]);
  assert.equal(s.progress, 0.5);
  s = holdReduce(s, tick(HOLD_MS_REDUCED, HOLD_MS_REDUCED));
  assert.equal(s.phase, "full");
  assert.ok(HOLD_MS_REDUCED > 0 && HOLD_MS_REDUCED < HOLD_MS);
});

test("a full ring starts at once, or arms and starts on release when the activation lapsed", () => {
  const full = run([press(), tick(HOLD_MS)]);
  assert.equal(holdReduce(full, { type: "start" }).phase, "starting");
  const armed = holdReduce(full, { type: "arm" });
  assert.equal(armed.phase, "armed");
  assert.equal(holdReduce(armed, { type: "release", source: "pointer:2" }).phase, "armed");
  assert.equal(holdReduce(armed, { type: "release", source: "pointer:1" }).phase, "starting");
  assert.equal(holdReduce(armed, { type: "cancel" }).phase, "idle");
  assert.equal(holdReduce(holdInit(), { type: "start" }).phase, "idle"); // only a full ring can start
});

test("only a mic that opened finishes; a failure keeps the reason and allows a retry", () => {
  const starting = run([press(), tick(HOLD_MS), { type: "start" }]);
  assert.equal(holdReduce(starting, { type: "started", ok: true }).phase, "done");
  const failed = holdReduce(starting, { type: "started", ok: false, error: "Microphone blocked: allow it" });
  assert.deepEqual([failed.phase, failed.error, failed.progress], ["failed", "Microphone blocked: allow it", 0]);
  const retry = holdReduce(failed, press("key:Enter", 50));
  assert.deepEqual([retry.phase, retry.error], ["holding", ""]);
  const done = holdReduce(starting, { type: "started", ok: true });
  assert.equal(holdReduce(done, press()), done); // no restart after the cut began
  assert.equal(holdReduce(holdInit(), { type: "started", ok: true }).phase, "idle"); // stray results are ignored
});

test("activationLive trusts navigator.userActivation when present", () => {
  assert.equal(activationLive({}), true);
  assert.equal(activationLive({ userActivation: { isActive: true } }), true);
  assert.equal(activationLive({ userActivation: { isActive: false } }), false);
});

test("a saved mic cuts to Listen, a first run to Setup", () => {
  const store = (v) => ({ getItem: (k) => (k === "hearing.mic" ? v : null) });
  assert.equal(startTarget(store("abc123")), "listen");
  assert.equal(startTarget(store(null)), "setup");
  assert.equal(startTarget({ getItem: () => { throw new Error("denied"); } }), "setup");
});

test("field light maps the scene contract's -60..-6 dBFS and is dark without a level", () => {
  assert.equal(levelLight(-60), 0);
  assert.equal(levelLight(-80), 0);
  assert.equal(levelLight(-6), 1);
  assert.equal(levelLight(0), 1);
  assert.equal(levelLight(-33), 0.5);
  assert.equal(levelLight(null), 0);
  assert.equal(levelLight(NaN), 0);
});

test("install hint only outside standalone; iPhone gets the Share steps", () => {
  assert.equal(installHint({ standalone: true, ios: true }), null);
  assert.match(installHint({ standalone: false, ios: true }), /tap Share, then Add to Home Screen/);
  assert.match(installHint({ standalone: false, ios: false }), /iPhone: Share, then Add to Home Screen/);
});

test("every phase has disc copy", () => {
  for (const p of ["idle", "holding", "full", "armed", "starting", "done", "failed"]) {
    const [a, b] = discCopy(p);
    assert.ok(a && b, p);
  }
  assert.equal(discCopy("idle")[0], "Hold to start");
});

test("mic problem lines name the held Start", () => {
  assert.equal(holdWording("allow it, then press Start again."), "allow it, then hold Start again.");
  assert.equal(holdWording("Unavailable. Press Start to try again."), "Unavailable. Hold Start to try again.");
  assert.equal(holdWording(""), "");
});

// ---------------------------------------------------------------- T-429: the cold open and the field's pure maths
import {
  BANDS, COLD_OPEN, GRAINS, ambientAt, bounceStep, collapseEase, collapseTarget, fieldLabel, grainTable, heroWords,
  holdSoundAt, introAt, releasedDb, wordDelay,
} from "../screens/welcome.js";

test("the hero line is eight words and the beat lands once, before 'without'", () => {
  const words = heroWords();
  assert.equal(words.length, 8);
  assert.equal(words[4], "without");
  const d = words.map((_, i) => wordDelay(i));
  assert.equal(d[0], COLD_OPEN.words);
  assert.equal(d[1] - d[0], COLD_OPEN.wordGap);
  assert.equal(d[4] - d[3], COLD_OPEN.wordGap + COLD_OPEN.beat); // the beat
  assert.equal(d[5] - d[4], COLD_OPEN.wordGap); // only once
  assert.ok(d.at(-1) + COLD_OPEN.wordMs < COLD_OPEN.ui, "the last word has landed before the disc arrives");
  assert.ok(COLD_OPEN.end >= 6000 && COLD_OPEN.end <= 10_000, "a 6-10 s cold open");
});

test("the grain table is seeded (same field on every load), in the unit square, bands interleaved", () => {
  const a = grainTable();
  const b = grainTable();
  assert.equal(a.x.length, GRAINS);
  assert.deepEqual([...a.x.slice(0, 5)], [...b.x.slice(0, 5)]);
  for (let i = 0; i < GRAINS; i++) {
    assert.ok(a.x[i] >= 0 && a.x[i] <= 1 && a.y[i] >= 0 && a.y[i] <= 1, `grain ${i} inside the frame`);
    assert.equal(a.band[i], i % BANDS.length);
    assert.ok(a.order[i] >= 0 && a.order[i] <= 1);
  }
  assert.notDeepEqual([...grainTable(GRAINS, 7).x.slice(0, 5)], [...a.x.slice(0, 5)]);
});

test("the collapse ease is monotonic from 0 to 1 and clamps", () => {
  assert.equal(collapseEase(0), 0);
  assert.equal(collapseEase(1), 1);
  assert.equal(collapseEase(-1), 0);
  assert.equal(collapseEase(2), 1);
  let prev = 0;
  for (let p = 0; p <= 1.0001; p += 0.05) { const v = collapseEase(p); assert.ok(v >= prev); prev = v; }
  assert.ok(collapseEase(0.25) < 0.25 && collapseEase(0.75) > 0.75, "slow to start, decisive at the end");
});

test("the intro: a black frame, then the hairline grows, then the grains scatter and are all home by `scattered`", () => {
  assert.deepEqual(introAt(0), { line: 0, scatter: 0 });
  assert.deepEqual(introAt(COLD_OPEN.line), { line: 0, scatter: 0 });
  const mid = introAt((COLD_OPEN.line + COLD_OPEN.scatter) / 2);
  assert.ok(mid.line > 0.5 && mid.line < 1 && mid.scatter === 0, "ease-out: past half way at half time");
  assert.deepEqual(introAt(COLD_OPEN.scatter), { line: 1, scatter: 0 });
  assert.deepEqual(introAt(COLD_OPEN.scattered), { line: 1, scatter: 1 });
  assert.deepEqual(introAt(1e6), { line: 1, scatter: 1 });
});

test("the drift is ambient through the cold open and settles to still by its end", () => {
  assert.equal(ambientAt(0), 1);
  assert.equal(ambientAt(COLD_OPEN.settle), 1);
  const half = ambientAt((COLD_OPEN.settle + COLD_OPEN.end) / 2);
  assert.ok(half > 0 && half < 1);
  assert.equal(ambientAt(COLD_OPEN.end), 0);
  assert.equal(ambientAt(COLD_OPEN.end + 5000), 0);
});

test("the field collapses with the ring, stays one voice through the start, and lets go on idle or failure", () => {
  assert.equal(collapseTarget("idle", 0), 0);
  assert.equal(collapseTarget("failed", 0), 0);
  assert.equal(collapseTarget("holding", 0), 0);
  assert.equal(collapseTarget("holding", 0.5), collapseEase(0.5));
  assert.equal(collapseTarget("holding", 1), 1);
  for (const p of ["full", "armed", "starting", "done"]) assert.equal(collapseTarget(p, 0), 1, p);
});

test("the release spring overshoots a little, then settles exactly on its target", () => {
  const st = { x: 1, v: 0 };
  let min = 1;
  let t = 0;
  while (t < 2 && !(st.x === 0 && st.v === 0)) { bounceStep(st, 0, 1 / 60); min = Math.min(min, st.x); t += 1 / 60; }
  assert.ok(min < 0 && min > -0.3, `overshoot ${min}`);
  assert.deepEqual(st, { x: 0, v: 0 });
  assert.ok(t < 1.5, `settled in ${t.toFixed(2)} s`);
  // critically damped (zeta 1): no overshoot on the way to a full ring
  const up = { x: 0.3, v: 0 };
  let max = 0;
  for (let i = 0; i < 120; i++) { bounceStep(up, 1, 1 / 60, { omega: 24, zeta: 1 }); max = Math.max(max, up.x); }
  assert.ok(max <= 1 + 1e-6);
  const same = { x: 0, v: 0 };
  assert.equal(bounceStep(same, 0, 0.5), same);
  assert.deepEqual(same, { x: 0, v: 0 });
});

test("the field's label names what drives it: ambient, still, holding, or the real mic level", () => {
  assert.equal(fieldLabel({ cold: true, live: false, phase: "idle" }), "ambient, not the mic");
  assert.equal(fieldLabel({ cold: false, live: false, phase: "idle" }), "still, mic off");
  assert.equal(fieldLabel({ cold: false, live: false, phase: "holding" }), "holding, mic off");
  assert.equal(fieldLabel({ cold: false, live: false, phase: "starting" }), "opening the mic");
  assert.equal(fieldLabel({ cold: false, live: true, phase: "done", db: -31.4 }), "mic live, -31 dBFS");
  assert.equal(fieldLabel({ cold: false, live: true, phase: "done", db: null }), "mic live, waiting for frames");
  assert.equal(fieldLabel({ cold: true, live: false, replaying: true, phase: "idle" }), "replay, not the mic");
});

test("the level keeps the meter's 24 dB/s release once events stop, down to the floor", () => {
  assert.equal(releasedDb(-20, 0), -20);
  assert.equal(releasedDb(-20, 1), -44);
  assert.equal(releasedDb(-20, 10), -60);
  assert.equal(releasedDb(null, 1), null);
  assert.equal(releasedDb(NaN, 1), null);
});

test("the hold sound narrows a band of noise into one rising tone", () => {
  const a = holdSoundAt(0);
  const z = holdSoundAt(1);
  assert.ok(a.noiseHz > 3000 && a.noiseGain > 0 && a.toneGain === 0, "noise only at the start");
  assert.ok(z.noiseGain === 0 && z.toneGain > 0 && z.toneHz > a.toneHz, "one tone at the end");
  let prevHz = Infinity;
  for (let p = 0; p <= 1; p += 0.1) { const s = holdSoundAt(p); assert.ok(s.noiseHz <= prevHz); prevHz = s.noiseHz; }
  assert.ok(holdSoundAt(1).toneGain < 0.1 && holdSoundAt(0).noiseGain < 0.1, "quiet: a UI sound, not a signal");
});
