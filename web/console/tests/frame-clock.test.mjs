// T-138: the background-tab notice logic (node --test web/console/tests/frame-clock.test.mjs).
import { test } from "node:test";
import assert from "node:assert/strict";
import { BackgroundWatch } from "../frame-clock.js";

test("hidden page with frames at 1 Hz says the camera is paused, once, and clears on the next fast frame", () => {
  const seen = [];
  const w = new BackgroundWatch(5, (p) => seen.push(p));
  w.frame(0);
  w.check(150, true);
  assert.deepEqual(seen, []);  // within 200 ms: fine
  w.check(1000, true);
  w.check(1500, true);
  assert.deepEqual(seen, [true]);
  w.frame(1600);
  assert.deepEqual(seen, [true, false]);
});

test("a visible page never shows the notice (backpressure skips are not a background pause)", () => {
  const seen = [];
  const w = new BackgroundWatch(5, (p) => seen.push(p));
  w.frame(0);
  w.check(5000, false);
  assert.deepEqual(seen, []);
});
