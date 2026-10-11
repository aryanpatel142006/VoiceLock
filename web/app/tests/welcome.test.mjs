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
