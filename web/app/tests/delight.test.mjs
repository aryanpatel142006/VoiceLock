// Design pass E (T-427): the pure half of the direct-manipulation and delight layer (web/app/delight.js).
import { test } from "node:test";
import assert from "node:assert/strict";
import { captionWords, dialValue, dropAction, haptic, revealWords, roomAmount, roomDb, roomLabel, seenWords } from "../delight.js";
import { recedeTarget } from "../scene.js";

test("room dial follows the T-093 curve: 3 dB per 10 points to 90, then a knee to silence", () => {
  assert.equal(roomDb(0), 0);
  assert.equal(roomDb(50), 15);
  assert.ok(Math.abs(roomDb(90) - 27) < 1e-9);
  assert.ok(roomDb(95) > 27 && Number.isFinite(roomDb(95)));
  assert.equal(roomDb(100), Infinity);
  assert.equal(roomLabel(0), "as it is");
  assert.equal(roomLabel(50), "−15 dB");
  assert.equal(roomLabel(100), "silent");
  assert.equal(roomAmount(0), 0);
  assert.equal(roomAmount(100), 1);
  assert.equal(dialValue(52), 50);
  assert.equal(dialValue(-9), 0);
  assert.equal(dialValue(117), 100);
  assert.equal(dialValue("x"), 100);
});

test("scene: the room recedes by the engine's confidence times the dial; the dial alone never pulls it back in Raw", () => {
  assert.equal(recedeTarget("focus", 1), 1, "default dial (100) keeps pass C's numbers");
  assert.equal(recedeTarget("focus", 1, 0), 0, "room as it is: nothing recedes");
  assert.equal(recedeTarget("focus", null, 0.5), 0.225);
  assert.equal(recedeTarget("raw", 1, 1), 0);
});

test("captions: only words the engine added fade in; a growing last word does not re-fade", () => {
  assert.equal(seenWords("see you at", []), 0);
  assert.equal(seenWords("see you at noon", ["see you at"]), 3);
  assert.equal(seenWords("see you tomorrow", ["see you tomor"]), 3);
  assert.equal(seenWords("bring an umbrella", ["see you at noon", "bring an"]), 2);
  assert.deepEqual(captionWords("meet me there", ["meet me"]).map((w) => w.fresh), [false, false, true]);
  assert.deepEqual(captionWords("", ["x"]), []);
});

test("drag a voice: into the ring chooses, the chosen one dragged far out releases, else it springs home", () => {
  assert.equal(dropAction({ dx: 10, dy: 5, r: 60 }), "choose");
  assert.equal(dropAction({ dx: 200, dy: 0, r: 60 }), "back");
  assert.equal(dropAction({ dx: 70, dy: 0, r: 60, current: true }), "back");
  assert.equal(dropAction({ dx: 120, dy: 0, r: 60, current: true }), "release");
});

test("haptics: only where vibrate exists, never under reduced motion, never for an unknown event", () => {
  const calls = [];
  const nav = { vibrate: (p) => { calls.push(p); return true; } };
  assert.equal(haptic("isolate", { nav }), true);
  assert.equal(haptic("start", { nav, reduced: true }), false);
  assert.equal(haptic("nope", { nav }), false);
  assert.equal(haptic("press", { nav: {} }), false, "iOS Safari has no vibrate");
  assert.deepEqual(calls, [14]);
});

test("hero reveal keeps the words in order", () => {
  assert.deepEqual(revealWords(" Turn down  the world "), [
    { word: "Turn", i: 0 }, { word: "down", i: 1 }, { word: "the", i: 2 }, { word: "world", i: 3 }]);
});
