// T-356: a People card says "Following now." only while that person is what plays.
import { test } from "node:test";
import assert from "node:assert/strict";

import { cardMeta } from "../screens/people.js";

const aryan = { current: true, key: "1", version: "v2" };

test("following only while playing; Raw and Quiet say what plays instead", () => {
  assert.equal(cardMeta(aryan, { playing: true, streaming: true, keys: false }), "Following now. Profile v2.");
  assert.match(cardMeta(aryan, { streaming: true, held: "raw" }), /^Chosen\. Raw plays the room for now\./);
  assert.match(cardMeta(aryan, { streaming: true, held: "quiet" }), /^Chosen\. Quiet: captions only\./);
  assert.match(cardMeta(aryan, { streaming: true }), /^Chosen\. Finding them\./);
  assert.match(cardMeta(aryan, {}), /^Chosen\. Plays when you start\./);
  assert.match(cardMeta({ current: false }, { playing: true }), /^Saved voice\./);
});
