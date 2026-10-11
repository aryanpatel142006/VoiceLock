// T-356: the phone companion keeps Listen's rules: the name only while the stage plays that person, no session id.
import { test } from "node:test";
import assert from "node:assert/strict";

import { companionModel } from "../screens/phone.js";
import { Store } from "../store.js";

const telemetry = (extra = {}) => ({
  type: "telemetry", latency_ms: { total: 1500 }, hq_backend: "auto", hq_active: "local", device: "mps",
  enrollment: { profiles: ["aryan", "joseph"], source: "profile", profile: "aryan" },
  acquisition: { state: "locked", detail: "known voice: aryan" }, tiers: { active: "enrolled" }, ...extra,
});

function mirror({ ab = "processed", watch = { state: "open", stale: false } } = {}) {
  const store = new Store();
  store.set({ connection: "open", started: true, telemetry: telemetry(), mode: "focus_hq", ab, watch });
  store.derive(1000, { sinceFrameMs: 20, rttMs: 40 });
  return companionModel(store.state);
}

test("companion isolated: green tone and the person", () => {
  const m = mirror();
  assert.equal(m.tone, "focus");
  assert.equal(m.who, "Aryan");
  assert.equal(m.stage, "Stage hears Aryan");
});

test("companion under Raw: amber, and no name under the word (the room plays, not Aryan)", () => {
  const m = mirror({ ab: "raw" });
  assert.equal(m.tone, "raw");
  assert.equal(m.who, "");
  assert.equal(m.sentence, "Selective on the laptop plays Aryan alone."); // no Space key, no Selective button here
});

test("companion not joined yet: no colour, no name", () => {
  const m = mirror({ watch: { state: "connecting" } });
  assert.equal(m.word, "Connecting");
  assert.equal(m.tone, "idle");
  assert.equal(m.who, "");
});
