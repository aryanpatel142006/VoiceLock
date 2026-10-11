// T-150: the /judge Focus / Quiet modes.
import assert from "node:assert/strict";
import { test } from "node:test";
import { DEGRADED_NOTICE, MODES, QUIET_CONTROL, captionFeed, modeControl, modeForKey } from "../judge-modes.js";

test("Focus sends focus_hq, Quiet sends the engine's quiet mode (T-149)", () => {
  assert.deepEqual(modeControl("focus_hq"), { mode: "focus_hq" });
  assert.deepEqual(modeControl("quiet"), { mode: "quiet" });
  assert.deepEqual(QUIET_CONTROL, { mode: "quiet" });
  modeControl("quiet").mode = "x"; // a copy, never the shared object
  assert.equal(QUIET_CONTROL.mode, "quiet");
});

test("Focus after a camera loss without a healthy Enrolled tier plays Natural; Quiet ignores the camera", () => {
  assert.deepEqual(modeControl("focus_hq", { cameraNatural: true }), { mode: "natural" });
  assert.deepEqual(modeControl("quiet", { cameraNatural: true }), QUIET_CONTROL);
});

test("F and Q pick the modes, other keys none", () => {
  assert.equal(modeForKey("f"), "focus_hq");
  assert.equal(modeForKey("F"), "focus_hq");
  assert.equal(modeForKey("q"), "quiet");
  for (const k of ["c", "r", "n", " ", "", undefined]) assert.equal(modeForKey(k), null);
  assert.equal(MODES.focus_hq.copy, "hear who matters");
  assert.equal(MODES.quiet.copy, "stay connected without listening");
});

test("caption header: the profile name, and a notice when the feed degraded to the mixture", () => {
  assert.deepEqual(captionFeed(null), { speaker: "Target", degraded: false, paused: "" });
  const ok = { synth_feed: "focus", synth_feed_request: "focus", enrollment: { profile: "Aryan", source: "profile" } };
  assert.deepEqual(captionFeed(ok), { speaker: "Aryan", degraded: false, paused: "" });
  assert.deepEqual(captionFeed({ ...ok, synth_feed: "mixture" }), { speaker: "Aryan", degraded: true, paused: "" });
  assert.equal(captionFeed({ quiet: { feed: "paused", reason: "gate tier" } }).paused, "gate tier");
  assert.match(DEGRADED_NOTICE, /whole room/);
});
