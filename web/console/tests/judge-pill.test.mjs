// T-088: the judge page's plain-language pill.
import assert from "node:assert/strict";
import { test } from "node:test";
import { judgePill } from "../judge-pill.js";

const t = (state, detail = "", extra = {}) => ({ acquisition: { state, detail, since_s: 0 }, ...extra });

test("walks the presenter through look -> speak -> calibrating -> isolated", () => {
  assert.equal(judgePill(null).text, "Press Start");
  assert.equal(judgePill(t("searching")).text, "Look at the person");
  assert.equal(judgePill(t("candidate")).text, "Look at the person");
  assert.equal(judgePill(t("locked-visual")).text, "Ask them to say one sentence");
  assert.equal(judgePill(t("acquiring", "enrolling 0.4/2.0 s")).text, "Ask them to say one sentence");
  assert.equal(judgePill(t("acquiring", "calibrating 2.3/3.0 s")).text, "Calibrating 2/3 s");
  // T-135: calibrated, the ladder still on its way to Enrolled: no step back to "speak"
  const done = judgePill(t("locked-visual", "face selected; waiting for isolation", { auto_calibrate: { state: "done" } }));
  assert.equal(done.state, "calibrating");
  assert.equal(done.text, "Calibrated — locking on…");
  assert.equal(judgePill(t("locked", "calibrated")).text, "Isolated — now let the other person talk");
  assert.equal(judgePill(t("locked")).state, "isolated");
});

test("fallbacks and the same-voice caveat are one line, never a dialog", () => {
  assert.equal(judgePill(t("locked"), { cameraLost: true }).text, "Camera lost — noise suppression only");
  assert.equal(judgePill(t("locked"), { connected: false }).state, "lost");
  const mac = t("locked", "", { tiers: { same_bucket: true, active: "gate" }, hq_active: "local" });
  assert.equal(judgePill(mac).note, "similar voices — limited isolation");
  const gpu = t("locked", "", { tiers: { same_bucket: true, active: "focus_hq" }, hq_active: "remote" });
  assert.equal(judgePill(gpu).note, "similar voices — using lips");
  assert.equal(judgePill(t("locked", "", { tiers: { same_bucket: false } })).note, "");
});

test("a voice profile names the known voice (T-125)", () => {
  assert.equal(judgePill(t("locked", "known voice: aryan")).text, "Isolated — known voice: aryan");
  assert.equal(judgePill(t("locked", "known voice: aryan")).state, "isolated");
});

test("camera lost keeps Isolated while the Enrolled tier is healthy (T-135)", async () => {
  const { enrolledHealthy } = await import("../judge-pill.js");
  const enrolled = t("locked", "", { tiers: { active: "enrolled", tiers: [{ name: "enrolled", eligible: true }] } });
  assert.ok(enrolledHealthy(enrolled));
  assert.deepEqual(judgePill(enrolled, { cameraLost: true }),
    { state: "isolated", text: "Camera lost — voice profile still isolating", note: "" });
  const gate = t("locked", "", { tiers: { active: "gate", tiers: [{ name: "enrolled", eligible: false }] } });
  assert.equal(judgePill(gate, { cameraLost: true }).text, "Camera lost — noise suppression only");
  assert.equal(enrolledHealthy(null), false);
});

test("pill hysteresis: non-isolated states hold 500 ms, isolated is instant (T-135)", async () => {
  const { PillHysteresis } = await import("../judge-pill.js");
  const h = new PillHysteresis();
  const p = (state, text = state) => ({ state, text, note: "" });
  assert.equal(h.update(p("idle"), 0).state, "idle");
  assert.equal(h.update(p("look"), 100).state, "idle"); // not yet
  assert.equal(h.update(p("speak"), 300).state, "idle"); // a different state restarts the clock
  assert.equal(h.update(p("speak"), 700).state, "idle");
  assert.equal(h.update(p("speak"), 800).state, "speak");
  assert.equal(h.update(p("isolated"), 900).state, "isolated"); // instant
  assert.equal(h.update(p("reacquiring"), 1000).state, "isolated"); // one dropped window
  assert.equal(h.update(p("isolated"), 1100).state, "isolated");
  assert.equal(h.update(p("reacquiring"), 1200).state, "isolated");
  assert.equal(h.update(p("isolated", "Isolated — known voice: aryan"), 1300).text, "Isolated — known voice: aryan");
  assert.equal(h.changes, 2); // idle -> speak -> isolated
  assert.deepEqual(h.history, ["idle", "speak", "isolated"]);
});

test("drawer numbers: EMA rounded to 50 ms (T-135)", async () => {
  const { Steady } = await import("../judge-pill.js");
  const s = new Steady(50, 0.2);
  assert.equal(s.update(1530), 1550);
  assert.equal(s.update(1560), 1550); // 1536 -> 1550
  assert.equal(s.update(NaN), 1550);
  for (let i = 0; i < 40; i++) s.update(700);
  assert.equal(s.update(700), 700);
});

test("a GPU session names the remote tier; the Mac fallback does not (T-156)", () => {
  const gpu = { hq_active: "remote", tiers: { active: "enrolled", remote: true }, remote: { source: "remote" } };
  assert.equal(judgePill(t("locked", "known voice: aryan", gpu)).text, "Isolated — known voice: aryan · iLab GPU · enrolled");
  assert.equal(judgePill(t("searching", "", gpu)).state, "look");
  const fallback = { hq_active: "local", tiers: { active: "gate" }, remote: { source: "local" } };
  assert.equal(judgePill(t("locked", "known voice: aryan", fallback)).text, "Isolated — known voice: aryan");
});

// T-158: the judge page must never look live while the mic is not streaming.
test("connected with telemetry but Start not pressed: Press Start, never the acquisition text", async () => {
  const { micPill, PillHysteresis } = await import("../judge-pill.js");
  const tele = t("searching", "", { seq: null, tiers: { active: "gate", remote: true }, hq_active: "remote" });
  assert.equal(judgePill(tele).text.startsWith("Look at the person"), true); // what the A100 sessions showed
  const h = new PillHysteresis();
  for (let ms = 0; ms <= 5000; ms += 100) {
    const shown = h.update(micPill({ connected: true, started: false }) || judgePill(tele), ms);
    assert.equal(shown.text, "Press Start");
    assert.equal(shown.state, "idle");
  }
  assert.equal(micPill({ connected: false, started: false }), null); // "Reconnecting…" wins
  assert.equal(micPill({ connected: true, started: false, replaying: true }), null); // the replay clip streams
});

test("after Start, no audio frame for 2 s while connected: Mic not streaming", async () => {
  const { micPill, MIC_STALL_MS, PillHysteresis } = await import("../judge-pill.js");
  assert.equal(micPill({ connected: true, started: true, sinceFrameMs: 20 }), null);
  assert.equal(micPill({ connected: true, started: true, sinceFrameMs: MIC_STALL_MS }), null);
  const stalled = micPill({ connected: true, started: true, sinceFrameMs: MIC_STALL_MS + 1 });
  assert.equal(stalled.state, "nomic");
  assert.match(stalled.text, /^Mic not streaming/);
  assert.equal(micPill({ connected: true, started: true, sinceFrameMs: Infinity }).state, "nomic"); // track ended
  const h = new PillHysteresis();
  h.update(judgePill(t("locked")), 0);
  assert.equal(h.update(stalled, 2100).state, "isolated"); // hysteresis: holds 500 ms
  assert.match(h.update(stalled, 2600).text, /^Mic not streaming/);
});

test("a failed or hung Start names the problem for the red banner", async () => {
  const { micProblem, withTimeout, MIC_START_TIMEOUT_MS } = await import("../judge-pill.js");
  const err = (name) => Object.assign(new Error("x"), { name });
  assert.match(micProblem(err("NotAllowedError")), /^Microphone blocked/);
  assert.match(micProblem(err("NotFoundError")), /^No microphone found/);
  assert.match(micProblem(err("NotReadableError")), /busy/);
  assert.match(micProblem(new Error("boom")), /^Microphone unavailable: boom/);
  assert.equal(MIC_START_TIMEOUT_MS, 8000);
  // a getUserMedia that never settles rejects with a TimeoutError after the timeout (fake timer)
  let fire;
  const hung = withTimeout(new Promise(() => {}), MIC_START_TIMEOUT_MS, (fn, ms) => { assert.equal(ms, 8000); fire = fn; return 1; }, () => {});
  fire();
  await assert.rejects(hung, (e) => e.name === "TimeoutError" && /did not start within 8 s/.test(micProblem(e)));
  assert.equal(await withTimeout(Promise.resolve(7), 50), 7);
});

test("audio_only saved voice: Loading voice during the cold start, then Isolated (T-329)", () => {
  const loading = judgePill(t("acquiring", "loading saved voice: aryan"));
  assert.equal(loading.state, "loading");
  assert.equal(loading.text, "Loading voice — aryan");
  const held = judgePill(t("acquiring", "saved voice aryan not playing: overrun"));
  assert.equal(held.state, "fallback");
  assert.equal(held.text, "Voice profile paused — overrun");
  const on = judgePill(t("locked", "known voice: aryan", { tiers: { active: "enrolled" } }));
  assert.equal(on.state, "isolated");
  assert.equal(on.text, "Isolated — known voice: aryan");
  // never "Look at the person" while a saved voice is the lock
  for (const p of [loading, held, on]) assert.notEqual(p.text, "Look at the person");
});
