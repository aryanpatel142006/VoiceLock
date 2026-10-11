// T-347: the Listen screen's pure parts: the stage keys (identical to /judge) and what the screen shows for a state.
import { test } from "node:test";
import assert from "node:assert/strict";

import { keyAction } from "../keys.js";
import { STATE_WORDS, levelUnit, listenModel } from "../screens/listen.js";
import { Store, initialState } from "../store.js";

const k = (key, extra = {}) => keyAction({ key, code: "", ...extra }, extra);

test("stage keys match /judge: Space, Shift+F/Q, 1/2, Shift+0, D, Shift+R", () => {
  assert.deepEqual(k(" "), { type: "ab", preventDefault: true });
  assert.equal(k("f"), null); // bare F/Q do nothing (T-150)
  assert.equal(k("q"), null);
  assert.deepEqual(k("F", { shiftKey: true }), { type: "mode", mode: "focus_hq" });
  assert.deepEqual(k("Q", { shiftKey: true }), { type: "mode", mode: "quiet" });
  assert.deepEqual(k("1"), { type: "person", index: 0 });
  assert.deepEqual(k("2"), { type: "person", index: 1 });
  assert.equal(k("0"), null); // Release needs Shift (or the open sheet)
  assert.deepEqual(keyAction({ key: ")", code: "Digit0", shiftKey: true }), { type: "release" });
  assert.deepEqual(keyAction({ key: "0", code: "Digit0" }, { drawerOpen: true }), { type: "release" });
  assert.deepEqual(k("d"), { type: "diag" });
  assert.equal(k("r"), null);
  assert.deepEqual(k("R", { shiftKey: true }), { type: "replay" });
  assert.deepEqual(k("C", { shiftKey: true }), { type: "calibrate" });
  assert.equal(k("n", { shiftKey: true }), null); // Name Call is /judge only
});

test("keys are ignored with modifiers, in text fields and on repeat where /judge ignores them", () => {
  assert.equal(k(" ", { metaKey: true }), null);
  assert.equal(k("1", { inField: true }), null);
  assert.equal(k("2", { repeat: true }), null);
  assert.equal(k("F", { shiftKey: true, repeat: true }), null);
  assert.equal(keyAction({ key: ")", code: "Digit0", shiftKey: true, repeat: true }), null);
  assert.deepEqual(k(" ", { repeat: true }), { type: "none", preventDefault: true }); // no scroll, no flip
});

function telemetry(extra = {}) {
  return {
    type: "telemetry", latency_ms: { total: 1500 }, hq_backend: "auto", hq_active: "local", device: "mps",
    enrollment: { profiles: ["aryan", "joseph"], source: "profile", profile: "aryan" },
    acquisition: { state: "locked", detail: "known voice: aryan" }, tiers: { active: "enrolled" },
    tse: { confidence: 0.82 }, ...extra,
  };
}

function live(store, t, { started = true, mode = "focus_hq", ab = "processed" } = {}) {
  store.set({ connection: "open", started, telemetry: t, mode, ab });
  store.derive(1000, { sinceFrameMs: 20, rttMs: 40 });
  return store.state;
}

test("idle before Start: grey word, no colour promise, numbers are dashes", () => {
  const m = listenModel(initialState());
  assert.equal(m.word, "Idle");
  assert.equal(m.live, false);
  assert.equal(m.start, "start");
  assert.equal(m.truth.delay, "—");
  assert.equal(m.truth.backend, "—");
  assert.equal(m.people[0].enabled, false);
  assert.equal(m.people[0].label, "Person 1");
});

test("isolated: green word, the person, the target line, measured numbers", () => {
  const s = live(new Store(), telemetry());
  const m = listenModel(s);
  assert.equal(m.word, "Isolated");
  assert.equal(m.tone, "focus");
  assert.equal(m.who, "Aryan");
  assert.equal(m.hear, "You hear Aryan");
  assert.equal(m.targetLine, "Target: Aryan");
  assert.equal(m.people[0].pressed, true);
  assert.equal(m.people[1].label, "Joseph");
  assert.equal(m.truth.delay, "1500 ms");
  assert.equal(m.truth.confidence, undefined); // "Voice match" lives in Diagnostics, not on stage (T-365 P0-4)
  assert.equal(m.sentence, ""); // the word and the name say it; no "Isolated — known voice: aryan"
  assert.equal(m.truth.backend, "Mac"); // the device name is engine language; Diagnostics shows it (T-388)
});

test("loading voice names the saved voice; Raw makes the word Raw", () => {
  const t = telemetry({ acquisition: { state: "acquiring", detail: "loading saved voice: joseph" } });
  const m = listenModel(live(new Store(), t));
  assert.equal(m.word, "Loading voice");
  assert.equal(m.detail, "joseph");
  const raw = listenModel(live(new Store(), telemetry(), { ab: "raw" }));
  assert.equal(raw.ab, "raw");
  assert.equal(raw.word, "Raw"); // the room plays: the word says so, amber
  assert.equal(raw.tone, "raw");
  assert.match(raw.hear, /room/);
  assert.equal(raw.sentence, "Space to hear Aryan alone."); // Raw: one state, one message (T-365 P1-2)
  const touch = listenModel(live(new Store(), telemetry(), { ab: "raw" }), { keys: false });
  assert.equal(touch.sentence, "Tap Selective to hear Aryan alone."); // no Space key on a touch screen
});

test("no mic, reconnecting, keys off and mic banners come from the store", () => {
  const store = new Store();
  store.set({ connection: "open", started: true, telemetry: telemetry() });
  store.derive(0, { sinceFrameMs: 5000 });
  assert.equal(listenModel(store.state).word, "No mic");
  const lost = { ...initialState(), connection: "reconnecting", attempt: 2, retryInMs: 1000,
    pill: { state: "lost", text: "Reconnecting to the engine…", note: "" } };
  const m = listenModel(lost, { keysOff: true });
  assert.equal(m.word, "Reconnecting");
  assert.match(m.banners.engine, /Retrying in 1\.0 s \(attempt 2\)/);
  assert.equal(m.banners.keysOff, "Keys off: click the page");
  const mic = listenModel({ ...initialState(), mic: { ...initialState().mic, banner: "Microphone blocked" } });
  assert.equal(mic.banners.mic, "Microphone blocked");
});

test("Quiet with paused captions shows the reason; REPLAY badge follows the server", () => {
  const t = telemetry({ quiet: { feed: "paused", reason: "target not isolated" } });
  const store = new Store();
  live(store, t, { mode: "quiet" });
  store.set({ captions: [{ text: "see you at noon", final: true }], replaying: true, replayClip: "team-mix/clip.wav" });
  store.derive(1100, { sinceFrameMs: 20 });
  const m = listenModel(store.state);
  assert.equal(m.mode, "quiet");
  assert.equal(m.captions.show, true);
  assert.equal(m.captions.paused, "Captions wait until Aryan is isolated.");
  assert.equal(m.captions.speaker, "Aryan");
  assert.equal(m.captions.lines[0].text, "see you at noon");
  assert.equal(m.replay, "Replay · team-mix");
  assert.equal(m.start, "hidden");
});

test("every pill state has a word; the halo maps dBFS to 0..1", () => {
  for (const state of ["idle", "lost", "nomic", "loading", "calibrating", "isolated", "look", "speak", "reacquiring", "fallback"]) {
    assert.ok(STATE_WORDS[state]?.word, state);
  }
  assert.equal(levelUnit(-60), 0);
  assert.equal(levelUnit(-90), 0);
  assert.equal(levelUnit(0), 1);
  assert.equal(levelUnit(-30), 0.5);
  assert.equal(levelUnit(null), 0);
});

test("polish (T-356): engine words never reach the stage; nothing is pressed or timed while idle", async () => {
  const { plainSentence, cleanLine, pausedCopy } = await import("../screens/listen.js");
  assert.match(plainSentence({ state: "fallback", text: "Voice profile paused — overrun" }, { word: "Not isolated" }), /^Catching up/);
  assert.equal(plainSentence({ state: "fallback", text: "Voice profile paused — warming" }, { word: "Not isolated", who: "Aryan" }),
    "You hear the cleaned room until Aryan's voice is back.");
  assert.equal(plainSentence({ state: "loading", text: "Loading voice — aryan" }, { word: "Loading voice", who: "Aryan" }),
    "Loading Aryan's saved voice…");
  assert.equal(plainSentence({ state: "look", text: "Searching" }, { word: "Searching" }), "");
  // T-403: audio-only, so never the judge pill's camera instructions
  assert.equal(plainSentence({ state: "look", text: "Look at the person" }, { word: "Searching" }), "Choose a saved person below.");
  assert.equal(plainSentence({ state: "look", text: "Look at the person" }, { word: "Searching", who: "Aryan" }), "Waiting to hear Aryan.");
  assert.equal(plainSentence({ state: "reacquiring", text: "Look at the person again" }, { word: "Finding them" }),
    "Finding them again: keep them talking.");
  assert.equal(cleanLine("Traffic. //"), "Traffic.");
  assert.equal(pausedCopy("not isolating the selected person yet (audio-only tier): captions paused", ""),
    "Captions wait until the chosen voice is isolated.");
  // a stopped session: the engine still names Aryan, but nothing plays, so nothing is pressed and Delay is "—"
  const store = new Store();
  store.set({ connection: "open", started: false, telemetry: telemetry() });
  store.derive(0, { sinceFrameMs: 0 });
  const m = listenModel(store.state);
  assert.equal(m.people[0].pressed, false);
  assert.equal(m.truth.delay, "—");
  assert.equal(m.who, ""); // nobody is heard, so nobody is named in the hero
});

test("T-382 gate pill: SOLO / OVERLAP-DUCKED / UNSURE from tse fields, hidden without them", () => {
  const gate = (tse, opts) => listenModel(live(new Store(), telemetry({ tse }), opts)).gate;
  assert.equal(gate({ confidence: 0.82 }), null); // older engine: no gated field, nothing invented
  assert.equal(gate({ confidence: 0.82, gated: false, soft_gated: false }), "solo");
  assert.equal(gate({ confidence: 0.2, gated: false, soft_gated: false }), "unsure");
  assert.equal(gate({ confidence: 0, gated: true, soft_gated: false }), "unsure");
  assert.equal(gate({ confidence: 0, gated: true, soft_gated: true }), "ducked");
  assert.equal(gate({ confidence: 0.9, gated: false }, { ab: "raw" }), null); // raw plays the room
});
