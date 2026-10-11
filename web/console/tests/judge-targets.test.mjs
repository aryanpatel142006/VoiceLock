// T-177: the /judge saved-person / Release buttons, the target line and the "Keys off" banner.
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  KEYS_OFF_TEXT, KeyFocus, RELEASE_CONTROL, SEARCHING_PILL, abControl, isReleaseKey, isTextField, personControl,
  personLabel, savedPeople, targetLine,
} from "../judge-targets.js";

const tel = (enrollment) => ({ type: "telemetry", enrollment });

test("each person button sends the message keys 1/2 send; no profile, no message", () => {
  const names = savedPeople(tel({ profiles: ["aryan", "joseph", "third"] }));
  assert.deepEqual(names, ["aryan", "joseph"]);
  assert.deepEqual(personControl(names, 0), { enroll: { action: "profile", name: "aryan" } });
  assert.deepEqual(personControl(names, 1), { enroll: { action: "profile", name: "joseph" } });
  assert.equal(personControl(["aryan"], 1), null);
  assert.deepEqual(savedPeople(null), []);
  assert.equal(personLabel("joseph"), "Joseph");
  assert.equal(personLabel(""), "");
});

test("Release sends enroll clear; key 0 / Shift+0 is Release", () => {
  assert.deepEqual(RELEASE_CONTROL, { enroll: { action: "clear" } });
  assert.equal(isReleaseKey("0", "Digit0"), true);
  assert.equal(isReleaseKey(")", "Digit0"), true); // Shift+0 on a US layout
  assert.equal(isReleaseKey("0", "Numpad0"), true);
  assert.equal(isReleaseKey("1", "Digit1"), false);
  assert.equal(SEARCHING_PILL.text, "Searching");
});

test("Raw / Isolated buttons set their side, Space flips", () => {
  assert.deepEqual(abControl("processed", "raw"), { ab: "raw" });
  assert.deepEqual(abControl("raw", "processed"), { ab: "processed" });
  assert.deepEqual(abControl("raw", "raw"), { ab: "raw" });
  assert.deepEqual(abControl("processed"), { ab: "raw" });
  assert.deepEqual(abControl("raw"), { ab: "processed" });
});

test("the target line names the saved person, says none after Release", () => {
  assert.equal(targetLine(null), "Target: —");
  assert.equal(targetLine(tel({ source: null })), "Target: none yet");
  assert.equal(targetLine(tel({ source: "profile", profile: "joseph" })), "Target: Joseph");
  assert.equal(targetLine(tel({ source: "calibrated" })), "Target: calibrated voice");
  assert.match(targetLine(tel({ source: "profile", profile: "joseph" }), { released: true }), /^Target: none/);
});

test("banner: off when the window blurs or a text field has focus, back on focus", () => {
  const k = new KeyFocus(true);
  assert.equal(k.off, false);
  k.blur();
  assert.equal(k.off, true);
  k.focus();
  assert.equal(k.off, false);
  k.focusOn("INPUT");
  assert.equal(k.off, true);
  k.focusOn("BUTTON"); // a clicked button keeps the keys (onKey only skips text fields)
  assert.equal(k.off, false);
  k.focusOn("select");
  assert.equal(k.off, true);
  k.focusOn(null); // focusout to the page
  assert.equal(k.off, false);
  assert.equal(new KeyFocus(false).off, true); // the page opened without focus
  assert.equal(isTextField("TEXTAREA"), true);
  assert.equal(KEYS_OFF_TEXT, "Keys off: click the page");
});
