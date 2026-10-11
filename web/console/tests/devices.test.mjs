// T-121: the pure parts of the device picker (Bluetooth names, level bands, remembered choices).
import assert from "node:assert/strict";
import { test } from "node:test";
import { TOO_QUIET_DB, audioConstraints, deviceWarnings, isBluetoothName, levelBand, loadChoice, peakDb, saveChoice } from "../devices.js";

test("Bluetooth-looking output names raise the badge", () => {
  for (const name of ["AirPods Pro", "Default - Aryan's AirPods", "Bluetooth Headset", "JBL BT 500", "WH-1000XM5 (BT)"]) {
    assert.equal(isBluetoothName(name), true, name);
  }
  for (const name of ["USB-C Headphones", "MacBook Pro Speakers", "Built-in Output", "", null]) {
    assert.equal(isBluetoothName(name), false, String(name));
  }
});

test("level bands: too quiet / good / clipping", () => {
  assert.equal(levelBand(-70), "quiet");
  assert.equal(levelBand(-20), "good");
  assert.equal(levelBand(-1), "clipping");
  assert.equal(levelBand(0), "clipping");
  // T-145: built-in MacBook mic speech at -45 dBFS (meter p10 -45.2) is good, the -58 dBFS room is quiet
  assert.equal(TOO_QUIET_DB, -48);
  assert.equal(levelBand(-45.2), "good");
  assert.equal(levelBand(-40), "good");
  assert.equal(levelBand(-48.5), "quiet");
  assert.equal(levelBand(-58), "quiet");
  assert.ok(Math.abs(peakDb(Float32Array.of(0.1, -0.5, 0.2)) - 20 * Math.log10(0.5)) < 1e-6);
  assert.equal(peakDb(new Float32Array(160)), -90);
});

test("choices survive in storage and a broken storage never throws", () => {
  const store = new Map();
  const storage = { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, v), removeItem: (k) => store.delete(k) };
  saveChoice("k", "dev1", storage);
  assert.equal(loadChoice("k", storage), "dev1");
  saveChoice("k", "", storage);
  assert.equal(loadChoice("k", storage), "");
  const broken = { getItem() { throw new Error("denied"); }, setItem() { throw new Error("denied"); } };
  assert.doesNotThrow(() => saveChoice("k", "x", broken));
  assert.equal(loadChoice("k", broken), "");
});

test("the saved mic is an exact constraint (T-133: an unplugged one fails loudly, then falls back with a banner)", () => {
  assert.deepEqual(audioConstraints("abc").deviceId, { exact: "abc" });
  assert.equal("deviceId" in audioConstraints(""), false);
  assert.equal(audioConstraints("").echoCancellation, false);
});

// T-143: demo path = MacBook Pro Microphone in, AirPods Pro 3 (A2DP) out.
const DEMO = { input: "MacBook Pro Microphone", output: "AirPods" };

test("the demo path (built-in mic, AirPods out) raises no banner", () => {
  for (const [input, output] of [["MacBook Pro Microphone", "Aryan's AirPods Pro 3"], ["Default - MacBook Pro Microphone (Built-in)", "Default - AirPods Pro"]]) {
    const w = deviceWarnings({ input, output, expected: DEMO });
    assert.deepEqual(w.lines, [], `${input} -> ${output}`);
    assert.equal(w.hfp, false);
  }
});

test("a Bluetooth / AirPods mic raises the HFP banner, with or without the preset", () => {
  for (const input of ["Aryan's AirPods Pro 3", "Bluetooth Headset", "WH-1000XM5 (BT)"]) {
    for (const expected of [DEMO, {}]) {
      const w = deviceWarnings({ input, output: "Aryan's AirPods Pro 3", expected });
      assert.equal(w.hfp, true, input);
      assert.match(w.lines.join(" "), /HFP/);
      assert.match(w.lines.join(" "), /MacBook Pro Microphone/);
    }
  }
  assert.equal(deviceWarnings({ input: "AirPods", running: false }).hfp, false); // stopped: no live label
});

test("an output other than the preset's raises the banner with the picker", () => {
  const w = deviceWarnings({ input: "MacBook Pro Microphone", output: "External Headphones", expected: DEMO });
  assert.equal(w.pickOutput, true);
  assert.match(w.lines.join(" "), /demo output is AirPods/);
  assert.deepEqual(deviceWarnings({ input: "MacBook Pro Microphone", output: "External Headphones" }).lines, []); // no preset
});

test("the laptop speakers still raise the feedback banner (once), and missing devices are named", () => {
  const w = deviceWarnings({ input: "MacBook Pro Microphone", output: "MacBook Pro Speakers", expected: DEMO });
  assert.equal(w.speakers, true);
  assert.equal(w.pickOutput, true);
  assert.equal(w.lines.length, 1);
  assert.match(w.lines[0], /feed back/);
  const gone = deviceWarnings({ input: "MacBook Pro Microphone", output: "", missing: { output: "AirPods Pro" }, expected: DEMO });
  assert.equal(gone.lines.length, 1);
  assert.match(gone.lines[0], /Demo output not found \(AirPods Pro\)/);
});

test("a mic other than the preset's is named in the banner", () => {
  const w = deviceWarnings({ input: "RØDE VideoMic GO II", output: "AirPods Pro", expected: DEMO });
  assert.equal(w.hfp, false);
  assert.match(w.lines.join(" "), /the demo mic is MacBook Pro Microphone/);
});
