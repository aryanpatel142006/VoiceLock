// T-287: the iPhone mic on /judge. Start opens the preset's expected mic by label; when it drops mid-session the
// page reopens the MacBook mic within 1 s with a red banner, and Retry switches back. A fake mediaDevices.
import assert from "node:assert/strict";
import { test } from "node:test";
import { FALLBACK_MIC, MIC_KEY, MIC_LABEL_KEY, createMic, deviceWarnings, lostLine } from "../devices.js";

const IPHONE = { deviceId: "iphone-id", label: "Aryan patel's iPhone Microphone" };
const MACBOOK = { deviceId: "macbook-id", label: "MacBook Pro Microphone (Built-in)" };
const RODE = { deviceId: "rode-id", label: "RØDE VideoMic GO II" };
const EXPECTED = "Aryan patel's iPhone Microphone"; // demo.sh --mic / preset.pins.input_device

class FakeTrack extends EventTarget {
  constructor(dev) { super(); this.label = dev.label; this.deviceId = dev.deviceId; this.readyState = "live"; }
  getSettings() { return { deviceId: this.deviceId }; }
  stop() { this.readyState = "ended"; } // stop() never fires "ended" (as in browsers)
  unplug() { this.readyState = "ended"; this.dispatchEvent(new Event("ended")); }
}

/** Chrome-like: a "default" entry plus the real inputs; no labels before the first getUserMedia unless `granted`. */
function fakeMedia(devices, { granted = true, defaultId = MACBOOK.deviceId } = {}) {
  return {
    devices: [...devices], opened: [], streams: [], granted,
    async enumerateDevices() {
      const real = this.devices.map((d) => ({ kind: "audioinput", deviceId: d.deviceId, label: this.granted ? d.label : "" }));
      const def = this.devices.find((d) => d.deviceId === defaultId) || this.devices[0];
      const out = { kind: "audiooutput", deviceId: "default", label: this.granted ? "Default - AirPods Pro" : "" };
      return def ? [{ kind: "audioinput", deviceId: "default", label: this.granted ? `Default - ${def.label}` : "" }, ...real, out] : [out];
    },
    async getUserMedia({ audio }) {
      const want = audio.deviceId?.exact;
      const dev = want ? this.devices.find((d) => d.deviceId === want)
        : this.devices.find((d) => d.deviceId === defaultId) || this.devices[0];
      if (!dev) throw Object.assign(new Error("device gone"), { name: want ? "OverconstrainedError" : "NotFoundError" });
      this.granted = true;
      const track = new FakeTrack(dev);
      const stream = { getAudioTracks: () => [track], getTracks: () => [track], track };
      this.opened.push(want || "default");
      this.streams.push(stream);
      return stream;
    },
  };
}

function memoryStorage(init = {}) {
  const m = new Map(Object.entries(init));
  return { getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, v), removeItem: (k) => m.delete(k) };
}

/** The controller as /judge wires it: onSwap is judge.js swapMic, which re-sends the hello with the new label. */
function judgeLike(media, storage) {
  const hellos = [];
  const mic = createMic({
    media, storage,
    onSwap: (stream, info) => {
      hellos.push({ type: "hello", mic_switch: `${info.reason}: ${info.from} -> ${info.to}`, input_device: mic.label(), devices_missing: mic.lost() ? ["mic"] : [] });
      return true;
    },
  });
  return { mic, hellos };
}

const banner = (mic, expected = EXPECTED) =>
  deviceWarnings({ input: mic.label() || "", output: "AirPods Pro", missing: { mic: mic.missing() }, expected: { input: expected }, running: Boolean(mic.label()), lost: mic.lost() });

async function startOn(media, storage) {
  const { mic, hellos } = judgeLike(media, storage);
  await mic.setExpected(EXPECTED);
  const stream = await mic.open();
  mic.attach(stream);
  return { mic, hellos, stream };
}

test("(a) expected mic plugged in: Start opens it by exact deviceId and saves it, over a different saved mic", async () => {
  const media = fakeMedia([MACBOOK, IPHONE, RODE]);
  const storage = memoryStorage({ [MIC_KEY]: RODE.deviceId, [MIC_LABEL_KEY]: RODE.label });
  const { mic, stream } = await startOn(media, storage);
  assert.deepEqual(media.opened, [IPHONE.deviceId]);
  assert.equal(stream.track.label, IPHONE.label);
  assert.equal(storage.getItem(MIC_KEY), IPHONE.deviceId);
  assert.equal(storage.getItem(MIC_LABEL_KEY), IPHONE.label);
  assert.equal(mic.label(), IPHONE.label);
  assert.deepEqual(banner(mic).lines, []);
});

test("(a) first Start before permission (no labels): opens the default, then re-opens the expected mic once", async () => {
  const media = fakeMedia([MACBOOK, IPHONE], { granted: false });
  const storage = memoryStorage();
  const { mic } = await startOn(media, storage);
  assert.deepEqual(media.opened, ["default", IPHONE.deviceId]);
  assert.equal(media.streams[0].track.readyState, "ended", "the default stream is released");
  assert.equal(mic.label(), IPHONE.label);
  assert.equal(storage.getItem(MIC_KEY), IPHONE.deviceId);
});

test("(b) expected mic absent: the default opens and the banner names the expected mic", async () => {
  const media = fakeMedia([MACBOOK, RODE]);
  const storage = memoryStorage();
  const { mic } = await startOn(media, storage);
  assert.deepEqual(media.opened, ["default"]);
  assert.equal(mic.label(), MACBOOK.label);
  assert.equal(storage.getItem(MIC_KEY), null, "nothing saved");
  const w = banner(mic);
  assert.equal(w.wrongMic, true);
  assert.deepEqual(w.lines, [`Mic is ${MACBOOK.label}, the demo mic is ${EXPECTED}.`]);
});

test("(b) expected absent and the saved mic unplugged: the default, with the not-found line", async () => {
  const media = fakeMedia([MACBOOK]);
  const storage = memoryStorage({ [MIC_KEY]: RODE.deviceId, [MIC_LABEL_KEY]: RODE.label });
  const { mic } = await startOn(media, storage);
  assert.deepEqual(media.opened, ["default"]); // the exact RØDE open threw (unplugged)
  assert.match(banner(mic).lines[0], /Demo mic not found \(RØDE VideoMic GO II\): using MacBook Pro Microphone/);
});

test("(c) the iPhone track ends mid-session: MacBook mic within 1 s, exact banner, hello names it", async () => {
  const media = fakeMedia([MACBOOK, IPHONE]);
  const storage = memoryStorage();
  const { mic, hellos, stream } = await startOn(media, storage);
  media.devices = media.devices.filter((d) => d !== IPHONE);
  const t0 = performance.now();
  stream.track.unplug();
  const ok = await mic.busy();
  assert.equal(ok, true);
  assert.ok(performance.now() - t0 < 1000, "reopened within 1 s");
  assert.equal(media.opened.at(-1), MACBOOK.deviceId, "the MacBook mic, by exact deviceId");
  assert.equal(mic.label(), MACBOOK.label);
  const w = banner(mic);
  assert.deepEqual(w.lines, ["iPhone mic lost: using MacBook mic"]);
  assert.equal(hellos.length, 1);
  assert.equal(hellos[0].input_device, MACBOOK.label);
  assert.deepEqual(hellos[0].devices_missing, ["mic"]);
  assert.equal(hellos[0].mic_switch, `lost (ended): ${IPHONE.label} -> ${MACBOOK.label}`);
  assert.equal(storage.getItem(MIC_KEY), IPHONE.deviceId, "the saved choice stays the iPhone");
});

test("(c) a devicechange that removes the live mic (no 'ended' yet) also falls back", async () => {
  const media = fakeMedia([MACBOOK, IPHONE]);
  const { mic, hellos } = await startOn(media, memoryStorage());
  media.devices = [MACBOOK];
  assert.equal(await mic.deviceChange(), true);
  assert.equal(mic.label(), MACBOOK.label);
  assert.match(hellos[0].mic_switch, /^lost \(removed\)/);
  assert.equal(await mic.deviceChange(), false, "the MacBook mic is still listed: nothing to do");
});

test("(c) no MacBook mic listed: the system default, named in the banner", async () => {
  const media = fakeMedia([RODE, IPHONE], { defaultId: RODE.deviceId });
  const { mic, stream } = await startOn(media, memoryStorage());
  media.devices = [RODE];
  stream.track.unplug();
  await mic.busy();
  assert.equal(media.opened.at(-1), "default");
  assert.deepEqual(banner(mic).lines, [`iPhone mic lost: using ${RODE.label}`]);
});

test("(d) the iPhone comes back: Retry switches back, saves it and the banner clears", async () => {
  const media = fakeMedia([MACBOOK, IPHONE]);
  const storage = memoryStorage();
  const { mic, hellos, stream } = await startOn(media, storage);
  media.devices = [MACBOOK];
  stream.track.unplug();
  await mic.busy();
  assert.equal(await mic.retry(), false, "still unplugged: stays on the MacBook mic");
  assert.deepEqual(banner(mic).lines, ["iPhone mic lost: using MacBook mic"]);
  const macStream = media.streams.at(-1);
  media.devices = [MACBOOK, IPHONE];
  assert.equal(await mic.retry(), true);
  assert.equal(media.opened.at(-1), IPHONE.deviceId);
  assert.equal(mic.label(), IPHONE.label);
  assert.equal(mic.lost(), null);
  assert.deepEqual(banner(mic).lines, []);
  assert.equal(hellos.at(-1).input_device, IPHONE.label);
  assert.equal(hellos.at(-1).mic_switch, `retry: ${MACBOOK.label} -> ${IPHONE.label}`);
  assert.deepEqual(hellos.at(-1).devices_missing, []);
  assert.equal(storage.getItem(MIC_KEY), IPHONE.deviceId);
  // the switched-back track is watched again: a second dropout falls back again
  macStream.track.stop(); // the page (swapMic) stops the old stream
  media.devices = [MACBOOK];
  media.streams.at(-1).track.unplug();
  assert.equal(await mic.busy(), true);
  assert.equal(mic.label(), MACBOOK.label);
});

test("a mic picked by hand in the drawer wins over the preset on that restart", async () => {
  const media = fakeMedia([MACBOOK, IPHONE]);
  const storage = memoryStorage({ [MIC_KEY]: MACBOOK.deviceId });
  const { mic } = judgeLike(media, storage);
  await mic.setExpected(EXPECTED);
  mic.pickedByHand();
  mic.attach(await mic.open());
  assert.deepEqual(media.opened, [MACBOOK.deviceId]);
  assert.equal(await mic.setExpected(EXPECTED), false, "same preset: no switch");
  mic.detach();
  mic.attach(await mic.open()); // the next Start goes back to the preset's mic
  assert.equal(mic.label(), IPHONE.label);
});

test("the preset arriving after Start switches a running session to the expected mic once", async () => {
  const media = fakeMedia([MACBOOK, IPHONE]);
  const { mic, hellos } = judgeLike(media, memoryStorage());
  mic.attach(await mic.open());
  assert.equal(mic.label(), MACBOOK.label);
  assert.equal(await mic.setExpected(EXPECTED), true);
  assert.equal(mic.label(), IPHONE.label);
  assert.match(hellos[0].mic_switch, /^expected: /);
});

test("Stop during a fallback: the new stream is released and nothing is attached", async () => {
  const media = fakeMedia([MACBOOK, IPHONE]);
  const { mic, hellos, stream } = await startOn(media, memoryStorage());
  media.devices = [MACBOOK];
  stream.track.unplug();
  mic.detach(); // Stop pressed while getUserMedia is pending
  assert.equal(await mic.busy(), false);
  assert.equal(media.streams.at(-1).track.readyState, "ended");
  assert.equal(mic.label(), null);
  assert.equal(hellos.length, 0);
});

test("the page cannot use the new stream: no mic open, the banner says so", async () => {
  const media = fakeMedia([MACBOOK, IPHONE]);
  const mic = createMic({ media, storage: memoryStorage(), onSwap: () => false });
  await mic.setExpected(EXPECTED);
  const stream = await mic.open();
  mic.attach(stream);
  media.devices = [MACBOOK];
  stream.track.unplug();
  assert.equal(await mic.busy(), false);
  assert.equal(mic.label(), null);
  assert.deepEqual(banner(mic).lines, ["iPhone mic lost: no microphone open, press Retry"]);
});

test("lost lines: short names on the demo path, full labels otherwise", () => {
  assert.equal(lostLine(IPHONE.label, "MacBook Pro Microphone"), "iPhone mic lost: using MacBook mic");
  assert.equal(lostLine(RODE.label, MACBOOK.label), `Mic ${RODE.label} lost: using MacBook mic`);
  assert.equal(lostLine(IPHONE.label, ""), "iPhone mic lost: using the system default");
  assert.equal(FALLBACK_MIC, "MacBook Pro Microphone");
});
