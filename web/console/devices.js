// Audio device pickers + input level meter (T-121), shared by /judge (inside the D drawer) and /simple.
// Mic and output choices are remembered in localStorage (deviceId), the meter reads the frames the page
// already sends, and a Bluetooth-looking output name shows an info badge (it adds 150-250 ms; T-143: the demo
// output is AirPods over A2DP, so that is expected). A Bluetooth INPUT raises the red banner: picking the AirPods
// mic drops them to HFP (16 kHz, no ANC). With the team preset the page also knows the expected devices.

export const MIC_KEY = "hearing.mic";
export const SINK_KEY = "hearing.sink";
export const MIC_LABEL_KEY = "hearing.mic.label"; // T-133: the saved mic's name, for the "not found" banner
export const SINK_LABEL_KEY = "hearing.sink.label";
// T-145: peak below this: move closer / raise the gain. The built-in MacBook mic hears speech at -50..-35 dBFS;
// with the target at -45 dBFS RMS over a -58 dBFS room (Aryan's and Joseph's takes, eval/enroll_eval.py --team)
// this meter reads p10 -45.2 dBFS while they talk, minus 3 dB. Was -40 (T-121), amber for most normal speech.
export const TOO_QUIET_DB = -48;
export const CLIPPING_DB = -2; // peak above this: the mic is clipping
const METER_FLOOR_DB = -60;
const RELEASE_DB_PER_S = 24;
const BLUETOOTH = /airpods|bluetooth|(^|[^a-z0-9])bt([^a-z0-9]|$)/i;

export const isBluetoothName = (label) => BLUETOOTH.test(label || "");
const BUILTIN_SPEAKERS = /(macbook|imac|built-?in|internal).*speakers?|^(default - )?speakers?$/i; // T-140
export const isBuiltinSpeakerName = (label) => BUILTIN_SPEAKERS.test(label || "");
const fold = (s) => (s || "").normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
const labelHas = (label, want) => fold(label).includes(fold(want));

/**
 * The red-banner lines for the devices in use (T-133, T-140, T-143): saved device not found, laptop speakers
 * (feedback), a Bluetooth mic (HFP), and, when the team preset names them, a mic/output other than the expected one.
 * `missing` holds the saved label of a device that was not found, else null.
 */
export function deviceWarnings({ input = "", output = "", missing = {}, expected = {}, running = true, lost = null } = {}) {
  const lines = [];
  if (lost) lines.push(lostLine(lost.from, lost.to)); // T-287: the mic dropped mid-session; this line names the one in use
  else if (missing.mic) lines.push(`Demo mic not found (${missing.mic}): using ${input || "the system default"}`);
  if (missing.output) lines.push(`Demo output not found (${missing.output}): using ${output || "the system default"}`);
  const speakers = !missing.output && isBuiltinSpeakerName(output);
  if (speakers) lines.push(`Audio plays through the laptop speakers (${output}): the mic will hear it and feed back. Pick headphones.`);
  let wrongMic = false;
  if (running && isBluetoothName(input)) {
    lines.push(`Mic is ${input}: a Bluetooth mic switches the AirPods to HFP (no noise cancelling, phone-call audio). Pick ${expected.input || "MacBook Pro Microphone"}.`);
  } else if (running && !lost && expected.input && input && !missing.mic && !labelHas(input, expected.input)) {
    wrongMic = true;
    lines.push(`Mic is ${input}, the demo mic is ${expected.input}.`);
  }
  const wrongOutput = Boolean(running && expected.output && !missing.output && !speakers && !labelHas(output, expected.output));
  if (wrongOutput) lines.push(`Output is ${output || "the system default"}, the demo output is ${expected.output}: pick it.`);
  return { lines, speakers, hfp: running && isBluetoothName(input), pickOutput: speakers || wrongOutput, wrongMic };
}

// T-287: the demo mic is the target's iPhone over USB ("Aryan patel's iPhone Microphone", --mic / preset pins);
// when it drops mid-session the page reopens this one (else the system default) and keeps the session running.
export const FALLBACK_MIC = "MacBook Pro Microphone";
const micShort = (label) => (/iphone/i.test(label || "") ? "iPhone" : labelHas(label, FALLBACK_MIC) ? "MacBook" : "");

/** The dropout banner line: "iPhone mic lost: using MacBook mic" on the demo path, the full labels otherwise. */
export function lostLine(from, to) {
  const was = micShort(from) ? `${micShort(from)} mic` : `Mic ${from || ""}`.trim();
  if (to == null) return `${was} lost: no microphone open, press Retry`;
  return `${was} lost: using ${micShort(to) ? `${micShort(to)} mic` : to || "the system default"}`;
}

/** Peak level of a frame in dBFS, floored at -90. */
export function peakDb(samples) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i]));
  return Math.max(-90, 20 * Math.log10(peak + 1e-9));
}

/** "quiet" | "good" | "clipping" for a peak level in dBFS. */
export function levelBand(db) {
  if (db >= CLIPPING_DB) return "clipping";
  return db < TOO_QUIET_DB ? "quiet" : "good";
}
const BAND_TEXT = { quiet: "too quiet", good: "good", clipping: "clipping" };

export function loadChoice(key, storage = globalThis.localStorage) {
  try { return storage.getItem(key) || ""; } catch { return ""; }
}

export function saveChoice(key, value, storage = globalThis.localStorage) {
  try { if (value) storage.setItem(key, value); else storage.removeItem(key); } catch { /* private mode */ }
}

/** getUserMedia audio constraints for the raw room; the saved mic is `exact` (T-133), so an unplugged one fails
 * loudly (see `openMic`) instead of silently becoming the laptop mic. */
export function audioConstraints(deviceId) {
  const audio = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false };
  if (deviceId) audio.deviceId = { exact: deviceId };
  return audio;
}

const MISSING_ERRORS = new Set(["OverconstrainedError", "NotFoundError", "NotReadableError"]);

/**
 * getUserMedia with the saved mic exact. When that device is gone, open the default instead and return
 * `{stream, missing: true}` so the page shows the banner; permission errors still throw.
 */
export async function openMic(deviceId, gum = (c) => navigator.mediaDevices.getUserMedia(c)) {
  try {
    return { stream: await gum({ audio: audioConstraints(deviceId) }), missing: false };
  } catch (e) {
    if (!deviceId || !MISSING_ERRORS.has(e?.name)) throw e;
    return { stream: await gum({ audio: audioConstraints("") }), missing: true };
  }
}

const realDevice = (d) => d.deviceId && d.deviceId !== "default" && d.deviceId !== "communications";
const stopStream = (stream) => stream?.getTracks?.().forEach((t) => t.stop());

/**
 * T-287: which mic is open, without the DOM (node tests drive it with a fake `media`). `open()` on Start picks
 * the preset's expected mic by label when it is plugged in (exact deviceId, saved as the choice), else the saved
 * or default mic; labels are only readable after permission, so it enumerates again after getUserMedia and
 * re-opens the expected mic once. When the live track ends or a devicechange removes it, `dropout()` opens
 * FALLBACK_MIC (else the default) and hands the stream to `onSwap(stream, {reason, from, to})`, which feeds the
 * running session from it; `retry()` switches back once the expected (or saved) mic is listed again.
 */
export function createMic({ media, storage = globalThis.localStorage, onSwap = async () => true, onChange = () => {}, onNote = () => {} } = {}) {
  const gum = (c) => media.getUserMedia(c);
  let all = [], track = null, expected = null, lost = null, missing = null;
  let manualNext = false, manual = false, switching = null, session = 0;

  const inputs = () => all.filter((d) => d.kind === "audioinput");
  const findExpected = () => (expected ? inputs().find((d) => realDevice(d) && d.label && labelHas(d.label, expected)) : undefined);
  const findSaved = () => { const id = loadChoice(MIC_KEY, storage); return id ? inputs().find((d) => d.deviceId === id) : undefined; };
  const liveId = () => track?.getSettings?.().deviceId || "";
  const save = (d) => { saveChoice(MIC_KEY, d.deviceId, storage); saveChoice(MIC_LABEL_KEY, d.label, storage); };
  const onEnded = () => { dropout("ended"); };

  async function enumerate() {
    try { all = await media.enumerateDevices(); } catch { all = []; }
    return all;
  }

  function detachTrack() {
    track?.removeEventListener?.("ended", onEnded);
    track = null;
  }

  function attach(stream) {
    detachTrack();
    track = stream.getAudioTracks()[0] || null;
    track?.addEventListener?.("ended", onEnded);
  }

  function detach() {
    detachTrack();
    lost = null; missing = null; session++;
  }

  /** Start: the expected mic if listed, else the saved one exact, else the default (`missing` raises the banner). */
  async function open() {
    const honour = !manualNext;
    manualNext = false; manual = !honour; lost = null; missing = null;
    if (honour && expected) {
      await enumerate();
      const want = findExpected();
      if (want) {
        try {
          const stream = await gum({ audio: audioConstraints(want.deviceId) });
          save(want);
          return stream;
        } catch (e) {
          if (!MISSING_ERRORS.has(e?.name)) throw e; // unplugged between enumerate and open: the saved/default path
        }
      }
    }
    const got = await openMic(loadChoice(MIC_KEY, storage), gum);
    missing = got.missing ? savedName(MIC_LABEL_KEY, MIC_KEY, storage) : null;
    if (!honour || !expected) return got.stream;
    // first Start: no labels before the permission prompt; now they are readable, so take the expected mic once
    await enumerate();
    const want = findExpected();
    const opened = got.stream.getAudioTracks()[0];
    if (!want || labelHas(opened?.label, expected)) return got.stream;
    try {
      const stream = await gum({ audio: audioConstraints(want.deviceId) });
      stopStream(got.stream);
      save(want);
      missing = null;
      onNote(`opened the demo mic ${want.label} (was ${opened?.label || "the default"})`);
      return stream;
    } catch {
      return got.stream; // keep what is open; the banner names the expected mic
    }
  }

  /** Open `device` (or the default when undefined) and hand it to the page; false when it could not be used. */
  async function switchTo(device, reason, from, mine = session) {
    const stream = await gum({ audio: audioConstraints(device?.deviceId || "") });
    if (mine !== session) { stopStream(stream); return false; } // stopped meanwhile
    const prev = track;
    attach(stream);
    const to = track?.label || device?.label || "";
    const ok = await onSwap(stream, { reason, from, to });
    if (ok === false) { // the page could not use it: back to the previous mic if it still runs, else none
      stopStream(stream);
      detachTrack();
      if (prev?.readyState !== "ended") { track = prev; track?.addEventListener?.("ended", onEnded); }
      return false;
    }
    return true;
  }

  function serial(fn) {
    if (switching) return switching;
    switching = fn().finally(() => { switching = null; onChange(); });
    return switching;
  }

  /** The live mic ended (unplugged, iPhone locked): reopen FALLBACK_MIC, else the default, keep the session. */
  function dropout(reason) {
    if (!track) return Promise.resolve(false);
    return serial(async () => {
      const from = lost?.from || track.label, mine = session; // a Stop meanwhile drops the new stream
      track.removeEventListener?.("ended", onEnded);
      lost = { from, to: null }; // set before onSwap, so the page's hello lists the mic as missing
      await enumerate();
      const fallback = inputs().find((d) => realDevice(d) && labelHas(d.label, FALLBACK_MIC));
      let ok = false;
      for (const device of fallback ? [fallback, undefined] : [undefined]) {
        try { ok = await switchTo(device, `lost (${reason})`, from, mine); } catch (e) { onNote(`mic fallback ${device?.label || "default"}: ${e?.message || e}`); }
        if (ok || mine !== session) break;
      }
      if (mine !== session) return false; // stopped: detach() already cleared the state
      lost = { from, to: ok ? track?.label || "" : null };
      onNote(`mic lost (${from}, ${reason}): ${ok ? `using ${lost.to}` : "no microphone could be opened"}`);
      return ok;
    });
  }

  /** The banner's Retry: back to the expected (else saved) mic when it is listed again. */
  function retry() {
    return serial(async () => {
      await enumerate();
      const want = findExpected() || findSaved();
      if (!want) { onNote(`retry: ${expected || "the saved mic"} is still not listed`); return false; }
      const from = track?.label || lost?.from || "";
      const was = { lost, missing }, mine = session;
      lost = null; missing = null; // before onSwap, so the page's hello no longer lists the mic as missing
      let ok = false;
      try { ok = await switchTo(want, "retry", from, mine); } catch (e) { onNote(`retry ${want.label}: ${e?.message || e}`); }
      if (ok) { manual = false; save(want); } else if (mine === session) ({ lost, missing } = was);
      return ok;
    });
  }

  /** A devicechange: the live mic is gone when its track ended or its deviceId is no longer listed. */
  async function deviceChange() {
    await enumerate();
    if (!track || switching) return false;
    const id = liveId();
    const gone = track.readyState === "ended" || (id && id !== "default" && !inputs().some((d) => d.deviceId === id));
    return gone ? dropout("removed") : false;
  }

  return {
    open, attach, detach, enumerate, dropout, retry, deviceChange,
    /** The user picked a mic in the drawer: the next open() uses it, not the preset's. */
    pickedByHand() { manualNext = true; },
    /** The preset's expected mic (label substring); while running, switch to it once if it is listed. */
    async setExpected(input) {
      if (input === expected) return false;
      expected = input;
      if (!track || manual || lost || switching || !expected || labelHas(track.label, expected)) return false;
      await enumerate();
      const want = findExpected();
      return want ? serial(async () => {
        const ok = await switchTo(want, "expected", track?.label || "").catch(() => false);
        if (ok) { save(want); missing = null; }
        return ok;
      }) : false;
    },
    devices: () => all,
    liveId,
    label: () => track?.label || null,
    missing: () => missing,
    lost: () => lost,
    busy: () => switching,
  };
}

const MARKUP = `
  <label>Microphone <select data-dev="in" aria-label="Microphone"><option value="">System default</option></select></label>
  <label>Output <select data-dev="out" aria-label="Output device"><option value="">System default</option></select></label>
  <p class="dev-meter" data-dev="meter" data-band="idle">
    <span class="dev-bar" role="meter" aria-label="Input level" aria-valuemin="-60" aria-valuemax="0" aria-valuenow="-60"><i></i></span>
    <output data-dev="db">— dBFS</output> <b data-dev="band">mic off</b>
  </p>
  <p class="dev-missing" data-dev="missing-note" hidden></p>
  <p class="dev-bt" data-dev="bt" hidden>Bluetooth output (the demo AirPods): about 150–250&nbsp;ms of delay. Keep the mic on the laptop.</p>`;

/**
 * Fill `root` with the pickers, meter and badge. `onMicChange()` runs when the user picks another mic
 * (the page restarts capture if it is running). Returns the handle the page calls from its media code.
 */
export function mountDevices(root, { onMicChange = () => {}, onNote = () => {}, onSwap = null } = {}) {
  root.classList.add("devices");
  root.innerHTML = MARKUP;
  const q = (name) => root.querySelector(`[data-dev="${name}"]`);
  const selIn = q("in"), selOut = q("out");
  const hasSink = typeof globalThis.AudioContext?.prototype?.setSinkId === "function";
  let ctx = null, inputs = [], outputs = [], level = METER_FLOOR_DB, lastFrame = 0;
  let missing = { output: null }; // saved label of an output that was not found, else null (the mic's: mic.missing())
  let expected = { input: null, output: null }; // T-143: the team preset's demo devices (telemetry preset.pins)
  // T-287: which mic is open (expected mic on Start, MacBook fallback on dropout, Retry back); refresh() repaints
  // without onSwap (/simple) a dropout only raises the banner and Retry restarts capture, as before
  const mic = createMic({ media: navigator.mediaDevices, onSwap: onSwap || (() => false), onNote, onChange: () => refresh() });
  const live = () => Boolean(mic.label());
  selOut.disabled = !hasSink;
  let wrongMic = false; // a mic other than the preset's is open (its Retry switches, no restart)
  const banner = makeBanner(() => {
    if (onSwap && (mic.lost() || wrongMic)) { mic.retry(); return; } // T-287: switch the running session back, no restart
    showBanner(banner, false); onMicChange();
  });
  if (!hasSink) selOut.title = "This browser cannot pick the output device";

  const labelOf = (list, id, fallback) => list.find((d) => d.deviceId === id)?.label || fallback;
  function fill(select, list, kind, saved) {
    select.replaceChildren(new Option("System default", ""));
    list.forEach((d, i) => {
      if (d.deviceId && d.deviceId !== "default") select.append(new Option(d.label || `${kind} ${i + 1}`, d.deviceId));
    });
    const known = [...select.options].some((o) => o.value === saved);
    select.value = known ? saved : "";
    return known;
  }

  function showMissing() {
    const warn = deviceWarnings({ input: mic.label() || "", output: outputLabel(), missing: { mic: mic.missing(), output: missing.output },
      expected, running: live(), lost: mic.lost() });
    const { lines, speakers, hfp, pickOutput } = warn;
    wrongMic = warn.wrongMic;
    root.dataset.speakers = String(speakers);
    root.dataset.hfp = String(hfp);
    root.dataset.micLost = String(Boolean(mic.lost()));
    const bannerText = banner.querySelector("span");
    const text = lines.join(" · ");
    if (bannerText.textContent !== text) bannerText.textContent = text; // only a change is re-announced
    banner.querySelector("button").hidden = !(mic.missing() || missing.output || mic.lost() || wrongMic);
    const pick = banner.querySelector("select");
    pick.hidden = !pickOutput;
    if (pickOutput) {
      pick.replaceChildren(new Option("Choose output…", ""));
      outputs.forEach((d) => {
        if (d.deviceId && d.deviceId !== "default" && !isBuiltinSpeakerName(d.label)) pick.append(new Option(d.label || d.deviceId, d.deviceId));
      });
      pick.onchange = () => { if (pick.value) { selOut.value = pick.value; selOut.dispatchEvent(new Event("change")); } };
    }
    showBanner(banner, lines.length > 0);
    q("missing-note").textContent = lines.join(" · ");
    q("missing-note").hidden = !lines.length;
    root.dataset.missing = missingKinds().join(",");
  }

  async function refresh() {
    const all = await mic.enumerate(); // [] without permission or API
    inputs = all.filter((d) => d.kind === "audioinput");
    outputs = all.filter((d) => d.kind === "audiooutput");
    // T-287: while running the select shows the mic actually in use (the fallback too), else the saved choice
    fill(selIn, inputs, "Microphone", live() ? mic.liveId() : loadChoice(MIC_KEY));
    const sinkKnown = fill(selOut, outputs, "Speaker", loadChoice(SINK_KEY));
    // labels (and output ids) are only trustworthy once capture runs; a saved output that is gone is loud too
    if (live() && hasSink) missing.output = loadChoice(SINK_KEY) && !sinkKnown ? savedName(SINK_LABEL_KEY, SINK_KEY) : null;
    showBluetooth();
    showMissing();
  }

  function outputLabel() {
    if (selOut.value) return selOut.selectedOptions[0]?.textContent || "";
    return outputs.find((d) => d.deviceId === "default")?.label || ""; // Chrome: "Default - AirPods Pro"
  }

  function showBluetooth() {
    const bt = isBluetoothName(outputLabel());
    q("bt").hidden = !bt;
    root.dataset.bluetooth = String(bt);
  }

  async function applySink() {
    if (!ctx || !hasSink) return;
    try {
      await ctx.setSinkId(selOut.value || "");
    } catch (e) {
      onNote(`output device: ${e?.message || e}`);
      if (selOut.value) { missing.output = savedName(SINK_LABEL_KEY, SINK_KEY); showMissing(); }
    }
  }

  const chosenText = (select) => (select.value ? select.selectedOptions[0]?.textContent || "" : "");
  selIn.addEventListener("change", () => {
    saveChoice(MIC_KEY, selIn.value); saveChoice(MIC_LABEL_KEY, chosenText(selIn)); mic.pickedByHand(); onMicChange();
  });
  selOut.addEventListener("change", () => {
    saveChoice(SINK_KEY, selOut.value); saveChoice(SINK_LABEL_KEY, chosenText(selOut));
    missing.output = null; showBluetooth(); showMissing(); applySink();
  });
  navigator.mediaDevices?.addEventListener?.("devicechange", async () => { await mic.deviceChange(); refresh(); });
  const missingKinds = () => [(mic.missing() || mic.lost()) && "mic", missing.output && "output"].filter(Boolean);

  // Perf: the meter paints only while it can be seen (not inside a closed <details> drawer or a hidden ancestor),
  // moves its bar with a transform (compositor only, no layout) and writes each field only when it changed.
  const meterEl = q("meter"), bar = meterEl.querySelector("i"), meterRole = meterEl.querySelector("[role=meter]");
  const dbEl = q("db"), bandEl = q("band");
  const shown = { band: "", pct: "", db: "", text: "" };
  const visible = () => root.isConnected && !root.closest?.("[hidden]") && (root.closest?.("details")?.open ?? true);
  function paint(now) {
    const dt = lastFrame ? Math.min(0.5, (now - lastFrame) / 1000) : 0;
    lastFrame = now;
    level = Math.max(METER_FLOOR_DB, level - RELEASE_DB_PER_S * dt);
    if (!live() || !visible()) return;
    const band = levelBand(level);
    const pct = Math.max(0, Math.min(1, (level - METER_FLOOR_DB) / -METER_FLOOR_DB)).toFixed(2);
    const db = level.toFixed(0);
    if (shown.band !== band) { shown.band = band; meterEl.dataset.band = band; }
    if (shown.pct !== pct) { shown.pct = pct; bar.style.transform = `scaleX(${pct})`; }
    if (shown.db !== db) {
      shown.db = db;
      meterRole.setAttribute("aria-valuenow", db);
      dbEl.textContent = `${level <= METER_FLOOR_DB ? "≤ -60" : db} dBFS`;
    }
    if (shown.text !== BAND_TEXT[band]) { shown.text = BAND_TEXT[band]; bandEl.textContent = BAND_TEXT[band]; }
  }
  setInterval(() => paint(performance.now()), 50);

  refresh();
  return {
    refresh,
    inputId: () => selIn.value,
    outputId: () => selOut.value,
    /** Constraints for the Start tap's getUserMedia (the saved mic, exact, even while its option is not listed yet). */
    constraints: () => audioConstraints(loadChoice(MIC_KEY)),
    /** T-287: the preset's expected mic when it is plugged in (saved as the choice), else the saved mic exactly;
     * when that is gone, the default with the banner (never silently). */
    openMic: () => mic.open(),
    /** Saved devices that were not found or dropped ("mic", "output"), for the hello and telemetry. */
    missing: missingKinds,
    /** The mic actually in use (the stream's track label), or null while stopped. */
    inputLabel: () => mic.label(),
    outputLabel: () => (live() ? outputLabel() || null : null),
    /** T-287: the banner's Retry (switch back to the expected mic); tests and the page call it too. */
    retryMic: () => mic.retry(),
    /** T-143: the team preset's demo devices (label substrings); a different mic or output raises the banner. */
    setExpected({ input = null, output = null } = {}) {
      if (input === expected.input && output === expected.output) return;
      expected = { input, output };
      mic.setExpected(input); // T-287: a running session switches to it once if it is plugged in
      showMissing();
    },
    /** After getUserMedia + AudioContext exist: labels are readable now, route the output. */
    async started(stream, audioCtx) {
      ctx = audioCtx;
      mic.attach(stream);
      level = METER_FLOOR_DB;
      await refresh();
      await applySink();
      showMissing();
    },
    stopped() {
      ctx = null; mic.detach();
      missing = { output: null };
      showMissing();
      const meter = q("meter");
      meter.dataset.band = "idle";
      meter.querySelector("i").style.width = "0%";
      q("db").textContent = "— dBFS";
      q("band").textContent = "mic off";
    },
    /** Feed one captured frame (Float32Array). */
    feed(samples) {
      const db = peakDb(samples);
      if (db > level) level = db;
    },
  };
}

function savedName(labelKey, idKey, storage = globalThis.localStorage) {
  return loadChoice(labelKey, storage) || `saved device ${loadChoice(idKey, storage).slice(0, 8)}`;
}

/** The page-level banner (T-133): outside the D drawer so it is seen, with a one-click retry. */
function makeBanner(onRetry) {
  let el = document.querySelector("[data-dev-banner]");
  if (!el) {
    el = document.createElement("div");
    el.dataset.devBanner = "";
    el.setAttribute("role", "alert");
    el.style.cssText = "display:none;position:fixed;top:0;left:0;right:0;z-index:1000;padding:10px 14px;background:#b00020;"
      + "color:#fff;font:600 15px/1.3 system-ui,sans-serif;gap:12px;align-items:center;justify-content:center";
    el.innerHTML = '<span></span><select data-dev-pick aria-label="Output device" hidden></select><button type="button" data-dev-retry style="font:inherit;padding:4px 12px">Retry</button>';
    document.body.append(el);
  }
  el.querySelector("button").onclick = onRetry;
  return el;
}

function showBanner(el, on) {
  el.style.display = on ? "flex" : "none";
}
