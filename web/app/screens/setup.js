// Setup screen (M14 task 5, T-348): the mic and output pickers, the live input meter, the device warnings and
// the four checks a stage run needs (secure context, audio worklet, engine, backend).
//
// Honesty: the meter reads only the store's "level" events (the peak of each frame session.js actually sends,
// 24 dB/s release); with no frames it says so instead of resting at a made-up level. The pickers are the real
// web/console/devices.js element owned by the session (it remembers `hearing.mic` / `hearing.sink`), restyled
// here. Warnings are `deviceWarnings` from the same module, checks come from the browser, `/health` and telemetry.

import { createSession } from "../session.js";
import { isStageRig, showDevFacts, touchDevice } from "../platform.js";
import { SETTLING, expectedDelayCopy, fmtDelay } from "../store.js";
import {
  CLIPPING_DB, MIC_LABEL_KEY, MIC_KEY, SINK_KEY, SINK_LABEL_KEY, TOO_QUIET_DB, deviceWarnings, isBluetoothName, loadChoice,
} from "../../console/devices.js";

export const title = "Setup";

export const METER_MIN_DB = -60;
export const STALL_MS = 600; // no level event for this long while started: "no frames"
const HEALTH_EVERY_MS = 5000;
const BAND_WORD = { idle: "Mic off", quiet: "Too quiet", good: "Good", clipping: "Too loud", stalled: "No sound" };
// plain words on screen; the dBFS thresholds and frame facts sit in the closed Details disclosure
const BAND_NOTE = {
  idle: "Press Start mic. Nothing is measured until the mic is on.",
  quiet: "Move the mic closer or turn its volume up.",
  good: "Your voice comes through clearly.",
  clipping: "Turn the mic volume down a little.",
  stalled: "The mic is on but no sound comes through. Choose the mic again or press Start mic.",
};
const MIC_PLAIN = { idle: "off", quiet: "too quiet", good: "ok", clipping: "too loud", stalled: "no sound" };

/**
 * The plain lines Setup shows by default: Microphone, Sound output, and Ready. `band` is meterBand's, `output` the
 * output device's name (devices.js label), `checks` setupChecks'. Ready needs a good mic level and every check ok.
 */
export function setupPlain({ band, output, checks }) {
  const fail = checks.some((c) => c.status === "fail");
  const ready = band === "good" && checks.every((c) => c.status === "ok");
  return [
    { id: "mic", label: "Microphone", value: MIC_PLAIN[band] || "off", status: band === "good" ? "ok" : band === "idle" ? "wait" : "warn" },
    { id: "output", label: "Sound output", value: output || "this device", status: "ok" },
    { id: "ready", label: ready ? "Ready" : fail ? "Not ready: something needs fixing (see Details)" : "Not ready yet", value: "",
      status: ready ? "ok" : fail ? "fail" : "wait" },
  ];
}

/** The app's one session, shared by every screen (Listen, Setup, People): created and connected on first use. */
export function sharedSession(app) {
  if (!app.session) {
    app.session = createSession();
    app.session.connect();
  }
  return app.session;
}

/** 0..1 position of a dBFS value on the meter scale (-60..0). */
export function meterPos(db) {
  return Math.max(0, Math.min(1, (db - METER_MIN_DB) / -METER_MIN_DB));
}

/** The meter's band: idle while the mic is off, "stalled" when started but frames stopped, else the level band. */
export function meterBand({ started, band, sinceLevelMs }) {
  if (!started) return "idle";
  if (!(sinceLevelMs <= STALL_MS)) return "stalled";
  return band === "quiet" || band === "clipping" ? band : "good";
}

/**
 * The warning lines for the devices in use, from devices.js `deviceWarnings`: HFP (a Bluetooth mic), the laptop
 * speakers, a saved device not found, and a mic/output other than the team preset's. `kind` tags each line.
 */
export function setupWarnings(mic, telemetry, started, storage = globalThis.localStorage) {
  const missing = mic?.missing || [];
  const pins = telemetry?.preset?.pins || {};
  const w = deviceWarnings({
    input: mic?.label || "",
    output: mic?.output || "",
    missing: {
      mic: missing.includes("mic") ? loadChoice(MIC_LABEL_KEY, storage) || "the saved mic" : null,
      output: missing.includes("output") ? loadChoice(SINK_LABEL_KEY, storage) || "the saved output" : null,
    },
    expected: { input: pins.input_device || null, output: pins.output_device || null },
    running: Boolean(started),
  });
  const kindOf = (line) => (/HFP/.test(line) ? "hfp" : /speakers/.test(line) ? "speakers" : /not found/.test(line) ? "missing"
    : /^Output is/.test(line) ? "output" : /^Mic is/.test(line) ? "mic" : "other");
  return {
    lines: w.lines.map((text) => ({ kind: kindOf(text), text })),
    hfp: w.hfp,
    wrongOutput: w.lines.some((l) => kindOf(l) === "output"),
    bluetoothOut: Boolean(started) && isBluetoothName(mic?.output || ""),
  };
}

/**
 * The four checks, each {id, label, status: "ok"|"warn"|"fail"|"wait", value, detail}. Inputs are all measured:
 * `secure` (window.isSecureContext), `worklet` {supported, reachable, running}, `health` (/health JSON, or
 * {error}), `connection` (store), `truth` (store.truthStrip).
 */
export function setupChecks({ secure, worklet, health, connection, truth }) {
  const checks = [];
  checks.push(secure
    ? { id: "secure", label: "Secure context", status: "ok", value: "Yes", detail: "The browser may open the mic (https or localhost)." }
    : { id: "secure", label: "Secure context", status: "fail", value: "No",
      detail: "The mic is blocked on plain http. Open the https address demo.sh --lan prints, or localhost." });

  let w;
  if (!worklet.supported) w = { status: "fail", value: "Missing", detail: "This browser has no AudioWorklet (iOS needs 14.5 or later)." };
  else if (worklet.running) w = { status: "ok", value: "Running", detail: "Loaded; frames go to the engine every 20 ms." };
  else if (worklet.reachable === true) w = { status: "ok", value: "Ready", detail: "Supported and served; it loads when the mic starts." };
  else if (worklet.reachable === false) w = { status: "fail", value: "Not served", detail: "/console/audio-worklet.js did not load from this server." };
  else w = { status: "wait", value: "Checking", detail: "Fetching /console/audio-worklet.js." };
  checks.push({ id: "worklet", label: "Audio worklet", ...w });

  let e;
  if (!health) e = { status: "wait", value: "Checking", detail: "Asking /health." };
  else if (health.error) e = { status: "fail", value: "Offline", detail: "Not reachable." };
  else {
    const warm = health.warm || "off";
    const socket = connection === "open" ? "session open" : connection === "reconnecting" ? "session reconnecting" : "session connecting";
    const status = connection !== "open" || warm === "warming" ? "warn" : "ok";
    const models = { ready: "models loaded", warming: "models loading", off: "models load on first use" }[warm] || `warm-up ${warm}`;
    e = { status, value: warm === "warming" ? "Warming" : connection === "open" ? "Connected" : "Reachable",
      detail: [health.version ? `version ${health.version}` : null, models, socket].filter(Boolean).join(", ") };
  }
  checks.push({ id: "engine", label: "Engine", ...e });

  let b;
  if (!health || health.error) b = { status: health?.error ? "fail" : "wait", value: "—", detail: "Needs the engine." };
  else {
    const device = health.device ? String(health.device) : ""; // unknown: no "?" on screen
    const where = device === "mps" ? "Mac (Apple GPU)" : device === "cuda" ? "CUDA GPU" : device === "cpu" ? "CPU" : device;
    const plays = truth?.backend; // what plays now, from telemetry hq_active; null before the first telemetry
    // T-463: the GPU's rev against this checkout only matters while the GPU plays; on the Mac (the decided stage
    // backend) it is a Diagnostics fact, not a stage warning
    const rev = plays === "GPU" ? health.remote_rev_warning || truth?.revWarning : null;
    let detail;
    if (plays === "GPU") detail = `Focus plays on the GPU${truth.gpuRttMs != null ? `, network round trip ${truth.gpuRttMs} ms` : ""}`;
    else if (plays === "Mac") detail = truth.fallback ? `The GPU is not answering, so ${where || "the laptop"} plays` : `Focus plays on this ${where || "laptop"}`;
    else detail = where ? `Engine on ${where}; waiting for it to say what plays` : "Waiting for the engine to say what plays";
    if (rev) detail += `. The GPU runs older code than this laptop (details in Diagnostics)`;
    b = { status: rev || truth?.fallback ? "warn" : plays ? "ok" : "wait", value: plays || (device ? device.toUpperCase() : "—"), detail: `${detail}.` };
  }
  checks.push({ id: "backend", label: "Backend", ...b });
  return checks;
}

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

const TICKS = [-60, -48, -36, -24, -12, -2];

/** The lede under the title: the laptop's stage setup, or what a phone visitor does (T-458). */
export function setupLede(touch) {
  return touch ? "Headphones on. Speak normally and watch the level." : "Headphones on, mic on the laptop. Speak normally and watch the level.";
}

/**
 * Setup's engine-delay line (T-463): the store's one rule and format (store.js delayText): "settling…" until the
 * EMA has held for 2 s, then the same "1.5 s" Listen and Diagnostics show. `steady` overrides truth.delaySteady.
 */
export function delayNowCopy(truth, steady = truth?.delaySteady) {
  const measured = truth?.delayMs;
  if (measured == null) return "The engine's delay shows here once sound flows.";
  if (steady !== "steady") return `Engine delay now: ${SETTLING} (the number shows once it holds for 2 s).`;
  return `Engine delay now: ${fmtDelay(measured)}${truth.latencyPreset ? ` (${truth.latencyPreset})` : ""}, before device buffers.`;
}

export function mount(root, { app, signal }) {
  const session = sharedSession(app);
  const store = session.store;
  // T-458: a visitor's phone gets phone copy; the stage checklist (engine revs, gpu_up) only on the laptop or ?dev=1
  const touch = touchDevice();
  const devFacts = showDevFacts({ touch });
  const stageRig = isStageRig({ touch });

  // ---- meter (the one moving thing on this screen; only real frames move it)
  const word = h("p", { class: "su-word" });
  const reading = h("p", { class: "su-reading" }, h("span", { class: "su-db" }), h("span", { class: "su-unit", text: "dBFS peak" }));
  const fill = h("i", { class: "su-fill" });
  const scale = h("div", { class: "su-scale", role: "meter", "aria-label": "Input level", "aria-valuemin": "-60", "aria-valuemax": "0",
    "aria-valuenow": "-60", style: `--quiet:${meterPos(TOO_QUIET_DB)};--clip:${meterPos(CLIPPING_DB)}` },
  h("span", { class: "su-zone su-zone-quiet" }), h("span", { class: "su-zone su-zone-good" }), h("span", { class: "su-zone su-zone-clip" }),
  fill,
  h("ol", { class: "su-ticks", "aria-hidden": "true" },
    ...TICKS.map((db) => h("li", { style: `--x:${meterPos(db)}`, text: String(db) }))));
  const bandNote = h("p", { class: "su-band-note" });
  const micBtn = h("button", { type: "button", class: "su-mic-btn" });
  const micBanner = h("p", { class: "su-mic-problem", role: "alert", hidden: true });

  // ---- pickers: the session's real devices.js element (remembers hearing.mic / hearing.sink)
  const pickers = h("div", { class: "su-pickers" });
  if (session.devicesElement) pickers.append(session.devicesElement);
  else pickers.append(h("p", { class: "su-muted", text: "Device pickers are not available in this browser." }));
  const saved = h("p", { class: "su-saved" });
  const warnList = h("ul", { class: "su-warnings", "aria-live": "polite" });

  // ---- plain status lines (visible), then the checks and numbers in a closed Details disclosure
  const plainList = h("ul", { class: "su-plain" });
  const checkList = h("ol", { class: "su-checks" });
  const delay = h("p", { class: "su-delay" });
  const delayNow = h("p", { class: "su-delay-now" });
  const details = h("details", { class: "disclose su-details" },
    h("summary", { text: "Details" }),
    h("div", { class: "su-details-body" },
      reading,
      h("p", { class: "su-thresholds", text: `Too quiet under ${TOO_QUIET_DB} dBFS peak; too loud (clipping) at ${CLIPPING_DB} dBFS or above. `
        + `No sound means no audio frames reached the engine for ${STALL_MS} ms.` }),
      devFacts ? checkList : null,
      delayNow));

  root.append(h("section", { class: "setup", "aria-labelledby": "setup-title" },
    h("header", { class: "su-head" },
      h("h1", { id: "setup-title", class: "su-title", text: "Set up the sound" }),
      h("p", { class: "su-lede", text: setupLede(touch) })),
    h("div", { class: "su-grid" },
      h("section", { class: "su-meter", "aria-label": "Microphone level" }, word, scale, bandNote,
        h("div", { class: "su-mic-row" }, micBtn, micBanner)),
      h("section", { class: "su-devices", "aria-labelledby": "su-devices-title" },
        h("h2", { id: "su-devices-title", class: "su-h2", text: "Microphone and output" }), pickers, saved, warnList),
      h("section", { class: "su-check-panel", "aria-labelledby": "su-checks-title" },
        h("h2", { id: "su-checks-title", class: "su-h2", text: "Status" }), plainList, delay, details)),
    h("footer", { class: "su-foot" },
      h("a", { href: "#listen", class: "su-next" }, h("span", { text: "Start listening" })))));

  // ---- live state
  let level = { db: METER_MIN_DB, band: "idle", at: 0 };
  let health = null;
  let worklet = { supported: typeof globalThis.AudioWorkletNode === "function", reachable: null };
  let raf = 0;
  let warnSig = "";
  let checkSig = "";
  let plainSig = "";
  let shownBand = "idle";
  let shownChecks = [];
  const now = () => performance.now();

  // the plain lines change only when the band, the output or a check status does (not every meter frame)
  function paintPlain() {
    const lines = setupPlain({ band: shownBand, output: store.state.mic?.output || "", checks: shownChecks });
    const sig = JSON.stringify(lines);
    if (sig === plainSig) return;
    plainSig = sig;
    plainList.replaceChildren(...lines.map((l) => h("li", { class: "su-plain-row", "data-status": l.status, "data-line": l.id },
      h("span", { class: "su-plain-label", text: l.value ? `${l.label}:` : l.label }),
      l.value ? h("span", { class: "su-plain-value", text: l.value }) : null)));
  }

  function paintMeter() {
    raf = 0;
    const s = store.state;
    const band = meterBand({ started: s.started && !s.replaying, band: level.band, sinceLevelMs: level.at ? now() - level.at : Infinity });
    const live = band !== "idle" && band !== "stalled";
    const db = live ? level.db : null;
    root.querySelector(".setup").dataset.band = band;
    const wordText = s.replaying ? "Replay" : BAND_WORD[band];
    if (word.textContent !== wordText) word.textContent = wordText;
    reading.querySelector(".su-db").textContent = db == null ? "—" : db <= METER_MIN_DB ? "≤ −60" : db.toFixed(0).replace("-", "−");
    fill.style.transform = `scaleX(${db == null ? 0 : meterPos(db).toFixed(4)})`;
    scale.setAttribute("aria-valuenow", db == null ? "-60" : db.toFixed(0));
    scale.setAttribute("aria-valuetext", db == null ? BAND_WORD[band] : `${db.toFixed(0)} dBFS, ${BAND_WORD[band]}`);
    const note = s.replaying ? "A recording is playing; this meter shows the recording, not your mic." : BAND_NOTE[band];
    if (bandNote.textContent !== note) bandNote.textContent = note;
    if (band !== shownBand) { shownBand = band; paintPlain(); }
  }
  const schedule = () => { if (!raf) raf = requestAnimationFrame(paintMeter); };

  function paintMic() {
    const s = store.state;
    micBtn.textContent = s.starting ? "Starting mic…" : s.started && !s.replaying ? "Stop mic" : "Start mic";
    micBtn.disabled = s.starting;
    micBtn.dataset.on = String(s.started && !s.replaying);
    micBanner.textContent = s.mic.banner || "";
    micBanner.hidden = !s.mic.banner;
    const w = setupWarnings(s.mic, s.telemetry, s.started && !s.replaying);
    root.querySelector(".setup").dataset.hfp = String(w.hfp);
    const items = w.lines.map((l) => h("li", { class: "su-warn", "data-kind": l.kind }, h("span", { class: "su-warn-tag", text:
      { hfp: "Bluetooth mic", speakers: "Echo", missing: "Missing", output: "Wrong output", mic: "Wrong mic", other: "Check" }[l.kind] }),
    h("span", { text: l.text })));
    if (w.bluetoothOut && !w.lines.length && stageRig) {
      items.push(h("li", { class: "su-warn", "data-kind": "info" }, h("span", { class: "su-warn-tag", text: "Bluetooth" }),
        h("span", { text: `Output is ${s.mic.output}: about 150–250 ms more delay. Expected for the demo AirPods; keep the mic on the laptop.` })));
    }
    const sig = JSON.stringify(w);
    if (sig !== warnSig) { warnSig = sig; warnList.replaceChildren(...items); }
    warnList.hidden = !items.length;
    const mic = loadChoice(MIC_LABEL_KEY) || (loadChoice(MIC_KEY) ? "a saved device" : "");
    const out = loadChoice(SINK_LABEL_KEY) || (loadChoice(SINK_KEY) ? "a saved device" : "");
    saved.textContent = mic || out ? `Remembered on this browser: mic ${mic || "system default"}, output ${out || "system default"}.`
      : "Your choices are remembered on this browser.";
    paintPlain();
  }

  function paintChecks() {
    const s = store.state;
    const checks = setupChecks({ secure: globalThis.isSecureContext === true, worklet: { ...worklet, running: s.started && !s.replaying },
      health, connection: s.connection, truth: s.truth });
    const sig = JSON.stringify(checks);
    if (sig !== checkSig) {
      checkSig = sig;
      checkList.replaceChildren(...checks.map((c) => h("li", { class: "su-check", "data-status": c.status, "data-check": c.id },
        h("span", { class: "su-check-mark", "aria-hidden": "true" }),
        h("span", { class: "su-check-label", text: c.label }),
        h("span", { class: "su-check-value", text: c.value }),
        h("span", { class: "su-check-detail", text: c.detail }))));
    }
    root.querySelector(".setup").dataset.ready = String(checks.every((c) => c.status === "ok"));
    shownChecks = checks;
    paintPlain();
    // T-463: the expected delay is the server's (/health tse_window, T-419), not a fixed "1.5 s"
    delay.replaceChildren(h("span", { class: "su-delay-big", text: `${expectedDelayCopy(health && !health.error ? health : null)}, like live captions.` }));
    delayNow.textContent = delayNowCopy(s.truth);
  }

  async function poll() {
    try {
      const res = await fetch("/health", { cache: "no-store", signal });
      health = res.ok ? await res.json() : { error: `HTTP ${res.status}` };
    } catch (err) {
      if (signal.aborted) return;
      health = { error: err?.message || String(err) };
    }
    paintChecks();
  }
  async function probeWorklet() {
    if (!worklet.supported) return paintChecks();
    try {
      const res = await fetch("/console/audio-worklet.js", { cache: "no-store", signal });
      worklet = { ...worklet, reachable: res.ok };
    } catch {
      if (signal.aborted) return;
      worklet = { ...worklet, reachable: false };
    }
    paintChecks();
  }

  micBtn.addEventListener("click", () => {
    // inside the click: session.start() creates the AudioContext synchronously (iOS wants the gesture)
    const p = store.state.started && !store.state.replaying ? (session.stop(), null) : session.start();
    paintMic();
    p?.catch(() => false).then(() => { paintMic(); paintChecks(); }); // a rejected start (a replay that would not stop) is no unhandled rejection
  }, { signal });

  const offs = [
    store.on("level", (lv) => { level = { db: lv.db, band: lv.band, at: now() }; schedule(); }),
    store.on("change", ({ keys }) => {
      if (keys.some((k) => k === "mic" || k === "started" || k === "starting" || k === "replaying" || k === "telemetry")) paintMic();
      if (keys.some((k) => k === "connection" || k === "truth" || k === "started" || k === "replaying")) paintChecks();
      schedule();
    }),
  ];
  const stallTimer = setInterval(schedule, 250); // the "no frames" state needs time to pass without events
  const healthTimer = setInterval(poll, HEALTH_EVERY_MS);
  session.devices?.refresh?.();

  paintMeter();
  paintMic();
  paintChecks();
  poll();
  probeWorklet();

  return () => {
    offs.forEach((off) => off());
    clearInterval(stallTimer);
    clearInterval(healthTimer);
    cancelAnimationFrame(raf);
    session.devicesElement?.remove(); // the session keeps the element (and its choices) for the next visit
  };
}
