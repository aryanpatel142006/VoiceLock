import { BackgroundWatch, frameClock } from "./frame-clock.js";
import { loudestMode, renderGate } from "./gate-state.js";
import { autoFocusState } from "./auto-focus.js";
import { showServerInfo } from "./server-info.js";
import { audioOnly, helloVideo } from "./audio-only.js";
import { replayControls } from "./replay.js";
import { mountDevices } from "./devices.js";
import {
  MIC_START_TIMEOUT_MS, PillHysteresis, Steady, enrolledHealthy, judgePill, micPill, micProblem, withTimeout,
} from "./judge-pill.js";
import { DEGRADED_NOTICE, captionFeed, modeControl, modeForKey } from "./judge-modes.js";
import {
  KeyFocus, RELEASE_CONTROL, SEARCHING_PILL, abControl, targetLine,
} from "./judge-targets.js";
import { mountNameGuards, nameCallControl, nameCallPill, saveName, savedName } from "./name-call.js";
import {
  KIND_AUDIO_IN, KIND_AUDIO_OUT, KIND_VIDEO_IN, abDelayMs, audioContextOptions, control, floatToPcm16, hello,
  hqBackendInfo, packFrame, pcm16ToFloat, sessionUrl, tierInfo, unpackFrame,
} from "./protocol.js";

// Judge demo page (/judge, T-088): a huge RAW | VOICELOCK switch, the plain-language pill and
// live captions; diagnostics live in a small drawer. One control ({profile: "judge"}) locks the session
// defaults on the server: Focus HQ, suppression 100, strict isolation, sticky attention, auto calibration.
// Keys: Space = A/B, D = drawer; Shift+C = calibrate (backup for the automatic one), Shift+R = replay clip,
// Shift+N = Name Call, Shift+F / Shift+Q = Focus / Quiet (T-150), Shift+0 = Release (T-177) (T-135: these
// alone only while the drawer is open, so a stray key never fires); 1/2 = saved person.
// T-177: 1/2/Release/Space are buttons too, and a banner says when the keys cannot reach the page.
// Fallbacks are one line in the drawer, never a dialog. Capture/playback code mirrors simple.js.

const FRAME_MS = 20;
const VIDEO_FPS = 25;
const VIDEO_W = 640;
const VIDEO_H = 360;
const VIDEO_BACKPRESSURE_BYTES = 256 * 1024;
const BG_MIN_FPS = 5;  // T-138: below this while hidden the page says the camera is paused
const RETRY_MS = [500, 1000, 2000, 4000];
const RENDER_MS = 100;
const MODE_GRACE_MS = 1500; // the phone companion can switch Focus/Quiet; adopt the engine's mode after our own settles
const CALIBRATE_S = 3;
// T-135: playback jitter buffer per mode. Focus HQ already plays ~0.6-1.5 s behind, so 120 ms more
// buffer costs nothing audible and absorbs the GPU/ladder bursts that underran at 40; Natural keeps 40.
const JITTER_MS = { focus_hq: 120, natural: 40 };
const JITTER_MAX_MS = { focus_hq: 240, natural: 160 };
// Offline evidence for the drawer (docs/BENCHMARKS.md): Enrolled tier, 16 LibriSpeech two-talker + noise mixes.
const SI_SDRI_CLAIM = "+11.8 dB enrolled TSE (offline, BENCHMARKS.md)";

const $ = (id) => document.getElementById(id);
/** Writes only on change: render() runs every 100 ms and the pill is an aria-live region. */
const setText = (el, text) => { if (el.textContent !== text) el.textContent = text; };
const now = () => (performance.timeOrigin + performance.now()) / 1000;

const state = {
  url: sessionUrl(window.location),
  ws: null,
  attempt: 0,
  retryTimer: null,
  connected: false,
  ab: "processed",
  mode: "focus_hq", // T-150: the Focus / Quiet buttons (F/Q)
  modeAt: 0, // performance.now() of this page's last mode request; telemetry may override it only after MODE_GRACE_MS
  telemetry: null,
  audio: null,
  video: null,
  seq: 0,
  frameAt: 0, // T-158: performance.now() of the last audio frame sent (or of Start finishing)
  startToken: 0, // T-158: bumped per Start; a getUserMedia that resolves after its timeout is torn down
  videoSeq: 0,
  rttMs: null,
  cameraLost: false,
  cameraNatural: false, // T-135: the camera loss switched the session to Natural (no healthy Enrolled tier)
  jitter: "focus_hq", // the worklet's jitter-buffer profile (JITTER_MS)
  pill: new PillHysteresis(),
  steadyDelay: new Steady(50),
  steadyRtt: new Steady(10),
  hqActive: null,
  sent: [],
  log: [],
  nameCall: false, // T-120: camera-free Name Call mode instead of the judge profile's Focus HQ
  name: savedName() || "Aryan",
  released: false, // T-177: Release sent; the server stops auto-calibrating until 1 or 2
  isolation: null, // T-181: picked in the drawer; resent with the profile so a reconnect keeps it
  keyFocus: new KeyFocus(document.hasFocus()), // T-177: "Keys off: click the page"
};

// ------------------------------------------------------------------ devices (T-121)

// Another mic while running: restart capture on it (the change event is a user gesture).
const devices = mountDevices($("devices"), {
  onMicChange: async () => { if (state.audio && !state.replaying) { stopMedia(); await startMedia(); } },
  onNote: (text) => note(text),
  onSwap: (stream, info) => swapMic(stream, info),
});

/** T-287: the mic changed mid-session (the iPhone dropped, or Retry switched back): feed the running worklet from
 * the new stream without restarting the socket or session, and send a hello naming it (`mic_switch`), so the
 * recording and session_report show both switches. False when nothing is capturing (the controller drops it). */
function swapMic(stream, info) {
  const a = state.audio;
  if (!a?.source) return false;
  let source;
  try {
    source = a.ctx.createMediaStreamSource(stream);
  } catch (e) {
    note(`mic switch to ${info.to} failed: ${e?.message || e}`);
    return false;
  }
  a.source.disconnect();
  a.stream.getTracks().forEach((t) => t.stop());
  source.connect(a.node);
  Object.assign(a, { source, stream });
  state.frameAt = performance.now();
  note(`mic ${info.reason}: ${info.from || "?"} → ${info.to}`);
  sendHello({ mic_switch: `${info.reason}: ${info.from || "?"} -> ${info.to}`.slice(0, 300) });
  return true;
}

// ------------------------------------------------------------------ drawer log (every fallback, one line)

function note(text) {
  const stamp = new Date().toLocaleTimeString([], { hour12: false });
  state.log.push(`${stamp} ${text}`);
  const li = document.createElement("li");
  li.textContent = `${stamp} ${text}`;
  $("log").append(li);
  while ($("log").children.length > 30) $("log").firstElementChild.remove();
}

// ------------------------------------------------------------------ socket

function connect() {
  clearTimeout(state.retryTimer);
  let ws;
  try {
    ws = new WebSocket(state.url + (state.url.includes("?") ? "&" : "?") + "replay=1");
  } catch (e) {
    lost(e.message);
    return;
  }
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  ws.onopen = () => {
    if (state.attempt > 0) note("engine back");
    state.attempt = 0;
    state.connected = true;
    state.frameAt = performance.now(); // T-158: the stall clock starts again on reconnect
    sendHello();
    state.modeAt = performance.now(); // a fresh session reports its default mode until this control lands
    sendJudgeProfile({ ab: state.ab, ...modeControl(state.mode, state) });
    if (state.nameCall) sendControl(nameCallControl(state.name));
  };
  ws.onmessage = (e) => (typeof e.data === "string" ? onText(e.data) : onBinary(e.data));
  ws.onclose = () => { if (state.ws === ws) lost("socket closed"); };
}

function lost(why) {
  replay.disconnect();
  if (state.connected || state.attempt === 0) note(`engine lost (${why}); retrying`);
  state.ws = null;
  state.connected = false;
  state.telemetry = null;
  const delay = RETRY_MS[Math.min(state.attempt, RETRY_MS.length - 1)];
  state.attempt++;
  state.retryTimer = setTimeout(connect, delay);
}

function send(obj) {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

function sendControl(fields) {
  const msg = control(fields);
  state.sent.push(msg);
  send(msg);
}

/** The judge profile (it loads the first saved person and turns auto-calibration on); after Release
 * (T-177) the release follows, so a reconnect or a camera return never re-targets on its own. */
function sendJudgeProfile(fields = {}) {
  sendControl({ profile: "judge", ...(state.isolation ? { isolation: state.isolation } : {}), ...fields });
  if (state.released) sendControl(RELEASE_CONTROL);
}

/** The mic and output actually in use (T-121); recordings and session_report show them. */
function deviceLabels() {
  return { input_device: devices.inputLabel()?.slice(0, 200), output_device: devices.outputLabel()?.slice(0, 200),
    devices_missing: devices.missing() }; // T-133: saved devices that were not found
}

function sendHello(extra = {}) {
  const video = helloVideo(isAudioOnly(), state.video?.stream);
  // extra (T-287 mic_switch) right after "type": session_report prints the first 160 characters of each hello
  send({ type: "hello", ...extra, ...hello(state.audio ? state.audio.rate : 16000, FRAME_MS, video), client: "judge", ...deviceLabels() });
}

function onText(text) {
  let msg;
  try { msg = JSON.parse(text); } catch { return; }
  replay.text(msg);
  if (msg.type === "telemetry") {
    state.telemetry = msg;
    expectDevices(msg); // T-287: before the next render, so a Start right after the first telemetry knows the mic
    showServerInfo(msg);
    enforceAudioOnly();
    postAb();
    setJitter(msg.effective_mode === "natural" ? "natural" : "focus_hq");
    followMode(msg.mode);
    watchBackend(msg);
    const floorRule = !!msg.agc?.floor_rule; // T-179: the server's AGC floor rule, mirrored on the RAW side
    if (state.audio && floorRule !== !!state.agcFloorRule) {
      state.agcFloorRule = floorRule;
      state.audio.node.port.postMessage({ type: "agc", on: true, floorRule });
    }
  } else if (msg.type === "caption") {
    addCaption(msg);
  } else if (msg.type === "error") {
    note(`${msg.stage}: ${msg.message}${msg.degraded_to ? ` → ${msg.degraded_to}` : ""}`);
  }
}

function watchBackend(t) {
  const active = t.hq_active || null;
  if (state.hqActive && active && active !== state.hqActive) {
    note(active === "local" ? `GPU lost → Mac ladder (crossfade): ${t.hq_reason || ""}` : `GPU back: ${t.hq_reason || ""}`);
  }
  state.hqActive = active;
}

function onBinary(buffer) {
  let frame;
  try { frame = unpackFrame(buffer); } catch { return; }
  if (replay.binary(frame)) return;
  if (frame.kind !== KIND_AUDIO_OUT) return;
  if (frame.captureTs > 0) {
    const rtt = 1000 * (now() - frame.captureTs);
    state.rttMs = state.rttMs == null ? rtt : 0.9 * state.rttMs + 0.1 * rtt;
  }
  const samples = pcm16ToFloat(frame.payload);
  state.audio?.node.port.postMessage({ type: "play", samples }, [samples.buffer]);
}

// ------------------------------------------------------------------ media

/** T-158: a Start that failed is a red line above the pill, never only a drawer note. */
function micBanner(text) {
  $("mic-banner").textContent = text || "";
  $("mic-banner").hidden = !text;
}

async function startMedia() {
  const token = ++state.startToken;
  $("start").disabled = true;
  micBanner("");
  try {
    // T-158: a hung permission prompt or addModule must not leave Start disabled for ever
    await withTimeout(startAudio(false, token), MIC_START_TIMEOUT_MS);
  } catch (e) {
    if (token !== state.startToken) return; // a newer Start owns the button
    state.startToken++; // abandon this attempt: if it resolves later, startAudio tears it down
    stopMedia();
    note(`microphone unavailable: ${e?.message || e}`);
    micBanner(micProblem(e));
    $("start").disabled = false;
    return;
  }
  if (!state.telemetry && state.connected) await firstTelemetry(1500); // T-168: learn the preset first
  if (isAudioOnly()) {
    setCameraLost(false); // T-168: no camera by design, nothing was lost
  } else {
    try {
      await startVideo();
      setCameraLost(false);
    } catch (e) {
      note(`camera unavailable: ${e?.message || e}`);
      setCameraLost(true);
    }
    enforceAudioOnly(); // the preset may have arrived while the camera was opening
  }
  sendHello();
  $("start").disabled = false;
  $("start").setAttribute("aria-pressed", "true");
}

async function startAudio(playbackOnly = false, token = null) {
  let ctx = new AudioContext(audioContextOptions());
  ctx.resume();
  let stream;
  try {
    stream = playbackOnly ? new MediaStream() : await devices.openMic();
  } catch (e) {
    ctx.close();
    throw e;
  }
  let source;
  try {
    if (!playbackOnly) source = ctx.createMediaStreamSource(stream);
  } catch {
    await ctx.close();
    ctx = new AudioContext({ latencyHint: "interactive" });
    if (!playbackOnly) source = ctx.createMediaStreamSource(stream);
  }
  await ctx.audioWorklet.addModule("audio-worklet.js");
  const rate = ctx.sampleRate;
  const node = new AudioWorkletNode(ctx, "hearing-io", {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
    processorOptions: {
      frameSamples: Math.round((rate * FRAME_MS) / 1000),
      targetSamples: Math.round((rate * JITTER_MS[state.jitter]) / 1000),
      maxSamples: Math.round((rate * JITTER_MAX_MS[state.jitter]) / 1000),
    },
  });
  node.port.postMessage({ type: "agc", on: true }); // T-135: RAW levelled like the server's output AGC
  node.port.onmessage = (e) => {
    if (e.data.type !== "frame" || state.replaying) return;
    devices.feed(e.data.samples);
    if (state.ws?.readyState !== WebSocket.OPEN) return;
    state.ws.send(packFrame(KIND_AUDIO_IN, state.seq++, now(), floatToPcm16(e.data.samples)));
    state.frameAt = performance.now();
  };
  source?.connect(node);
  node.connect(ctx.destination);
  await ctx.resume();
  if (token !== null && token !== state.startToken) { // T-158: timed out or superseded while pending
    stream.getTracks().forEach((t) => t.stop());
    ctx.close();
    throw new Error("abandoned: Start timed out or was pressed again");
  }
  state.audio = { ctx, node, stream, rate, source };
  state.frameAt = performance.now();
  if (!playbackOnly) await devices.started(stream, ctx);
  postAb();
  state.seq = 0;
}

/** T-168: audio only under the team preset (telemetry.preset.pins.audio_only) or with ?audio=1. */
function isAudioOnly() {
  return audioOnly({ search: window.location.search, telemetry: state.telemetry });
}

/** Resolves once telemetry arrived (or after `ms`), so Start knows the preset before opening the camera. */
function firstTelemetry(ms) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    const poll = () => (state.telemetry || performance.now() - t0 > ms ? resolve() : setTimeout(poll, 50));
    poll();
  });
}

/** T-168: the camera never runs under audio only; one opened before the preset was known is closed. */
function enforceAudioOnly() {
  const on = isAudioOnly();
  document.body.classList.toggle("audio-only", on); // hides the camera peek (no face UI)
  if (!on || !state.video) return;
  stopVideo();
  note("audio only (team preset): camera off");
  sendHello();
}

async function startVideo() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: VIDEO_W }, height: { ideal: VIDEO_H }, frameRate: { ideal: VIDEO_FPS }, facingMode: "user" },
  });
  const video = $("video");
  video.srcObject = stream;
  await video.play();
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, VIDEO_W / video.videoWidth, VIDEO_H / video.videoHeight);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  for (const track of stream.getVideoTracks()) track.addEventListener("ended", () => setCameraLost(true));
  state.video = { stream, canvas, busy: false, clock: frameClock(1000 / VIDEO_FPS, sendVideoFrame) };
  state.videoSeq = 0;
}

// T-138: say so when the browser pauses the camera in a background tab instead of degrading silently.
const bgWatch = new BackgroundWatch(BG_MIN_FPS, (paused) => {
  $("bg-notice").hidden = !paused;
  note(paused ? "page in background: camera paused" : "camera back (page in foreground)");
});

/** Playback jitter buffer for the mode playing (JITTER_MS); the worklet applies it on the next re-buffer. */
function setJitter(kind) {
  if (kind === state.jitter) return;
  state.jitter = kind;
  const rate = state.audio?.rate;
  if (!rate) return;
  state.audio.node.port.postMessage({ type: "config", targetSamples: Math.round((rate * JITTER_MS[kind]) / 1000),
    maxSamples: Math.round((rate * JITTER_MAX_MS[kind]) / 1000) });
}

/**
 * Camera gone. T-135: a healthy Enrolled tier needs no camera, so Focus HQ keeps isolating the voice
 * profile; only without it the page falls back to Natural (noise suppression only). Back to the profile
 * when the camera returns (only if it had left it).
 */
function setCameraLost(isLost) {
  if (isLost === state.cameraLost) return;
  state.cameraLost = isLost;
  if (state.nameCall || state.mode === "quiet") {
    note(isLost ? `camera lost (${state.nameCall ? "Name Call" : "Quiet"} does not need it)` : "camera back");
    return;
  }
  if (isLost && enrolledHealthy(state.telemetry)) {
    note("camera lost, voice profile still isolating (Enrolled tier, no camera needed)");
  } else if (isLost) {
    state.cameraNatural = true;
    note("camera lost → Natural, noise suppression only");
    sendControl({ mode: "natural" });
  } else {
    note("camera back");
    if (state.cameraNatural) sendJudgeProfile(modeControl(state.mode));
    state.cameraNatural = false;
  }
}

function sendVideoFrame() {
  const v = state.video;
  const ws = state.ws;
  bgWatch.check(performance.now(), document.hidden);
  if (!v?.canvas || v.busy || ws?.readyState !== WebSocket.OPEN || ws.bufferedAmount > VIDEO_BACKPRESSURE_BYTES) return;
  const ts = now();
  v.canvas.getContext("2d").drawImage($("video"), 0, 0, v.canvas.width, v.canvas.height);
  v.busy = true;
  v.canvas.toBlob(async (blob) => {
    v.busy = false;
    if (!blob || state.ws !== ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(packFrame(KIND_VIDEO_IN, state.videoSeq++, ts, await blob.arrayBuffer()));
    bgWatch.frame(performance.now());
  }, "image/jpeg", 0.7);
}

function stopVideo() {
  if (!state.video) return;
  state.video.clock.stop();
  bgWatch.lastFrameMs = null;
  $("bg-notice").hidden = true;
  state.video.stream.getTracks().forEach((t) => t.stop());
  $("video").srcObject = null;
  state.video = null;
}

function stopMedia() {
  stopVideo();
  if (state.audio) {
    state.audio.stream.getTracks().forEach((t) => t.stop());
    state.audio.ctx.close();
    state.audio = null;
    devices.stopped();
  }
  state.rttMs = null;
  $("start").setAttribute("aria-pressed", "false");
  sendHello();
}

// ------------------------------------------------------------------ A/B, keys

function showMode(mode) {
  state.mode = mode;
  for (const b of document.querySelectorAll("#modes button")) b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  document.body.dataset.mode = mode;
}

/** The phone companion forwards Focus/Quiet into this session: show what the engine plays, never only our last click. */
function followMode(mode) {
  if (mode !== "quiet" && mode !== "focus_hq") return; // natural (camera loss) / name_call are this page's own states
  if (mode === state.mode || state.nameCall || state.cameraNatural || performance.now() - state.modeAt < MODE_GRACE_MS) return;
  showMode(mode);
  note(`mode ${mode === "quiet" ? "Quiet" : "Focus"} (set from the phone)`);
}

/** T-150: Focus (focus_hq) or Quiet (captions of the target, no audio). Leaves Name Call. */
function setMode(mode) {
  state.modeAt = performance.now();
  showMode(mode);
  state.cameraNatural = mode !== "quiet" && state.cameraLost && !enrolledHealthy(state.telemetry);
  if (state.nameCall) setNameCall(false); // sends the profile with this mode
  else sendControl(modeControl(mode, state));
  render();
}

function postAb() {
  state.audio?.node.port.postMessage({ type: "ab", state: state.ab, delayMs: abDelayMs(state.telemetry) });
}

function setAb(ab) {
  ({ ab } = abControl(state.ab, ab));
  state.ab = ab;
  postAb();
  sendControl({ ab });
}

function calibrate() {
  state.released = false; // T-177: a calibration picks a target too; leave Searching, stop re-sending clear
  sendControl({ enroll: { action: "start", seconds: CALIBRATE_S } });
  note("manual calibration (C)");
  render();
}

function onKey(e) {
  if (e.altKey || e.ctrlKey || e.metaKey || e.target.closest?.("input, select, textarea")) return;
  const key = e.key.toLowerCase();
  // T-135: C/R/N act only with Shift or the drawer open (a judge leaning on the keyboard fires nothing)
  // T-150: F/Q (Focus / Quiet) are guarded the same way
  if ("crnfq".includes(key) && key.length === 1 && !(e.shiftKey || $("drawer").open)) return;
  if (key === " ") {
    e.preventDefault();
    if (!e.repeat) toggleAutoFocus(); // Q-2010: Space is the Auto voice focus switch
  } else if (modeForKey(key)) {
    if (!e.repeat) setMode(modeForKey(key));
  } else if (key === "c") {
    calibrate();
  } else if (key === "r") {
    $("replay").click();
  } else if (key === "n") {
    setNameCall(!state.nameCall);
  } else if (key === "d") {
    $("drawer").open = !$("drawer").open;
  }
  // Q-2010: no 1/2 person keys and no Release (Shift+0): nobody chooses a name, the closest voice is isolated
}

/** Auto voice focus (Q-2010): flip from what the engine reports playing (telemetry ab), so a press always moves the
 * switch the way it reads. ON = processed (isolated), OFF = raw (the whole room). */
function toggleAutoFocus() {
  setAb(autoFocusState(state.telemetry?.ab, state.ab).next);
  renderAutoFocus();
}

/** The switch shows the engine's ab (telemetry), never only the click; `pending` until the engine agrees. */
function renderAutoFocus() {
  const a = autoFocusState(state.telemetry?.ab, state.ab);
  const sw = $("ab");
  sw.setAttribute("aria-checked", String(a.on));
  sw.dataset.on = String(a.on);
  sw.dataset.pending = String(a.pending);
  sw.dataset.live = String(Boolean(state.audio && state.connected));
  const word = sw.querySelector(".autofocus-word");
  if (word.textContent !== (a.on ? "ON" : "OFF")) word.textContent = a.on ? "ON" : "OFF";
  if ($("autofocus-line").textContent !== a.line) $("autofocus-line").textContent = a.line;
}

/** T-177: the target line (Q-2010: no person buttons; under HEARING_PICK=loudest the closest voice). */
function renderPeople(t) {
  // loudest mode: the pill, the chip and the switch already say "closest voice"; a fourth line was noise
  const line = loudestMode(t) ? "" : targetLine(t, { released: state.released });
  $("target-line").hidden = !line;
  if ($("target-line").textContent !== line) $("target-line").textContent = line;
}

/** T-177: keys never fail silently. Another window/tab or a text field has them -> the banner. */
function renderKeysOff() {
  $("keys-off").hidden = !state.keyFocus.off;
}

// ------------------------------------------------------------------ rendering

function render() {
  const t = state.telemetry;
  if (state.video) bgWatch.check(performance.now(), document.hidden);
  // T-158: never look live while the mic is not streaming (Start not pressed, denied, hung or stalled)
  const mic = micPill({ connected: state.connected, started: Boolean(state.audio), replaying: Boolean(state.replaying),
    sinceFrameMs: state.audio?.stream.getAudioTracks().some((tr) => tr.readyState === "ended")
      ? Infinity : performance.now() - state.frameAt });
  const raw = mic || (state.nameCall && state.connected && t
    ? { ...nameCallPill(t, state.name.split(",")[0]), note: "" }
    : state.released && state.connected ? SEARCHING_PILL // T-177: after Release, until 1 or 2
      : judgePill(t, { cameraLost: state.cameraLost, connected: state.connected }));
  $("namecall-release").hidden = !(state.nameCall && t?.name_call?.target);
  // T-134: the enrolled tier's pick confidence (0 = plays the mix, 1 = residual 40 dB down) + the saved profile
  const conf = t?.tse?.confidence;
  if (Number($("d-conf").value) !== (conf ?? 0)) $("d-conf").value = conf ?? 0;
  const prof = t?.enrollment?.source === "profile" && t.enrollment.profile ? ` · profile: ${t.enrollment.profile} (saved)` : "";
  // T-178: the pick's contrast floor and cos(target, other profile); "dropped" = above SAME_VOICE 0.7, LDA off
  const pc = t?.tse?.pair_cos;
  const pair = pc == null ? "" : ` · cos(pair) ${pc.toFixed(2)}${t.tse.negative_active ? "" : " (negative dropped)"}`;
  const ctr = t?.tse?.contrast == null ? "" : ` · contrast ${t.tse.contrast}`;
  setText($("d-conf-text"), (conf == null ? "—" : `${Math.round(conf * 100)}%`) + prof + ctr + pair);
  $("namecall").hidden = !(state.nameCall || $("drawer").open); // T-135: only once N armed it (or in the drawer)
  const pill = state.pill.update(raw, performance.now()); // T-135: hysteresis, isolated instant
  $("start").hidden = Boolean(state.replaying);
  // Q-2010: Auto voice focus off (the engine reports ab raw) plays the whole room, so the pill never says Isolated then
  const roomPlays = pill.state === "isolated" && !autoFocusState(t?.ab, state.ab).on;
  const pillState = roomPlays ? "raw" : pill.state;
  if ($("pill").dataset.state !== pillState) $("pill").dataset.state = pillState;
  setText($("pill-text"), roomPlays ? "Whole room — auto voice focus off" : pill.text);
  setText($("pill-note"), roomPlays ? "" : pill.note);
  $("pill-note").hidden = roomPlays || !pill.note;
  // loudest mode: the pill already reads "Isolated — closest voice", so the gate chip would only repeat it
  renderGate($("gate-pill"), t, pill.state === "isolated" && !roomPlays && state.ab !== "raw" && !loudestMode(t));
  const hq = hqBackendInfo(t);
  // T-109: the live backend (hq_active) and its RTT, so the presenter sees the auto switch to the GPU.
  // T-170: the GPU serving older code than this checkout cost a whole live test; say so here
  const stale = t?.remote?.rev_warning || t?.preset?.pins?.rev_warning || null;
  setText($("d-backend"), t
    ? `${hq.active === "remote" ? "iLab GPU" : `Mac (${t.device})`} · hq_active ${hq.active || "local"} · ${hq.label}${hq.reason ? ` · ${hq.reason}` : ""}${stale ? ` · ⚠ ${stale}` : ""}`
    : "—");
  const tiers = t?.tiers;
  setText($("d-tier"), t ? `${tierInfo(tiers?.active || t.tier).label}${tiers?.same_bucket ? " · same voice group" : ""}` : "—");
  const total = state.steadyDelay.update(t?.latency_ms?.total); // T-135: EMA, 50 ms steps
  setText($("d-delay"), t && total != null ? `${total} ms (${t.latency_preset}, ${t.isolation})` : "—");
  if (t?.isolation && document.activeElement !== $("isolation") && $("isolation").value !== t.isolation) $("isolation").value = t.isolation;
  const rtt = t?.remote_rtt_ms ?? null;
  const pageRtt = state.steadyRtt.update(state.rttMs ?? NaN);
  setText($("d-rtt"), `${state.rttMs == null ? "—" : `${pageRtt} ms page↔engine`}${rtt == null ? (t?.hq_backend && t.hq_backend !== "local" ? " · GPU —" : "") : ` · GPU ${Math.round(rtt)} ms`}`);
  renderPreset(t?.preset); // d-claim is a constant, set once in init()
  expectDevices(t);
  renderCaptions(t);
  renderPeople(t);
  renderAutoFocus();
  renderKeysOff();
}

/** T-143/T-287: the team preset's demo mic and output (preset.pins); Start opens that mic when it is plugged in. */
function expectDevices(t) {
  devices.setExpected({ input: t?.preset?.pins?.input_device || null, output: t?.preset?.pins?.output_device || null });
}

/** T-150: captions show in Quiet (and whenever a line arrived, e.g. a replay) with the target's name; a
 * notice when the feed fell back to the room mixture (Focus HQ ladder failed). */
function renderCaptions(t) {
  const quiet = state.mode === "quiet" && !state.nameCall;
  $("captions-panel").hidden = !quiet && !$("captions").querySelector("p:not(.empty)");
  const feed = captionFeed(t);
  $("caption-speaker").hidden = !quiet;
  if ($("caption-speaker").textContent !== feed.speaker) $("caption-speaker").textContent = feed.speaker;
  $("caption-warning").hidden = !(quiet && t && (feed.degraded || feed.paused));
  setText($("caption-warning"), feed.paused || DEGRADED_NOTICE);
}

/** T-130: HEARING_DEMO_PRESET=team shows "TEAM PRESET" and every pin it holds; hidden when the preset is off. */
function renderPreset(preset) {
  const on = Boolean(preset?.pins);
  $("d-preset-label").hidden = $("d-preset").hidden = !on;
  if (!on) return;
  const p = preset.pins;
  $("d-preset").textContent = [
    `HQ ${p.hq_backend} (${p.hq_reason})${p.remote_git_rev ? `, A100 rev ${p.remote_git_rev.slice(0, 7)}` : ""}${p.rev_warning ? ` ⚠ ${p.rev_warning}` : ""}`,
    `same-bucket guard ${p.same_bucket_guard}`,
    `lock on Start${p.lock_on_start ? "" : " off"}, hold ${p.hold_lock_s} s`,
    `name ${p.name} (${(p.aliases || []).join(", ")}; ${p.match}), quiet ${p.namecall_quiet_s} s, release ${p.namecall_timeout_s} s`,
    `voice ${p.voice_override}`,
    `min switch ${p.min_switch_s} s`,
    `feed ${p.synth_feed}, ASR ${p.asr}, T=${p.caption_temperature}${p.caption_prompt ? ", vocab prompt" : ""}`,
    `noise ${p.demo_noise_level_db} dBFS`,
    `devices ${p.input_device} → ${p.output_device} (BT ≤ ${p.bt_round_trip_max_ms} ms)`,
    p.audio_only ? "audio only (no camera)" : "camera on",
  ].join(" · ");
}

let partial = null;
function addCaption(c) {
  const box = $("captions");
  box.querySelector(".empty")?.remove();
  if (!partial) {
    partial = document.createElement("p");
    box.append(partial);
  }
  partial.textContent = c.text;
  partial.classList.toggle("partial", !c.final);
  if (c.final) partial = null;
  while (box.children.length > 4) box.firstElementChild.remove();
}

// ------------------------------------------------------------------ wiring

const replay = replayControls(state, {
  startAudio,
  stopMedia,
  error: (message) => note(`replay: ${message}`),
});

/** Name Call on: the session plays name_call (no camera needed); off: back to the judge profile. */
function setNameCall(on) {
  state.nameCall = on;
  $("namecall-on").checked = on;
  note(on ? `Name Call on: listening for “${state.name}”` : "Name Call off");
  if (on) sendControl(nameCallControl(state.name));
  else {
    state.cameraNatural = state.mode !== "quiet" && state.cameraLost && !enrolledHealthy(state.telemetry);
    sendJudgeProfile(modeControl(state.mode, state));
  }
}

function init() {
  $("namecall-name").value = state.name;
  $("namecall-on").addEventListener("change", (e) => setNameCall(e.target.checked));
  $("namecall-name").addEventListener("change", (e) => {
    state.name = e.target.value.trim() || "Aryan";
    saveName(state.name);
    if (state.nameCall) sendControl(nameCallControl(state.name));
  });
  mountNameGuards($("namecall-match"), $("namecall-exclude"), () => {
    if (state.nameCall) sendControl(nameCallControl(state.name));
  });
  $("namecall").addEventListener("submit", (e) => e.preventDefault());
  $("namecall-release").addEventListener("click", () => sendControl({ name_call: "release" }));
  $("isolation").addEventListener("change", (e) => {
    state.isolation = e.target.value;
    sendControl({ isolation: state.isolation });
  });
  $("ab").addEventListener("click", () => toggleAutoFocus());
  $("modes").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (b) setMode(b.dataset.mode);
  });

  // T-177: the banner follows window focus (another window or tab) and text fields (onKey ignores them)
  window.addEventListener("blur", () => { state.keyFocus.blur(); renderKeysOff(); });
  window.addEventListener("focus", () => { state.keyFocus.focus(); renderKeysOff(); });
  document.addEventListener("focusin", (e) => { state.keyFocus.focusOn(e.target?.tagName); renderKeysOff(); });
  document.addEventListener("focusout", () => { state.keyFocus.focusOn(null); renderKeysOff(); });

  $("start").addEventListener("click", () => (state.audio ? stopMedia() : startMedia()));
  $("d-claim").textContent = SI_SDRI_CLAIM;
  document.addEventListener("keydown", onKey);
  document.addEventListener("keyup", (e) => { if (e.key === " ") e.preventDefault(); });
  setInterval(render, RENDER_MS);
  connect();
}

init();
window.hearingJudge = state; // for the headless check and devtools
