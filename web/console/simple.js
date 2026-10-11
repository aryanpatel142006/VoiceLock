import { frameClock } from "./frame-clock.js";
import { showServerInfo } from './server-info.js';
import { replayControls } from './replay.js';
import { mountDevices } from "./devices.js";
import { renderAcquisition } from "./acquisition.js";
import { mountCalibration } from "./calibrate.js";
import { mountNameGuards, nameCallControl, nameCallPill, saveName, savedName } from "./name-call.js";

// Simple test page for /ws/session (served at /simple): five big modes + Auto, the suppression
// slider, where Focus HQ runs, and one status line. The wire format comes from protocol.js and
// the mic framing / playback buffer from audio-worklet.js, the same modules the full console uses.
// Query parameters: ?ws=<url> picks the session socket (default: /ws/session on this host).

import {
  KIND_AUDIO_IN, KIND_AUDIO_OUT, KIND_VIDEO_IN, MODES, SLIDER_MODES,
  abDelayMs, control, dbfs, floatToPcm16, hello, hqBackendInfo, packFrame, pcm16ToFloat, sessionUrl,
  splitLatency, tierInfo, unpackFrame,
  audioContextOptions,
} from "./protocol.js";

const FRAME_MS = 20;
const PLAYBACK_TARGET_MS = 40;
const PLAYBACK_MAX_MS = 160;
const VIDEO_FPS = 25;
const VIDEO_W = 640;
const VIDEO_H = 360;
const JPEG_QUALITY = 0.7;
const VIDEO_BACKPRESSURE_BYTES = 256 * 1024;
const TELEMETRY_STALE_MS = 1500;
const RETRY_MS = [1000, 2000, 4000, 8000];
const RENDER_MS = 100;
const NATURAL_AIM_MS = 100; // the Natural mic-to-ear target; above twice that the number turns red
const CAPTION_MODES = new Set(["synthetic", "text"]);
const MODE_LABELS = {
  natural: "Natural", focus_hq: "Focus HQ", synthetic: "Synthetic",
  text: "Text", quiet: "Quiet", auto: "Auto", name_call: "Name Call",
};
const VOICE_LABELS = {
  neutral: "neutral", female_young: "young female", female_adult: "adult female",
  female_older: "older female", male_young: "young male", male_adult: "adult male",
  male_older: "older male",
};
const SOURCE_LABELS = {
  "face+voice": "from face + voice", face: "from face", voice: "from voice",
  override: "your choice", default: "default",
};

const $ = (id) => document.getElementById(id);
const now = () => (performance.timeOrigin + performance.now()) / 1000;
const fmt = (ms, digits = 0) => (ms == null || Number.isNaN(ms) ? "—" : ms.toFixed(digits));

const state = {
  url: sessionUrl(window.location),
  ws: null,
  attempt: 0,
  retryTimer: null,
  everConnected: false,
  mode: "natural",
  ab: "processed", // what the headphones play: "raw" mic or the engine's "processed" output
  suppression: 50,
  voiceOverride: null,
  latencyPreset: "balanced", // Focus HQ delay/quality: speed | balanced | accuracy (independent of mode)
  hqBackend: null, // null until the user picks: the server's default applies
  telemetry: null,
  telemetryAt: 0,
  faces: [],
  audio: null, // {ctx, node, stream, rate}
  video: null, // {stream, timer, canvas, busy}
  seq: 0,
  videoSeq: 0,
  framesOut: 0,
  framesBack: 0,
  rttMs: null,
  playback: { buffered: 0, underruns: 0, dropped: 0 },
  sent: [], // every control message, for the headless check
  name: savedName() || "Aryan", // Name Call: the wearer's name (and aliases after commas)
};
// Another mic while running: restart capture on it (the change event is a user gesture).
const devices = mountDevices($("devices"), {
  onMicChange: async () => { if (state.audio && !state.replaying) { stopMedia(); await startMedia(); } },
  onNote: (text) => showBanner(text, false),
});
const calibration = mountCalibration($("acquisition"), (fields) => sendControl(fields));

// ------------------------------------------------------------------ socket

function connect() {
  clearTimeout(state.retryTimer);
  if (state.ws) {
    state.ws.onclose = state.ws.onerror = state.ws.onmessage = null;
    state.ws.close();
  }
  setConn("connecting", "Connecting…");
  let ws;
  try {
    ws = new WebSocket(state.url + (state.url.includes("?") ? "&" : "?") + "replay=1");
  } catch (e) {
    connectionLost(e.message);
    return;
  }
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  ws.onopen = () => {
    state.attempt = 0;
    state.everConnected = true;
    setConn("open", "Connected");
    $("banner").hidden = true;
    sendHello();
    sendControl({
      ab: state.ab, mode: state.mode, suppression: state.suppression, voice_override: state.voiceOverride,
      ...(state.mode === "name_call" ? nameCallControl(state.name, state.suppression) : {}),
      ...(state.hqBackend ? { hq_backend: state.hqBackend } : {}),
      latency_preset: state.latencyPreset,
      ...calibration.initialFields(),
    });
  };
  ws.onmessage = (e) => (typeof e.data === "string" ? onText(e.data) : onBinary(e.data));
  ws.onclose = (e) => {
    if (state.ws === ws) connectionLost(e.code === 1006 || !e.code ? "no answer" : `closed with code ${e.code}`);
  };
}

function connectionLost(why) {
  replay.disconnect();
  state.ws = null;
  state.telemetry = null;
  const delay = RETRY_MS[Math.min(state.attempt, RETRY_MS.length - 1)];
  state.attempt++;
  const lead = state.everConnected ? "Lost the engine" : "Can't reach the engine";
  setConn("error", lead);
  showBanner(`${lead} at ${state.url} (${why}). Is \`hearing serve\` running? Retrying in ${Math.round(delay / 1000)} s.`);
  state.retryTimer = setTimeout(connect, delay);
  render();
}

function send(obj) {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

function sendControl(fields) {
  const msg = control(fields);
  state.sent.push(msg);
  send(msg);
}

function sendHello() {
  const rate = state.audio ? state.audio.rate : 16000;
  send({
    ...hello(rate, FRAME_MS, Boolean(state.video)), client: "simple",
    input_device: devices.inputLabel()?.slice(0, 200), output_device: devices.outputLabel()?.slice(0, 200), // T-121: which mic the recording used
    devices_missing: devices.missing(), // T-133: saved devices that were not found
  });
}

function onText(text) {
  let msg;
  try {
    msg = JSON.parse(text);
  } catch {
    return;
  }
  replay.text(msg);
  if (msg.type === "telemetry") {
    state.telemetry = msg;
    showServerInfo(msg);
    postAb(); // keeps the raw side's delay on the engine's current latency
    state.telemetryAt = performance.now();
    state.faces = msg.faces || [];
  } else if (msg.type === "caption") {
    addCaption(msg);
  } else if (msg.type === "error" && msg.stage === "hq_backend") {
    $("hq-active").title = msg.message;
  }
}

function onBinary(buffer) {
  let frame;
  try {
    frame = unpackFrame(buffer);
  } catch {
    return;
  }
  if (replay.binary(frame)) return;
  if (frame.kind !== KIND_AUDIO_OUT) return;
  state.framesBack++;
  if (frame.captureTs > 0) {
    const rtt = 1000 * (now() - frame.captureTs);
    state.rttMs = state.rttMs == null ? rtt : 0.9 * state.rttMs + 0.1 * rtt;
  }
  const samples = pcm16ToFloat(frame.payload);
  state.audio?.node.port.postMessage({ type: "play", samples }, [samples.buffer]);
}

// ------------------------------------------------------------------ media

async function startMedia() {
  const btn = $("start");
  btn.disabled = true;
  try {
    await startAudio();
  } catch (e) {
    stopMedia();
    showBanner(`Microphone unavailable: ${mediaError(e)}`, false);
    btn.disabled = false;
    return;
  }
  try {
    await startVideo();
  } catch (e) {
    $("camera-placeholder").textContent = `Camera unavailable: ${mediaError(e)}. Audio still streams.`;
  }
  sendHello();
  btn.disabled = false;
  btn.setAttribute("aria-pressed", "true");
  btn.textContent = "Stop";
  render();
}

function mediaError(e) {
  if (e?.name === "NotAllowedError") return "permission denied. Allow it in the address bar, then press Start";
  if (e?.name === "NotFoundError") return "no device found";
  if (!window.isSecureContext) return "the page must be served from localhost or https";
  return e?.message || String(e);
}

async function startAudio(playbackOnly = false) {
  if (!playbackOnly && !navigator.mediaDevices?.getUserMedia) throw new Error("this browser has no getUserMedia");
  // iOS only runs an AudioContext made inside the Start tap: create (and resume) it before any await.
  let ctx = new AudioContext(audioContextOptions());
  ctx.resume();
  let stream;
  try {
    stream = playbackOnly ? new MediaStream() : await devices.openMic(); // raw room, saved mic exact (T-133)
  } catch (e) {
    ctx.close();
    throw e;
  }
  let source;
  try {
    if (!playbackOnly) source = ctx.createMediaStreamSource(stream);
  } catch {
    // Firefox cannot connect a mic to a context at another rate: send the native rate (hello says so).
    await ctx.close();
    ctx = new AudioContext({ latencyHint: "interactive" });
    if (!playbackOnly) source = ctx.createMediaStreamSource(stream);
  }
  await ctx.audioWorklet.addModule("audio-worklet.js");
  const rate = ctx.sampleRate;
  const node = new AudioWorkletNode(ctx, "hearing-io", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      frameSamples: Math.round((rate * FRAME_MS) / 1000),
      targetSamples: Math.round((rate * PLAYBACK_TARGET_MS) / 1000),
      maxSamples: Math.round((rate * PLAYBACK_MAX_MS) / 1000),
    },
  });
  node.port.onmessage = (e) => onWorklet(e.data);
  source?.connect(node);
  node.connect(ctx.destination);
  await ctx.resume();
  state.audio = { ctx, node, stream, rate };
  if (!playbackOnly) await devices.started(stream, ctx);
  // iOS has no background audio: the context is suspended or "interrupted" while hidden.
  document.onvisibilitychange = () => {
    if (document.visibilityState === "visible") state.audio?.ctx.resume();
  };
  postAb();
  state.seq = 0;
}

function onWorklet(msg) {
  if (msg.type === "frame") {
    if (state.replaying) return;
    devices.feed(msg.samples);
    if (state.ws?.readyState !== WebSocket.OPEN) return;
    state.ws.send(packFrame(KIND_AUDIO_IN, state.seq++, now(), floatToPcm16(msg.samples)));
    state.framesOut++;
  } else if (msg.type === "stats") {
    state.playback = msg;
  }
}

async function startVideo() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: VIDEO_W }, height: { ideal: VIDEO_H }, frameRate: { ideal: VIDEO_FPS },
      facingMode: "user",
    },
  });
  const video = $("video");
  video.srcObject = stream;
  await video.play();
  const canvas = document.createElement("canvas");
  const scale = Math.min(1, VIDEO_W / video.videoWidth, VIDEO_H / video.videoHeight);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  $("view").style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  $("view").classList.add("live");
  state.video = { stream, canvas, busy: false, clock: frameClock(1000 / VIDEO_FPS, sendVideoFrame) };
  state.videoSeq = 0;
}

function sendVideoFrame() {
  const v = state.video;
  const ws = state.ws;
  if (!v || v.busy || ws?.readyState !== WebSocket.OPEN) return;
  if (ws.bufferedAmount > VIDEO_BACKPRESSURE_BYTES) return;
  const ts = now();
  v.canvas.getContext("2d").drawImage($("video"), 0, 0, v.canvas.width, v.canvas.height);
  v.busy = true;
  v.canvas.toBlob(async (blob) => {
    v.busy = false;
    if (!blob || state.ws !== ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(packFrame(KIND_VIDEO_IN, state.videoSeq++, ts, await blob.arrayBuffer()));
  }, "image/jpeg", JPEG_QUALITY);
}

function stopMedia() {
  if (state.video) {
    state.video.clock.stop();
    state.video.stream.getTracks().forEach((t) => t.stop());
    $("video").srcObject = null;
    $("view").classList.remove("live");
    $("view").style.aspectRatio = "";
    state.video = null;
  }
  if (state.audio) {
    state.audio.stream.getTracks().forEach((t) => t.stop());
    state.audio.ctx.close();
    state.audio = null;
    devices.stopped();
  }
  state.rttMs = null;
  const btn = $("start");
  btn.setAttribute("aria-pressed", "false");
  btn.innerHTML = 'Start<span class="long"> mic and camera</span>';
  sendHello();
  render();
}


// ------------------------------------------------------------------ A/B (raw vs selective hearing)
// The worklet plays the raw mic (delayed by the engine's latency) or the engine output, crossfading
// over 100 ms. Audio keeps going to the engine either way, so lock and enrollment never reset.

function postAb() {
  state.audio?.node.port.postMessage({ type: "ab", state: state.ab, delayMs: abDelayMs(state.telemetry) });
}

function setAb(ab, notify = true) {
  state.ab = ab;
  for (const b of document.querySelectorAll("#ab button")) {
    b.setAttribute("aria-pressed", String(b.dataset.ab === ab));
  }
  postAb();
  if (notify) sendControl({ ab });
}

function toggleAb() {
  setAb(state.ab === "raw" ? "processed" : "raw");
}

/** Space toggles A/B anywhere but a text field; a focused button must not also click on keyup. */
function abKey(e) {
  if (e.key !== " " || e.altKey || e.ctrlKey || e.metaKey || e.target.closest("input, select, textarea")) return false;
  if (e.type === "keydown" && !e.repeat) toggleAb();
  e.preventDefault();
  return true;
}

// ------------------------------------------------------------------ controls

function setMode(mode, notify = true) {
  state.mode = mode;
  for (const b of document.querySelectorAll("#modes button")) {
    b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  }
  const usesSlider = SLIDER_MODES.has(mode);
  $("suppression").disabled = !usesSlider;
  $("slider-box").classList.toggle("off", !usesSlider);
  if (notify) sendControl(mode === "name_call" ? nameCallControl(state.name, state.suppression) : { mode });
  render();
}

let sliderTimer = null;
function onSlider() {
  state.suppression = Number($("suppression").value);
  paintSlider();
  if (sliderTimer) return; // at most one control message per 50 ms while dragging; the last value wins
  sliderTimer = setTimeout(() => {
    sliderTimer = null;
    sendControl({ suppression: state.suppression });
  }, 50);
}

function paintSlider() {
  $("suppression-out").textContent = String(state.suppression);
  $("suppression").style.setProperty("--fill", `${state.suppression}%`);
}

function setLatencyPreset(preset) {
  state.latencyPreset = preset;
  sendControl({ latency_preset: preset });
}

function setHqBackend(backend) {
  state.hqBackend = backend;
  sendControl({ hq_backend: backend });
}

function onKey(e) {
  if (e.altKey || e.ctrlKey || e.metaKey || e.target.closest("select, input[type=range]")) return;
  if (abKey(e)) return;
  const i = Number(e.key) - 1;
  if (i >= 0 && i < MODES.length) {
    setMode(MODES[i]);
    e.preventDefault();
  }
}

function onOverlayClick(e) {
  const rect = e.currentTarget.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const y = (e.clientY - rect.top) / rect.height;
  const hit = state.faces.find(({ box: [bx, by, bw, bh] }) => x >= bx && x <= bx + bw && y >= by && y <= by + bh);
  if (hit) sendControl({ target: { lock_track_id: hit.track_id } });
  else if (state.telemetry?.target?.locked) sendControl({ target: null }); // tap elsewhere releases the lock
}

// ------------------------------------------------------------------ rendering

function setConn(kind, text) {
  $("conn").dataset.state = kind;
  $("conn-text").textContent = text;
}

function showBanner(text, retry = true) {
  $("banner-text").textContent = text;
  $("banner-retry").hidden = !retry;
  $("banner").hidden = false;
}

/** Mic-to-ear once audio comes back (same parts as the console's bar), else the server's estimate. */
function latency(t) {
  const lat = splitLatency(t?.latency_ms);
  const a = state.audio;
  if (!t) return { ms: null, measured: false };
  if (!a || state.rttMs == null) return { ms: lat.total, measured: false };
  const network = Math.max(0, state.rttMs - lat.server);
  const playback = (1000 * state.playback.buffered) / a.rate;
  const device = 1000 * ((a.ctx.baseLatency || 0) + (a.ctx.outputLatency || 0));
  return { ms: FRAME_MS + lat.algorithmic + lat.resample + lat.server + network + playback + device, measured: true };
}

function render() {
  const t = state.telemetry;
  renderAcquisition($("acquisition"), t?.acquisition);
  calibration.render(t);
  const stale = !t || performance.now() - state.telemetryAt > TELEMETRY_STALE_MS;
  document.body.classList.toggle("stale", stale);
  const effective = t?.effective_mode || state.mode;

  // Status line: mode / tier / latency.
  const modeLabel = !t?.mode
    ? MODE_LABELS[state.mode]
    : t.mode === "auto" ? `Auto → ${MODE_LABELS[effective] || effective}` : MODE_LABELS[t.mode] || t.mode;
  $("st-mode").textContent = modeLabel;
  const tier = tierInfo(t?.tier);
  $("st-tier").textContent = tier.label;
  if (tier.av == null) delete $("st-tier").dataset.av;
  else $("st-tier").dataset.av = String(tier.av);
  const lat = latency(t);
  $("st-latency").textContent = fmt(lat.ms);
  $("st-latency-kind").textContent = !t ? "no engine" : lat.measured ? "mic to ear" : "server estimate";
  const status = $("status");
  status.dataset.level = lat.ms == null || effective !== "natural" ? ""
    : lat.ms <= NATURAL_AIM_MS ? "ok" : lat.ms <= 2 * NATURAL_AIM_MS ? "warn" : "slow";
  status.title = t?.pipeline ? `pipeline ${t.pipeline} on ${t.device}` : "";
  $("auto-hint").textContent = t?.mode === "auto" ? `Now: ${MODE_LABELS[effective] || effective}` : "Picks a mode for the room";

  // Captions and voice only where they mean something.
  $("captions-box").hidden = !CAPTION_MODES.has(effective);
  $("voice-box").hidden = effective !== "synthetic";
  renderVoice(t);
  renderHqBackend(t);
  renderLatencyPreset(t);
  drawFaces(t);
  $("namecall-box").hidden = effective !== "name_call";
  if (effective === "name_call") {
    const pill = nameCallPill(t, state.name.split(",")[0]);
    $("namecall-pill").dataset.state = pill.state;
    $("namecall-pill").textContent = pill.text;
  }
}

function renderLatencyPreset(t) {
  const picked = t?.latency_preset || state.latencyPreset;
  for (const input of document.querySelectorAll("input[name=latency-preset]")) input.checked = input.value === picked;
}

function renderHqBackend(t) {
  const info = hqBackendInfo(t);
  const picked = state.hqBackend || info.requested;
  for (const input of document.querySelectorAll("input[name=hq-backend]")) input.checked = input.value === picked;
  const out = $("hq-active");
  out.textContent = info.requested ? `Playing: ${info.label}` : "—";
  if (info.reason) out.title = info.reason;
  out.dataset.active = info.active || "";
  out.dataset.fallback = String(info.fallback);
}

function renderVoice(t) {
  const v = t?.voice;
  if (!v) return;
  const bucket = VOICE_LABELS[v.bucket] || v.bucket.replace(/_/g, " ");
  $("voice-chip").textContent = `Voice: ${bucket} (${SOURCE_LABELS[v.source] || v.source})`;
}

function drawFaces(t) {
  const canvas = $("overlay");
  const view = $("view");
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(view.clientWidth * dpr);
  const h = Math.round(view.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const g = canvas.getContext("2d");
  g.clearRect(0, 0, w, h);
  if (!state.video) return; // boxes only make sense over this camera's picture
  const targetId = t?.target?.track_id;
  const targetColor = getComputedStyle(document.documentElement).getPropertyValue("--target");
  g.font = `600 ${Math.round(13 * dpr)}px system-ui, sans-serif`;
  g.textBaseline = "bottom";
  for (const face of state.faces) {
    const [x, y, bw, bh] = face.box;
    const isTarget = face.track_id === targetId;
    g.strokeStyle = isTarget ? targetColor : "rgba(232, 236, 241, 0.6)";
    g.lineWidth = (isTarget ? 3 : 1.5) * dpr;
    g.strokeRect(x * w, y * h, bw * w, bh * h);
    if (!isTarget) continue;
    const label = `TARGET${t.target.locked ? " (locked)" : ""}`;
    const pad = 5 * dpr;
    const th = 20 * dpr;
    const ly = Math.max(th, y * h);
    g.fillStyle = targetColor;
    g.fillRect(x * w - g.lineWidth / 2, ly - th, g.measureText(label).width + 2 * pad, th);
    g.fillStyle = "#0f1216";
    g.fillText(label, x * w + pad, ly - 4 * dpr);
  }
}

let partial = null;
function addCaption(c) {
  const log = $("captions");
  log.querySelector(".empty")?.remove();
  if (!partial) {
    partial = document.createElement("p");
    log.append(partial);
  }
  partial.textContent = c.text;
  partial.classList.toggle("partial", !c.final);
  if (c.final) partial = null;
  while (log.children.length > 30) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
}

// ------------------------------------------------------------------ wiring

function init() {
  $("modes").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-mode]");
    if (b) setMode(b.dataset.mode);
  });
  $("suppression").addEventListener("input", onSlider);
  $("ab").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-ab]");
    if (b) setAb(b.dataset.ab);
  });
  $("hq-backend").addEventListener("change", (e) => setHqBackend(e.target.value));
  $("latency-preset").addEventListener("change", (e) => setLatencyPreset(e.target.value));
  $("voice-override").addEventListener("change", (e) => {
    state.voiceOverride = e.target.value || null;
    sendControl({ voice_override: state.voiceOverride });
  });
  $("start").addEventListener("click", () => (state.audio ? stopMedia() : startMedia()));
  $("namecall-name").value = state.name;
  $("namecall-name").addEventListener("change", (e) => {
    state.name = e.target.value.trim() || "Aryan";
    saveName(state.name);
    if (state.mode === "name_call") sendControl(nameCallControl(state.name, state.suppression));
  });
  mountNameGuards($("namecall-match"), $("namecall-exclude"), () => {
    if (state.mode === "name_call") sendControl(nameCallControl(state.name, state.suppression));
  });
  $("namecall-release").addEventListener("click", () => sendControl({ name_call: "release" }));
  $("banner-retry").addEventListener("click", () => {
    state.attempt = 0;
    connect();
  });
  $("headphones-ok").addEventListener("click", () => ($("headphones").hidden = true));
  $("overlay").addEventListener("click", onOverlayClick);
  document.addEventListener("keydown", onKey);
  document.addEventListener("keyup", (e) => { if (e.key === " " && !e.target.closest("input, select, textarea")) e.preventDefault(); });

  paintSlider();
  setMode(state.mode, false);
  setInterval(render, RENDER_MS);
  connect();
}

const replay = replayControls(state, {
  startAudio, stopMedia, error: (message) => showBanner(message, false),
});
init();
window.hearingSimple = state; // for the headless check and for poking at in devtools
