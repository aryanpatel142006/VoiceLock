import { showServerInfo } from './server-info.js';
import { replayControls } from './replay.js';
import { renderAcquisition } from "./acquisition.js";
import { mountCalibration } from "./calibrate.js";

// Debug console for /ws/session: mic + camera up, processed audio down, telemetry on screen.
// Query parameters: ?ws=<url> picks the session socket (default: /ws/session on this host).

import {
  KIND_AUDIO_IN, KIND_AUDIO_OUT, KIND_VIDEO_IN, MODES, SLIDER_MODES,
  abDelayMs, control, dbfs, floatToPcm16, headroomInfo, hello, hqBackendInfo, ladderInfo, packFrame, pcm16ToFloat, sessionUrl,
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
const VIDEO_BACKPRESSURE_BYTES = 256 * 1024; // skip camera frames while the socket is backed up
const TELEMETRY_STALE_MS = 1500;
const RETRY_MS = [1000, 2000, 4000, 8000];
const BAR_MAX_MS = 150;
const RENDER_MS = 100; // telemetry arrives at ~10 Hz; redrawing faster only burns CPU
const TOAST_MS = 12000; // a recommendation the user ignores goes away on its own
const MODE_LABELS = {
  natural: "Natural", focus_hq: "Focus HQ", synthetic: "Synthetic",
  text: "Text", quiet: "Quiet", auto: "Auto",
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
const fmt = (ms, digits = 1) => (ms == null || Number.isNaN(ms) ? "—" : ms.toFixed(digits));

const state = {
  attention: null, // glance | normal | sticky once picked; null = server default
  ab: "processed", // what the headphones play: "raw" mic or the engine's "processed" output
  url: sessionUrl(window.location),
  ws: null,
  attempt: 0,
  retryTimer: null,
  everConnected: false,
  mode: "natural",
  beforeFocus: "natural", // the mode the Focus HQ toggle returns to
  suppression: 50,
  voiceOverride: null,
  latencyPreset: "balanced", // Focus HQ delay/quality: speed | balanced | accuracy (independent of mode)
  hqBackend: null, // where Focus HQ runs: null until the user picks (the server's default applies)
  telemetry: null,
  telemetryAt: 0,
  faces: [],
  audio: null, // {ctx, node, stream, source, rate, frameSamples}
  video: null, // {stream, timer, canvas, busy}
  seq: 0,
  videoSeq: 0,
  framesOut: 0,
  framesBack: 0,
  rttMs: null,
  playback: { buffered: 0, underruns: 0, dropped: 0 },
  levelIn: -90,
  levelOut: -90,
  errors: [],
  recommendation: null,
};
const calibration = mountCalibration($("acquisition"), (fields) => sendControl(fields));

// ------------------------------------------------------------------ socket

function connect() {
  clearTimeout(state.retryTimer);
  if (state.ws) {
    state.ws.onclose = state.ws.onerror = state.ws.onmessage = null;
    state.ws.close();
  }
  setConn("connecting", `Connecting to ${state.url}…`);
  let ws;
  try {
    ws = new WebSocket(state.url + (state.url.includes("?") ? "&" : "?") + "replay=1");
  } catch (e) {
    connectionLost(`${e.message}`);
    return;
  }
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  ws.onopen = () => {
    state.attempt = 0;
    state.everConnected = true;
    setConn("open", "Connected");
    hideBanner();
    sendHello();
    send(control({
      ab: state.ab,
      mode: state.mode, suppression: state.suppression, voice_override: state.voiceOverride,
      ...(state.attention ? { attention: { ...ATTENTION_PRESETS[state.attention], preset: state.attention } } : {}),
      ...(state.hqBackend ? { hq_backend: state.hqBackend } : {}),
      latency_preset: state.latencyPreset,
      ...calibration.initialFields(),
    }));
  };
  ws.onmessage = (e) => (typeof e.data === "string" ? onText(e.data) : onBinary(e.data));
  ws.onclose = (e) => {
    if (state.ws !== ws) return;
    const why = e.code === 1006 || !e.code ? "no answer" : `closed with code ${e.code}${e.reason ? `: ${e.reason}` : ""}`;
    connectionLost(why);
  };
}

function connectionLost(why) {
  replay.disconnect();
  state.ws = null;
  state.telemetry = null;
  const delay = RETRY_MS[Math.min(state.attempt, RETRY_MS.length - 1)];
  state.attempt++;
  const lead = state.everConnected ? "Lost the session server" : "Can't reach the session server";
  setConn("error", lead);
  showBanner(
    `${lead} at ${state.url} (${why}). Check that \`hearing serve\` is running and the address is right. ` +
    `Retrying in ${Math.round(delay / 1000)} s.`,
  );
  state.retryTimer = setTimeout(connect, delay);
  render();
}

function send(obj) {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(obj));
}

function sendHello() {
  if (!state.audio) {
    send(hello(16000, FRAME_MS, false));
    return;
  }
  send(hello(state.audio.rate, FRAME_MS, Boolean(state.video)));
}

function onText(text) {
  let msg;
  try {
    msg = JSON.parse(text);
  } catch {
    return;
  }
  replay.text(msg);
  switch (msg.type) {
    case "telemetry":
      state.telemetry = msg;
      showServerInfo(msg);
      postAb();
      state.telemetryAt = performance.now();
      state.faces = msg.faces || [];
      break;
    case "caption":
      addCaption(msg);
      break;
    case "context":
      renderContext(msg);
      break;
    case "recommendation":
      showRecommendation(msg);
      break;
    case "error":
      addError(msg);
      break;
    default:
      break;
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
  state.levelOut = Math.max(dbfs(samples), state.levelOut - 3);
  if (state.audio && $("play").checked) {
    state.audio.node.port.postMessage({ type: "play", samples }, [samples.buffer]);
  }
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
    stream = playbackOnly ? new MediaStream() : await navigator.mediaDevices.getUserMedia({
      // The pipeline wants the raw room: browser processing would fight the enhancement stages.
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (e) {
    ctx.close();
    throw e;
  }
  let source;
  try {
    if (!playbackOnly) source = ctx.createMediaStreamSource(stream);
  } catch {
    // Firefox cannot connect a mic to a context at another rate; send the native rate and
    // let the server's stream resampler convert (hello carries the rate).
    await ctx.close();
    ctx = new AudioContext({ latencyHint: "interactive" });
    if (!playbackOnly) source = ctx.createMediaStreamSource(stream);
  }
  await ctx.audioWorklet.addModule("audio-worklet.js");
  const rate = ctx.sampleRate;
  const frameSamples = Math.round((rate * FRAME_MS) / 1000);
  const node = new AudioWorkletNode(ctx, "hearing-io", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    processorOptions: {
      frameSamples,
      targetSamples: Math.round((rate * PLAYBACK_TARGET_MS) / 1000),
      maxSamples: Math.round((rate * PLAYBACK_MAX_MS) / 1000),
    },
  });
  node.port.onmessage = (e) => onWorklet(e.data);
  source?.connect(node);
  node.connect(ctx.destination);
  await ctx.resume();
  state.audio = { ctx, node, stream, source, rate, frameSamples };
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
    const samples = msg.samples;
    state.levelIn = Math.max(dbfs(samples), state.levelIn - 3);
    if (state.ws?.readyState !== WebSocket.OPEN) return;
    state.ws.send(packFrame(KIND_AUDIO_IN, state.seq++, now(), floatToPcm16(samples)));
    state.framesOut++;
  } else if (msg.type === "stats") {
    state.playback = msg;
  }
}

async function startVideo() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: VIDEO_W }, height: { ideal: VIDEO_H }, frameRate: { ideal: VIDEO_FPS } },
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
  state.video = { stream, canvas, busy: false, timer: setInterval(sendVideoFrame, 1000 / VIDEO_FPS) };
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
    clearInterval(state.video.timer);
    state.video.stream.getTracks().forEach((t) => t.stop());
    $("video").srcObject = null;
    $("view").classList.remove("live");
    state.video = null;
  }
  if (state.audio) {
    state.audio.stream.getTracks().forEach((t) => t.stop());
    state.audio.ctx.close();
    state.audio = null;
  }
  state.rttMs = null;
  const btn = $("start");
  btn.setAttribute("aria-pressed", "false");
  btn.textContent = "Start mic and camera";
  $("camera-placeholder").textContent = "Camera starts with the microphone.";
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

function sendControl(fields) {
  send(control(fields));
}

// ------------------------------------------------------------------ controls

function buildModes() {
  const grid = $("modes");
  MODES.forEach((mode, i) => {
    const label = document.createElement("label");
    label.className = "mode";
    label.innerHTML = `<input type="radio" name="mode" value="${mode}"><span>${MODE_LABELS[mode]}</span><kbd>${i + 1}</kbd>`;
    grid.append(label);
  });
  grid.addEventListener("change", (e) => setMode(e.target.value));
  setMode(state.mode, false);
}

function setMode(mode, notify = true) {
  state.mode = mode;
  const input = document.querySelector(`input[name=mode][value=${mode}]`);
  if (input) input.checked = true;
  const usesSlider = SLIDER_MODES.has(mode);
  $("suppression").disabled = !usesSlider;
  $("slider-box").classList.toggle("off", !usesSlider);
  $("suppression-hint").textContent = usesSlider
    ? "0 leaves the room untouched, 100 keeps only the target."
    : `${MODE_LABELS[mode]} mode plays no room audio, so suppression does not apply.`;
  if (mode !== "focus_hq") state.beforeFocus = mode;
  $("focus-hq").setAttribute("aria-pressed", String(mode === "focus_hq"));
  if (notify) send(control({ mode }));
}

const ATTENTION_PRESETS = {
  glance: { dwell_s: 0.3, hold_s: 1.0, switch_margin: 0.02 },
  normal: { dwell_s: 0.5, hold_s: 2.0, switch_margin: 0.04 },
  sticky: { dwell_s: 1.0, hold_s: 4.0, switch_margin: 0.1 },
};

function setAttention(preset, notify = true) {
  state.attention = preset;
  for (const b of document.querySelectorAll("#attention button")) {
    b.setAttribute("aria-pressed", String(b.dataset.attention === preset));
  }
  if (notify) send(control({ attention: { ...ATTENTION_PRESETS[preset], preset } }));
}

function setLatencyPreset(preset) {
  state.latencyPreset = preset;
  send(control({ latency_preset: preset }));
}

function setHqBackend(backend) {
  state.hqBackend = backend;
  send(control({ hq_backend: backend }));
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
  out.title = info.reason;
  out.dataset.active = info.active || "";
  out.dataset.fallback = String(info.fallback);
  return info;
}

function toggleFocus() {
  setMode(state.mode === "focus_hq" ? state.beforeFocus || "natural" : "focus_hq");
}

let sliderTimer = null;
function onSlider() {
  state.suppression = Number($("suppression").value);
  $("suppression-out").textContent = String(state.suppression);
  if (sliderTimer) return; // at most one control message per 50 ms while dragging
  sliderTimer = setTimeout(() => {
    sliderTimer = null;
    send(control({ suppression: state.suppression }));
  }, 50);
}

function onKey(e) {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.target.closest("input[type=url], input[type=text], select")) return;
  if (abKey(e)) return;
  if (e.key === "f" || e.key === "F") {
    toggleFocus();
    e.preventDefault();
    return;
  }
  if (e.key === "c" || e.key === "C") {
    calibration.start();
    e.preventDefault();
    return;
  }
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
  if (hit) send(control({ target: { lock_track_id: hit.track_id } }));
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

function hideBanner() {
  $("banner").hidden = true;
}

/** Bar segments in latency order; client parts only once audio is flowing. */
function latencyParts(t) {
  const lat = splitLatency(t?.latency_ms);
  const parts = [];
  const audio = state.audio;
  if (audio) parts.push(["capture", "Mic frame", FRAME_MS]);
  parts.push(["algorithmic", "Stage lookahead", lat.algorithmic]);
  if (lat.resample) parts.push(["resample", "Resampling", lat.resample]);
  parts.push(["server", "Server compute", lat.server]);
  if (audio && state.rttMs != null) {
    parts.push(["network", "Socket + queues", Math.max(0, state.rttMs - lat.server)]);
    parts.push(["playback", "Playback buffer", (1000 * state.playback.buffered) / audio.rate]);
    const device = 1000 * ((audio.ctx.baseLatency || 0) + (audio.ctx.outputLatency || 0));
    if (device > 0) parts.push(["device", "Audio device", device]);
  }
  return { parts, lat };
}

function render() {
  const t = state.telemetry;
  const stale = !t || performance.now() - state.telemetryAt > TELEMETRY_STALE_MS;
  document.body.classList.toggle("stale", stale);

  const { parts, lat } = latencyParts(t);
  const measured = state.audio && state.rttMs != null;
  const total = !t ? null : measured ? parts.reduce((s, p) => s + p[2], 0) : lat.total;
  $("total").textContent = fmt(total, 0);
  $("total-label").textContent = !t
    ? "No telemetry: not connected"
    : measured
      ? "Mouth to ear, measured in this browser"
      : "Server estimate. Start the mic to measure the full round trip.";
  $("budget-bar").classList.toggle("over", total != null && total > 100);
  renderBudget(t ? parts : []);

  renderTier(t, lat);
  $("device").textContent = t?.device || "—";
  $("server-mode").textContent = !t?.mode
    ? "—"
    : t.mode === "auto" && t.effective_mode
      ? `Auto → ${MODE_LABELS[t.effective_mode] || t.effective_mode}`
      : MODE_LABELS[t.mode] || t.mode;
  $("pipeline").textContent = t?.pipeline || "—";
  $("stages").textContent = lat.stages.length
    ? lat.stages.map(([k, v]) => `${k} ${fmt(v, 2)} ms`).join(", ")
    : "—";
  $("audio-rate").textContent = state.audio
    ? `${state.audio.rate / 1000} kHz, ${FRAME_MS} ms frames${state.video ? ", camera on" : ""}`
    : "off";
  $("frames").textContent = `${state.framesOut} / ${state.framesBack}`;
  const droppedMs = state.audio ? (1000 * state.playback.dropped) / state.audio.rate : 0;
  $("glitches").textContent = `${state.playback.underruns} / ${fmt(droppedMs, 0)} ms`;

  $("level-in").value = state.levelIn;
  $("level-out").value = state.levelOut;
  state.levelIn = Math.max(-90, state.levelIn - 6); // meters fall back between frames
  state.levelOut = Math.max(-90, state.levelOut - 6);

  renderTarget(t);
  renderVoice(t);
  drawFaces(t);
}

function renderBudget(parts) {
  const seg = $("segments");
  const legend = $("legend");
  seg.replaceChildren(...parts.map(([key, , ms]) => {
    const el = document.createElement("span");
    el.className = `seg seg-${key}`;
    el.style.width = `${(100 * Math.max(0, ms)) / BAR_MAX_MS}%`;
    return el;
  }));
  legend.replaceChildren(...parts.map(([key, label, ms]) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="swatch seg-${key}"></span>${label} <b>${fmt(ms)}</b>`;
    return li;
  }));
}

function renderTier(t, lat) {
  renderAcquisition($("acquisition"), t?.acquisition);
  calibration.render(t);
  const tier = tierInfo(t?.tier);
  for (const el of [$("tier"), $("view-tier")]) {
    el.textContent = tier.label;
    if (tier.av == null) delete el.dataset.av;
    else el.dataset.av = String(tier.av);
  }
  $("view-tier").hidden = tier.av == null || !state.video;
  renderLadder(t);
  const v = t?.vision;
  let text = "—";
  if (v?.status === "live") {
    text = `${v.detector} + ${v.asd} · ${fmt(v.fps, 0)} fps · ${fmt(lat.vision)} ms/frame` +
      (v.dropped ? ` · ${v.dropped} dropped` : "");
  } else if (v?.status === "unavailable") {
    text = `unavailable: ${v.reason || "no face detector"}. Audio-only.`;
  } else if (v) {
    text = state.video ? "waiting for camera frames" : "camera off: audio-only";
  }
  $("vision").textContent = text;
  const room = headroomInfo(t?.tiers);
  $("headroom").textContent = room.text + (room.low ? " · low: close other apps" : "");
}

const CANDIDATE_LABELS = { av_mossformer2: "AV-MossFormer2", dolphin: "Dolphin", enrolled: "ECAPA + MossFormer2" };

function renderLadder(t) {
  const info = ladderInfo(t?.tiers);
  const list = $("ladder");
  if (!info.rows.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = t?.mode === "focus_hq" ? "Waiting for the ladder…" : "Off. Turn on Focus HQ to see the tiers.";
    list.replaceChildren(li);
  } else {
    list.replaceChildren(...info.rows.map((r) => {
      const li = document.createElement("li");
      li.dataset.state = r.active ? "active" : r.eligible ? "ready" : "blocked";
      li.title = r.reason || (r.eligible ? "ready" : "");
      li.innerHTML = `<b></b><span></span>`;
      li.querySelector("b").textContent = r.label;
      li.querySelector("span").textContent = r.active ? "playing" : r.eligible ? "ready" : r.reason || "not ready";
      return li;
    }));
  }
  const hint = $("focus-hint");
  const tr = t?.transition;
  const hq = renderHqBackend(t);
  renderLatencyPreset(t);
  if (tr && tr.priming_ms > 0) {
    hint.textContent = `Warming up: ${fmt(tr.priming_ms / 1000, 1)} s until the switch. The previous mode keeps playing.`;
  } else if (t?.mode === "focus_hq" && hq.active === "remote") {
    hint.textContent = `Playing on the remote GPU${hq.rttMs != null ? `, ${fmt(hq.rttMs, 0)} ms round trip` : ""}. ` +
      "Falls back to this laptop if it stops answering.";
  } else if (hq.fallback && t?.mode === "focus_hq") {
    hint.textContent = `GPU unavailable, playing locally: ${hq.reason}.`;
  } else if (info.rows.length) {
    const active = info.rows.find((r) => r.active);
    const hq = info.hq;
    const model = hq ? `${CANDIDATE_LABELS[hq.candidate] || hq.candidate}${hq.mock ? " (mock)" : ""}` +
      (hq.device ? ` on ${hq.device}` : "") +
      (hq.compute_p50_ms != null ? `, ${fmt(hq.compute_p50_ms, 0)} ms per window` : "") : "no model";
    hint.textContent = `Playing ${active?.label || "—"}, ${fmt(info.delayMs / 1000, 1)} s behind. ${model}.` +
      (active?.name !== "focus_hq" && info.rows[0]?.reason ? ` Focus HQ: ${info.rows[0].reason}.` : "");
  } else {
    hint.textContent = "Pulls the target's voice out of the room with the camera. Adds about 1.5 s of delay.";
  }
}

function renderTarget(t) {
  const target = t?.target;
  const unlock = $("unlock");
  if (!target || target.track_id == null) {
    $("target").textContent = state.faces.length
      ? "No target. Click a face box to lock it."
      : "No faces tracked yet.";
    unlock.hidden = true;
    return;
  }
  $("target").textContent =
    `Target: face ${target.track_id}, confidence ${fmt(target.confidence * 100, 0)}%` +
    (target.locked ? ", locked by you" : ", picked automatically");
  unlock.hidden = !target.locked;
}

function renderVoice(t) {
  const v = t?.voice;
  if (!v) return;
  const bucket = VOICE_LABELS[v.bucket] || v.bucket.replace(/_/g, " ");
  const source = SOURCE_LABELS[v.source] || v.source;
  $("voice-chip").textContent = `Voice: ${bucket} (${source})`;
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
  const targetId = t?.target?.track_id;
  const css = getComputedStyle(document.documentElement);
  g.font = `${Math.round(14 * dpr)}px system-ui, sans-serif`;
  g.textBaseline = "bottom";
  for (const face of state.faces) {
    const [x, y, bw, bh] = face.box;
    const isTarget = face.track_id === targetId;
    const color = isTarget ? css.getPropertyValue("--target") : "rgba(232, 236, 241, 0.7)";
    g.strokeStyle = color;
    g.lineWidth = (isTarget ? 3 : 1.5) * dpr;
    g.strokeRect(x * w, y * h, bw * w, bh * h);
    const label = `${isTarget ? "TARGET " : ""}${face.track_id} · speaking ${fmt(face.p_speaking * 100, 0)}%`;
    const pad = 4 * dpr;
    const tw = g.measureText(label).width + 2 * pad;
    const th = 20 * dpr;
    const ly = Math.max(th, y * h);
    g.fillStyle = isTarget ? color : "rgba(15, 18, 22, 0.8)";
    g.fillRect(x * w, ly - th, tw, th);
    g.fillStyle = isTarget ? "#0f1216" : "#e8ecf1";
    g.fillText(label, x * w + pad, ly - 4 * dpr);
  }
}

function renderContext(c) {
  $("noise").textContent = c.noise_db == null ? "—" : `${fmt(c.noise_db)} dB`;
  $("snr").textContent = c.snr_db == null ? "—" : `${fmt(c.snr_db)} dB`;
  $("speakers").textContent = c.speakers ?? "—";
  $("overlap").textContent = c.overlap == null ? "—" : `${fmt(c.overlap * 100, 0)}%`;
  $("target-conf").textContent = c.target_conf == null ? "—" : `${fmt(c.target_conf * 100, 0)}%`;
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
  while (log.children.length > 50) log.firstElementChild.remove();
  log.scrollTop = log.scrollHeight;
}

function addError(e) {
  const list = $("errors");
  list.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  const time = new Date().toLocaleTimeString();
  const degraded = e.degraded_to ? ` Degraded to ${e.degraded_to}.` : "";
  li.innerHTML = `<time></time> <b></b> <span></span>`;
  li.querySelector("time").textContent = time;
  li.querySelector("b").textContent = e.stage || "server";
  li.querySelector("span").textContent = `${e.message}${degraded}`;
  list.prepend(li);
  while (list.children.length > 8) list.lastElementChild.remove();
}

let toastTimer = null;
function showRecommendation(r) {
  state.recommendation = r;
  const label = MODE_LABELS[r.mode] || r.mode;
  const sure = `${fmt(r.confidence * 100, 0)}% sure`;
  // Auto already switched: just say so. Otherwise offer the switch (accepting it is logged
  // by the server as a choice, which is how the preference model learns).
  $("toast-text").textContent = r.auto
    ? `Auto switched to ${label}: ${r.reason} (${sure})`
    : `Try ${label}: ${r.reason} (${sure})`;
  $("toast-accept").textContent = `Switch to ${label}`;
  $("toast-accept").hidden = Boolean(r.auto);
  $("toast-dismiss").textContent = r.auto ? "OK" : "Dismiss";
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("toast").hidden = true), TOAST_MS);
}

// ------------------------------------------------------------------ wiring

function init() {
  buildModes();
  $("focus-hq").addEventListener("click", toggleFocus);
  $("hq-backend").addEventListener("change", (e) => setHqBackend(e.target.value));
  $("latency-preset").addEventListener("change", (e) => setLatencyPreset(e.target.value));
  $("ws-url").value = state.url;
  $("server-form").addEventListener("submit", (e) => {
    e.preventDefault();
    state.url = $("ws-url").value.trim() || sessionUrl(window.location);
    const params = new URLSearchParams(window.location.search);
    params.set("ws", state.url);
    history.replaceState(null, "", `?${params}`);
    state.attempt = 0;
    state.everConnected = false;
    connect();
  });
  $("banner-retry").addEventListener("click", () => {
    state.attempt = 0;
    connect();
  });
  $("start").addEventListener("click", () => (state.audio ? stopMedia() : startMedia()));
  $("suppression").addEventListener("input", onSlider);
  $("voice-override").addEventListener("change", (e) => {
    state.voiceOverride = e.target.value || null;
    send(control({ voice_override: state.voiceOverride }));
  });
  $("play").addEventListener("change", (e) => {
    if (!e.target.checked) state.audio?.node.port.postMessage({ type: "flush" });
  });
  $("unlock").addEventListener("click", () => send(control({ target: null })));
  $("overlay").addEventListener("click", onOverlayClick);
  $("toast-accept").addEventListener("click", () => {
    if (state.recommendation) setMode(state.recommendation.mode);
    $("toast").hidden = true;
  });
  $("toast-dismiss").addEventListener("click", () => ($("toast").hidden = true));
  document.addEventListener("keydown", onKey);
  document.addEventListener("keyup", (e) => { if (e.key === " " && !e.target.closest("input[type=url], input[type=text], select")) e.preventDefault(); });
  $("attention").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-attention]");
    if (b) setAttention(b.dataset.attention);
  });
  $("ab").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-ab]");
    if (b) setAb(b.dataset.ab);
  });

  setInterval(render, RENDER_MS);
  connect();
}

const replay = replayControls(state, {
  startAudio, stopMedia, error: (message) => showBanner(message, false),
});
init();
window.hearingConsole = state; // for the headless check and for poking at in devtools
