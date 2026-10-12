// Who's speaking (/who): the demo mic -> /ws/who -> a calm "Aryan" / "Joseph" / "No one speaking" label.
// Nothing is played back: the worklet's output goes through a gain of 0. The mic is the one the judge page saved
// (localStorage hearing.mic, the iPhone over USB on the demo), or ?mic=<label part>, else the default.
// ?autostart=1 starts without a click (the screenshot run). Space toggles Start / Stop.
// Live voice: an AnalyserNode on the same mic input draws a calm waveform; each person's match meter eases toward the
// message's per-profile match (server `match`, else mapped from raw.cos) and their voiceprint bars move with the mic
// only while they hold the label. The Mic dropdown picks this page's own mic (localStorage hearing.who.mic, falling
// back to the app's hearing.mic); changing it restarts capture and reconnects /ws/who.

import { MIC_KEY, MIC_LABEL_KEY, loadChoice, openMic, saveChoice } from "./devices.js";
import { KIND_AUDIO_IN, audioContextOptions, floatToPcm16, packFrame } from "./protocol.js";
import { CAPTION, MATCH_CAPTION, Timeline, WHO_MIC_KEY, confidenceOf, displayName, ease, labelFor, matchTargets, matchText, micOptions,
  pickMicId, statusLine, waveformPoints } from "./who-view.js";

const FRAME_MS = 20;
const SPAN_S = 60;
const VP_BARS = 20; // voiceprint bars per person
const WAVE_POINTS = 96;
const REDUCED = window.matchMedia?.("(prefers-reduced-motion: reduce)");
const $ = (id) => document.getElementById(id);
const state = { ws: null, audio: null, seq: 0, names: ["aryan", "joseph"], last: null, lastT: 0, lastAt: 0, ready: false,
  mic: "", micMissing: false, sessionsActive: 0, compute: null, device: "", timeline: new Timeline(SPAN_S), running: false,
  micId: pickMicId([], loadChoice(WHO_MIC_KEY), loadChoice(MIC_KEY)), analyser: null,
  match: { target: [0, 0], shown: [0, 0], pct: [-1, -1] }, wave: new Float32Array(WAVE_POINTS), vp: [new Float32Array(VP_BARS), new Float32Array(VP_BARS)] };

function whoUrl() {
  const explicit = new URLSearchParams(location.search).get("ws");
  if (explicit) return explicit;
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${location.host || "127.0.0.1:8000"}/ws/who`;
}

function render(msg) {
  const label = labelFor(msg, state.names);
  $("stage").dataset.cls = label.cls;
  if ($("big").textContent !== label.big) $("big").textContent = label.big;
  $("sub").textContent = label.sub;
  $("conf").style.width = `${Math.round(100 * confidenceOf(msg))}%`;
  $("status").textContent = statusLine(state);
}

function drawTimeline() {
  const canvas = $("timeline");
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.round(36 * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const now = state.running ? state.lastT + (performance.now() - state.lastAt) / 1000 : state.lastT;
  const colours = { [state.names[0]]: getComputedStyle(document.documentElement).getPropertyValue("--p0").trim() || "#4fa3ff",
    [state.names[1]]: getComputedStyle(document.documentElement).getPropertyValue("--p1").trim() || "#ffb84f" };
  for (const b of state.timeline.bars(now)) {
    ctx.fillStyle = colours[b.speaker] || "#888";
    ctx.fillRect(Math.round(b.x * w), Math.round(0.2 * h), Math.max(1, Math.round(b.w * w)), Math.round(0.6 * h));
  }
}

function onMessage(ev) {
  let msg;
  try { msg = JSON.parse(ev.data); } catch { return; }
  if (msg.type === "who_ready") {
    state.ready = true;
    state.names = msg.names || state.names;
    nameCards();
    state.device = msg.device || "";
    state.sessionsActive = msg.sessions_active || 0;
    $("notice").hidden = !(state.sessionsActive > 0);
    $("notice").textContent = state.sessionsActive > 0
      ? `A Listen session is still open on this server (${state.sessionsActive}). Stop it on the other tab first: both would share the Mac's GPU.`
      : "";
    render({ type: "who", speaker: null, state: "listening" });
    return;
  }
  if (msg.type === "who_status") { render(msg); return; }
  if (msg.type !== "who") return;
  if (typeof msg.compute_ms === "number") state.compute = msg.compute_ms;
  state.last = msg;
  state.lastT = msg.t || state.lastT;
  state.lastAt = performance.now();
  state.timeline.add(state.lastT, msg.speaker || null);
  const targets = matchTargets(msg, state.names);
  if (targets) state.match.target = state.names.map((n) => targets[n] ?? 0);
  render(msg);
}

async function startAudio() {
  const want = loadChoice(WHO_MIC_KEY) ? "" : new URLSearchParams(location.search).get("mic"); // a pick here beats ?mic
  let { stream, missing } = await openMic(want ? "" : state.micId);
  if (want) { // pick by label once labels are readable
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
    const hit = devices.find((d) => (d.label || "").toLowerCase().includes(want.toLowerCase()));
    if (hit) { stream.getTracks().forEach((t) => t.stop()); ({ stream, missing } = await openMic(hit.deviceId)); }
  }
  state.mic = stream.getAudioTracks()[0]?.label || loadChoice(MIC_LABEL_KEY) || "default mic";
  state.micMissing = missing;
  const opened = stream.getAudioTracks()[0]?.getSettings?.().deviceId;
  if (opened && !missing && opened !== "default") state.micId = opened;
  refreshMics(); // labels are readable now that the mic is open
  let ctx = new AudioContext(audioContextOptions());
  let source;
  try { source = ctx.createMediaStreamSource(stream); } catch {
    await ctx.close();
    ctx = new AudioContext({ latencyHint: "interactive" });
    source = ctx.createMediaStreamSource(stream);
  }
  await ctx.audioWorklet.addModule("audio-worklet.js");
  const rate = ctx.sampleRate;
  const node = new AudioWorkletNode(ctx, "hearing-io", {
    numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
    processorOptions: { frameSamples: Math.round((rate * FRAME_MS) / 1000), targetSamples: rate, maxSamples: 2 * rate },
  });
  node.port.onmessage = (e) => {
    if (e.data.type !== "frame" || state.ws?.readyState !== WebSocket.OPEN) return;
    state.ws.send(packFrame(KIND_AUDIO_IN, state.seq++, Date.now() / 1000, floatToPcm16(e.data.samples)));
  };
  const mute = ctx.createGain();
  mute.gain.value = 0; // never any audio out, whatever the worklet's playback ring holds (it holds nothing)
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.8;
  source.connect(node);
  source.connect(analyser);
  node.connect(mute);
  analyser.connect(mute); // pulled by the graph, still silent
  mute.connect(ctx.destination);
  await ctx.resume();
  state.analyser = { node: analyser, time: new Float32Array(analyser.fftSize), freq: new Uint8Array(analyser.frequencyBinCount), rate };
  state.audio = { ctx, node, stream, rate, source };
  return rate;
}

async function start() {
  if (state.running) return;
  state.running = true;
  $("go").textContent = "Stop";
  $("go").dataset.on = "true";
  render({ type: "who_status", state: "loading" });
  try {
    const ws = new WebSocket(whoUrl());
    state.ws = ws;
    ws.onmessage = onMessage;
    ws.onclose = () => { if (state.running && state.ws === ws) stop("connection closed"); }; // not a mic switch's old socket
    await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = () => reject(new Error("cannot reach /ws/who")); });
    const rate = await startAudio();
    ws.send(JSON.stringify({ type: "hello", sample_rate: rate, frame_ms: FRAME_MS, video: false, client: "who", input_device: state.mic }));
    state.seq = 0;
    state.lastAt = performance.now();
    render(state.last || { type: "who", speaker: null, state: "listening" });
  } catch (e) {
    stop(e?.message || String(e));
  }
}

function stop(reason) {
  state.running = false;
  $("go").textContent = "Start";
  $("go").dataset.on = "false";
  state.ws?.close();
  state.ws = null;
  state.audio?.stream.getTracks().forEach((t) => t.stop());
  state.audio?.ctx.close();
  state.audio = null;
  state.analyser = null;
  state.match.target = [0, 0];
  state.ready = false;
  render(null);
  if (reason) $("status").textContent = reason;
}

async function preflight() {
  try {
    const r = await fetch("/who/status", { cache: "no-store" });
    const s = await r.json();
    state.sessionsActive = s.sessions_active || 0;
    if (s.names?.length === 2) { state.names = s.names; nameCards(); }
    $("notice").hidden = !(state.sessionsActive > 0);
    if (state.sessionsActive > 0) $("notice").textContent = `A Listen session is still open on this server (${state.sessionsActive}). Stop it on the other tab first: both would share the Mac's GPU.`;
  } catch { /* the page still works; /ws/who says the same on connect */ }
}

// ---- live voice visuals -----------------------------------------------------------------------------------------

const cssVar = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
const vpSpans = [0, 1].map((i) => {
  const box = $(`vp${i}`);
  box.replaceChildren(...Array.from({ length: VP_BARS }, () => document.createElement("span")));
  return [...box.children];
});

function speakerIndex() {
  const msg = state.last;
  return state.running && msg?.speaker ? state.names.indexOf(msg.speaker) : -1;
}

function drawWave(dt) {
  const canvas = $("wave");
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.round(72 * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const a = state.analyser;
  const still = REDUCED?.matches || !a;
  let pts = null;
  if (!still) {
    a.node.getFloatTimeDomainData(a.time);
    pts = waveformPoints(a.time, WAVE_POINTS);
  }
  for (let i = 0; i < WAVE_POINTS; i++) state.wave[i] = still ? 0 : ease(state.wave[i], pts[i], dt, 0.07);
  const who = speakerIndex();
  ctx.strokeStyle = who === 0 ? cssVar("--p0", "#4fa3ff") : who === 1 ? cssVar("--p1", "#ffb84f") : cssVar("--dim", "#8a94a3");
  ctx.globalAlpha = who >= 0 ? 0.95 : 0.6;
  ctx.lineWidth = 2 * dpr;
  ctx.lineJoin = ctx.lineCap = "round";
  const mid = h / 2, amp = 0.42 * h, xAt = (i) => (i / (WAVE_POINTS - 1)) * w;
  const yAt = (i) => mid - Math.tanh(2.5 * state.wave[i]) * amp; // soft-clipped so loud speech stays inside the strip
  ctx.beginPath();
  ctx.moveTo(0, yAt(0));
  for (let i = 1; i < WAVE_POINTS - 1; i++) { // midpoint quadratic curves: a smooth line, no corners
    ctx.quadraticCurveTo(xAt(i), yAt(i), (xAt(i) + xAt(i + 1)) / 2, (yAt(i) + yAt(i + 1)) / 2);
  }
  ctx.lineTo(w, yAt(WAVE_POINTS - 1));
  ctx.stroke();
  ctx.globalAlpha = 1;
}

/** Voiceprint targets 0..1: the speech band (~90 Hz-4 kHz) of the analyser spectrum, in VP_BARS log-spaced bands. */
function voiceprintLevels() {
  const a = state.analyser;
  const out = new Float32Array(VP_BARS);
  if (!a) return out;
  a.node.getByteFrequencyData(a.freq);
  const hz = a.rate / a.node.fftSize, lo = 90, hi = 4000;
  for (let b = 0; b < VP_BARS; b++) {
    const f0 = lo * (hi / lo) ** (b / VP_BARS), f1 = lo * (hi / lo) ** ((b + 1) / VP_BARS);
    const i0 = Math.max(1, Math.floor(f0 / hz)), i1 = Math.max(i0 + 1, Math.ceil(f1 / hz));
    let peak = 0;
    for (let i = i0; i < i1 && i < a.freq.length; i++) peak = Math.max(peak, a.freq[i]);
    out[b] = peak / 255;
  }
  return out;
}

function drawPeople(dt) {
  const still = REDUCED?.matches;
  const who = speakerIndex();
  const levels = who >= 0 && !still ? voiceprintLevels() : null;
  for (let p = 0; p < 2; p++) {
    const m = state.match;
    m.shown[p] = still ? m.target[p] : ease(m.shown[p], m.target[p], dt, 0.45);
    $(`fill${p}`).style.transform = `scaleX(${m.shown[p].toFixed(4)})`;
    const pct = Math.round(100 * m.shown[p]);
    if (pct !== m.pct[p]) {
      m.pct[p] = pct;
      $(`pct${p}`).textContent = matchText(m.shown[p]);
      $(`meter${p}`).setAttribute("aria-valuenow", String(pct));
    }
    const on = who === p;
    const card = $(`person${p}`);
    if (card.dataset.on !== String(on)) card.dataset.on = String(on);
    const bars = state.vp[p];
    for (let b = 0; b < VP_BARS; b++) {
      const target = on && levels ? 0.08 + 0.92 * levels[b] : still && on ? 0.5 : 0.08;
      bars[b] = still ? target : ease(bars[b], target, dt, on ? 0.09 : 0.3);
      vpSpans[p][b].style.transform = `scaleY(${bars[b].toFixed(3)})`;
    }
  }
}

// ---- mic picker -------------------------------------------------------------------------------------------------

async function refreshMics() {
  let devices = [];
  try { devices = await navigator.mediaDevices.enumerateDevices(); } catch { return; }
  const sel = $("mic");
  const options = micOptions(devices);
  if (!options.some((o) => o.value === "")) options.unshift({ value: "", text: "Default microphone" });
  if (state.micId && !options.some((o) => o.value === state.micId)) options.push({ value: state.micId, text: "Saved mic (not connected)" });
  sel.replaceChildren(...options.map((o) => Object.assign(document.createElement("option"), { value: o.value, textContent: o.text })));
  sel.value = state.micId;
}

async function chooseMic() {
  state.micId = $("mic").value;
  saveChoice(WHO_MIC_KEY, state.micId || "default"); // "default" = picked on purpose, do not fall back to hearing.mic
  if (!state.running) return;
  stop();
  await start();
}

$("caption").textContent = CAPTION;
$("match-caption").textContent = MATCH_CAPTION;
function nameCards() { state.names.forEach((n, i) => { if ($(`pname${i}`)) $(`pname${i}`).textContent = displayName(n); }); }
nameCards();
$("mic").addEventListener("change", chooseMic);
navigator.mediaDevices?.addEventListener?.("devicechange", refreshMics);
refreshMics();
$("go").addEventListener("click", () => (state.running ? stop() : start()));
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && e.target === document.body) { e.preventDefault(); state.running ? stop() : start(); }
});
window.addEventListener("resize", drawTimeline);
let lastFrame = performance.now();
(function loop(now) {
  const dt = Math.min(0.1, Math.max(0, ((now ?? performance.now()) - lastFrame) / 1000));
  lastFrame = now ?? performance.now();
  drawTimeline();
  drawWave(dt);
  drawPeople(dt);
  requestAnimationFrame(loop);
})();
preflight().then(() => { if (new URLSearchParams(location.search).get("autostart") === "1") start(); });
