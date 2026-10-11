// T-285: rehearsal testbed. Live A/B matrix, script prompter, telemetry, take recorder.
// Pure helpers are exported (tested in tests/rehearsal.test.mjs); the page wiring only runs in a browser
// that has #rehearsal-root. Audio comes from the phone's microphone through getUserMedia.
import {
  KIND_AUDIO_IN, KIND_AUDIO_OUT, audioContextOptions, control, floatToPcm16, hello, packFrame,
  pcm16ToFloat, sessionUrl, unpackFrame,
} from "./protocol.js";

export const EVENT_BATCH_MS = 1000;
export const AUDIO_BATCH_MS = 500;

/** What the telemetry gauges show, from one `telemetry` message. Missing fields stay null. */
export function telemetryView(t) {
  if (!t) return { speaker: null, margin: null, confidence: null, latencyMs: null, rtf: null, tier: null };
  const num = (x) => (typeof x === "number" && Number.isFinite(x) ? x : null);
  const profile = t.enrollment?.profile;
  const trackId = t.target?.track_id;
  const headroom = num(t.tiers?.compute_headroom);
  return {
    speaker: profile || (trackId != null ? `track ${trackId}` : null),
    margin: num(t.tse?.margin) ?? num(t.tse_similarity),
    confidence: num(t.tse?.confidence) ?? num(t.target?.confidence),
    latencyMs: num(t.latency_ms?.total),
    rtf: headroom == null ? null : Math.max(0, 1 - headroom), // compute / hop of the slowest tier
    tier: t.tier ?? t.tiers?.active ?? null,
  };
}

/** Clamp a gauge value to a 0..1 fill; null (no data) fills nothing. */
export function gaugeFill(value, lo = 0, hi = 1) {
  if (value == null || !Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, (value - lo) / (hi - lo)));
}

/** The prompter's state `elapsedS` seconds into a script: which card is on, how long is left, done. */
export function promptAt(script, elapsedS) {
  const cards = script?.cards ?? [];
  if (!cards.length) return { index: -1, card: null, left: 0, done: true };
  if (script.kind === "freeform") return { index: 0, card: cards[0], left: Infinity, done: false };
  let t = 0;
  for (let i = 0; i < cards.length; i++) {
    t += cards[i].seconds;
    if (elapsedS < t) return { index: i, card: cards[i], left: t - elapsedS, done: false };
  }
  return { index: cards.length - 1, card: cards[cards.length - 1], left: 0, done: true };
}

/** Cards that are on screen at the same time as `index` (overlap scripts show both lines). */
export function simultaneous(script, index) {
  const card = script?.cards?.[index];
  if (!card?.overlap_with) return [];
  return script.cards.filter((c, i) => i !== index && c.who === card.overlap_with);
}

/** One events.jsonl line: `t` is ms since the take started. */
export function eventLine(kind, tMs, payload = {}) {
  return { kind, t_ms: Math.round(tMs), ...payload };
}

/** Folder name the server will create, for display before the take starts. */
export function takeLabel(mode, scenario) {
  const slug = (s) => String(s || "").replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40);
  return `${slug(mode) || "mode"}_${slug(scenario) || "scenario"}`;
}

/** Concatenate Float32 chunks. */
export function concatFloat(chunks) {
  const out = new Float32Array(chunks.reduce((n, c) => n + c.length, 0));
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/**
 * Buffers audio and events for a take and uploads them in batches through `post(path, body, headers)`.
 * A failed upload keeps its data and retries with the next flush, so a blip loses nothing; `pending`
 * and `failures` are shown on the page.
 */
export class TakeUploader {
  constructor(post, takeId, rate) {
    this.post = post; this.base = `/api/rehearsal/take/${takeId}`; this.rate = rate;
    this.audio = { mic: [], out: [] };
    this.events = [];
    this.failures = 0;
    this.sent = { mic: 0, out: 0, events: 0 };
  }

  addAudio(stream, samples) { (this.audio[stream] ||= []).push(samples); }
  addEvent(event) { this.events.push(event); }
  get pending() { return this.events.length + Object.values(this.audio).reduce((n, a) => n + a.length, 0); }

  async flush() {
    const jobs = [];
    for (const [stream, chunks] of Object.entries(this.audio)) {
      if (!chunks.length) continue;
      const taken = chunks.splice(0);
      const body = floatToPcm16(concatFloat(taken));
      jobs.push(this.post(`${this.base}/audio?stream=${stream}&rate=${this.rate}`, body, { "Content-Type": "application/octet-stream" })
        .then(() => { this.sent[stream] += body.byteLength / 2; })
        .catch(() => { this.audio[stream].unshift(...taken); this.failures++; }));
    }
    if (this.events.length) {
      const taken = this.events.splice(0);
      jobs.push(this.post(`${this.base}/events`, JSON.stringify(taken), { "Content-Type": "application/json" })
        .then(() => { this.sent.events += taken.length; })
        .catch(() => { this.events.unshift(...taken); this.failures++; }));
    }
    await Promise.all(jobs);
    return this.pending === 0;
  }
}

// ------------------------------------------------------------------ page

const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const res = await fetch(path, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}

function boot() {
  const S = {
    options: null, matrix: null, ws: null, ctx: null, node: null, stream: null, seq: 0, rate: 16000,
    telemetry: null, take: null, uploader: null, takeStart: 0, script: null, scriptStart: 0, timers: [],
    sink: null,
  };
  const now = () => performance.now();
  const note = (msg) => { $("status").textContent = msg; };

  const radio = (group, table, key) => {
    const el = $(group);
    el.replaceChildren(...Object.entries(table).map(([id, row]) => {
      const b = document.createElement("button");
      b.type = "button"; b.dataset.value = id; b.setAttribute("aria-pressed", "false");
      const extra = row.window_s ? ` · ${row.window_s}s/${row.hop_s}s` : (row.applied === false ? " · record only" : "");
      b.textContent = row.label + extra;
      b.onclick = () => setMatrix({ [key]: id });
      return b;
    }));
  };
  const paintMatrix = () => {
    for (const [group, key] of [["m-repr", "representation"], ["m-policy", "policy"], ["m-ladder", "ladder"]]) {
      for (const b of $(group).children) b.setAttribute("aria-pressed", String(b.dataset.value === S.matrix[key]));
    }
    $("m-supp").value = Math.round(S.matrix.suppression * 100);
    $("m-supp-out").textContent = `${Math.round(S.matrix.suppression * 100)} %`;
  };

  async function setMatrix(patch) {
    try {
      const res = await api("/api/rehearsal/config", patch);
      S.matrix = res.matrix; S.controls = res.controls;
      paintMatrix();
      if (S.ws?.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(control(res.controls)));
      event("matrix", { matrix: res.matrix, controls: res.controls, gated_play: res.gated_play });
      note(res.representation_applied ? "matrix applied live" : "matrix applied; the representation is recorded only (the engine has no switch yet)");
    } catch (e) { note(`matrix not applied: ${e.message}`); }
  }

  function event(kind, payload) {
    if (S.uploader) S.uploader.addEvent(eventLine(kind, now() - S.takeStart, payload));
  }

  // -------- engine socket
  function connect() {
    const ws = new WebSocket(sessionUrl(location));
    ws.binaryType = "arraybuffer";
    S.ws = ws;
    let resolve, reject;
    S.connection = new Promise((ok, fail) => { resolve = ok; reject = fail; });
    S.connection.catch(() => {}); // reconnects may run without a waiting click handler
    ws.onopen = () => {
      ws.send(JSON.stringify({ ...hello(S.rate, 20, false), client: "rehearsal" }));
      ws.send(JSON.stringify(control({ mode: "focus_hq", ...(S.controls || {}) })));
      $("engine").textContent = "engine: connected";
      resolve();
    };
    ws.onclose = () => { reject(new Error("engine disconnected")); $("engine").textContent = "engine: lost, retrying"; setTimeout(connect, 1500); };
    ws.onmessage = (e) => (typeof e.data === "string" ? onText(JSON.parse(e.data)) : onBinary(e.data));
  }
  function onText(msg) {
    if (msg.type === "telemetry") {
      S.telemetry = msg;
      paintTelemetry(telemetryView(msg));
      event("telemetry", { seq: msg.seq, tier: msg.tier, latency_ms: msg.latency_ms, target: msg.target,
        tse: msg.tse, tse_similarity: msg.tse_similarity, enrollment: msg.enrollment,
        compute_headroom: msg.tiers?.compute_headroom, latency_preset: msg.latency_preset });
    } else if (msg.type === "error") {
      event("error", msg); note(`engine error: ${msg.message}`);
    } else if (msg.type === "caption") {
      event("caption", msg);
    }
  }
  function onBinary(buffer) {
    let frame;
    try { frame = unpackFrame(buffer); } catch { return; }
    if (frame.kind !== KIND_AUDIO_OUT) return;
    const samples = pcm16ToFloat(frame.payload);
    S.uploader?.addAudio("out", samples.slice());
    S.node?.port.postMessage({ type: "play", samples }, [samples.buffer]);
  }

  function paintTelemetry(v) {
    $("t-speaker").textContent = v.speaker ?? "none";
    $("t-latency").textContent = v.latencyMs == null ? "—" : `${Math.round(v.latencyMs)} ms`;
    $("t-rtf").textContent = v.rtf == null ? "—" : v.rtf.toFixed(2);
    $("t-tier").textContent = v.tier ?? "—";
    $("g-margin").value = gaugeFill(v.margin, 0, 0.5);
    $("g-margin-n").textContent = v.margin == null ? "—" : v.margin.toFixed(2);
    $("g-conf").value = gaugeFill(v.confidence);
    $("g-conf-n").textContent = v.confidence == null ? "—" : `${Math.round(100 * v.confidence)} %`;
  }

  // -------- phone microphone
  async function openAudio() {
    if (S.ctx) return S.connection;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    const ctx = new AudioContext(audioContextOptions());
    await ctx.audioWorklet.addModule("audio-worklet.js");
    S.rate = ctx.sampleRate;
    const node = new AudioWorkletNode(ctx, "hearing-io", {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
      processorOptions: { frameSamples: Math.round(ctx.sampleRate * 0.02), targetSamples: Math.round(ctx.sampleRate * 0.06),
        maxSamples: Math.round(ctx.sampleRate * 0.4) },
    });
    node.port.onmessage = (e) => {
      if (e.data.type !== "frame") return;
      S.uploader?.addAudio("mic", e.data.samples.slice());
      if (S.ws?.readyState === WebSocket.OPEN) {
        S.ws.send(packFrame(KIND_AUDIO_IN, S.seq++, performance.timeOrigin / 1000 + now() / 1000, floatToPcm16(e.data.samples)));
      }
    };
    ctx.createMediaStreamSource(stream).connect(node);
    node.connect(ctx.destination);
    await ctx.resume();
    Object.assign(S, { ctx, node, stream });
    connect();
    $("mic").textContent = `phone mic open · ${S.rate} Hz`;
    await S.connection;
  }

  // -------- script prompter
  function paintPrompt() {
    const s = S.script;
    if (!s) return;
    const p = S.take ? promptAt(s, (now() - S.scriptStart) / 1000) : { index: 0, card: s.cards[0], left: 0, done: false };
    const cards = $("cards");
    cards.replaceChildren(...s.cards.map((c, i) => {
      const li = document.createElement("li");
      li.className = "card" + (i === p.index ? " on" : "") + (simultaneous(s, p.index).includes(c) ? " together" : "");
      li.innerHTML = `<b></b><span></span>`;
      li.firstChild.textContent = c.who;
      li.lastChild.textContent = c.line + (c.interrupts ? "  (cut in!)" : "");
      return li;
    }));
    $("p-left").textContent = !S.take || s.kind === "freeform" ? "" : p.done ? "script finished" : `${p.left.toFixed(1)} s`;
    if (S.take && p.index !== S.lastCard) { S.lastCard = p.index; event("card", { script: s.id, index: p.index, who: p.card?.who }); }
  }
  function pickScript(id) {
    S.script = S.options.scripts.find((x) => x.id === id) || S.options.scripts[0];
    $("script-select").value = S.script.id;
    paintPrompt();
  }

  // -------- recording
  async function startTake() {
    try {
      await openAudio();
      if (S.ws?.readyState !== WebSocket.OPEN) throw new Error("engine reconnecting; try again when connected");
      const mode = S.script.kind === "freeform" ? "freeform" : "script";
      const take = await api("/api/rehearsal/take", { mode, scenario: S.script.id });
      S.take = take;
      S.uploader = new TakeUploader((path, body, headers) => fetch(path, { method: "POST", headers, body }).then((r) => {
        if (!r.ok) throw new Error(String(r.status));
      }), take.id, S.rate);
      S.takeStart = now(); S.scriptStart = now(); S.lastCard = -1;
      event("take_start", { matrix: S.matrix, rate: S.rate, ua: navigator.userAgent });
      S.timers.push(setInterval(() => S.uploader.flush().then(() => {
        $("rec-info").textContent = `${S.uploader.failures} failed uploads, ${S.uploader.pending} chunks queued`;
      }), AUDIO_BATCH_MS), setInterval(paintPrompt, 200));
      $("rec").textContent = "Stop take"; $("rec").dataset.on = "1";
      $("take-dir").textContent = take.dir;
      note("recording");
    } catch (e) { note(`could not start: ${e.message}`); }
  }
  async function stopTake() {
    S.timers.splice(0).forEach(clearInterval);
    event("take_stop", {});
    for (let i = 0; i < 5 && !(await S.uploader.flush()); i++); // retry queued chunks
    try {
      const sum = await api(`/api/rehearsal/take/${S.take.id}/stop`, {});
      $("take-dir").textContent = `${sum.dir} (${sum.events} events)`;
      note(S.uploader.pending ? `saved, but ${S.uploader.pending} chunks never uploaded` : "take saved");
    } catch (e) { note(`stop failed: ${e.message}`); }
    S.take = null; S.uploader = null;
    $("rec").textContent = "Start take"; delete $("rec").dataset.on;
  }

  async function init() {
    S.options = await api("/api/rehearsal/options");
    const config = await api("/api/rehearsal/config");
    S.matrix = config.matrix; S.controls = config.controls;
    radio("m-repr", S.options.representations, "representation");
    radio("m-policy", S.options.policies, "policy");
    radio("m-ladder", S.options.ladder, "ladder");
    paintMatrix();
    $("m-supp").oninput = (e) => { $("m-supp-out").textContent = `${e.target.value} %`; };
    $("m-supp").onchange = (e) => setMatrix({ suppression: Number(e.target.value) / 100 });
    $("script-select").replaceChildren(...S.options.scripts.map((s) => new Option(s.title, s.id)));
    $("script-select").onchange = (e) => pickScript(e.target.value);
    pickScript(S.options.scripts[0].id);
    $("rec").onclick = () => (S.take ? stopTake() : startTake());
    $("mark").onclick = () => event("marker", { note: $("mark-note").value });
    $("mic-open").onclick = () => openAudio().catch((e) => note(`mic: ${e.message}`));
    if (!S.options.projection_head_present) $("repr-warn").hidden = false;
    $("engine").textContent = "engine: waiting for mic";
  }
  init().catch((e) => note(`rehearsal API missing: ${e.message}`));
}

if (typeof document !== "undefined" && document.getElementById("rehearsal-root")) boot();
