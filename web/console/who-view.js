// Who's speaking (/who): the DOM-free part of the page, so node tests can drive it.
// Server messages ({type: "who", speaker, state, confidence, t, ...}, hearing/server/who.py) -> what the page shows,
// plus the running timeline strip of who spoke when.

export const IDLE = { big: "Press Start", sub: "the demo mic listens, nothing plays back", cls: "idle" };
export const LOADING = { big: "Loading model…", sub: "fine-tuned separator + voice profiles", cls: "idle" };
export const CAPTION = "Fine-tuned on our two voices · speaker ID only, no audio output";

/** "aryan" -> "Aryan". */
export const displayName = (name) => (name ? name.charAt(0).toUpperCase() + name.slice(1) : "");

/** What the big label, its sub line and the colour class should be for a `who` message. */
export function labelFor(msg, names = ["aryan", "joseph"]) {
  if (!msg) return IDLE;
  if (msg.type === "who_status") return msg.state === "error" ? { big: "Model unavailable", sub: msg.message || "", cls: "bad" } : LOADING;
  if (msg.type !== "who") return IDLE;
  const idx = names.indexOf(msg.speaker);
  if (msg.speaker && idx >= 0) {
    return { big: displayName(msg.speaker), sub: msg.state === "hold" ? "pausing" : "speaking", cls: `p${idx}` };
  }
  if (msg.state === "unsure") return { big: "Listening…", sub: "someone is speaking", cls: "none" };
  return { big: "No one speaking", sub: "Listening…", cls: "none" };
}

/** Confidence bar width, 0..1, only while someone is labelled. */
export const confidenceOf = (msg) => (msg?.type === "who" && msg.speaker ? Math.max(0, Math.min(1, msg.confidence || 0)) : 0);

/** The status line under the caption. */
export function statusLine({ ready, mic, micMissing, sessionsActive, compute, device } = {}) {
  const parts = [];
  if (ready) parts.push(`model on ${device || "the Mac"}`);
  if (mic) parts.push(`mic: ${mic}${micMissing ? " (saved mic not found: default)" : ""}`);
  if (compute) parts.push(`${compute} ms per window`);
  if (sessionsActive > 0) parts.push(`⚠ ${sessionsActive} Listen session still open: stop it first (shares the GPU)`);
  return parts.join(" · ");
}

/** Who spoke when: merged segments [{from, to, speaker}] in the message clock, trimmed to the last `span` seconds. */
export class Timeline {
  constructor(span = 60) {
    this.span = span;
    this.segments = [];
  }

  add(t, speaker) {
    if (typeof t !== "number" || !Number.isFinite(t)) return;
    const last = this.segments[this.segments.length - 1];
    if (last && last.speaker === speaker && t >= last.to) last.to = t;
    else if (!last || t >= last.to) this.segments.push({ from: last ? last.to : t, to: t, speaker });
    this.trim(t);
  }

  trim(now) {
    const cut = now - this.span;
    while (this.segments.length && this.segments[0].to < cut) this.segments.shift();
    if (this.segments.length && this.segments[0].from < cut) this.segments[0].from = cut;
  }

  /** Segments as fractions of the strip (0 = span seconds ago, 1 = now), speakers only. */
  bars(now) {
    const start = now - this.span;
    return this.segments
      .filter((s) => s.speaker && s.to > s.from)
      .map((s) => ({ x: Math.max(0, (s.from - start) / this.span), w: (s.to - Math.max(s.from, start)) / this.span, speaker: s.speaker }));
  }

  /** Seconds each speaker held the label inside the strip. */
  totals(now) {
    const out = {};
    for (const b of this.bars(now)) out[b.speaker] = (out[b.speaker] || 0) + b.w * this.span;
    return out;
  }
}

// ---- live voice match (meters, voiceprint bars) and the page's own mic choice ------------------------------------

export const MATCH_CAPTION = "Match = similarity to each saved voice profile (speaker embedding + our fine-tuned model)";
export const WHO_MIC_KEY = "hearing.who.mic"; // this page's mic, separate from the main app's hearing.mic
// ECAPA cosine -> match: 0.2 (an unrelated voice on the demo mics) reads 0 %, 0.8 (the same voice) reads 100 %.
// Same mapping as hearing.who.match_of on the server; used when the message has raw.cos but no match.
export const COS_LOW = 0.2;
export const COS_HIGH = 0.8;

const clip01 = (v) => Math.max(0, Math.min(1, v));

/** ECAPA cosine -> 0..1 (linear, 0.2 -> 0, 0.8 -> 1). */
export const matchOfCos = (cos) => (typeof cos === "number" && Number.isFinite(cos) ? clip01((cos - COS_LOW) / (COS_HIGH - COS_LOW)) : null);

/**
 * Per-person match targets {name: 0..1} for a `who` message: the server's `match` when present, else mapped from
 * `raw.cos` (one cosine per profile, in `names` order). Outside speech every target is 0 (the meters settle down);
 * null when the message carries no scores at all (the page keeps easing toward the last targets).
 */
export function matchTargets(msg, names = ["aryan", "joseph"]) {
  if (msg?.type !== "who") return null;
  if (msg.speech === false) return Object.fromEntries(names.map((n) => [n, 0]));
  if (msg.match && typeof msg.match === "object") {
    const out = {};
    for (const n of names) out[n] = typeof msg.match[n] === "number" ? clip01(msg.match[n]) : 0;
    return out;
  }
  const cos = msg.raw?.cos;
  if (!Array.isArray(cos) || cos.length < names.length) return null;
  return Object.fromEntries(names.map((n, i) => [n, matchOfCos(cos[i]) ?? 0]));
}

/** Exponential ease of `value` toward `target` over `dt` seconds with time constant `tau` (frame-rate independent). */
export const ease = (value, target, dt, tau = 0.35) => value + (target - value) * (1 - Math.exp(-Math.max(0, dt) / tau));

/** "Match 72%". */
export const matchText = (v) => `Match ${Math.round(100 * clip01(v || 0))}%`;

/** The deviceId to open: this page's saved mic, else the main app's, else "" (the default), if it is still listed. */
export function pickMicId(devices, whoSaved, appSaved) {
  if (whoSaved === "default") return ""; // the default was picked on purpose here: do not fall back to the app's mic
  const ids = new Set((devices || []).filter((d) => d.kind === "audioinput").map((d) => d.deviceId));
  for (const id of [whoSaved, appSaved]) if (id && (!ids.size || ids.has(id))) return id;
  return "";
}

/** Dropdown options [{value, text}] for the audio inputs; labels are empty until mic permission, so number them. */
export function micOptions(devices) {
  const inputs = (devices || []).filter((d) => d.kind === "audioinput" && d.deviceId !== "communications");
  return inputs.map((d, i) => ({ value: d.deviceId === "default" ? "" : d.deviceId, text: d.label || `Microphone ${i + 1}` }));
}

/** Waveform points: `n` evenly spaced peak-preserving samples of a time-domain frame, each in -1..1. */
export function waveformPoints(samples, n = 96) {
  const out = new Float32Array(n);
  if (!samples?.length) return out;
  const step = samples.length / n;
  for (let i = 0; i < n; i++) {
    let peak = 0;
    const from = Math.floor(i * step), to = Math.max(from + 1, Math.floor((i + 1) * step));
    for (let j = from; j < to && j < samples.length; j++) if (Math.abs(samples[j]) > Math.abs(peak)) peak = samples[j];
    out[i] = Math.max(-1, Math.min(1, peak));
  }
  return out;
}
