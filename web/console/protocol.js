// Wire format of /ws/session (docs/CONTRACTS.md#events), browser side.
// Binary: u8 kind · u32 seq · f64 capture_ts (little endian, 13 bytes) · payload.

export const KIND_AUDIO_IN = 1;
export const KIND_VIDEO_IN = 2;
export const KIND_AUDIO_OUT = 3;
export const HEADER_BYTES = 13;

export const MODES = ["natural", "focus_hq", "synthetic", "text", "quiet", "auto", "name_call"];
export const SLIDER_MODES = new Set(["natural", "focus_hq", "auto", "name_call"]);

/** Header + payload (an ArrayBuffer or typed array) as one ArrayBuffer. */
export function packFrame(kind, seq, captureTs, payload) {
  const bytes = payload instanceof ArrayBuffer
    ? new Uint8Array(payload)
    : new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
  const out = new ArrayBuffer(HEADER_BYTES + bytes.byteLength);
  const view = new DataView(out);
  view.setUint8(0, kind);
  view.setUint32(1, seq >>> 0, true);
  view.setFloat64(5, captureTs, true);
  new Uint8Array(out, HEADER_BYTES).set(bytes);
  return out;
}

/** {kind, seq, captureTs, payload: ArrayBuffer}; throws on a short frame. */
export function unpackFrame(buffer) {
  if (buffer.byteLength < HEADER_BYTES) {
    throw new Error(`binary frame is ${buffer.byteLength} bytes; the header alone is ${HEADER_BYTES}`);
  }
  const view = new DataView(buffer);
  return {
    kind: view.getUint8(0),
    seq: view.getUint32(1, true),
    captureTs: view.getFloat64(5, true),
    payload: buffer.slice(HEADER_BYTES),
  };
}

/** Float32 -> PCM16 little endian bytes, scaled and clipped like hearing.audio.io. */
export function floatToPcm16(samples) {
  const out = new ArrayBuffer(samples.length * 2);
  const view = new DataView(out);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.round(samples[i] * 32768);
    view.setInt16(2 * i, Math.max(-32768, Math.min(32767, v)), true);
  }
  return out;
}

/** PCM16 little endian (ArrayBuffer) -> Float32 [-1, 1). */
export function pcm16ToFloat(buffer) {
  const n = buffer.byteLength >> 1;
  const view = new DataView(buffer);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getInt16(2 * i, true) / 32768;
  return out;
}

export function hello(sampleRate, frameMs, video) {
  return { type: "hello", sample_rate: sampleRate, frame_ms: frameMs, video, client: "console" };
}

/** Partial control update: only the keys given are sent (the server applies only those). */
export function control(fields) {
  return { type: "control", ...fields };
}

/** WebSocket URL: ?ws= override, else /ws/session on the page's host. */
export function sessionUrl(location) {
  const params = new URLSearchParams(location.search);
  const explicit = params.get("ws");
  if (explicit) return explicit;
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const host = location.host || "127.0.0.1:8000"; // file:// has no host
  return `${scheme}//${host}/ws/session`;
}

/** True on iPhone/iPad (every iOS browser is WebKit; iPadOS reports itself as a Mac with touch). */
export function isIOS(nav = globalThis.navigator) {
  if (!nav) return false;
  return /iPad|iPhone|iPod/.test(nav.userAgent || "") || (nav.platform === "MacIntel" && nav.maxTouchPoints > 1);
}

/**
 * AudioContext options: 16 kHz where the browser resamples the mic itself (Chrome, Android), the
 * native rate (48 kHz) on iOS, where a mic on a context at another rate is unreliable. The rate
 * goes in `hello` and the server's stream resampler converts (hearing.server.resample).
 */
export function audioContextOptions(nav = globalThis.navigator) {
  return isIOS(nav) ? { latencyHint: "interactive" } : { sampleRate: 16000, latencyHint: "interactive" };
}

/** Root-mean-square level in dBFS, floored at -90. */
export function dbfs(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  const rms = Math.sqrt(sum / Math.max(1, samples.length));
  return Math.max(-90, 20 * Math.log10(rms + 1e-12));
}

/**
 * Split telemetry latency_ms into per-stage compute and the fixed parts.
 * Keys besides algorithmic/resample/server/total are stage names.
 */
// "vision" is per video frame and off the audio path, so it is not a per-hop stage.
const FIXED = new Set(["algorithmic", "resample", "server", "total", "vision"]);
export function splitLatency(latency) {
  const stages = [];
  for (const [key, value] of Object.entries(latency || {})) {
    if (!FIXED.has(key)) stages.push([key, value]);
  }
  return {
    stages,
    algorithmic: latency?.algorithmic ?? 0,
    resample: latency?.resample ?? 0,
    server: latency?.server ?? 0,
    total: latency?.total ?? null,
    vision: latency?.vision ?? null,
  };
}

/** Delay for the raw A/B side: the engine's declared total latency, so A and B line up. */
export function abDelayMs(telemetry) {
  const total = telemetry?.latency_ms?.total;
  return Number.isFinite(total) ? Math.max(0, total) : 0;
}

const TIER_LABELS = {
  focus_hq: "Focus HQ", enrolled: "Saved voice", gate: "AV · gate", av: "AV", "audio-only": "Audio-only",
};
const AUDIO_TIERS = new Set(["audio-only", "enrolled"]); // no camera in the audio path

/** Telemetry tier -> {av, label}: av is true when the camera drives the audio path. */
export function tierInfo(tier) {
  if (!tier) return { av: null, label: "—" };
  return { av: !AUDIO_TIERS.has(tier), label: TIER_LABELS[tier] || `AV · ${tier.replace(/_/g, " ")}` };
}

const HQ_LABELS = { local: "Local", remote: "GPU", auto: "Auto" };

/**
 * Where Focus HQ runs, from telemetry ``hq_backend`` (requested), ``hq_active`` (playing now),
 * ``hq_reason`` and ``remote_rtt_ms``: {requested, active, fallback, rttMs, label, reason}.
 * ``fallback`` is true when the user asked for the GPU (or Auto) but local plays.
 */
export function hqBackendInfo(t) {
  if (!t?.hq_backend) return { requested: null, active: null, fallback: false, rttMs: null, label: "—", reason: "" };
  const active = t.hq_active || "local";
  const rttMs = typeof t.remote_rtt_ms === "number" ? t.remote_rtt_ms : null;
  const rtt = rttMs == null ? "" : ` · RTT ${rttMs < 10 ? rttMs.toFixed(1) : Math.round(rttMs)} ms`;
  const auto = t.hq_backend === "auto" ? "Auto → " : "";
  return {
    requested: t.hq_backend,
    active,
    fallback: t.hq_backend === "remote" && active !== "remote",
    rttMs,
    label: `${auto}${HQ_LABELS[active] || active}${rtt}`,
    reason: t.hq_reason || "",
  };
}

/**
 * The Focus HQ ladder from telemetry ``tiers`` (hearing.fusion.tiers.TierLadder.status) as
 * rows {name, label, active, eligible, reason} best first, plus the HQ model's health (or null).
 */
/**
 * Compute headroom of the Focus HQ ladder (T-089): `tiers.compute_headroom` = 1 - compute/hop of the
 * slowest tier, plus the enrolled hop and its p50/p95. `{text, low}`; low = under 30 % headroom.
 */
export function headroomInfo(tiers) {
  const h = tiers?.compute_headroom;
  if (h == null) return { text: "—", low: false };
  const tse = (tiers.tiers || []).flatMap((t) => t.health || []).find((r) => r.utilisation != null);
  const pct = (x) => `${Math.round(100 * x)} %`;
  let text = `${pct(h)} free`;
  if (tse) {
    text += ` · ${tse.candidate} ${Math.round(tse.compute_p50_ms)}/${Math.round(tse.compute_p95_ms)} ms per window` +
      ` (p50/p95) every ${tse.hop_s} s` + (tse.hop_s > tse.hop_base_s ? ` (widened from ${tse.hop_base_s} s)` : "");
  }
  return { text, low: h < 0.3 };
}

export function ladderInfo(tiers) {
  if (!tiers?.tiers) return { rows: [], hq: null, delayMs: null, reason: "" };
  const rows = tiers.tiers.map((t) => ({
    name: t.name,
    label: tierInfo(t.name).label,
    active: t.name === tiers.active,
    eligible: Boolean(t.eligible),
    reason: t.reason || "",
  }));
  const hqTier = tiers.tiers.find((t) => t.name === "focus_hq");
  const hq = hqTier?.health?.find((h) => h.candidate) || null;
  return { rows, hq, delayMs: tiers.delay_ms ?? null, reason: tiers.reason || "" };
}
