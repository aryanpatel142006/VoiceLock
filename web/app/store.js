// App store (M14, T-346): one state object, one event bus, and the pure rules the screens render from.
// session.js is the only writer (socket, mic, worklet); screens read `store.state` and subscribe with `store.on`.
// Pure: no DOM, no WebSocket, no AudioContext, so `node --test web/app/tests/session.test.mjs` covers every rule.
// Every value here comes from the engine's telemetry or the page's own measurements; nothing is invented, and a
// value nobody measured yet is `null` (screens show "—").
//
// Event API (stable; screens 4/5/6/10 build against it):
//   store.on(event, fn) -> off()        subscribe; fn(payload, state)
//   store.state                         the current AppState (read only; replace via store.set)
// Events:
//   "change"     {keys: [...]}          after any store.set; keys = top-level fields that changed
//   "telemetry"  msg                    each engine telemetry (~10 Hz), after state is updated
//   "caption"    {text, final, t0, t1}  each caption message (Quiet captions; never synthesised)
//   "error"      msg                    engine `error` messages ({stage, message, degraded_to})
//   "note"       {at, text}             one line of the fallback log (Diagnostics); also kept in state.log
//   "level"      {db, band}             mic peak of each frame sent (dBFS, 24 dB/s release), ~50 Hz
//   "playback"   {db}                   peak of each audio-out frame queued to play (dBFS, 24 dB/s release; -60 at Stop)
//   "connection" {connection, attempt, retryInMs}
//   "session"    {sessionId}            the server's session id (null on disconnect)
//   "replay"     {active, clip}         the server's replay state (the REPLAY badge must follow it)

import { PillHysteresis, Steady, judgePill, micPill } from "../console/judge-pill.js";
import { captionFeed, modeControl } from "../console/judge-modes.js";
import { SEARCHING_PILL, abControl, personLabel, savedPeople, targetLine } from "../console/judge-targets.js";
import { abDelayMs, hqBackendInfo, tierInfo } from "../console/protocol.js";
import { levelBand } from "../console/devices.js";

export const EVENTS = ["change", "telemetry", "caption", "error", "note", "level", "playback", "connection", "session", "replay"];

// Same backoff as /judge (judge.js RETRY_MS): 0.5, 1, 2, then every 4 s for ever.
export const RETRY_MS = [500, 1000, 2000, 4000];
// T-135 jitter profiles, as /judge: Focus HQ already plays ~0.6-1.5 s behind, so 120 ms of buffer is inaudible and
// absorbs GPU/ladder bursts; Natural (the camera-lost / fallback path) keeps 40 ms.
export const JITTER_MS = { focus_hq: 120, natural: 40 };
export const JITTER_MAX_MS = { focus_hq: 240, natural: 160 };
export const APP_MODES = ["focus_hq", "quiet"]; // Focus / Quiet (judge-modes.js MODES)
export const METER_FLOOR_DB = -60;
export const METER_RELEASE_DB_PER_S = 24; // devices.js meter release
const MAX_LOG = 30;
const MAX_CAPTIONS = 4;

/** The reconnect delay after `attempt` failed tries (0 = first retry). */
export function backoffMs(attempt) {
  return RETRY_MS[Math.min(Math.max(0, attempt), RETRY_MS.length - 1)];
}

/** Which jitter profile the worklet should run for this telemetry: Natural when the engine plays Natural. */
export function jitterFor(t) {
  return t?.effective_mode === "natural" ? "natural" : "focus_hq";
}

/** Worklet `config` message fields for a jitter profile at `rate` Hz. */
export function jitterSamples(kind, rate) {
  return {
    targetSamples: Math.round((rate * JITTER_MS[kind]) / 1000),
    maxSamples: Math.round((rate * JITTER_MAX_MS[kind]) / 1000),
  };
}

/** The A/B message for the worklet: RAW is delayed by the engine's declared total latency so A and B line up. */
export function abWorklet(ab, t) {
  return { type: "ab", state: ab, delayMs: abDelayMs(t) };
}

/** Next A/B side: a button sets its side, `undefined` (Space) flips. */
export function nextAb(current, pressed) {
  return abControl(current, pressed || null).ab;
}

/** The control fields for an app mode; anything but Quiet is Focus (focus_hq). */
export function modeFields(mode) {
  return modeControl(mode === "quiet" ? "quiet" : "focus_hq");
}

/**
 * The raw (pre-hysteresis) pill, exactly as /judge decides it: the mic rule first (never look live while the mic
 * is not streaming), then Searching after Release, then the acquisition pill. The app is audio-only (no camera).
 */
export function rawPill({ connected, started, replaying, sinceFrameMs, released, telemetry, attempt = 1 }) {
  if (!connected && attempt === 0) return { state: "lost", text: "Connecting to the engine…", note: "" };
  const mic = micPill({ connected, started, replaying, sinceFrameMs });
  if (mic) return mic;
  if (released && connected) return SEARCHING_PILL;
  return judgePill(telemetry, { cameraLost: false, connected });
}

/**
 * People: the first two saved profiles (keys 1/2), who is targeted, and the target line. `streaming` false (no mic
 * frames flowing): nobody is "current", because nobody plays; the engine's choice is kept as `last` (T-365 P1-15).
 */
export function personState(t, { released = false, streaming = true } = {}) {
  const names = savedPeople(t);
  const chosen = !released && t?.enrollment?.source === "profile" ? t.enrollment.profile || null : null;
  const current = streaming ? chosen : null;
  return {
    names,
    labels: names.map(personLabel),
    current,
    currentLabel: current ? personLabel(current) : null,
    last: chosen,
    lastLabel: chosen ? personLabel(chosen) : null,
    calibrated: !released && Boolean(t?.enrollment?.source) && t?.enrollment?.source !== "profile",
    released,
    line: targetLine(t, { released }),
  };
}

/**
 * The truth strip (Listen) and Diagnostics numbers. Every field is null until measured or reported:
 * backend: "GPU" | "Mac" (what plays now, `hq_active`); delayMs: engine total latency (Steady EMA, 50 ms steps);
 * rttMs: page<->engine round trip from audio-out capture stamps (Steady, 10 ms); gpuRttMs: laptop<->GPU;
 * recording: the server's `recording` telemetry field (null when the server does not report it).
 */
export function truthStrip(t, { connected = false, sessionId = null, replaying = false, delayMs = null, rttMs = null, streaming = true } = {}) {
  const hq = hqBackendInfo(t);
  const active = t ? hq.active || "local" : null;
  return {
    connected,
    sessionId,
    recording: typeof t?.recording === "boolean" ? t.recording : null,
    replay: Boolean(replaying),
    backend: t ? (active === "remote" ? "GPU" : "Mac") : null,
    device: t?.device || null,
    hqActive: t ? active : null,
    hqLabel: t ? hq.label : null,
    hqReason: hq.reason || null,
    fallback: Boolean(hq.fallback),
    delayMs: t && streaming ? delayMs : null, // "—" until audio flows: an idle engine's 0 ms is not a delay (P1-1)
    latencyPreset: t?.latency_preset || null,
    isolation: t?.isolation || null,
    rttMs,
    gpuRttMs: typeof t?.remote_rtt_ms === "number" ? Math.round(t.remote_rtt_ms) : null,
    tier: t && streaming ? tierInfo(t.tiers?.active || t.tier).label : null,
    sameBucket: Boolean(t?.tiers?.same_bucket),
    confidence: typeof t?.tse?.confidence === "number" ? t.tse.confidence : null,
    preset: t?.preset?.pins ? "team" : null,
    revWarning: t?.remote?.rev_warning || t?.preset?.pins?.rev_warning || null,
    gitRev: t?.git_rev || null,
  };
}

/** Quiet's caption header and fallback notice (judge-modes.js captionFeed). */
export function captionState(t, mode) {
  const feed = captionFeed(t);
  const quiet = mode === "quiet";
  return { quiet, speaker: feed.speaker, degraded: quiet && Boolean(t) && feed.degraded, paused: quiet && t ? feed.paused : "" };
}

/** Meter: peak dBFS with a 24 dB/s release (devices.js), and its band. `dtS` = seconds since the last update. */
export function meterStep(levelDb, peakDb, dtS) {
  const released = Math.max(METER_FLOOR_DB, levelDb - METER_RELEASE_DB_PER_S * Math.min(0.5, Math.max(0, dtS)));
  const db = Math.max(released, peakDb);
  return { db, band: levelBand(db) };
}

/** Captions: the in-progress line is replaced until final, at most MAX_CAPTIONS lines kept. */
export function addCaption(lines, c) {
  const out = lines.slice();
  if (out.length && !out[out.length - 1].final) out.pop();
  out.push({ text: String(c.text || ""), final: Boolean(c.final), t0: c.t0 ?? null, t1: c.t1 ?? null });
  return out.slice(-MAX_CAPTIONS);
}

/** The initial AppState. */
export function initialState() {
  return {
    connection: "idle", // idle | connecting | open | reconnecting
    attempt: 0,
    retryInMs: null,
    sessionId: null,
    started: false, // Start pressed and the mic opened (or a replay owns playback)
    starting: false,
    replaying: false,
    replayClip: null,
    mic: { label: null, output: null, banner: "", level: METER_FLOOR_DB, band: "idle", missing: [] },
    ab: "processed",
    abDelayMs: 0,
    mode: "focus_hq",
    effectiveMode: null,
    jitter: "focus_hq",
    released: false,
    suppression: 100, // the room dial (T-427): what this page last sent; the judge profile's own value is 100
    isolation: null,
    person: personState(null),
    pill: { state: "idle", text: "Press Start", note: "" },
    pillChanges: 0,
    truth: truthStrip(null),
    captions: [],
    caption: captionState(null, "focus_hq"),
    telemetry: null,
    log: [],
  };
}

/**
 * The store: state + event bus + the pill hysteresis and steady numbers (which need memory across ticks).
 * `derive(nowMs, live)` recomputes the rendered fields from the latest telemetry and the page's measurements.
 */
export class Store {
  constructor() {
    this.state = initialState();
    this.listeners = new Map(EVENTS.map((e) => [e, new Set()]));
    this.pill = new PillHysteresis();
    this.steadyDelay = new Steady(50);
    this.steadyRtt = new Steady(10);
  }

  on(event, fn) {
    const set = this.listeners.get(event);
    if (!set) throw new Error(`unknown store event "${event}" (known: ${EVENTS.join(", ")})`);
    set.add(fn);
    return () => set.delete(fn);
  }

  emit(event, payload) {
    for (const fn of [...(this.listeners.get(event) || [])]) {
      try { fn(payload, this.state); } catch (e) { console.error(`store "${event}" listener`, e); }
    }
  }

  /** Shallow-merge `patch`; emits "change" with the keys whose value changed (by identity). */
  set(patch) {
    const keys = Object.keys(patch).filter((k) => this.state[k] !== patch[k]);
    if (!keys.length) return keys;
    this.state = { ...this.state, ...patch };
    this.emit("change", { keys });
    return keys;
  }

  note(text, at = new Date()) {
    const stamp = at.toLocaleTimeString?.([], { hour12: false }) ?? "";
    const line = { at: stamp, text: String(text) };
    this.set({ log: [...this.state.log, line].slice(-MAX_LOG) });
    this.emit("note", line);
  }

  /**
   * Recompute pill, person, truth strip and caption state. `live`: {sinceFrameMs, rttMs} measured by the page.
   * Called on every telemetry and on a 100 ms render tick (the pill's mic rule needs time to pass).
   */
  derive(nowMs, { sinceFrameMs = 0, rttMs = null, streaming: audioFlows = true } = {}) {
    const s = this.state;
    const t = s.telemetry;
    const connected = s.connection === "open";
    const raw = rawPill({ connected, started: s.started, replaying: s.replaying, sinceFrameMs, released: s.released, telemetry: t,
      attempt: s.attempt });
    const pill = this.pill.update(raw, nowMs);
    const delay = this.steadyDelay.update(t?.latency_ms?.total);
    const rtt = rttMs == null ? null : this.steadyRtt.update(rttMs);
    // `audioFlows` false: the caller knows no audio reaches the engine (the phone mirror's seq stopped, T-458)
    const streaming = connected && (s.started || s.replaying) && audioFlows !== false;
    const truth = truthStrip(t, { connected, sessionId: s.sessionId, replaying: s.replaying, delayMs: delay, rttMs: rtt, streaming });
    const patch = { pillChanges: this.pill.changes, abDelayMs: abDelayMs(t), effectiveMode: t?.effective_mode || null };
    if (!same(pill, s.pill)) patch.pill = { state: pill.state, text: pill.text, note: pill.note || "" };
    const person = personState(t, { released: s.released, streaming });
    if (!same(person, s.person)) patch.person = person;
    if (!same(truth, s.truth)) patch.truth = truth;
    const caption = captionState(t, s.mode);
    if (!same(caption, s.caption)) patch.caption = caption;
    return this.set(patch);
  }

  /** Forget the engine (disconnect): telemetry-derived numbers go back to "—". */
  resetEngine() {
    this.steadyDelay.reset();
    this.steadyRtt.reset();
  }
}

function same(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** The app's one store (session.js writes it; screens import it). */
export const store = new Store();
