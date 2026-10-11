// Companion mirror (M14, T-351): the phone follows the stage's engine session without owning one.
// `WS /ws/watch?session=<id>` (hearing/server/watch.py, docs/CONTRACTS.md) mirrors that session's telemetry,
// caption, error and replay JSON (never audio) and forwards our `control` messages into it, recorded like the
// stage page's own. Closing the watcher never touches the stage session, and no engine session is created here,
// which is why the companion (not Solo) is the stage role.
//
// The mirror feeds its own Store (store.js), so the phone's state word, target line and captions go through the
// same rules (pill hysteresis, person, captions) as the stage screen. Honesty: the only clock the phone has for
// "is the stage streaming" is the telemetry itself (the engine sends it per processed chunk), so a mirror that
// stops ticking reads "Stage not streaming", never a frozen Isolated.

import { Store, addCaption, backoffMs, modeFields, nextAb } from "./store.js";
import { RELEASE_CONTROL, personControl, personLabel, savedPeople } from "../console/judge-targets.js";

/** The `hello.client` names of a stage session (`/judge`, the app). Solo refuses while one is open. */
export const STAGE_CLIENTS = ["judge", "app"];
/** No mirrored telemetry for this long: the stage is not streaming (judge-pill's MIC_STALL_MS rule). */
export const STALE_MS = 1500;
const TICK_MS = 100;

/**
 * Whether the mirrored stage is streaming audio (T-458). Telemetry ticks at 10 Hz even for an idle session, but its
 * `seq` is the last audio frame the engine received, so only a seq that moved within STALE_MS means audio flows.
 * `track` is {seq, at}: the last seq seen and when it last changed (null until it changes once after joining).
 */
export function seqStep(track, seq, nowMs) {
  if (!Number.isFinite(seq)) return track;
  if (!track) return { seq, at: null }; // the joining snapshot says nothing about now
  return seq === track.seq ? track : { seq, at: nowMs };
}

/** ms since the stage's audio last moved, Infinity when it has not moved since we joined. */
export function sinceAudioMs(track, nowMs) {
  return track?.at == null ? Infinity : nowMs - track.at;
}

/** What a short look at a session's telemetry says: {live: true|false|null (no telemetry), who: "Aryan"|""}. */
export function probeVerdict(telemetries) {
  const last = telemetries[telemetries.length - 1];
  const who = last?.enrollment?.source === "profile" ? personLabel(last.enrollment.profile) : "";
  if (telemetries.length < 2) return { live: null, who };
  // a session that never received audio reports seq null: that is idle too
  const seqs = telemetries.map((t) => (Number.isFinite(t?.seq) ? t.seq : null));
  return { live: seqs[seqs.length - 1] != null && seqs[seqs.length - 1] !== seqs[0], who };
}

/**
 * Watch session `id` for `ms` (read-only; closing a watcher never touches the stage) and say whether its audio
 * moves (probeVerdict). The lobby labels idle sessions with it, and Solo waits only for a live one (T-458).
 */
export function probeSession(id, { loc = globalThis.location, WS = globalThis.WebSocket, ms = 1300, setTimer = setTimeout } = {}) {
  return new Promise((resolve) => {
    const seen = [];
    let sock;
    const done = () => { try { sock?.close(1000); } catch { /* closed */ } resolve(probeVerdict(seen)); };
    try { sock = new WS(watchUrl(loc, id)); } catch { resolve({ live: null, who: "" }); return; }
    sock.onmessage = (e) => {
      if (typeof e.data !== "string") return;
      try { const m = JSON.parse(e.data); if (m.type === "telemetry") seen.push(m); } catch { /* not JSON */ }
    };
    sock.onerror = () => undefined;
    setTimer(done, ms);
  });
}

/** "3 min" since an ISO start time (the /sessions `started` field), "" when unknown. */
export function ageText(started, nowMs = Date.now()) {
  const t = Date.parse(started || "");
  if (!Number.isFinite(t)) return "";
  const s = Math.max(0, Math.round((nowMs - t) / 1000));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  return `${Math.round(s / 3600)} h`;
}

/** wss://host/ws/watch?session=<id> for a page at `loc`. */
export function watchUrl(loc, sessionId) {
  const scheme = loc.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${loc.host}/ws/watch?session=${encodeURIComponent(sessionId)}`;
}

/** The stage sessions in a `GET /sessions` list, newest first. */
export function stageSessions(list) {
  return (Array.isArray(list) ? list : [])
    .filter((s) => s && STAGE_CLIENTS.includes(s.client))
    .sort((a, b) => String(b.started || "").localeCompare(String(a.started || "")));
}

/** `GET /sessions` (an empty list when the server is unreachable or too old to have the route). */
export async function listSessions(fetchImpl = globalThis.fetch) {
  try {
    const res = await fetchImpl("/sessions", { cache: "no-store" });
    return res.ok ? await res.json() : [];
  } catch {
    return [];
  }
}

/**
 * The control a companion button sends, or null when it cannot (no such saved profile).
 * action: "focus" | "quiet" | "release" | "person" (with index) | "ab" (flip, or `side`).
 */
export function companionControl(action, telemetry, { index = 0, side = null } = {}) {
  if (action === "focus") return modeFields("focus_hq");
  if (action === "quiet") return modeFields("quiet");
  if (action === "release") return { ...RELEASE_CONTROL };
  if (action === "person") return personControl(savedPeople(telemetry), index);
  if (action === "ab") return { ab: nextAb(telemetry?.ab === "raw" ? "raw" : "processed", side) };
  return null;
}

/** The store patch one mirrored telemetry gives: what the stage is doing, read from the engine, not assumed. */
export function telemetryPatch(t) {
  return {
    telemetry: t,
    mode: t?.mode === "quiet" ? "quiet" : "focus_hq",
    ab: t?.ab === "raw" ? "raw" : "processed",
  };
}

/**
 * Attach to stage session `sessionId`. Returns {store, act(action, opts), close(), get ended}.
 * store.state gains `watch`: {state: "connecting"|"open"|"reconnecting"|"ended", reason, stale, sinceMs, sent}.
 */
export function createWatch(sessionId, { loc = globalThis.location, WS = globalThis.WebSocket, now = () => performance.now() } = {}) {
  const store = new Store();
  let ws = null;
  let closed = false;
  let attempt = 0;
  let retryTimer = 0;
  let lastTelemetry = null;
  let seqTrack = null;
  let sent = 0;
  const watch = (patch) => store.set({ watch: { ...store.state.watch, ...patch } });
  store.set({ sessionId, started: true, watch: { state: "connecting", reason: "", stale: true, streaming: false, sinceMs: null, sent: 0 } });

  function derive() {
    const since = lastTelemetry == null ? Infinity : now() - lastTelemetry;
    // the stage's frames are invisible to us; its telemetry is the link's heartbeat (STALE_MS = MIC_STALL_MS) and
    // its seq the audio's (T-458: an idle stage still ticks, so the heartbeat alone read as live)
    const audioMs = store.state.replaying ? 0 : sinceAudioMs(seqTrack, now());
    const streaming = audioMs <= STALE_MS;
    store.derive(now(), { sinceFrameMs: Math.max(since, audioMs), streaming });
    if (streaming !== store.state.watch.streaming) watch({ streaming });
    const stale = !(since <= STALE_MS);
    if (stale !== store.state.watch.stale) watch({ stale });
  }

  function onMessage(text) {
    let msg;
    try { msg = JSON.parse(text); } catch { return; }
    if (msg.type === "telemetry") {
      lastTelemetry = now();
      seqTrack = seqStep(seqTrack, msg.seq, lastTelemetry);
      store.set(telemetryPatch(msg));
      derive();
      store.emit("telemetry", msg);
    } else if (msg.type === "caption") {
      store.set({ captions: addCaption(store.state.captions, msg) });
      store.emit("caption", msg);
    } else if (msg.type === "replay") {
      store.set({ replaying: Boolean(msg.active), replayClip: msg.active ? msg.clip || null : null });
      store.emit("replay", { active: Boolean(msg.active), clip: msg.clip || null });
    } else if (msg.type === "error") {
      if (msg.stage === "watch" && /disconnected/.test(msg.message || "")) {
        closed = true; // the stage session is gone: nothing to reconnect to
        watch({ state: "ended", reason: "The stage session ended. Scan the new code on the laptop." });
      } else {
        store.note(`${msg.stage || "engine"}: ${msg.message || "error"}${msg.degraded_to ? ` (now ${msg.degraded_to})` : ""}`);
      }
      store.emit("error", msg);
    }
  }

  function connect() {
    if (closed) return;
    watch({ state: attempt ? "reconnecting" : "connecting" });
    const sock = new WS(watchUrl(loc, sessionId));
    ws = sock;
    sock.onopen = () => {
      attempt = 0;
      store.set({ connection: "open", attempt: 0, retryInMs: null });
      watch({ state: "open", reason: "" });
    };
    sock.onmessage = (e) => { if (typeof e.data === "string") onMessage(e.data); };
    sock.onclose = (e) => {
      if (ws !== sock) return;
      ws = null;
      store.set({ connection: closed ? "idle" : "reconnecting", telemetry: null, attempt: attempt + 1 });
      store.resetEngine();
      lastTelemetry = null;
      seqTrack = null;
      if (closed || e.code === 1000) {
        closed = true;
        if (store.state.watch.state !== "ended") watch({ state: "ended", reason: "The stage session ended. Scan the new code on the laptop." });
        return;
      }
      const delay = backoffMs(attempt++);
      store.set({ retryInMs: delay });
      watch({ state: "reconnecting", reason: `Lost the laptop. Retrying in ${(delay / 1000).toFixed(1)} s.` });
      retryTimer = setTimeout(connect, delay);
    };
  }

  const tick = setInterval(() => { if (!closed) derive(); }, TICK_MS);
  store.set({ connection: "connecting" });
  connect();

  return {
    store,
    sessionId,
    get ended() { return store.state.watch.state === "ended"; },
    /** Forward a button into the stage session; false when it could not be sent. */
    act(action, opts = {}) {
      const fields = companionControl(action, store.state.telemetry, opts);
      if (!fields || ws?.readyState !== 1) return false;
      ws.send(JSON.stringify({ type: "control", ...fields }));
      sent += 1;
      watch({ sent });
      store.note(`sent ${action}${action === "person" ? ` ${opts.index + 1}` : ""} to the stage`);
      return true;
    },
    close() {
      closed = true;
      clearInterval(tick);
      clearTimeout(retryTimer);
      const sock = ws;
      ws = null;
      sock?.close(1000);
    },
  };
}
