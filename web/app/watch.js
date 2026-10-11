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
import { RELEASE_CONTROL, personControl, savedPeople } from "../console/judge-targets.js";

/** The `hello.client` names of a stage session (`/judge`, the app). Solo refuses while one is open. */
export const STAGE_CLIENTS = ["judge", "app"];
/** No mirrored telemetry for this long: the stage is not streaming (judge-pill's MIC_STALL_MS rule). */
export const STALE_MS = 1500;
const TICK_MS = 100;

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
  let sent = 0;
  const watch = (patch) => store.set({ watch: { ...store.state.watch, ...patch } });
  store.set({ sessionId, started: true, watch: { state: "connecting", reason: "", stale: true, sinceMs: null, sent: 0 } });

  function derive() {
    const since = lastTelemetry == null ? Infinity : now() - lastTelemetry;
    // the stage's frames are invisible to us; its telemetry is the heartbeat (STALE_MS = MIC_STALL_MS)
    store.derive(now(), { sinceFrameMs: since });
    const stale = !(since <= STALE_MS);
    if (stale !== store.state.watch.stale) watch({ stale });
  }

  function onMessage(text) {
    let msg;
    try { msg = JSON.parse(text); } catch { return; }
    if (msg.type === "telemetry") {
      lastTelemetry = now();
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
