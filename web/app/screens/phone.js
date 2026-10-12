// Phone (M14, T-351): the companion and Solo mode.
//
// Companion (`/app/?watch=latest`, the QR on the laptop's Listen screen and the home-screen app's start_url; or
// `/app/?watch=<session>` for one session): mirrors the stage session over
// `WS /ws/watch` (../watch.js): the state word, who is targeted, the captions, and the controls forwarded into the
// stage session: the Auto voice focus switch (board Q-2010: the stage's A/B, ON = the loudest voice isolated, OFF =
// the whole room; nobody chooses a name), Focus, Quiet. It creates no engine session and plays no audio: the phone
// is never the stage mic or the stage output.
//
// Solo (off stage, a labelled extra): the phone's own mic and its own engine session through the browser SDK
// (/sdk/hearing-client.js). It is refused while a stage session (`/judge` or the app) is open on the laptop, because
// a second session loads the same MPS device and can overrun the stage tier; the screen says why. Solo asks for a
// screen wake lock (iOS stops audio when the screen locks).
//
// `?watch=latest` joins the laptop's newest stage session and, when it ends, the next one, in place (the URL stays
// stable). While the companion is open its manifest (../companion.webmanifest, ../install.js) is the page's, so Add to
// Home Screen installs "VoiceLock" the companion; a dismissible hint says how (iOS Share, Android Install app). The
// installed app reconnects when it comes back to the front (iOS drops the socket of a hidden page).
//
// Honesty: every word and number here is mirrored telemetry, a caption message, or this page's own state. A mirror
// that stops ticking says so; nothing animates to look alive.

import { ageText, companionTarget, createWatch, listSessions, pickLatest, probeSession, stageSessions } from "../watch.js";
import { Store } from "../store.js";
import {
  COMPANION_URL, dismissHint, hintDismissed, installHint, installPrompt, isStandalone, platformOf, promptInstall, swapManifest,
} from "../install.js";
import { listenModel } from "./listen.js";
import { AUTO_FOCUS_LABEL } from "../../console/auto-focus.js";
import { expectedDelayCopy, fetchHealth, modeFields } from "../store.js";

export const title = "Phone";

const POLL_MS = 2000;
/** The stage session ended (the laptop reconnected or restarted): the next one it opens is followed on its own. */
export const ENDED_SENTENCE = "The stage session ended. This phone follows the laptop's next session as soon as it opens.";
/** `?watch=latest` before the laptop has a stage session (or between two). */
export const WAITING_SENTENCE = "Waiting for the laptop. Open Listen there and this phone joins on its own.";
export const SOS_SENTENCE = "The room is cut and calm sound plays on the laptop. Hold SOS there (or the band) to come back.";

/** Pure: one lobby row's words for a session and its probe ({live, who}). */
export function sessionRow(s, probe = null, nowMs = Date.now()) {
  const kind = "Laptop";
  const age = ageText(s.started, nowMs);
  const live = probe ? probe.live : null;
  const state = live === true ? (probe.who ? `hears ${probe.who}` : "live") : live === false ? "idle, not listening" : "checking…";
  return { title: kind, state, age: age ? `open ${age}` : "", live, go: live === false ? "Follow anyway" : "Follow" };
}

/** Pure: what the companion shows for its mirror store state. */
export function companionModel(s) {
  const m = listenModel(s, { keys: false });
  const w = s.watch || {};
  let word = m.word;
  let tone = m.tone;
  let sentence = m.sentence;
  if (w.state === "ended") {
    word = "Ended";
    tone = "idle";
    sentence = ENDED_SENTENCE;
  } else if (w.state !== "open") {
    // `?watch=latest` with no stage session yet (sessionId "", waitingState) waits rather than connects
    word = w.state === "reconnecting" ? "Reconnecting" : s.sessionId === "" ? "Waiting" : "Connecting";
    tone = w.state === "reconnecting" ? "bad" : "idle";
    sentence = w.reason || "Joining the stage session…";
  } else if (w.stale) {
    word = "Not streaming";
    sentence = "No telemetry from the stage for 1.5 s: the engine is busy or the link dropped.";
  } else if (m.state === "nomic" || w.streaming === false) {
    // T-458: telemetry ticks but no audio reaches the stage's engine (seq stands still): idle, not "Searching"
    word = "Not listening";
    tone = "idle";
    sentence = "This stage isn't listening right now: its mic is off. Press Start on the laptop.";
  }
  // Q-2010: the companion's Auto voice focus switch is the way back to one voice
  if (m.tone === "raw" && w.state === "open") sentence = "Turn Auto voice focus on to hear only the closest voice.";
  const live = w.state === "open" && !w.stale && w.streaming !== false;
  // T-462: the stage's SOS (telemetry.sos.on) cuts the room on the laptop; the mirror says so instead of "Isolated"
  const sos = live && Boolean(s.telemetry?.sos?.on);
  if (sos) {
    word = "SOS";
    tone = "idle";
    sentence = SOS_SENTENCE;
  }
  // the stage in Quiet plays nothing: the word says Quiet, and no name is shown (nobody is heard)
  if (live && !sos && m.mode === "quiet") {
    word = "Quiet";
    tone = "idle";
  }
  let stage = "";
  if (live) {
    if (sos) stage = "Stage is in SOS: the room is cut";
    else if (m.mode === "quiet") stage = "Stage is in Quiet: captions only";
    else if (m.ab === "raw") stage = "Stage plays the whole room";
    else if (m.starting) stage = "Stage is starting";
    else if (m.gate === "unsure") stage = `Stage is listening for ${m.who || "the chosen voice"}`; // T-463: not "hears"
    else if (m.loudest && m.state === "isolated") stage = "Stage hears the closest voice";
    else if (m.state === "isolated") stage = m.who ? `Stage hears ${m.who}` : "Stage hears the isolated voice";
    else stage = "Stage is not isolated yet";
  }
  return {
    word, tone, sentence, stage, live, sos,
    // the name only while it is what the stage plays (same as Listen: not under Raw, not under SOS)
    who: live && !sos && m.mode !== "quiet" && m.state === "isolated" && m.tone !== "raw" && m.gate !== "unsure" && m.who ? m.who : "",
    target: w.state === "open" ? m.targetLine : "", // an ended or lost link keeps no stale target or numbers
    mode: m.mode,
    ab: m.ab,
    autoFocus: m.autoFocus,
    people: m.people,
    released: m.released,
    captions: m.captions.lines,
    captionNote: m.captions.paused ? `Captions paused: ${m.captions.paused}` : m.captions.degraded,
    replay: m.replay,
    delay: w.state === "open" ? m.truth.delay : "—",
    tier: w.state === "open" ? m.truth.tier : "—",
    canSend: w.state === "open",
    session: String(s.sessionId || "").slice(0, 8),
  };
}

/**
 * Pure: Solo is refused while a stage session is live; the reason names it. A session `live: false` (probed: its
 * audio does not move) is idle and does not block (T-458); an unprobed one (`live` null/undefined) still does.
 */
export function soloGate(sessions) {
  const stage = stageSessions(sessions).filter((s) => s.live !== false);
  if (!stage.length) return { allowed: true, reason: "" };
  return {
    allowed: false,
    // a second engine session would share the laptop's GPU and could make it stutter, so Solo waits
    reason: "The laptop is already listening. Follow it here instead.",
    stage,
  };
}

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

export function mount(root, ctx = {}) {
  const params = ctx.params || new URLSearchParams(location.search);
  const target = companionTarget(params);
  root.classList.add("screen-phone");
  document.body.dataset.phone = target.follow ? "companion" : "lobby";
  const cleanups = [() => { delete document.body.dataset.phone; delete document.body.dataset.ab; }];
  ctx.signal?.addEventListener("abort", () => cleanups.splice(0).forEach((fn) => fn()), { once: true });
  if (target.follow) cleanups.push(mountCompanion(root, target, ctx));
  else cleanups.push(mountLobby(root, ctx));
  return () => cleanups.splice(0).forEach((fn) => fn());
}

// ------------------------------------------------------------------ companion

/** The mirror state shown while `?watch=latest` has no stage session to join yet. */
export function waitingState() {
  const store = new Store();
  store.set({ sessionId: "", started: true, watch: { state: "connecting", reason: WAITING_SENTENCE, stale: true, streaming: false, sinceMs: null, sent: 0 } });
  return store.state;
}

/** Pure: the install hint's props on this device now. */
function hintNow() {
  return installHint({
    platform: platformOf(navigator.userAgent, navigator.maxTouchPoints || 0),
    standalone: isStandalone(window),
    canPrompt: Boolean(installPrompt()),
    dismissed: hintDismissed(),
  });
}

function mountCompanion(root, target, ctx) {
  const latest = target.follow === "latest";
  // Add to Home Screen / Install app installs the companion (start_url ?watch=latest), not the laptop app
  const restoreManifest = swapManifest(document);
  let watch = null;
  let off = () => undefined;

  const btn = (label, attrs) => h("button", { type: "button", class: "ph-btn", "aria-pressed": "false", ...attrs },
    h("span", { class: "ph-btn-label", text: label }));
  const frame = h("div", { class: "phone", "data-role": "companion", "data-tone": "idle", "data-ab": "processed", "data-mode": "focus_hq" },
    h("header", { class: "ph-top" },
      h("p", { class: "ph-link" }, h("span", { class: "ph-dot", "aria-hidden": "true" }), h("span", { class: "ph-link-text", text: "Joining" })),
      h("p", { class: "ph-replay", hidden: true, text: "Replay" })),
    h("section", { class: "ph-hero", "aria-labelledby": "ph-word" },
      h("p", { class: "ph-stage" }),
      h("h1", { id: "ph-word", class: "ph-word", text: "Connecting" }),
      h("p", { class: "ph-who" }),
      h("p", { class: "ph-sentence" }),
      h("p", { class: "ph-target" })),
    h("section", { class: "ph-captions", "aria-label": "Captions" },
      h("p", { class: "ph-cap-head" }, h("span", { text: "Captions" }), h("span", { class: "ph-cap-note" })),
      h("div", { class: "ph-cap-lines" })),
    // screen readers hear the state word and each final caption once, not every partial or hero line
    h("p", { class: "sr-only ph-sr-word", role: "status" }),
    h("p", { class: "sr-only ph-sr-caption", "aria-live": "polite" }),
    h("nav", { class: "ph-controls", "aria-label": "Control the stage" },
      h("button", { type: "button", class: "ph-switch", role: "switch", "aria-checked": "true", "data-act": "ab", "data-on": "true",
        "aria-label": AUTO_FOCUS_LABEL, "aria-describedby": "ph-switch-line" },
      h("span", { class: "ph-switch-text" }, h("span", { class: "ph-switch-label", text: AUTO_FOCUS_LABEL }),
        h("span", { id: "ph-switch-line", class: "ph-switch-line" })),
      h("span", { class: "ph-switch-track", "aria-hidden": "true" }, h("span", { class: "ph-switch-word", text: "ON" }), h("span", { class: "ph-switch-knob" }))),
      h("div", { class: "ph-row ph-modes", role: "group", "aria-label": "Mode" },
        btn("Focus", { "data-act": "focus" }), btn("Quiet", { "data-act": "quiet" }))),
    h("footer", { class: "ph-foot" },
      h("span", { class: "ph-fact", "data-fact": "delay" }), h("span", { class: "ph-fact", "data-fact": "tier" }),
      h("a", { class: "ph-leave", href: "/app/#phone", text: "Leave" })),
    h("aside", { class: "ph-install", hidden: true, "aria-label": "Put VoiceLock on your home screen" },
      h("p", { class: "ph-install-text" }),
      h("button", { type: "button", class: "ph-btn ph-install-go", hidden: true }, h("span", { class: "ph-btn-label", text: "Install app" })),
      h("a", { class: "ph-install-trust", href: `/connect?page=${encodeURIComponent(COMPANION_URL)}` }),
      h("button", { type: "button", class: "ph-install-close", "aria-label": "Dismiss", text: "×" })));
  root.replaceChildren(frame);

  const q = (sel) => frame.querySelector(sel);
  const el = {
    link: q(".ph-link-text"), replay: q(".ph-replay"), stage: q(".ph-stage"), word: q(".ph-word"), who: q(".ph-who"),
    sentence: q(".ph-sentence"), target: q(".ph-target"), capLines: q(".ph-cap-lines"), capNote: q(".ph-cap-note"),
    srWord: q(".ph-sr-word"), srCaption: q(".ph-sr-caption"),
    delay: q('[data-fact="delay"]'), tier: q('[data-fact="tier"]'),
  };

  frame.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-act]");
    if (!b || b.disabled) return;
    // the switch flips from what the stage's telemetry says plays (watch.js companionControl reads telemetry ab)
    const ok = Boolean(watch) && watch.act(b.dataset.act, { index: Number(b.dataset.index || 0) });
    b.dataset.flash = ok ? "sent" : "failed";
    setTimeout(() => delete b.dataset.flash, 600);
  }, { signal: ctx.signal });

  const hint = q(".ph-install");
  function renderHint() {
    const m = hintNow();
    hint.hidden = !m.show;
    hint.dataset.kind = m.kind;
    setText(q(".ph-install-text"), m.text);
    q(".ph-install-go").hidden = m.kind !== "prompt";
    setText(q(".ph-install-trust"), m.trust);
  }
  q(".ph-install-go").addEventListener("click", async () => { await promptInstall(); renderHint(); }, { signal: ctx.signal });
  q(".ph-install-close").addEventListener("click", () => { dismissHint(); renderHint(); }, { signal: ctx.signal });
  addEventListener("voicelock:installable", renderHint, { signal: ctx.signal });
  renderHint();

  let shownCaptions = "";
  function render() {
    const state = watch ? watch.store.state : waitingState();
    const m = companionModel(state);
    frame.dataset.tone = m.tone;
    frame.dataset.ab = m.ab;
    frame.dataset.mode = m.mode;
    frame.dataset.live = String(m.live);
    frame.dataset.watch = state.watch?.state || "connecting";
    document.body.dataset.ab = m.live ? m.ab : "";
    setText(el.link, m.canSend ? "Following the stage" : m.session ? `Stage ${m.session} · ${state.watch?.state}` : "Looking for the laptop");
    setText(el.word, m.word);
    setText(el.srWord, m.word);
    setText(el.who, m.who);
    el.who.hidden = !m.who;
    setText(el.stage, m.stage);
    setText(el.sentence, m.sentence);
    setText(el.target, m.target);
    el.target.hidden = !m.target || Boolean(m.who) && m.target === `Target: ${m.who}`; // the name says it once
    el.replay.hidden = !m.replay;
    setText(el.replay, m.replay);
    setText(el.delay, `Engine delay ${m.delay}`);
    setText(el.tier, `Tier ${m.tier}`);
    setText(el.capNote, m.captionNote || ""); // status only, no instructions in the status slot
    const key = JSON.stringify(m.captions);
    if (key !== shownCaptions) {
      shownCaptions = key;
      if (!m.captions.length) {
        el.capLines.replaceChildren(h("p", { class: "ph-cap empty", text: "Captions show here when someone near you speaks." }));
      } else {
        // update the lines in place: rebuilding them restarted every line's rise animation on each partial (flicker);
        // only a line that is new to the column animates in
        const nodes = [...el.capLines.children].filter((n) => !n.classList.contains("empty"));
        if (nodes.length !== el.capLines.children.length) el.capLines.replaceChildren(...nodes);
        m.captions.forEach((c, i) => {
          const cls = c.final ? "ph-cap" : "ph-cap partial";
          const n = nodes[i] || el.capLines.appendChild(h("p", { class: cls }));
          if (n.className !== cls) n.className = cls;
          setText(n, c.text);
        });
        for (const n of nodes.slice(m.captions.length)) n.remove();
      }
      const lastFinal = m.captions.findLast((c) => c.final)?.text || "";
      if (lastFinal) setText(el.srCaption, lastFinal);
    }
    for (const b of frame.querySelectorAll("button[data-act]")) {
      const act = b.dataset.act;
      b.disabled = !m.canSend;
      if (act === "ab") {
        // Q-2010: the stage's A/B as read from its telemetry (the mirror never has a local guess)
        b.setAttribute("aria-checked", String(m.autoFocus.on));
        b.dataset.on = String(m.autoFocus.on);
        setText(b.querySelector(".ph-switch-word"), m.autoFocus.on ? "ON" : "OFF");
        setText(b.querySelector(".ph-switch-line"), m.autoFocus.line);
      } else {
        b.setAttribute("aria-pressed", String(m.live && m.mode === (act === "quiet" ? "quiet" : "focus_hq")));
      }
    }
  }
  // the stage reconnecting (or the server restarting) ends this session for good: follow the laptop's next stage
  // session instead of asking for a rescan. Only a different id; only while this screen is mounted. `?watch=latest`
  // follows in place (the URL, the home-screen app's start_url, stays); `?watch=<id>` moves to the next id's URL.
  let followTimer = 0;
  let resolving = 0;
  const dead = () => ctx.signal?.aborted || followTimer === -1;
  function attach(id) {
    off();
    watch?.close();
    watch = createWatch(id);
    window.hearingWatch = watch; // scripts/phone_check.py reads it
    const store = watch.store;
    off = store.on("change", () => {
      render();
      if (store.state.watch?.state === "ended" && !followTimer) followTimer = setTimeout(followNext, 300);
    });
    render();
  }
  async function followNext() {
    followTimer = 0;
    if (dead() || (watch && watch.store.state.watch?.state !== "ended")) return;
    const mine = ++resolving;
    const next = pickLatest(await listSessions(), { exclude: watch?.sessionId ?? null });
    if (dead() || mine !== resolving) return;
    if (next && latest) attach(next);
    else if (next) location.replace(`/app/?watch=${encodeURIComponent(next)}`);
    else followTimer = setTimeout(followNext, POLL_MS);
  }
  // the home-screen app coming back to the front (or a page restored from the back/forward cache): iOS froze it and
  // dropped its socket, so reconnect now instead of after the backoff, and join the laptop's newer session if it has one
  async function wake() {
    if (dead() || document.visibilityState !== "visible") return;
    renderHint(); // installed meanwhile: the hint goes
    if (!watch || watch.ended) {
      clearTimeout(followTimer);
      followNext();
      return;
    }
    if (latest) {
      const mine = ++resolving;
      const newest = pickLatest(await listSessions());
      if (dead() || mine !== resolving) return;
      if (newest && newest !== watch.sessionId) { attach(newest); return; }
    }
    watch.reconnect();
  }
  document.addEventListener("visibilitychange", wake, { signal: ctx.signal });
  addEventListener("pageshow", (e) => { if (e.persisted) wake(); }, { signal: ctx.signal });

  render();
  if (latest) followNext();
  else attach(target.id);
  return () => {
    off();
    clearTimeout(followTimer);
    followTimer = -1; // never re-armed after unmount
    watch?.close();
    if (window.hearingWatch === watch) delete window.hearingWatch;
    restoreManifest();
  };
}

// ------------------------------------------------------------------ lobby: pick a stage session, or Solo

/** Solo's lede with the server's expected delay (T-463). */
export function soloLede(health) {
  return `Your phone's own mic, its own session, your headphones. ${expectedDelayCopy(health)}, like live captions. Keep the screen on: iOS stops audio when it locks.`;
}

function mountLobby(root, ctx) {
  const list = h("ol", { class: "ph-sessions", "aria-live": "polite" });
  const soloBtn = h("button", { type: "button", class: "ph-btn ph-solo-start", "data-act": "solo" },
    h("span", { class: "ph-btn-label", text: "Start Solo" }));
  const soloWhy = h("p", { class: "ph-solo-why", role: "status" });
  const soloLive = h("div", { class: "ph-solo-live", hidden: true },
    h("p", { class: "ph-solo-state", text: "Starting" }),
    h("p", { class: "ph-solo-facts" }),
    h("p", { class: "ph-solo-lock" }),
    h("div", { class: "ph-row", role: "group", "aria-label": "Solo mode" },
      h("button", { type: "button", class: "ph-btn", "data-solo": "focus", "aria-pressed": "true" }, h("span", { class: "ph-btn-label", text: "Focus" })),
      h("button", { type: "button", class: "ph-btn", "data-solo": "quiet", "aria-pressed": "false" }, h("span", { class: "ph-btn-label", text: "Quiet" })),
      h("button", { type: "button", class: "ph-btn ph-release", "data-solo": "stop" }, h("span", { class: "ph-btn-label", text: "Stop" }))),
    h("div", { class: "ph-cap-lines ph-solo-caps", "aria-live": "polite" }));
  const frame = h("div", { class: "phone", "data-role": "lobby", "data-tone": "idle" },
    h("header", { class: "ph-top" }, h("p", { class: "ph-link" }, h("span", { class: "ph-dot", "aria-hidden": "true" }), h("span", { text: "Phone" }))),
    h("section", { class: "ph-lobby" },
      h("h1", { class: "ph-lobby-title" }, h("span", { text: "Follow" }), h("span", { class: "dim", text: " the stage." })),
      h("p", { class: "ph-lede", text: "Scan the code on the laptop's Listen screen, or pick the session here. The phone mirrors it and can turn Auto voice focus on or off, or press Focus or Quiet. It plays nothing." }),
      list),
    h("section", { class: "ph-solo", "aria-labelledby": "ph-solo-title" },
      h("h2", { id: "ph-solo-title", class: "ph-solo-title" }, h("span", { text: "Solo" }), h("span", { class: "ph-tag", text: "off stage" })),
      h("p", { class: "ph-lede ph-solo-lede", text: soloLede(null) }),
      soloBtn, soloWhy, soloLive));
  root.replaceChildren(frame);

  // T-463: the expected delay is the server's (/health tse_window), not a fixed figure
  fetchHealth().then((health) => { if (!ctx.signal?.aborted) setText(frame.querySelector(".ph-solo-lede"), soloLede(health)); });
  let gate = { allowed: false, reason: "Checking the laptop for a stage session…" };
  let solo = null; // {client, lock}
  let timer = 0;
  let sdk = null; // loaded now so the Start tap can create the AudioContext synchronously
  import("/sdk/hearing-client.js").then((m) => { sdk = m; }, (err) => console.warn("[phone] SDK not loaded", err));
  const signal = ctx.signal;

  // T-458: each row says whether that stage is listening (its telemetry seq moves) and for how long it has been open;
  // live sessions first, idle ones greyed and labelled idle
  const probes = new Map();
  function renderSessions(sessions) {
    const rank = (s) => ({ true: 0, null: 1, false: 2 })[String(probes.get(s.id)?.live ?? null)];
    const stage = stageSessions(sessions).sort((a, b) => rank(a) - rank(b));
    list.replaceChildren(...(stage.length ? stage.map((s) => {
      const r = sessionRow(s, probes.get(s.id));
      return h("li", { class: "ph-session", "data-live": String(r.live) },
        h("a", { href: `/app/?watch=${encodeURIComponent(s.id)}`, class: "ph-session-link" },
          h("span", { class: "ph-session-main" },
            h("span", { class: "ph-session-client", text: r.title }),
            h("span", { class: "ph-session-state", text: [r.state, r.age].filter(Boolean).join(" · ") })),
          h("span", { class: "ph-session-id", text: String(s.id).slice(0, 8) }),
          h("span", { class: "ph-session-go", text: r.go })));
    })
      : [h("li", { class: "ph-session empty", text: "No stage session is open on the laptop." })]));
  }
  async function probeAll(sessions) {
    const stage = stageSessions(sessions).slice(0, 6);
    const seen = await Promise.all(stage.map((s) => probeSession(s.id).then((p) => [s.id, p])));
    for (const [id, p] of seen) probes.set(id, p);
    for (const id of [...probes.keys()]) if (!stage.some((s) => s.id === id)) probes.delete(id);
  }
  function renderGate() {
    soloBtn.disabled = Boolean(solo) || !gate.allowed;
    soloBtn.hidden = Boolean(solo);
    soloWhy.textContent = solo ? "" : gate.reason;
    soloWhy.dataset.kind = gate.allowed ? "ok" : "refused";
    frame.dataset.solo = solo ? "on" : gate.allowed ? "ready" : "refused";
  }
  async function poll() {
    const sessions = await listSessions();
    if (signal?.aborted) return;
    renderSessions(sessions);
    await probeAll(sessions);
    if (signal?.aborted) return;
    const probed = sessions.map((s) => ({ ...s, live: probes.get(s.id)?.live ?? null }));
    renderSessions(probed);
    gate = soloGate(probed);
    if (gate.allowed && !solo) gate.reason = "No stage session is open. Solo can start.";
    renderGate();
    timer = setTimeout(poll, POLL_MS);
  }
  poll();

  async function requestWakeLock() {
    window.__soloWakeLock = { requested: true, state: "pending" };
    if (!navigator.wakeLock?.request) {
      window.__soloWakeLock.state = "unsupported";
      return { text: "This browser has no screen wake lock: keep the screen on yourself.", lock: null };
    }
    try {
      const lock = await navigator.wakeLock.request("screen");
      window.__soloWakeLock.state = "held";
      return { text: "Screen kept awake while Solo runs.", lock };
    } catch (err) {
      window.__soloWakeLock.state = `refused: ${err?.name || err}`;
      return { text: `Screen may lock (${err?.name || "wake lock refused"}): keep it on yourself.`, lock: null };
    }
  }

  async function startSolo() {
    if (solo || !gate.allowed) return;
    // inside the tap: the SDK creates the AudioContext before anything is awaited (iOS), and the wake lock asks now
    if (!sdk) {
      soloWhy.textContent = "The browser SDK (/sdk/hearing-client.js) did not load, so Solo cannot start.";
      return;
    }
    const client = new sdk.HearingClient({ client: "solo" });
    const started = client.start();
    const locked = requestWakeLock();
    solo = { client, lock: null };
    renderGate();
    soloLive.hidden = false;
    const state = soloLive.querySelector(".ph-solo-state");
    const facts = soloLive.querySelector(".ph-solo-facts");
    const caps = soloLive.querySelector(".ph-solo-caps");
    state.textContent = "Opening the mic";
    client.addEventListener("telemetry", (e) => {
      const t = e.detail;
      const total = t?.latency_ms?.total;
      facts.textContent = `Engine ${t?.tier || "—"} · ${typeof total === "number" ? `${Math.round(total)} ms` : "—"} · ${t?.device || "—"}`;
      state.textContent = t?.mode === "quiet" ? "Quiet: captions only" : "Listening through your phone";
    });
    client.addEventListener("caption", (e) => {
      caps.prepend(h("p", { class: e.detail.final ? "ph-cap" : "ph-cap partial", text: e.detail.text }));
      while (caps.children.length > 4) caps.lastChild.remove();
    });
    client.addEventListener("close", () => { if (solo?.client === client) stopSolo("The engine closed the session."); });
    try {
      await started;
      client.control({ profile: "judge", ...modeFields("focus_hq") });
      state.textContent = "Listening through your phone";
    } catch (err) {
      stopSolo(`Solo could not start: ${err?.message || err}`);
      return;
    }
    const lk = await locked;
    if (!solo) { lk.lock?.release?.(); return; }
    solo.lock = lk.lock;
    soloLive.querySelector(".ph-solo-lock").textContent = lk.text;
    // a stage session that opened while we were starting wins: the phone steps back
    const listed = await listSessions();
    await probeAll(listed);
    const after = soloGate(listed.map((s) => ({ ...s, live: probes.get(s.id)?.live ?? null })));
    if (!after.allowed && solo) stopSolo(after.reason);
  }

  function stopSolo(reason = "") {
    if (!solo) return;
    const { client, lock } = solo;
    solo = null;
    try { client.close(); } catch { /* already closed */ }
    lock?.release?.().catch?.(() => undefined);
    soloLive.hidden = true;
    renderGate();
    if (reason) soloWhy.textContent = reason;
  }

  soloBtn.addEventListener("click", () => { startSolo(); }, { signal });
  soloLive.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-solo]");
    if (!b || !solo) return;
    if (b.dataset.solo === "stop") return stopSolo();
    solo.client.control(modeFields(b.dataset.solo === "quiet" ? "quiet" : "focus_hq"));
    for (const x of soloLive.querySelectorAll('[data-solo="focus"], [data-solo="quiet"]')) x.setAttribute("aria-pressed", String(x === b));
  }, { signal });
  // the wake lock is dropped when the page hides; take it again when Solo is still running
  document.addEventListener("visibilitychange", async () => {
    if (document.visibilityState !== "visible" || !solo || solo.lock?.released === false) return;
    const lk = await requestWakeLock();
    if (solo) solo.lock = lk.lock; else lk.lock?.release?.();
  }, { signal });

  renderGate();
  return () => { clearTimeout(timer); stopSolo(); };
}
