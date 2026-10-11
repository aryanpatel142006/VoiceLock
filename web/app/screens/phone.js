// Phone (M14, T-351): the companion and Solo mode.
//
// Companion (`/app/?watch=<session>`, from the QR on the laptop's Listen screen): mirrors the stage session over
// `WS /ws/watch` (../watch.js): the state word, who is targeted, the captions, and four controls forwarded into the
// stage session: Focus, Quiet, a saved person, Release. It creates no engine session and plays no audio: the phone
// is never the stage mic or the stage output.
//
// Solo (off stage, a labelled extra): the phone's own mic and its own engine session through the browser SDK
// (/sdk/hearing-client.js). It is refused while a stage session (`/judge` or the app) is open on the laptop, because
// a second session loads the same MPS device and can overrun the stage tier; the screen says why. Solo asks for a
// screen wake lock (iOS stops audio when the screen locks).
//
// Honesty: every word and number here is mirrored telemetry, a caption message, or this page's own state. A mirror
// that stops ticking says so; nothing animates to look alive.

import { ageText, createWatch, listSessions, probeSession, stageSessions } from "../watch.js";
import { listenModel } from "./listen.js";
import { modeFields } from "../store.js";

export const title = "Phone";

const POLL_MS = 2000;

/** Pure: one lobby row's words for a session and its probe ({live, who}). */
export function sessionRow(s, probe = null, nowMs = Date.now()) {
  const kind = s.client === "judge" ? "Stage · /judge" : "Stage · app";
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
    sentence = w.reason;
  } else if (w.state !== "open") {
    word = w.state === "reconnecting" ? "Reconnecting" : "Connecting";
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
  // the companion has no Selective button: the way back to one voice is on the laptop
  if (m.tone === "raw" && w.state === "open") sentence = m.who ? `Selective on the laptop plays ${m.who} alone.` : "";
  const live = w.state === "open" && !w.stale && w.streaming !== false;
  let stage = "";
  if (live) {
    if (m.mode === "quiet") stage = "Stage is in Quiet: captions only";
    else if (m.ab === "raw") stage = "Stage plays the room (Raw)";
    else if (m.state === "isolated") stage = m.who ? `Stage hears ${m.who}` : "Stage hears the isolated voice";
    else stage = "Stage is not isolated yet";
  }
  return {
    word, tone, sentence, stage, live,
    // the name only while it is what the stage plays (same as Listen: not under Raw)
    who: live && m.state === "isolated" && m.tone !== "raw" && m.who ? m.who : "",
    target: m.targetLine,
    mode: m.mode,
    ab: m.ab,
    people: m.people,
    released: m.released,
    captions: m.captions.lines,
    captionNote: m.captions.paused ? `Captions paused: ${m.captions.paused}` : m.captions.degraded,
    replay: m.replay,
    delay: m.truth.delay,
    tier: m.truth.tier,
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
  const s = stage[0];
  const who = s.client === "judge" ? "/judge" : "the app";
  return {
    allowed: false,
    reason: `A stage session (${who}, ${String(s.id).slice(0, 8)}) is live on the laptop. A second engine session `
      + "would share its GPU and can make the stage stutter, so Solo waits until it ends. Follow the stage instead.",
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
  const id = params.get("watch");
  root.classList.add("screen-phone");
  document.body.dataset.phone = id ? "companion" : "lobby";
  const cleanups = [() => { delete document.body.dataset.phone; delete document.body.dataset.ab; }];
  ctx.signal?.addEventListener("abort", () => cleanups.splice(0).forEach((fn) => fn()), { once: true });
  if (id) cleanups.push(mountCompanion(root, id, ctx));
  else cleanups.push(mountLobby(root, ctx));
  return () => cleanups.splice(0).forEach((fn) => fn());
}

// ------------------------------------------------------------------ companion

function mountCompanion(root, sessionId, ctx) {
  const watch = createWatch(sessionId);
  window.hearingWatch = watch; // scripts/phone_check.py reads it
  const store = watch.store;

  const btn = (label, attrs) => h("button", { type: "button", class: "ph-btn", "aria-pressed": "false", ...attrs },
    h("span", { class: "ph-btn-label", text: label }));
  const frame = h("div", { class: "phone", "data-role": "companion", "data-tone": "idle", "data-ab": "processed", "data-mode": "focus_hq" },
    h("header", { class: "ph-top" },
      h("p", { class: "ph-link" }, h("span", { class: "ph-dot", "aria-hidden": "true" }), h("span", { class: "ph-link-text", text: "Joining" })),
      h("p", { class: "ph-replay", hidden: true, text: "Replay" })),
    h("section", { class: "ph-hero", "aria-labelledby": "ph-word", "aria-live": "polite" },
      h("p", { class: "ph-stage" }),
      h("h1", { id: "ph-word", class: "ph-word", text: "Connecting" }),
      h("p", { class: "ph-who" }),
      h("p", { class: "ph-sentence" }),
      h("p", { class: "ph-target" })),
    h("section", { class: "ph-captions", "aria-label": "Captions" },
      h("p", { class: "ph-cap-head" }, h("span", { text: "Captions" }), h("span", { class: "ph-cap-note" })),
      h("div", { class: "ph-cap-lines", "aria-live": "polite" })),
    h("nav", { class: "ph-controls", "aria-label": "Control the stage" },
      h("div", { class: "ph-row ph-modes", role: "group", "aria-label": "Mode" },
        btn("Focus", { "data-act": "focus" }), btn("Quiet", { "data-act": "quiet" })),
      h("div", { class: "ph-row ph-people", role: "group", "aria-label": "Person" },
        btn("Person 1", { "data-act": "person", "data-index": "0", disabled: true }),
        btn("Person 2", { "data-act": "person", "data-index": "1", disabled: true }),
        btn("Release", { "data-act": "release", class: "ph-btn ph-release" }))),
    h("footer", { class: "ph-foot" },
      h("span", { class: "ph-fact", "data-fact": "delay" }), h("span", { class: "ph-fact", "data-fact": "tier" }),
      h("a", { class: "ph-leave", href: "/app/#phone", text: "Leave" })));
  root.replaceChildren(frame);

  const q = (sel) => frame.querySelector(sel);
  const el = {
    link: q(".ph-link-text"), replay: q(".ph-replay"), stage: q(".ph-stage"), word: q(".ph-word"), who: q(".ph-who"),
    sentence: q(".ph-sentence"), target: q(".ph-target"), capLines: q(".ph-cap-lines"), capNote: q(".ph-cap-note"),
    delay: q('[data-fact="delay"]'), tier: q('[data-fact="tier"]'),
  };

  frame.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-act]");
    if (!b || b.disabled) return;
    const ok = watch.act(b.dataset.act, { index: Number(b.dataset.index || 0) });
    b.dataset.flash = ok ? "sent" : "failed";
    setTimeout(() => delete b.dataset.flash, 600);
  }, { signal: ctx.signal });

  let shownCaptions = "";
  function render() {
    const m = companionModel(store.state);
    frame.dataset.tone = m.tone;
    frame.dataset.ab = m.ab;
    frame.dataset.mode = m.mode;
    frame.dataset.live = String(m.live);
    frame.dataset.watch = store.state.watch?.state || "connecting";
    document.body.dataset.ab = m.live ? m.ab : "";
    setText(el.link, m.canSend ? "Following the stage" : `Stage ${m.session} · ${store.state.watch?.state}`);
    setText(el.word, m.word);
    setText(el.who, m.who);
    el.who.hidden = !m.who;
    setText(el.stage, m.stage);
    setText(el.sentence, m.sentence);
    setText(el.target, m.target);
    el.target.hidden = Boolean(m.who) && m.target === `Target: ${m.who}`; // the name says it once
    el.replay.hidden = !m.replay;
    setText(el.replay, m.replay);
    setText(el.delay, `Delay ${m.delay}`);
    setText(el.tier, `Tier ${m.tier}`);
    setText(el.capNote, m.captionNote || ""); // status only, no instructions in the status slot
    const key = JSON.stringify(m.captions);
    if (key !== shownCaptions) {
      shownCaptions = key;
      el.capLines.replaceChildren(...(m.captions.length
        ? m.captions.map((c) => h("p", { class: c.final ? "ph-cap" : "ph-cap partial", text: c.text }))
        : [h("p", { class: "ph-cap empty", text: "No captions yet. They appear when the target speaks." })]));
    }
    for (const b of frame.querySelectorAll("button[data-act]")) {
      const act = b.dataset.act;
      if (act === "person") {
        const p = m.people[Number(b.dataset.index)];
        setText(b.querySelector(".ph-btn-label"), p.label);
        b.disabled = !m.canSend || !p.enabled;
        b.setAttribute("aria-pressed", String(p.pressed));
      } else {
        b.disabled = !m.canSend;
        const pressed = act === "release" ? m.live && !m.people.some((p) => p.pressed) && m.target.startsWith("Target: none")
          : m.live && m.mode === (act === "quiet" ? "quiet" : "focus_hq");
        b.setAttribute("aria-pressed", String(pressed));
      }
    }
  }
  const off = store.on("change", render);
  render();
  return () => { off(); watch.close(); if (window.hearingWatch === watch) delete window.hearingWatch; };
}

// ------------------------------------------------------------------ lobby: pick a stage session, or Solo

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
      h("p", { class: "ph-lede", text: "Scan the code on the laptop's Listen screen, or pick the session here. The phone mirrors it and can press Focus, Quiet, a person or Release. It plays nothing." }),
      list),
    h("section", { class: "ph-solo", "aria-labelledby": "ph-solo-title" },
      h("h2", { id: "ph-solo-title", class: "ph-solo-title" }, h("span", { text: "Solo" }), h("span", { class: "ph-tag", text: "off stage" })),
      h("p", { class: "ph-lede", text: "Your phone's own mic, its own session, your headphones. About 1.5 s behind, like live captions. Keep the screen on: iOS stops audio when it locks." }),
      soloBtn, soloWhy, soloLive));
  root.replaceChildren(frame);

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
