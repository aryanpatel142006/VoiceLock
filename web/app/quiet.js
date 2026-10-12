// Quiet mode for the app (PR #402): the engine's existing Quiet (T-149 engine, T-150 /judge F/Q buttons), as a calm
// phone surface.
//
// enterQuiet() sends the very control /judge sends for Quiet (judge-modes.js QUIET_CONTROL = {type:"control",
// mode:"quiet"}): the engine stops playing the extracted voice (its output ramps to digital silence) and captions only
// the selected person. exitQuiet() goes back to Focus ({mode:"focus_hq"}), the existing processed mode.
//
// Honest by construction:
// - The header says "app audio paused": Quiet silences only this app's output. It is not noise cancellation.
// - Every card is an engine `caption` message ({text, final, t0, t1}); nothing here writes a caption. With no session
//   or no caption yet the surface shows the empty state.
// - The speaker label is the engine's enrolled profile (telemetry.enrollment.profile) when it reports one, else none.
// - No "Hear it": the engine saves no caption clips, so a replay button would never show (cut, Oct 11).
// - embedded:true (Comfort's Stay Quiet card inside SOS): no header of its own and no Back to Focus (↓ / the hold
//   already go back); the host card says "Stay Quiet · Nothing plays. Their words appear here."
// - This module never starts or stops sounds (music, guidance): it only sends the mode control. No sound on arrival.
//
// Pure parts (quietInit/quietReduce, speakerOf, timeLabel) run under node --test; mountQuiet is the browser
// half. Tests: web/app/tests/quiet.test.mjs.

import { QUIET_CONTROL } from "../console/judge-modes.js";
import { loudestMode } from "../console/gate-state.js";

export const MAX_CARDS = 5;
export const FOCUS_MODE = "focus_hq";
export const HEADER = "Quiet · app audio paused";
export const SUBHEAD = "Only this app's sound is paused. This is not noise cancellation.";
export const EMPTY = "Captions show here when someone near you speaks.";
export const EMPTY_EMBEDDED = "Waiting for them to speak.";
export const PAUSED_EMBEDDED = "Captions are paused until their voice is clear again.";

// ---------------------------------------------------------------- engine controls

function defaultSession() {
  return (typeof window !== "undefined" && window.hearingApp?.session) || null;
}

/** Quiet: the engine's own Quiet control, through the app session. Returns the fields sent, or null without a session. */
export function enterQuiet(session = defaultSession()) {
  if (!session) return null;
  if (typeof session.setMode === "function") session.setMode(QUIET_CONTROL.mode);
  else session.send({ ...QUIET_CONTROL });
  return { ...QUIET_CONTROL };
}

/** Back to Focus (the existing processed mode). */
export function exitQuiet(session = defaultSession()) {
  if (!session) return null;
  if (typeof session.setMode === "function") session.setMode(FOCUS_MODE);
  else session.send({ mode: FOCUS_MODE });
  return { mode: FOCUS_MODE };
}

// ---------------------------------------------------------------- pure helpers

/**
 * The engine's name for the captioned person, or null (never a guess). Under HEARING_PICK=loudest the closest voice is
 * captioned, whoever it is, so the profile the preset loaded is not named.
 */
export function speakerOf(msg, telemetry) {
  if (!msg?.speaker && loudestMode(telemetry)) return "Closest voice";
  const s = msg?.speaker || telemetry?.enrollment?.profile || null;
  if (typeof s !== "string" || !s.trim()) return null;
  const t = s.trim();
  return t[0].toUpperCase() + t.slice(1);
}

/** 12-hour local time of arrival, e.g. "6:42 pm". */
export function timeLabel(ms) {
  const d = new Date(ms);
  let h = d.getHours();
  const m = String(d.getMinutes()).padStart(2, "0");
  const ap = h >= 12 ? "pm" : "am";
  h = h % 12 || 12;
  return `${h}:${m} ${ap}`;
}

const key = (t) => String(t || "").toLowerCase().replace(/[^\p{L}\p{N}' ]+/gu, "").replace(/\s+/g, " ").trim();

export function quietInit() {
  return { cards: [], nextId: 1 };
}

/**
 * Events: {type:"caption", msg, at, speaker} · {type:"dismiss", id} · {type:"clear"}.
 * A partial caption updates the newest card while it is still in progress; the final settles it. A final repeating
 * the final just before it is dropped (store.js addCaption does the same). The newest MAX_CARDS cards are kept.
 */
export function quietReduce(s, ev) {
  switch (ev.type) {
    case "caption": {
      const text = String(ev.msg?.text || "").trim();
      if (!text) return s;
      const final = Boolean(ev.msg.final);
      let cards = s.cards.slice();
      const last = cards[cards.length - 1];
      if (last && !last.final) cards.pop();
      const prev = cards[cards.length - 1];
      if (final && prev?.final && key(prev.text) === key(text)) return { ...s, cards };
      const id = last && !last.final ? last.id : s.nextId;
      const card = { id, text, final, speaker: ev.speaker ?? null, at: last && !last.final ? last.at : ev.at ?? 0 };
      cards.push(card);
      return { cards: cards.slice(-MAX_CARDS), nextId: id === s.nextId ? s.nextId + 1 : s.nextId };
    }
    case "dismiss":
      return { ...s, cards: s.cards.filter((c) => c.id !== ev.id) };
    case "clear":
      return { ...s, cards: [] };
    default:
      return s;
  }
}

// ---------------------------------------------------------------- browser half

/**
 * Render the Quiet surface into `el` and follow the session's caption events.
 * opts: {session, doc, win, now, autoEnter, embedded} (defaults: window.hearingApp.session, document, window, Date.now,
 * true, false). embedded: the captions only (no header, no Back to Focus), for a host card that has its own title.
 * Returns {state, enter, exit, dismiss, clear, destroy}. destroy() unsubscribes; it does not change the mode.
 */
export function mountQuiet(el, opts = {}) {
  const doc = opts.doc || el.ownerDocument || document;
  const win = opts.win || (typeof window !== "undefined" ? window : {});
  const session = opts.session === undefined ? defaultSession() : opts.session;
  const store = session?.store || null;
  const now = opts.now || (() => Date.now());
  let s = quietInit();

  const href = new URL("./quiet.css", import.meta.url).href;
  const styled = () => Array.from(doc.head.querySelectorAll?.('link[rel="stylesheet"]') || []).some((l) => l.href === href);
  if (doc.head && !opts.noStyles && !styled()) { // once, however often mounted (index.html loads it with the page)
    const link = doc.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    doc.head.append(link);
  }
  const reduced = Boolean(win.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches);

  const node = (tag, cls, text) => {
    const n = doc.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  const embedded = Boolean(opts.embedded);
  const root = node("section", "qm");
  if (embedded) {
    root.dataset.embedded = "1";
    root.setAttribute("aria-label", "Captions");
  } else root.setAttribute("aria-labelledby", "qm-title");
  if (reduced) root.dataset.still = "1";
  const head = node("header", "qm-head");
  const title = node("h2", "qm-title", HEADER);
  title.id = "qm-title";
  const sub = node("p", "qm-sub", SUBHEAD);
  const back = node("button", "qm-back", "Back to Focus");
  back.type = "button";
  back.addEventListener("click", () => exitQuiet(session));
  head.append(title, sub, back);

  const status = node("p", "qm-status");
  status.hidden = true;
  const list = node("ol", "qm-list");
  list.setAttribute("aria-label", "Captions");
  const empty = node("p", "qm-empty", embedded ? EMPTY_EMBEDDED : EMPTY);
  // one Clear for the whole list (a per-card Dismiss was a row of buttons under every caption)
  const clearBtn = node("button", "qm-btn qm-clear", "Clear");
  clearBtn.type = "button";
  clearBtn.setAttribute("aria-label", "Clear captions");
  clearBtn.addEventListener("click", () => api.clear());
  const actions = node("div", "qm-actions");
  actions.append(clearBtn);
  // a polite live region announces each settled caption once (no sound)
  const live = node("p", "qm-live");
  live.setAttribute("aria-live", "polite");
  root.append(...(embedded ? [] : [head]), status, empty, list, actions, live);
  el.replaceChildren(root);

  const rendered = new Map(); // id -> {li, text}

  function cardNode(card) {
    const li = node("li", "qm-card");
    li.dataset.id = String(card.id);
    if (!reduced) li.dataset.enter = "1";
    const meta = node("p", "qm-meta");
    const who = node("span", "qm-who", card.speaker || "");
    who.hidden = !card.speaker;
    const when = node("time", "qm-time", timeLabel(card.at));
    when.setAttribute("datetime", new Date(card.at).toISOString());
    meta.append(who, when);
    const text = node("p", "qm-text", card.text);
    li.append(meta, text);
    // the fade-in plays once: re-inserting the card (reorder) must not replay it on every partial
    li.addEventListener?.("animationend", () => { delete li.dataset.enter; }, { once: true });
    return { li, text, final: false };
  }

  function render() {
    const ids = new Set(s.cards.map((c) => c.id));
    for (const [id, r] of rendered) if (!ids.has(id)) { r.li.remove(); rendered.delete(id); }
    for (const card of s.cards) {
      let r = rendered.get(card.id);
      if (!r) { r = cardNode(card); rendered.set(card.id, r); }
      if (r.text.textContent !== card.text) r.text.textContent = card.text;
      r.li.classList?.toggle?.("qm-partial", !card.final);
      if (card.final && !r.final) live.textContent = `${card.speaker ? `${card.speaker}: ` : ""}${card.text}`;
      r.final = card.final;
    }
    // newest first, so the latest caption sits under the header on a phone; touch the list only when the order
    // changed (a partial updates text in place, and re-inserting a card would restart its fade-in: flicker)
    const order = s.cards.slice().reverse().map((c) => rendered.get(c.id).li);
    const shown = Array.from(list.children || []);
    if (order.length !== shown.length || order.some((li, i) => li !== shown[i])) list.replaceChildren(...order);
    empty.hidden = s.cards.length > 0;
    list.hidden = s.cards.length === 0;
    actions.hidden = s.cards.length === 0;
  }

  function showStatus(t) {
    const q = t?.quiet;
    const paused = embedded ? PAUSED_EMBEDDED : q?.reason || "Captions paused";
    const text = !session ? "Not connected to the engine." : q?.feed === "paused" ? paused : "";
    if (status.textContent !== text) status.textContent = text; // 10 Hz telemetry: a DOM write only on a change
    status.hidden = !text;
  }

  // opts.paused(): the host's surface is not showing (Comfort mounts this once, at idle); nothing is rendered until it
  // is, and the host clears the cards when it opens
  const paused = typeof opts.paused === "function" ? opts.paused : () => false;
  const offs = [];
  if (store?.on) {
    offs.push(store.on("caption", (msg) => {
      if (paused()) return;
      s = quietReduce(s, { type: "caption", msg, at: now(), speaker: speakerOf(msg, store.state?.telemetry) });
      render();
    }));
    offs.push(store.on("telemetry", (t) => { if (!paused()) showStatus(t); }));
  }
  showStatus(store?.state?.telemetry);
  render();

  const api = {
    get state() { return s; },
    enter: () => enterQuiet(session),
    exit: () => exitQuiet(session),
    dismiss(id) {
      s = quietReduce(s, { type: "dismiss", id });
      render();
      // keep keyboard focus on the list's controls after a dismiss
      const next = s.cards.length ? clearBtn : embedded ? null : back;
      next?.focus?.();
    },
    clear() {
      s = quietReduce(s, { type: "clear" });
      render();
      if (!embedded) back.focus?.(); // Clear hid itself; focus goes to the header's Back, not the page
    },
    destroy() {
      for (const off of offs) off?.();
    },
  };
  if (opts.autoEnter !== false) api.enter();
  return api;
}
