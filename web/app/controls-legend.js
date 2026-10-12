// Contextual controls legend + key handler (Comfort Companion, PR #402).
//
// One small control surface that follows the screen: the same five actions (helpful, notHelpful, quiet, focus, stop)
// plus remember / notNow, mapped to a keyboard or a presenter remote, and shown as a legend of on-screen buttons that
// say in plain words what each one does RIGHT NOW. It is a slim, muted key hint (the big on-screen buttons are the
// touch path); controls-legend.css hides it under 600 px, where there is no keyboard or remote. Remote key codes are not confirmed yet, so the label says
// "Keyboard / remote" and the mapping is the plain arrow / page / escape keys a remote usually sends.
//
//   context    ArrowRight / PageDown   ArrowLeft / PageUp    ArrowUp   ArrowDown   Escape
//   focus      (nothing)               (nothing)             quiet     focus       stop
//   quiet      (nothing)               (nothing)             quiet     focus       stop
//   comfort    (nothing)               (nothing)             quiet     focus       stop
//   feedback   helpful                 notHelpful            quiet     focus       stop
//   remember   remember                notNow                quiet     focus       stop
//
// In 'focus' left/right do nothing on purpose: person picking was removed. A key whose handler is missing is not
// handled (and so keeps its browser meaning). Everything pure is exported for node --test; the DOM parts only touch
// `document` when it exists.

export const CONTEXTS = ["focus", "quiet", "comfort", "feedback", "remember"];
export const ACTIONS = ["helpful", "notHelpful", "quiet", "focus", "stop", "remember", "notNow"];
export const LEGEND_LABEL = "Keyboard / remote";
export const FLASH_MS = 150;
export const CONFIRM_MS = 1400;

/** What each action says on the page (plain words) and which key shows it. */
const ACTION_INFO = {
  helpful: { glyph: "→", label: "Helpful", key: "Right arrow" },
  notHelpful: { glyph: "←", label: "Not helpful", key: "Left arrow" },
  remember: { glyph: "→", label: "Remember this", key: "Right arrow" },
  notNow: { glyph: "←", label: "Not now", key: "Left arrow" },
  quiet: { glyph: "↑", label: "Quiet", key: "Up arrow" },
  focus: { glyph: "↓", label: "Focus", key: "Down arrow" },
  stop: { glyph: "Esc", label: "Stop", key: "Escape" },
};

/** The confirmation words flash() prints (past tense, still plain). */
const CONFIRM_TEXT = {
  helpful: "Marked helpful", notHelpful: "Marked not helpful", remember: "Remembered", notNow: "Not now",
  quiet: "Quiet", focus: "Focus", stop: "Stopped",
};

/** Which action a key means in a context, or null. Pure. */
export function actionForKey(ctx, key) {
  switch (key) {
    case "ArrowUp": return "quiet";
    case "ArrowDown": return "focus";
    case "Escape": return "stop";
    case "ArrowRight": case "PageDown":
      return ctx === "feedback" ? "helpful" : ctx === "remember" ? "remember" : null;
    case "ArrowLeft": case "PageUp":
      return ctx === "feedback" ? "notHelpful" : ctx === "remember" ? "notNow" : null;
    default: return null;
  }
}

/**
 * The legend for a context, left to right: [{action, glyph, label, key}]. What is already true is not offered:
 * in 'focus' there is no "Focus" button, in 'quiet' no "Quiet" button (the keys still work, harmlessly).
 */
export function legendItems(ctx) {
  const order = {
    focus: ["quiet", "stop"],
    quiet: ["focus", "stop"],
    comfort: ["quiet", "focus", "stop"],
    feedback: ["notHelpful", "helpful", "quiet", "focus", "stop"],
    remember: ["notNow", "remember", "quiet", "focus", "stop"],
  }[ctx] || [];
  return order.map((action) => ({ action, ...ACTION_INFO[action] }));
}

/** The one-line legend in words, e.g. "← Not helpful · → Helpful · ↑ Quiet · ↓ Focus · Esc Stop". */
export function legendText(ctx) {
  return legendItems(ctx).map((i) => `${i.glyph} ${i.label}`).join(" · ");
}

/** True when typing is happening: input, textarea, select or contenteditable (also in a shadow root's active element). */
export function isEditable(target) {
  if (!target) return false;
  const tag = String(target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return true;
  if (target.isContentEditable === true) return true;
  const ce = typeof target.getAttribute === "function" ? target.getAttribute("contenteditable") : null;
  return ce !== null && ce !== undefined && String(ce).toLowerCase() !== "false";
}

// ---- state ---------------------------------------------------------------------------------------------------------
let state = { ctx: "focus", handlers: {} };
const listeners = new Set();
let attached = null; // {doc, fn}

function emit() { for (const fn of [...listeners]) fn(); }

/** Run an action as if its key or button was pressed: flash, then call the handler. Returns true if one ran. */
export function trigger(action) {
  const fn = state.handlers?.[action];
  if (typeof fn !== "function") return false;
  flash(action);
  fn(action);
  return true;
}

/**
 * Keydown handler. Returns the action it handled, or null. Ignores: modified keys, auto-repeat, typing in a field,
 * keys the context does not map, and actions with no handler (none of those get preventDefault).
 */
export function onKeydown(e) {
  if (!e || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return null;
  if (isEditable(e.target)) return null;
  if (e.target?.closest?.("[role='slider']")) return null; // a dial's arrows are its own (the room dial on Listen)
  const action = actionForKey(state.ctx, e.key);
  if (!action || typeof state.handlers?.[action] !== "function") return null;
  if (e.repeat) { e.preventDefault?.(); return null; } // held key: swallow so a remote hold does not scroll, act once
  e.preventDefault?.();
  trigger(action);
  return action;
}

/**
 * Set the screen's control context and its handlers, and start listening on `document` (once). The mapping switches
 * with `ctx`; handlers are {helpful, notHelpful, quiet, focus, stop, remember, notNow}, any subset.
 * Returns a function that removes the listener.
 */
export function setControlContext(ctx, handlers = {}) {
  if (!CONTEXTS.includes(ctx)) throw new RangeError(`unknown control context: ${ctx}`);
  state = { ctx, handlers: { ...handlers } };
  const doc = globalThis.document;
  if (doc && !attached) {
    attached = { doc, fn: onKeydown };
    doc.addEventListener("keydown", attached.fn);
  }
  emit();
  return disposeControls;
}

export function disposeControls() {
  if (attached) attached.doc.removeEventListener("keydown", attached.fn);
  attached = null;
}

export function getControlContext() { return state.ctx; }

// ---- legend --------------------------------------------------------------------------------------------------------
const mounted = new Set(); // {el, status, timers}

function renderInto(m) {
  const { el } = m;
  const items = legendItems(state.ctx).filter((i) => typeof state.handlers?.[i.action] === "function");
  el.dataset.context = state.ctx;
  const list = el.querySelector(".cl-items");
  list.textContent = "";
  for (const it of items) {
    const b = el.ownerDocument.createElement("button");
    b.type = "button";
    b.className = "cl-btn";
    b.dataset.action = it.action;
    b.setAttribute("aria-label", `${it.label} (${it.key})`);
    b.setAttribute("aria-keyshortcuts", it.key === "Right arrow" ? "ArrowRight PageDown"
      : it.key === "Left arrow" ? "ArrowLeft PageUp" : it.key === "Up arrow" ? "ArrowUp"
      : it.key === "Down arrow" ? "ArrowDown" : "Escape");
    const g = el.ownerDocument.createElement("span");
    g.className = "cl-glyph"; g.setAttribute("aria-hidden", "true"); g.textContent = it.glyph;
    const t = el.ownerDocument.createElement("span");
    t.className = "cl-text"; t.textContent = it.label;
    b.append(g, t);
    b.addEventListener("click", () => trigger(it.action));
    list.append(b);
  }
}

/**
 * Render the persistent legend into `el`: a "Keyboard / remote" label and one button per action available right now.
 * It re-renders on every setControlContext. Returns {unmount}.
 */
export function mountLegend(el) {
  const doc = el.ownerDocument;
  el.classList.add("cl-legend");
  el.setAttribute("role", "group");
  el.setAttribute("aria-label", LEGEND_LABEL);
  el.textContent = "";
  const label = doc.createElement("span");
  label.className = "cl-label"; label.textContent = LEGEND_LABEL;
  const list = doc.createElement("div");
  list.className = "cl-items";
  const status = doc.createElement("span");
  status.className = "cl-status"; status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
  el.append(label, list, status);
  const m = { el, status, flashTimer: null, confirmTimer: null };
  mounted.add(m);
  const fn = () => renderInto(m);
  listeners.add(fn);
  m.fn = fn;
  renderInto(m);
  return {
    unmount() {
      listeners.delete(fn); mounted.delete(m);
      clearTimeout(m.flashTimer); clearTimeout(m.confirmTimer);
      el.textContent = ""; el.classList.remove("cl-legend");
    },
  };
}

/** A subtle confirmation of `action`: its legend button lights for 150 ms and the status line names it for ~1.4 s. */
export function flash(action) {
  for (const m of mounted) {
    const btn = m.el.querySelector(`.cl-btn[data-action="${action}"]`);
    if (btn) {
      btn.dataset.flash = "1";
      clearTimeout(m.flashTimer);
      m.flashTimer = setTimeout(() => { delete btn.dataset.flash; }, FLASH_MS);
    }
    m.status.textContent = CONFIRM_TEXT[action] || "";
    m.status.dataset.on = "1";
    clearTimeout(m.confirmTimer);
    m.confirmTimer = setTimeout(() => { m.status.textContent = ""; delete m.status.dataset.on; }, CONFIRM_MS);
  }
}
