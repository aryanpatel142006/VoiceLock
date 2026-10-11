// App keys (M14, T-347): the stage keys, identical to /judge (web/console/judge.js onKey) so STAGE_CARD.md holds.
//   Space      RAW <-> Selective (A/B)
//   Shift+F/Q  Focus / Quiet            (bare F/Q do nothing, T-150: a judge leaning on the keyboard fires nothing)
//   1 / 2      saved person 1 / 2
//   Shift+0    Release                  (or 0 with the Diagnostics sheet open, as with /judge's drawer)
//   D          Diagnostics sheet (the /judge drawer)
//   Shift+R    replay clip (REPLAY badge) on / off
//   Shift+C    calibrate a new voice (off the stage path, as on /judge)
// /judge's N (Name Call) has no app counterpart: it is ignored here.
// keyAction() is pure (node --test covers it); bindKeys()/watchKeyFocus() wire it to the document.

import { KeyFocus, isReleaseKey } from "../console/judge-targets.js";
import { modeForKey } from "../console/judge-modes.js";

export { KEYS_OFF_TEXT } from "../console/judge-targets.js";

/** The key legend the Listen screen prints next to each control. */
export const KEY_HINTS = {
  ab: "Space",
  focus_hq: "Shift F",
  quiet: "Shift Q",
  person0: "1",
  person1: "2",
  release: "Shift 0",
  diag: "D",
  replay: "Shift R",
};

/**
 * What a keydown asks for, exactly as /judge decides it, or null.
 * `e`: {key, code, shiftKey, altKey, ctrlKey, metaKey, repeat, inField}; `drawerOpen`: the Diagnostics sheet is open.
 * Returns {type: "ab"} | {type: "mode", mode} | {type: "person", index} | {type: "release"} | {type: "diag"}
 *       | {type: "replay"} | {type: "calibrate"}; `preventDefault` is true for Space (no page scroll).
 */
export function keyAction(e, { drawerOpen = false } = {}) {
  if (e.altKey || e.ctrlKey || e.metaKey || e.inField) return null;
  const key = String(e.key || "").toLowerCase();
  const guarded = e.shiftKey || drawerOpen;
  // T-135/T-150: C/R/N/F/Q act only with Shift or the drawer open
  if ("crnfq".includes(key) && key.length === 1 && !guarded) return null;
  if (isReleaseKey(key, e.code) || (e.shiftKey && e.key === ")")) { // T-177; ")" = Shift+0 on a US layout
    return guarded && !e.repeat ? { type: "release" } : null;
  }
  if (key === " ") return e.repeat ? { type: "none", preventDefault: true } : { type: "ab", preventDefault: true };
  const mode = modeForKey(key);
  if (mode) return e.repeat ? null : { type: "mode", mode };
  if (key === "c") return { type: "calibrate" };
  if (key === "r") return e.repeat ? null : { type: "replay" };
  if (key === "d") return e.repeat ? null : { type: "diag" };
  if ((key === "1" || key === "2") && !e.repeat) return { type: "person", index: Number(key) - 1 };
  return null;
}

/** /judge's onKey ignores keys typed into these. */
function inField(target) {
  return Boolean(target?.closest?.("input, select, textarea"));
}

/**
 * Listen for the stage keys on `doc` until `signal` aborts. `handlers[type](action, event)`; `drawerOpen()` says
 * whether the Diagnostics sheet is open. Space's keyup is swallowed too (a focused button would click on it).
 */
export function bindKeys(doc, handlers, { signal, drawerOpen = () => false } = {}) {
  doc.addEventListener("keydown", (e) => {
    const action = keyAction({
      key: e.key, code: e.code, shiftKey: e.shiftKey, altKey: e.altKey, ctrlKey: e.ctrlKey, metaKey: e.metaKey,
      repeat: e.repeat, inField: inField(e.target),
    }, { drawerOpen: drawerOpen() });
    if (!action) return;
    if (action.preventDefault) e.preventDefault();
    handlers[action.type]?.(action, e);
  }, { signal });
  doc.addEventListener("keyup", (e) => { if (e.key === " " && !inField(e.target)) e.preventDefault(); }, { signal });
}

/**
 * T-177: "Keys off: click the page" while another window/tab or a text field has the keys.
 * Calls `onChange(off)` now and on every change, until `signal` aborts.
 */
export function watchKeyFocus(win, onChange, { signal } = {}) {
  const doc = win.document;
  const focus = new KeyFocus(doc.hasFocus());
  focus.focusOn(doc.activeElement?.tagName);
  let last = null;
  const emit = () => {
    if (focus.off === last) return;
    last = focus.off;
    onChange(focus.off);
  };
  win.addEventListener("blur", () => { focus.blur(); emit(); }, { signal });
  win.addEventListener("focus", () => { focus.focus(); emit(); }, { signal });
  doc.addEventListener("focusin", (e) => { focus.focusOn(e.target?.tagName); emit(); }, { signal });
  doc.addEventListener("focusout", () => { focus.focusOn(null); emit(); }, { signal });
  emit();
  return focus;
}
