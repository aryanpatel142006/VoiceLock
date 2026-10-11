// Welcome screen (M14, T-366): the hero line, the press-and-hold Start disc, the honest delay line and the
// iPhone install hint. Spec: web/app/DESIGN_SPEC.md#51-welcome-welcome, comp web/app/comp/welcome.html.
//
// Hold to start (borrowed from Why Zero): the ring fills with pointer or key time, never a timer of its own, over
// --dur-hold-to-start (900 ms, 600 ms under reduced motion). Mouse, touch and Space/Enter held on the disc (or on
// the page when nothing else has focus) all drive the same pure state machine, `holdReduce` below, tested by
// web/app/tests/welcome.test.mjs. When the ring is full the session starts inside the user activation that the press
// opened, so the AudioContext is created in the gesture (iOS). If the browser says that activation has lapsed
// (`navigator.userActivation.isActive` false) the disc asks to be let go and starts on the release instead, which is
// itself a gesture. Only a mic that really opened cuts to Setup (or Listen when a mic choice is already saved); a
// failed getUserMedia keeps the screen and shows micProblem()'s line under the disc.
//
// The sound field is a static placeholder until scene.js (T-349) lands: flat and labelled "idle, mic off" until the
// mic streams; once it does, its light follows the store's "level" events (the peak of each frame sent) and nothing
// else. It never breathes or pulses on its own.

import { MIC_KEY, loadChoice } from "../../console/devices.js";
import { sharedSession } from "./setup.js";

export const title = "Welcome";

export const HOLD_MS = 900; // --dur-hold-to-start
export const HOLD_MS_REDUCED = 600; // --dur-hold-to-start under prefers-reduced-motion
export const CUT_HOLD_MS = 240; // --dur-hold: stillness before the cut
export const LEVEL_FLOOR_DB = -60; // --scene-floor-db
export const LEVEL_CEILING_DB = -6; // --scene-ceiling-db
export const HOLD_KEYS = new Set([" ", "Enter"]);

// ---------------------------------------------------------------- pure state machine

/**
 * Phases: idle (nothing pressed) -> holding (ring filling) -> full (ring complete; the caller starts the session now,
 * or arms it) -> armed (waiting for the release to start) -> starting (mic opening) -> done (cut) | failed (retry).
 * A release or cancel while holding drops back to idle and empties the ring.
 */
export function holdInit() {
  return { phase: "idle", source: null, t0: 0, progress: 0, error: "" };
}

/**
 * Next hold state for `ev`. Events: {type:"press", source, at} · {type:"tick", at, durationMs} ·
 * {type:"release", source} · {type:"cancel"} · {type:"arm"} · {type:"start"} · {type:"started", ok, error}.
 * `source` names the input that pressed ("pointer:<id>" or "key:<key>") so a second finger or another key cannot
 * end someone else's hold. Unknown or out-of-phase events return the same object.
 */
export function holdReduce(s, ev) {
  switch (ev.type) {
    case "press":
      if (s.phase !== "idle" && s.phase !== "failed") return s;
      return { ...s, phase: "holding", source: ev.source, t0: ev.at, progress: 0, error: "" };
    case "tick": {
      if (s.phase !== "holding") return s;
      const dur = ev.durationMs > 0 ? ev.durationMs : HOLD_MS;
      const progress = Math.min(1, Math.max(0, (ev.at - s.t0) / dur));
      return progress >= 1 ? { ...s, phase: "full", progress: 1 } : { ...s, progress };
    }
    case "release":
      if (ev.source !== s.source) return s;
      if (s.phase === "holding") return { ...s, phase: "idle", source: null, progress: 0 };
      if (s.phase === "armed") return { ...s, phase: "starting", source: null };
      if (s.phase === "full") return { ...s, source: null }; // let go in the same frame the ring filled
      return s;
    case "cancel":
      if (s.phase === "holding" || s.phase === "armed") return { ...s, phase: "idle", source: null, progress: 0 };
      return s;
    case "arm":
      return s.phase === "full" ? { ...s, phase: "armed" } : s;
    case "start":
      return s.phase === "full" ? { ...s, phase: "starting" } : s;
    case "started":
      if (s.phase !== "starting") return s;
      return ev.ok
        ? { ...s, phase: "done", source: null }
        : { ...s, phase: "failed", source: null, progress: 0, error: ev.error || "Microphone unavailable. Hold to try again." };
    default:
      return s;
  }
}

/** Whether a full ring may start now: true unless the browser reports the press's user activation has lapsed. */
export function activationLive(nav = globalThis.navigator) {
  const ua = nav?.userActivation;
  return ua ? ua.isActive !== false : true;
}

/** Where a successful Start cuts to: Listen when a microphone choice is already saved, else Setup. */
export function startTarget(storage = globalThis.localStorage) {
  try { return loadChoice(MIC_KEY, storage) ? "listen" : "setup"; } catch { return "setup"; }
}

/** 0..1 light of the field for a mic peak in dBFS (the scene contract's floor/ceiling); null/NaN = 0. */
export function levelLight(db) {
  if (!Number.isFinite(db)) return 0;
  return Math.max(0, Math.min(1, (db - LEVEL_FLOOR_DB) / (LEVEL_CEILING_DB - LEVEL_FLOOR_DB)));
}

/** The install hint, or null when the app already runs installed (display-mode standalone / iOS home screen). */
export function installHint({ standalone, ios }) {
  if (standalone) return null;
  return ios ? "For full screen: tap Share, then Add to Home Screen." : "On iPhone: Share, then Add to Home Screen.";
}

/** micProblem()'s lines say "press Start" (the /judge button); here Start is held. */
export function holdWording(text) {
  return String(text || "").replace(/\bpress Start\b/gi, (m) => (m[0] === "P" ? "Hold Start" : "hold Start"));
}

/** The disc's two lines for a phase. */
export function discCopy(phase) {
  switch (phase) {
    case "holding": return ["Keep holding", "opens the mic"];
    case "full": case "starting": return ["Opening the mic", "allow it if asked"];
    case "armed": return ["Let go to start", "opens the mic"];
    case "done": return ["Mic open", "cutting to the next step"];
    case "failed": return ["Hold to retry", "opens the mic"];
    default: return ["Hold to start", "opens the mic"];
  }
}

// ---------------------------------------------------------------- screen

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

export function mount(root, { app, signal }) {
  const session = sharedSession(app);
  const store = session.store;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  const standalone = matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  const ios = /iP(hone|ad|od)/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  // the stage laptop needs no install line (T-365 P2-1): phones and tablets only
  const hint = ios || matchMedia("(pointer: coarse)").matches ? installHint({ standalone, ios }) : null;

  const label = h("span", { class: "wl-disc-label" });
  const sub = h("span", { class: "wl-disc-sub" });
  const disc = h("button", { type: "button", class: "wl-disc", id: "start", "data-phase": "idle", "aria-describedby": "wl-facts wl-problem" },
    h("span", { class: "wl-ring", "aria-hidden": "true" }), label, sub);
  const problem = h("p", { class: "wl-problem", id: "wl-problem", role: "alert" });
  const status = h("p", { class: "wl-sr", role: "status" });
  const fieldLabel = h("span", { class: "wl-field-label" });
  const field = h("div", { class: "wl-field", "data-live": "idle", "aria-hidden": "true" }, h("span", { class: "wl-field-core" }));

  root.classList.add("screen-welcome");
  const section = h("section", { class: "wl", "data-phase": "idle", "aria-labelledby": "wl-title" },
    field, fieldLabel,
    h("div", { class: "wl-main" },
      h("h1", { id: "wl-title", class: "wl-hero", text: "Turn down the world without disconnecting from it." }),
      h("div", { class: "wl-row" },
        h("div", { class: "wl-start" }, disc, h("p", { class: "wl-key" }, "or hold ", h("kbd", { text: "Space" }))),
        h("div", { class: "wl-facts", id: "wl-facts" },
          h("p", { text: "Pick one voice. The room goes quiet around it." }),
          h("p", {}, h("strong", { text: "About 1.5 s behind" }), ", like live captions."),
          hint ? h("p", { class: "wl-install", text: hint }) : null,
          problem)),
      status));
  root.append(section);

  // ---- hold state machine wiring
  let s = holdInit();
  let raf = 0;
  let cutTimer = 0;
  let target = "setup";
  const duration = () => {
    const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--dur-hold-to-start"));
    return Number.isFinite(v) && v > 0 ? v : (reduced.matches ? HOLD_MS_REDUCED : HOLD_MS);
  };

  const render = (prev) => {
    disc.style.setProperty("--wl-p", s.progress.toFixed(3));
    if (prev && prev.phase === s.phase) return;
    disc.dataset.phase = s.phase;
    section.dataset.phase = s.phase;
    const [a, b] = discCopy(s.phase);
    label.textContent = a;
    sub.textContent = b;
    disc.setAttribute("aria-busy", String(s.phase === "starting" || s.phase === "full"));
    problem.textContent = s.phase === "failed" ? s.error : "";
    status.textContent = s.phase === "holding" ? "Holding. Keep holding to start." : s.phase === "armed" ? "Let go to start."
      : s.phase === "starting" ? "Opening the microphone." : s.phase === "done" ? "Microphone open." : "";
  };

  const dispatch = (ev) => {
    const prev = s;
    s = holdReduce(s, ev);
    if (s === prev) return;
    render(prev);
    if (s.phase === "holding" && !raf) raf = requestAnimationFrame(tick);
    if (s.phase !== "holding" && raf) { cancelAnimationFrame(raf); raf = 0; }
    if (s.phase === "full") dispatch({ type: activationLive() ? "start" : "arm" });
    else if (s.phase === "starting" && prev.phase !== "starting") begin();
    else if (s.phase === "done" && prev.phase !== "done") cut();
  };

  function tick(now) {
    raf = 0;
    dispatch({ type: "tick", at: now, durationMs: duration() });
    if (s.phase === "holding" && !raf) raf = requestAnimationFrame(tick);
  }

  function begin() {
    app.emit("start");
    // synchronous inside this handler: session.start() creates the AudioContext before its first await
    target = startTarget();
    let p;
    try { p = session.start(); } catch (err) { p = Promise.reject(err); }
    Promise.resolve(p).then((ok) => {
      if (signal.aborted) return;
      if (ok) dispatch({ type: "started", ok: true });
      else dispatch({ type: "started", ok: false, error: holdWording(store.state.mic.banner) });
    }, (err) => {
      if (!signal.aborted) dispatch({ type: "started", ok: false, error: String(err?.message || err) });
    });
  }

  function cut() {
    cutTimer = setTimeout(() => { if (!signal.aborted) app.go(target); }, reduced.matches ? 0 : CUT_HOLD_MS);
  }

  const press = (source, e) => {
    if (s.phase !== "idle" && s.phase !== "failed") return false;
    app.emit("start-gesture"); // the shell's contract: synchronous inside the pointerdown / keydown
    dispatch({ type: "press", source, at: e.timeStamp || performance.now() });
    return true;
  };

  disc.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (!press(`pointer:${e.pointerId}`, e)) return;
    e.preventDefault(); // no text selection or drag image while held
    try { disc.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
  }, { signal });
  const pointerEnd = (e) => dispatch({ type: "release", source: `pointer:${e.pointerId}` });
  disc.addEventListener("pointerup", pointerEnd, { signal });
  disc.addEventListener("pointercancel", () => dispatch({ type: "cancel" }), { signal });
  disc.addEventListener("lostpointercapture", (e) => { if (s.source === `pointer:${e.pointerId}`) dispatch({ type: "cancel" }); }, { signal });
  disc.addEventListener("click", (e) => e.preventDefault(), { signal }); // a click is not a hold
  disc.addEventListener("contextmenu", (e) => e.preventDefault(), { signal }); // long-press menu on touch

  // keys: on the disc, or on the page when nothing focusable has focus (the judge can just hold Space)
  const keyTarget = (t) => t === disc || t === document.body || t === document.documentElement || t?.id === "stage" || root.contains(t) && !t.closest("a, input, select, textarea, button:not(#start)");
  addEventListener("keydown", (e) => {
    if (!HOLD_KEYS.has(e.key) || e.metaKey || e.ctrlKey || e.altKey || !keyTarget(e.target)) return;
    e.preventDefault(); // Space must not scroll, Enter must not click
    if (e.repeat) return;
    press(`key:${e.key}`, e);
  }, { signal });
  addEventListener("keyup", (e) => {
    if (!HOLD_KEYS.has(e.key)) return;
    if (s.source === `key:${e.key}`) e.preventDefault();
    dispatch({ type: "release", source: `key:${e.key}` });
  }, { signal });
  addEventListener("blur", () => dispatch({ type: "cancel" }), { signal });
  document.addEventListener("visibilitychange", () => { if (document.hidden) dispatch({ type: "cancel" }); }, { signal });

  // ---- the field: idle until frames are sent, then the real peak (no loop of its own)
  let lightRaf = 0;
  let lastDb = null;
  let lastDraw = 0;
  const drawField = () => {
    lightRaf = 0;
    const live = store.state.started && !store.state.replaying;
    field.dataset.live = live ? "live" : store.state.replaying ? "replay" : "idle";
    field.style.setProperty("--wl-level", live ? levelLight(lastDb).toFixed(3) : "0");
    fieldLabel.textContent = live
      ? (Number.isFinite(lastDb) ? `mic live, ${Math.round(lastDb)} dBFS` : "mic live, waiting for frames")
      : store.state.replaying ? "replay, not the mic" : "idle, mic off";
  };
  const schedule = () => {
    if (lightRaf) return;
    const now = performance.now();
    if (reduced.matches && now - lastDraw < 100) return; // reduced motion: at most one static frame per 100 ms
    lastDraw = now;
    lightRaf = requestAnimationFrame(drawField);
  };
  const offLevel = store.on("level", ({ db }) => { lastDb = db; schedule(); });
  const offChange = store.on("change", ({ keys }) => { if (keys.includes("started") || keys.includes("replaying")) { if (!store.state.started) lastDb = null; schedule(); } });
  drawField();
  render(null);

  return () => {
    offLevel();
    offChange();
    cancelAnimationFrame(raf);
    cancelAnimationFrame(lightRaf);
    clearTimeout(cutTimer);
  };
}
