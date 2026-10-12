// Motion system (M14, T-349): screen cuts, shared elements, entrances and the number ticker for /app.
// Spec: web/app/DESIGN_SPEC.md section 6. The shell (app.js) imports this module at boot when the server lists it
// and calls `cut(update, {from, to})` on every screen change; importing it also mounts the sound scene (scene.js).
//
// Cuts: the View Transitions API, film pacing (hold --dur-hold, then a --dur-cut dissolve; motion.css). Two shared
// elements glide across the cut while the rest dissolves:
//   pill    the status pill   ([data-vt="pill"], else .pill, else the screen's [role="status"])
//   person  the chosen person ([data-vt="person"], else .person[aria-pressed="true"] / .person.is-current)
// Names are given to one element per side just for the cut (a duplicate name would cancel the transition), then
// removed. Screens only need the data-vt attribute (or those classes).
// Entrances (Motion, vendored): after the cut the screen's word fades up --motion-rise px and its controls follow
// in one short stagger. Elements opt in with [data-enter]; without any, the first h1/.display and the screen's buttons.
// Travel is a `transform` string, not Motion's `y`: Motion runs opacity/filter/transform on WAAPI (the compositor)
// but independent transforms (`x`, `y`) in JS every frame, which stalls while the new screen mounts (T-379).
// Nothing loops, nothing blurs. Reduced motion: a short linear cross-fade, no shared glide, no travel, opacity only.
// No View Transitions (Firefox < 144, old Safari): the update runs at once and the entrance fades in over --dur-ui.
//
// Pass B (T-389, docs/ux-review-shots/MOTION_PASS_B.md lists every motion with its trigger and timing):
// - cuts have a direction along the scene order (transitions.js cutDirection): forward pushes in, back pulls out,
//   About/Phone dissolve; html[data-cut-dir] picks the keyframes in motion.css.
// - People's cards arrive one by one on the calm curve (cardDelays) after the title.
// - pulses: one compositor-only accent when the engine reports a real change (Listen goes Isolated or loses it, a
//   card starts playing, the Start hold completes or fails), read from the attributes the screens write from
//   telemetry (transitions.js pulseFor). Nothing pulses on a timer.

import { animate, stagger } from "./vendor/motion.js";
import { mountScene } from "./scene.js";
import { cardDelays, cutDirection, pulseFor, pulseKeyframes, tickPlan } from "./transitions.js";

export const SHARED = {
  pill: ['[data-vt="pill"]', ".pill", '[role="status"]'],
  person: ['[data-vt="person"]', '.person[aria-pressed="true"]', ".person.is-current", '[data-person-current]'],
};
const MAX_ENTER = 14;
const reduced = matchMedia("(prefers-reduced-motion: reduce)");
/** --motion-ease (tokens.css) as a Motion cubic-bezier: calm ease-out, no overshoot. */
export const EASE = [0.2, 0, 0, 1];
/** --motion-rise in px (0 under reduced motion, where the token is 0px). */
function rise() {
  const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--motion-rise"));
  return Number.isFinite(v) ? v : 8;
}

/** CSS time token ("520ms" / "0.2s") in seconds. */
export function tokenS(name, fallbackS) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const n = parseFloat(v);
  if (!Number.isFinite(n)) return fallbackS;
  return v.endsWith("ms") ? n / 1000 : n;
}

function visible(el) {
  if (!el?.isConnected) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== "hidden";
}

/** The element of `root` that plays shared element `name` (first visible match by priority), or null. */
export function sharedElement(root, name) {
  for (const sel of SHARED[name] || []) {
    for (const el of root.querySelectorAll(sel)) if (visible(el)) return el;
  }
  return null;
}

const named = new Set();
function nameShared(root) {
  clearShared();
  if (!root || reduced.matches) return;
  for (const name of Object.keys(SHARED)) {
    const el = sharedElement(root, name);
    if (!el) continue;
    el.style.viewTransitionName = name;
    named.add(el);
  }
}

function clearShared() {
  for (const el of named) el.style.viewTransitionName = "";
  named.clear();
}

/**
 * The shell's cut hook: run `update` (which swaps the screen) inside a view transition. Resolves when the new
 * screen is in the DOM (the shell then focuses it); the animation finishes on its own.
 */
export async function cut(update, { from = null, to = null } = {}) {
  const stage = document.getElementById("stage");
  const html = document.documentElement;
  if (!document.startViewTransition || from === null || document.visibilityState === "hidden") {
    await update();
    if (from !== null) enter(stage);
    return;
  }
  const dir = cutDirection(from, to, { reduced: reduced.matches });
  html.dataset.cut = reduced.matches ? "fade" : "film";
  html.dataset.cutDir = dir;
  if (to) html.dataset.cutTo = to;
  nameShared(stage);
  const run = async () => {
    clearShared();
    await update();
    nameShared(stage);
  };
  let vt;
  try {
    vt = document.startViewTransition({ update: run, types: ["cut", reduced.matches ? "fade" : "film", dir] });
  } catch {
    vt = document.startViewTransition(run); // Chrome < 125, Safari 18.0: callback form only
  }
  vt.ready.catch(() => undefined); // skipped (tab hidden, a newer cut) is not an error
  vt.finished.catch(() => undefined).finally(() => {
    clearShared();
    if (html.dataset.cutTo === to) { delete html.dataset.cut; delete html.dataset.cutTo; delete html.dataset.cutDir; }
  });
  await vt.updateCallbackDone;
  // the entrance starts while the cross-fade is a third in, so the word lands as the old screen is gone
  enter(stage, { delay: reduced.matches ? 0 : tokenS("--dur-hold", 0) + tokenS("--motion-slow", 0.4) * 0.35 });
}

/** Which elements of a freshly mounted screen enter: [data-enter], else its word and its controls. */
export const CARD = ".pp-card, .person, [data-person-card]";
export function entranceTargets(root) {
  if (!root) return [];
  const marked = [...root.querySelectorAll("[data-enter]")];
  if (marked.length) return marked.slice(0, MAX_ENTER);
  const word = root.querySelector("h1, .display");
  const controls = [...root.querySelectorAll("button, [role='button'], .seg, .card")]
    .filter((el) => !word?.contains(el) && !el.closest("[data-enter-skip]") && !el.matches(CARD));
  return [word, ...controls].filter(Boolean).filter(visible).slice(0, MAX_ENTER);
}

const running = new Set();
/** The person cards of a screen, in order (they enter as their own staggered run). */
export function cardTargets(root) {
  return root ? [...root.querySelectorAll(CARD)].filter(visible).slice(0, MAX_ENTER) : [];
}

/** One entrance per screen: the word fades up a few px, then the controls in one short stagger, then the person cards. */
export function enter(root, { delay = 0 } = {}) {
  for (const a of running) a.stop?.();
  running.clear();
  const [word, ...rest] = entranceTargets(root);
  if (!word) return null;
  const cards = cardTargets(root);
  const ease = EASE;
  for (const el of [word, ...rest, ...cards]) el.style.opacity = "0"; // hidden from the first frame, not only after the delay
  if (reduced.matches) {
    const a = animate([word, ...rest, ...cards], { opacity: [0, 1] }, { duration: tokenS("--dur-ui", 0.16), delay, ease: "linear" });
    running.add(a);
    return a;
  }
  const dist = rise();
  const dur = tokenS("--motion-base", 0.24);
  const all = [
    animate(word, { opacity: [0, 1], transform: [`translateY(${dist}px)`, "none"] }, { duration: dur, delay, ease }),
  ];
  if (rest.length) {
    all.push(animate(rest, { opacity: [0, 1], transform: [`translateY(${dist}px)`, "none"] },
      { duration: dur, delay: stagger(0.03, { startDelay: delay + 0.06 }), ease }));
  }
  if (cards.length) all.push(...revealCards(cards, { delay: delay + 0.1, track: false }));
  for (const a of all) {
    running.add(a);
    // the entrance must not leave inline styles that fight the screen's own CSS once it is done
    a.finished?.then(() => {
      running.delete(a);
      for (const el of [word, ...rest, ...cards]) { el.style.transform = ""; el.style.opacity = ""; }
    }).catch(() => undefined);
  }
  return all;
}

/**
 * The people arrive one by one: rise --motion-rise px on the calm curve (Motion on WAAPI, compositor), cardDelays
 * apart (one short stagger). Runs in the screen's entrance, and again when the engine's first report turns the "—"
 * placeholder into cards (watchPulses). Reduced motion: opacity only, all at once.
 */
export function revealCards(cards, { delay = 0, track = true } = {}) {
  if (!cards.length) return [];
  const delays = cardDelays(cards.length, { start: delay, reduced: reduced.matches });
  const dist = rise();
  const dur = tokenS("--motion-base", 0.24);
  const out = cards.map((el, i) => {
    el.style.opacity = "0";
    const a = reduced.matches
      ? animate(el, { opacity: [0, 1] }, { duration: tokenS("--dur-ui", 0.16), delay: delays[i], ease: "linear" })
      : animate(el, { opacity: [0, 1], transform: [`translateY(${dist}px)`, "none"] },
        { duration: dur, delay: delays[i], ease: EASE });
    if (track) a.finished?.then(() => { el.style.transform = ""; el.style.opacity = ""; }).catch(() => undefined);
    return a;
  });
  return out;
}

/** A CSS linear() spring as a Motion easing function (Motion takes functions; WAAPI then samples it). */
const easeCache = new Map();
export function cssEase(linear) {
  if (easeCache.has(linear)) return easeCache.get(linear);
  const pts = linear.replace(/^linear\(|\)$/g, "").split(",").map((v) => parseFloat(v));
  const n = pts.length - 1;
  const fn = (t) => {
    if (t <= 0) return pts[0];
    if (t >= 1) return pts[n];
    const x = t * n;
    const i = Math.floor(x);
    return pts[i] + (pts[i + 1] - pts[i]) * (x - i);
  };
  easeCache.set(linear, fn);
  return fn;
}

/**
 * Number ticker (delay, RTT): tick `el`'s text from the previous measured value to the new one in --dur-quick.
 * Never from 0 and never past `to`: the first value (or a value after "—") is written as is.
 * `format(v)` -> string. Returns the new value, which the caller passes back as `from` next time.
 */
export function tick(el, from, to, format = (v) => String(Math.round(v))) {
  if (!el) return to;
  const prev = el.__tick;
  prev?.stop?.();
  const plan = tickPlan(from, to, { reduced: reduced.matches });
  if (plan.dash) { el.textContent = "—"; return null; }
  if (plan.jump) {
    el.textContent = format(to);
    return to;
  }
  el.__tick = animate(from, to, {
    duration: tokenS("--dur-quick", 0.18), ease: [0.16, 1, 0.3, 1],
    onUpdate: (v) => { el.textContent = format(Math.min(Math.max(v, plan.lo), plan.hi)); },
    onComplete: () => { el.textContent = format(to); },
  });
  return to;
}

/** The short acknowledgement of a pressed control (the state change itself follows telemetry, not the key). */
export function ack(el) {
  if (!el || reduced.matches) return null;
  return animate(el, { scale: [0.98, 1] }, { duration: tokenS("--motion-fast", 0.16), ease: EASE });
}

// ---------- pulses: real engine events only ----------
// Which attribute on which element means what; the screens write these from the store (telemetry), never a timer.
export const PULSE_SOURCES = [
  { kind: "listen", attr: "data-state", match: ".listen", target: (el) => el.querySelector(".ls-word") },
  { kind: "card", attr: "data-playing", match: ".pp-card", target: (el) => el.querySelector(".pp-card-name") || el },
  { kind: "hold", attr: "data-phase", match: ".wl-disc", target: (el) => el },
];
const pulsing = new WeakMap();

/** Play pulse `name` on `el` (Motion on WAAPI: transform/opacity only). A newer pulse replaces the running one. */
export function pulse(el, name) {
  const spec = pulseKeyframes(name, { reduced: reduced.matches });
  if (!el || !spec || document.visibilityState === "hidden") return null;
  pulsing.get(el)?.stop?.();
  const ease = spec.ease === "linear" ? "linear" : cssEase(spec.ease);
  const a = animate(el, spec.keyframes, { duration: spec.ms / 1000, ease });
  pulsing.set(el, a);
  el.dataset.pulse = name; // read by the frame-timing harness and the review doc's walk; removed when done
  a.finished?.then(() => {
    if (pulsing.get(el) !== a) return;
    pulsing.delete(el);
    delete el.dataset.pulse;
    el.style.transform = "";
    el.style.scale = "";
    el.style.opacity = "";
  }).catch(() => undefined);
  return a;
}

/**
 * People rebuilds its cards (replaceChildren) when what they show changes, so a card's data-playing flip arrives as
 * a removed card and an added one with the same data-person. The first cards after the "—" placeholder are revealed.
 */
function cardsReplaced(r) {
  const was = new Map();
  for (const n of r.removedNodes) if (n.matches?.(CARD)) was.set(n.dataset.person, n.getAttribute("data-playing"));
  const added = [...r.addedNodes].filter((n) => n.matches?.(CARD));
  if (!added.length) return;
  if (!was.size && r.target.isConnected) { revealCards(added); return; }
  for (const card of added) {
    const name = pulseFor("card", was.get(card.dataset.person) ?? null, card.getAttribute("data-playing"));
    if (name) pulse(card.querySelector(".pp-card-name") || card, name);
  }
}

/** Watch `root` for the engine-driven changes in PULSE_SOURCES (and People's card rebuilds) and pulse. Returns stop(). */
export function watchPulses(root) {
  if (!root || typeof MutationObserver === "undefined") return () => undefined;
  const filter = [...new Set(PULSE_SOURCES.map((s) => s.attr))];
  const mo = new MutationObserver((records) => {
    for (const r of records) {
      const el = r.target;
      if (r.type === "childList") {
        if (el.classList?.contains("pp-cards")) cardsReplaced(r);
        continue;
      }
      for (const src of PULSE_SOURCES) {
        if (r.attributeName !== src.attr || !el.matches?.(src.match)) continue;
        const name = pulseFor(src.kind, r.oldValue, el.getAttribute(src.attr));
        if (name) pulse(src.target(el), name);
      }
    }
  });
  mo.observe(root, { subtree: true, childList: true, attributes: true, attributeOldValue: true, attributeFilter: filter });
  return () => mo.disconnect();
}

export { animate, stagger };

// mount the scene behind the stage as soon as the shell loads this module
export const scene = typeof document !== "undefined" ? mountScene() : null;
export const stopPulses = typeof document !== "undefined" ? watchPulses(document.getElementById("stage")) : null;
