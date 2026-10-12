// Motion pass B (T-389): the pure half of the choreography. No DOM at import, so node tests load it
// (web/app/tests/transitions.test.mjs); motion.js applies what these functions decide.
//
// Cut direction: the app is one scene walked in order (Welcome > Setup > People > Listen). Forward is a camera push
// (the new screen settles from slightly large), back a pull-out (from slightly small); About and Phone sit off the
// axis and get a plain dissolve. Reduced motion is always "fade" (motion.css: 200 ms linear cross-fade).
//
// Pulses: one short, compositor-only accent when the engine reports a real state change, read from the attributes
// the screens already write from telemetry (Listen's data-state, a People card's data-playing, the Welcome disc's
// data-phase). A pulse never starts on its own, never loops, and an attribute that did not change plays nothing.

/** Spring easings generated with Motion's spring(visualDuration, bounce) (vendor/motion.js); same as motion.css. */
export const SPRINGS = {
  // spring(0.4, 0.12): a state that arrived (Isolated, Following now); 2.5 % overshoot, assistive app, no wobble
  settle: { ms: 500, ease: "linear(0, 0.0659, 0.2078, 0.3701, 0.5232, 0.6538, 0.7581, 0.8374, 0.8951, 0.9355, 0.9628, 0.9804, 0.9913, 0.9976, 1.001, 1.0025, 1)" },
  // spring(0.2, 0): a press acknowledged (the hold committed); critically damped
  press: { ms: 400, ease: "linear(0, 0.2175, 0.5207, 0.7361, 0.8631, 0.9317, 0.9668, 0.9842, 0.9926, 0.9966, 0.9984, 0.9993, 1)" },
  // spring(0.45, 0.08): staggered person cards arriving
  reveal: { ms: 650, ease: "linear(0, 0.0521, 0.1679, 0.3056, 0.4418, 0.5643, 0.6684, 0.7531, 0.82, 0.8713, 0.9099, 0.9382, 0.9586, 0.973, 0.983, 0.9897, 0.9941, 0.997, 0.9987, 0.9997, 1.0003, 1)" },
};

/** Narrative order of the screens; About and Phone are off the axis. */
export const ORDER = ["welcome", "setup", "people", "listen"];

/** "forward" | "back" | "dissolve" | "fade" for a cut from `from` to `to`. */
export function cutDirection(from, to, { reduced = false } = {}) {
  if (reduced) return "fade";
  const a = ORDER.indexOf(from);
  const b = ORDER.indexOf(to);
  if (a < 0 || b < 0 || a === b) return "dissolve";
  return b > a ? "forward" : "back";
}

/** Listen states (STATE_WORDS in screens/listen.js) that mean the isolated voice was lost or the engine fell back. */
const DROPS = new Set(["reacquiring", "fallback", "look", "lost", "nomic", "connecting"]);

/**
 * Which pulse an attribute change plays, or null. `kind`:
 *   "listen"  Listen's data-state (the pill state): into "isolated" = "lock"; out of it into a drop state = "drop";
 *             any other change between two states = "change" (the state word cross-fades to its new text)
 *   "card"    a People card's data-playing: "false" -> "true" = "lock" (the chosen person is now isolated and plays)
 *   "hold"    the Welcome disc's data-phase: "full" (the hold completed) = "commit"; "failed" = "drop"
 */
export function pulseFor(kind, prev, next) {
  if (prev === next || next == null) return null;
  if (kind === "listen") {
    if (next === "isolated") return prev == null ? null : "lock"; // first render of a live screen is not an event
    if (prev === "isolated" && DROPS.has(next)) return "drop";
    return prev == null ? null : "change";
  }
  if (kind === "card") return next === "true" && prev === "false" ? "lock" : null;
  if (kind === "hold") {
    if (next === "full") return "commit";
    if (next === "failed" && prev !== "failed") return "drop";
    return null;
  }
  return null;
}

/** Keyframes and timing for a pulse (compositor-only: transform and opacity). Reduced motion: opacity only. */
export function pulseKeyframes(name, { reduced = false } = {}) {
  if (name === "lock") {
    return reduced
      ? { keyframes: { opacity: [0.6, 1] }, ms: 160, ease: "linear" }
      // numeric `scale`, not transform strings: a spring from "scale(0.97)" to "none" committed scale(0) and the Listen
      // word vanished (h = 0) after every state change (T-403). Calm pass: a small settle, no spring.
      : { keyframes: { scale: [0.98, 1], opacity: [0.5, 1] }, ms: 400, ease: "linear(0, 0.4, 0.68, 0.84, 0.93, 0.97, 0.99, 1)" };
  }
  if (name === "commit") {
    return reduced
      ? { keyframes: { opacity: [0.7, 1] }, ms: 160, ease: "linear" }
      : { keyframes: { scale: [0.96, 1] }, ...SPRINGS.press };
  }
  if (name === "change") {
    // the word already shows its new text: it fades up from a third, no scale (a cross-fade, not an accent)
    return { keyframes: { opacity: [0.35, 1] }, ms: reduced ? 160 : 240, ease: "linear" };
  }
  if (name === "drop") {
    // a dip, not a shake: losing the voice is information, not an error to punish
    return { keyframes: { opacity: [1, 0.45, 1] }, ms: reduced ? 160 : 360, ease: "linear" };
  }
  return null;
}

/**
 * Stagger for the person cards: `n` start delays in seconds after `start`, `step` apart, the whole run capped at
 * `maxSpan` so a long list never makes the last card wait. Reduced motion: all at `start`.
 */
export function cardDelays(n, { start = 0, step = 0.07, maxSpan = 0.35, reduced = false } = {}) {
  if (n <= 0) return [];
  const gap = reduced ? 0 : Math.min(step, n > 1 ? maxSpan / (n - 1) : step);
  return Array.from({ length: n }, (_, i) => +(start + i * gap).toFixed(4));
}

/**
 * Honest ticker plan: tick from `from` to `to` only between two measured numbers; a first value, a value after a
 * dash, an equal value or reduced motion is written at once. Returns {jump: true} or {jump: false, lo, hi}, where
 * [lo, hi] clamps every intermediate frame so the ticker never shows a number outside the two measurements.
 */
export function tickPlan(from, to, { reduced = false } = {}) {
  const ok = (v) => typeof v === "number" && Number.isFinite(v);
  if (!ok(to)) return { jump: true, dash: true };
  if (!ok(from) || from === to || reduced) return { jump: true };
  return { jump: false, lo: Math.min(from, to), hi: Math.max(from, to) };
}
