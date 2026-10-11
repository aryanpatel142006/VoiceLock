// Design pass E (T-427): the pure half of the direct-manipulation and delight layer. No DOM at import, so node tests
// load it (web/app/tests/delight.test.mjs). docs/ux-review-shots/DESIGN_PASS_E.md lists every interaction with its
// trigger and the real signal behind it.
//
// Honesty: nothing here invents a state. The room dial's words are the T-093 curve (docs/DESIGN.md "Suppression
// slider") for the value the listener set; a caption word is "new" only when the engine's text grew; a haptic fires
// only for a real event the caller hands it (the hold committed, the engine reported Isolated).

/** Room attenuation in dB for slider `s` (0..100), docs/DESIGN.md T-093: 3 dB per 10 points to 90, then a soft knee to silence. */
export function roomDb(s) {
  const v = Math.max(0, Math.min(100, Number(s) || 0)) / 100;
  if (v <= 0.9) return 30 * v;
  if (v >= 1) return Infinity;
  return 27 - 20 * Math.log10(0.5 * (1 + Math.cos(Math.PI * (v - 0.9) / 0.1)));
}

/** The dial's words for slider `s`: "as it is" at 0, "−15 dB" in the middle, "silent" at 100. */
export function roomLabel(s) {
  const db = roomDb(s);
  if (db === 0) return "as it is";
  if (!Number.isFinite(db)) return "silent";
  return `−${Math.round(db)} dB`;
}

/** 0..1 how far the room is turned down (for the scene): dB over the 27 dB knee, 1 at silence. */
export function roomAmount(s) {
  const db = roomDb(s);
  return Number.isFinite(db) ? Math.min(1, db / 27) : 1;
}

/** Clamp and round a dial value to a 5-point step (keyboard and drag land on the same grid). */
export function dialValue(v, step = 5) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 100;
  return Math.max(0, Math.min(100, Math.round(n / step) * step));
}

function words(text) {
  return String(text || "").trim().split(/\s+/).filter(Boolean);
}

/**
 * How many leading words of `text` were already on screen: the longest word prefix it shares with any of the
 * previously rendered lines (`prev`, strings). The rest are new and fade in; nothing is drawn before the engine sent it.
 */
export function seenWords(text, prev = []) {
  const now = words(text);
  let best = 0;
  for (const p of prev) {
    const was = words(p);
    let k = 0;
    while (k < now.length && k < was.length && now[k] === was[k]) k += 1;
    // a partial's last word can grow ("tomor" -> "tomorrow"): it was on screen, so it does not fade in again
    if (k < now.length && k === was.length - 1 && k > 0 && now[k].startsWith(was[k])) k += 1;
    if (k > best) best = k;
  }
  return Math.min(best, now.length);
}

/** Split a caption line into [{word, fresh}] given what was on screen before. */
export function captionWords(text, prev = []) {
  const seen = seenWords(text, prev);
  return words(text).map((word, i) => ({ word, fresh: i >= seen }));
}

/**
 * Where a dragged voice lands: "choose" inside the focus ring (radius `r` from the scene centre), "release" when the
 * current target is dragged well out of it, else "back" (it springs home). `dx`, `dy` are from the centre.
 */
export function dropAction({ dx, dy, r, current = false }) {
  const d = Math.hypot(dx, dy);
  if (!current) return d <= r ? "choose" : "back";
  return d > r * 1.6 ? "release" : "back";
}

/** Haptic patterns (ms) for the two real events that earn one. */
export const HAPTICS = { start: [12, 40, 18], isolate: 14, press: 8 };

/**
 * Vibrate for a real event where the device supports it (Android Chrome; iOS Safari has no vibrate and is a no-op).
 * Never under reduced motion (a calm setting), never without a pattern. Returns whether it asked the device.
 */
export function haptic(name, { nav = globalThis.navigator, reduced = false } = {}) {
  const pattern = HAPTICS[name];
  if (pattern == null || reduced || typeof nav?.vibrate !== "function") return false;
  try { return nav.vibrate(pattern) !== false; } catch { return false; }
}

/** Split the hero line into words for the opening reveal: [{word, i}] with a stagger index. */
export function revealWords(text) {
  return words(text).map((word, i) => ({ word, i }));
}
