// T-177: the /judge "Choose saved person" buttons, Release, the target line and the "Keys off" banner.
// Pure helpers (no DOM) so node --test covers them; judge.js wires them to buttons, keys 0/1/2 and focus events.

export const RELEASE_CONTROL = { enroll: { action: "clear" } };
export const KEYS_OFF_TEXT = "Keys off: click the page";
export const SEARCHING_PILL = { state: "look", text: "Searching", note: "" };

/** The first two saved voice profiles (telemetry `enrollment.profiles`, the list keys 1/2 use). */
export function savedPeople(t) {
  return (t?.enrollment?.profiles || []).slice(0, 2);
}

/** "joseph" -> "Joseph": the button label for a profile name. */
export function personLabel(name) {
  const s = String(name || "");
  return s ? s[0].toUpperCase() + s.slice(1) : "";
}

/** The control for key/button 1 or 2 (index 0/1), or null when there is no such saved profile. */
export function personControl(profiles, i) {
  const name = profiles?.[i];
  return name ? { enroll: { action: "profile", name } } : null;
}

/** The A/B the Raw / Isolated buttons and Space ask for: a button sets its side, Space (no arg) flips. */
export function abControl(current, pressed = null) {
  return { ab: pressed || (current === "raw" ? "processed" : "raw") };
}

/** Key 0 / Shift+0 (`e.code` Digit0) is Release; guarded like C/R/N (Shift or the open drawer). */
export function isReleaseKey(key, code) {
  return key === "0" || code === "Digit0" || code === "Numpad0";
}

/** The target line next to the pill: who the session is isolating, "none" after Release. */
export function targetLine(t, { released = false } = {}) {
  if (released) return "Target: none (Release) · press 1 or 2";
  const e = t?.enrollment;
  if (e?.source === "profile" && e.profile) return `Target: ${personLabel(e.profile)}`;
  if (e?.source) return "Target: calibrated voice";
  return t ? "Target: none yet" : "Target: —";
}

/**
 * Whether the page's keys reach onKey. Tracks window focus/blur (another window or tab) and the focused
 * element (a text field, where onKey ignores keys). `off` -> show the "Keys off" banner.
 */
export class KeyFocus {
  constructor(hasFocus = true) {
    this.windowFocused = Boolean(hasFocus);
    this.inField = false;
  }

  blur() { this.windowFocused = false; }
  focus() { this.windowFocused = true; }

  /** `focusin` on an element (tagName, e.g. "INPUT"); `focusout` passes null. */
  focusOn(tagName) { this.inField = isTextField(tagName); }

  get off() { return !this.windowFocused || this.inField; }
}

/** The elements onKey ignores (judge.js: input, select, textarea). */
export function isTextField(tagName) {
  return ["INPUT", "SELECT", "TEXTAREA"].includes(String(tagName || "").toUpperCase());
}
