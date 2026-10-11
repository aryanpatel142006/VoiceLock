// Platform-aware copy (T-458, the phone touch review): which device the page runs on, so a hint names the control
// the visitor actually has (a tap, not Space), the install path of their OS, and their browser's mic settings.
// Pure helpers take their inputs (navigator, matchMedia, URL params) so tests run in Node.

/** "ios" | "android" | "other" from the user agent (iPadOS reports a Mac with touch points). */
export function platformOf(nav = globalThis.navigator) {
  const ua = String(nav?.userAgent || "");
  if (/iP(hone|ad|od)/.test(ua) || (nav?.platform === "MacIntel" && nav?.maxTouchPoints > 1)) return "ios";
  if (/Android/i.test(ua)) return "android";
  return "other";
}

/**
 * True on a touch-first device: a coarse primary pointer, or touch points with no hover or a phone/tablet user
 * agent (some Android browsers and emulators report a fine pointer). A touchscreen laptop stays a keyboard device.
 */
export function isTouch(mm = globalThis.matchMedia, nav = globalThis.navigator) {
  const q = (s) => (typeof mm === "function" ? mm(s).matches : false);
  if (q("(pointer: coarse)")) return true;
  if (!((nav?.maxTouchPoints || 0) > 0)) return false;
  return !q("(hover: hover)") || /Mobi|Android|iP(hone|ad|od)/.test(String(nav?.userAgent || "")) || platformOf(nav) === "ios";
}

let decided = null;
/** isTouch() decided once per page (app.js asks at boot), so every screen and the shell's CSS flag agree. */
export function touchDevice() {
  if (decided == null) decided = isTouch();
  return decided;
}

/** The URL flags: `?stage` marks the stage rig on a touch device, `?dev=1` shows developer facts on a phone. */
export function pageFlags(search = globalThis.location?.search || "") {
  const p = new URLSearchParams(search);
  return { stage: p.has("stage"), dev: p.get("dev") === "1" };
}

/**
 * Whether this page is the stage rig, where the demo-device check (preset mic / output pins) belongs. A visitor's
 * phone never is, unless the URL says `?stage`.
 */
export function isStageRig({ touch = touchDevice(), flags = pageFlags() } = {}) {
  return flags.stage || !touch;
}

/** Developer facts (stage checklist, engine revs, off-stage notes): always on the laptop, on a phone only with ?dev=1. */
export function showDevFacts({ touch = touchDevice(), flags = pageFlags() } = {}) {
  return !touch || flags.dev;
}

/** Where the browser's microphone permission lives, per platform. */
export function micSettingsSteps(platform) {
  if (platform === "ios") return "tap aA in the address bar → Website Settings → Microphone → Allow";
  if (platform === "android") return "tap the icon left of the address → Permissions → Microphone → Allow";
  return "address bar → site settings → Microphone";
}

/** A mic error line (judge-pill micProblem's wording) with the platform's own steps for a blocked or missing mic. */
export function platformMicProblem(line, err, platform) {
  const name = err?.name || "";
  if (platform === "other") return line;
  if (name === "NotAllowedError" || name === "SecurityError") {
    return `Microphone blocked: ${micSettingsSteps(platform)}, then press Start.`;
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No microphone found: close other apps using the mic, then press Start.";
  }
  return line;
}

/**
 * The install line under the Start disc, or null: iPhone gets its Share-sheet steps; Android gets nothing here
 * (its "Install app" button appears when Chrome fires beforeinstallprompt); installed apps and laptops get nothing.
 */
export function installHint({ standalone = false, platform = "other" } = {}) {
  if (standalone) return null;
  return platform === "ios" ? "For full screen: tap Share, then Add to Home Screen." : null;
}

/** The hint under the sound scene: what a finger can do, or what a mouse can. */
export function scenePrompt(touch) {
  return touch ? "Tap a person, or drag them into the ring" : "Drag a person into the ring";
}
