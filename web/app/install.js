// Home-screen companion: the phone page (`/app/?watch=latest`, screens/phone.js) installs as its own app, "VoiceLock",
// whose start_url is the stable companion URL (it joins whatever stage session the laptop has live), not the laptop
// app's `/app/`. While the companion is open its manifest replaces the shell's <link rel=manifest> (the server swaps it
// too for `?watch=` loads: hearing/server/static.py), so Add to Home Screen / Install app installs the companion.
//
// The hint (iOS: Share → Add to Home Screen; Android: the beforeinstallprompt Install button) is hidden once the page
// runs standalone. Chrome can fire beforeinstallprompt before the phone screen is imported: app.js imports this
// module, so captureInstallPrompt() runs with the page.

/** The companion URL that does not name one session: joins the laptop's live stage session and follows the next. */
export const COMPANION_URL = "/app/?watch=latest";
export const COMPANION_MANIFEST = "companion.webmanifest";
export const APP_MANIFEST = "manifest.webmanifest";
const DISMISS_KEY = "voicelock.install-hint.dismissed";

/**
 * Point the page's <link rel=manifest> at `href` (created when missing). Returns a restore function that puts the
 * previous href back, or the laptop app's manifest when the server already served the companion's (`?watch=` load).
 */
export function swapManifest(doc, href = COMPANION_MANIFEST, { fallback = APP_MANIFEST } = {}) {
  let link = doc.querySelector('link[rel="manifest"]');
  let created = false;
  if (!link) {
    link = doc.createElement("link");
    link.setAttribute("rel", "manifest");
    doc.head.append(link);
    created = true;
  }
  const before = link.getAttribute("href");
  if (before !== href) link.setAttribute("href", href);
  return () => {
    if (created) link.remove();
    else if (link.getAttribute("href") === href) link.setAttribute("href", before && before !== href ? before : fallback);
  };
}

/** Running as an installed app (Android: display-mode standalone; iOS home-screen: navigator.standalone). */
export function isStandalone(win = globalThis) {
  if (win.navigator?.standalone === true) return true;
  try {
    return ["standalone", "fullscreen", "minimal-ui"].some((m) => win.matchMedia?.(`(display-mode: ${m})`)?.matches);
  } catch {
    return false;
  }
}

/** "ios" | "android" | "other". iPadOS reports a Mac user agent, so a touch "Macintosh" is iOS too. */
export function platformOf(ua = "", maxTouchPoints = 0) {
  if (/iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && maxTouchPoints > 1)) return "ios";
  if (/Android/.test(ua)) return "android";
  return "other";
}

/**
 * Pure: what the companion's install hint shows. `canPrompt`: a beforeinstallprompt event is held.
 * Returns {show, kind: "ios"|"prompt"|"", text, trust}. `trust` is the certificate line (iPhone: the installed app
 * opens reliably only once the laptop's certificate is trusted).
 */
export function installHint({ platform = "other", standalone = false, canPrompt = false, dismissed = false } = {}) {
  const trust = "First time on this phone? Trust the laptop";
  if (standalone || dismissed) return { show: false, kind: "", text: "", trust };
  if (platform === "ios") return { show: true, kind: "ios", text: "Keep VoiceLock on your home screen: tap Share, then Add to Home Screen.", trust };
  if (canPrompt) return { show: true, kind: "prompt", text: "Keep VoiceLock on your home screen.", trust };
  return { show: false, kind: "", text: "", trust };
}

export function hintDismissed(storage = globalThis.localStorage) {
  try { return storage?.getItem(DISMISS_KEY) === "1"; } catch { return false; }
}

export function dismissHint(storage = globalThis.localStorage) {
  try { storage?.setItem(DISMISS_KEY, "1"); } catch { /* private mode */ }
}

let held = null;

/** The held beforeinstallprompt event (null when the browser has not offered one, or it was used). */
export function installPrompt() {
  return held;
}

/** Show the browser's install dialog once; resolves "accepted" | "dismissed" | "unavailable". */
export async function promptInstall() {
  const e = held;
  if (!e?.prompt) return "unavailable";
  held = null; // an event prompts once
  e.prompt();
  try { return (await e.userChoice)?.outcome || "dismissed"; } catch { return "dismissed"; }
}

/** Hold Chrome's beforeinstallprompt for the Install button; `voicelock:installable` tells a mounted hint. */
export function captureInstallPrompt(win = globalThis) {
  if (!win?.addEventListener) return;
  win.addEventListener("beforeinstallprompt", (e) => {
    // only the companion offers an Install button; elsewhere Chrome keeps its own install UI
    if (!new URLSearchParams(win.location?.search || "").has("watch")) return;
    e.preventDefault();
    held = e;
    win.dispatchEvent?.(new Event("voicelock:installable"));
  });
  win.addEventListener("appinstalled", () => { held = null; win.dispatchEvent?.(new Event("voicelock:installable")); });
}
