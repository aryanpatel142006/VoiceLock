// SOS: "Pair the band" (T-461). The wrist band has one button; today a presenter remote stands in for it.
// The user presses that button once; its {code, key, label} goes to localStorage `hearing.sos` with the calm sound
// (brown | pink | white) that SOS plays (T-460 reads the same key). Further presses light a test light (held-key
// repeats ignored). Escape is refused (it exits full screen); F5 / B / "." get a toggle warning.
// The pure parts (classifyKey, matchesBand, loadSos, saveSos, labelFor, pairWarning) are covered by node --test.

export const title = "Pair the band";

export const STORAGE_KEY = "hearing.sos";
export const SOUNDS = [
  { id: "brown", name: "Brown", note: "Deep, soft rumble. The default." },
  { id: "pink", name: "Pink", note: "Lighter, like steady rain." },
  { id: "white", name: "White", note: "Even hiss, like a fan." },
];
export const DEFAULT_SOUND = "brown";
export const LIGHT_MS = 450;

export const ESCAPE_REASON = "Escape can't be the band: it exits full screen, so the page would jump out under you. Press a different button.";
export const TAB_REASON = "Tab moves between controls on this page, so it can't be the band. Press a different button.";

const MODIFIERS = new Set(["Shift", "Control", "Alt", "Meta", "AltGraph", "CapsLock", "Fn", "FnLock", "OS"]);
const NAMES = {
  " ": "Space", PageDown: "Page Down", PageUp: "Page Up", ArrowRight: "Right arrow", ArrowLeft: "Left arrow",
  ArrowUp: "Up arrow", ArrowDown: "Down arrow", Enter: "Enter", Backspace: "Backspace", ".": "Full stop (.)",
  ",": "Comma (,)", MediaPlayPause: "Play/Pause", AudioVolumeUp: "Volume up", AudioVolumeDown: "Volume down",
};
// keys the Listen screen already uses (web/app/keys.js); pairing one makes the band do two things there
const STAGE_KEYS = new Set([" ", "1", "2", "d"]);

/** A human name for the key, for the screen and for `hearing.sos.label`. */
export function labelFor(key, code = "") {
  if (NAMES[key]) return NAMES[key];
  if (typeof key === "string" && key.length === 1) return key.toUpperCase();
  if (key && key !== "Unidentified") return key.replace(/([a-z])([A-Z0-9])/g, "$1 $2");
  return code ? code.replace(/([a-z])([A-Z0-9])/g, "$1 $2") : "Unknown button";
}

/** A caution about a paired key, or null. F5 and B/"." are often toggles on presenter remotes. */
export function pairWarning(key, code = "") {
  const k = String(key || "");
  if (k === "F5" || code === "F5") {
    return "This button sends F5 (start slideshow). Many remotes alternate F5 and Shift+F5 or Escape on the same button, so check the light comes on at every press below. Chrome reloads the page on F5 when this tab isn't focused.";
  }
  if (k.toLowerCase() === "b" || k === ".") {
    return `This button sends ${labelFor(k)} (black screen in slideshow apps). Some remotes send it as a toggle, so check the light comes on at every press below.`;
  }
  if (STAGE_KEYS.has(k.toLowerCase())) {
    return `${labelFor(k)} is also a key on the Listen screen. It will work as the band, but a keyboard press of ${labelFor(k)} counts too.`;
  }
  return null;
}

/**
 * What one keydown means while pairing or testing.
 * `e`: {key, code, repeat, altKey, ctrlKey, metaKey}. Returns
 *   {kind: "ignore"}                       a held key's repeat, a lone modifier, a browser shortcut (Cmd/Ctrl/Alt+…)
 * | {kind: "refuse", reason}               Escape, Tab
 * | {kind: "press", band: {code, key, label}, warning}
 */
export function classifyKey(e) {
  if (!e || e.repeat) return { kind: "ignore" };
  if (e.ctrlKey || e.metaKey || e.altKey) return { kind: "ignore" };
  if (MODIFIERS.has(e.key)) return { kind: "ignore" };
  if (e.key === "Escape" || e.code === "Escape") return { kind: "refuse", reason: ESCAPE_REASON };
  if (e.key === "Tab") return { kind: "refuse", reason: TAB_REASON };
  const band = { code: String(e.code || ""), key: String(e.key || ""), label: labelFor(e.key, e.code) };
  return { kind: "press", band, warning: pairWarning(band.key, band.code) };
}

/** Whether a keydown is the paired button: by `code` when the remote sent one, else by `key`. */
export function matchesBand(band, e) {
  if (!band || !e) return false;
  if (band.code) return e.code === band.code;
  return Boolean(band.key) && e.key === band.key;
}

/** `hearing.sos` as stored, with the sound defaulted; {code:null, key:null, label:null} when not paired. */
export function loadSos(storage) {
  let raw = null;
  try { raw = JSON.parse(storage?.getItem(STORAGE_KEY) || "null"); } catch { raw = null; }
  const v = raw && typeof raw === "object" ? raw : {};
  const paired = typeof v.key === "string" && v.key !== "" || typeof v.code === "string" && v.code !== "";
  return {
    code: paired ? String(v.code || "") : null,
    key: paired ? String(v.key || "") : null,
    label: paired ? String(v.label || labelFor(v.key, v.code)) : null,
    sound: SOUNDS.some((s) => s.id === v.sound) ? v.sound : DEFAULT_SOUND,
  };
}

/** Merge `patch` into `hearing.sos`; returns the stored value. Storage failures (private mode) keep it in memory. */
export function saveSos(storage, patch) {
  const next = { ...loadSos(storage), ...patch };
  try { storage?.setItem(STORAGE_KEY, JSON.stringify(next)); } catch { /* quota or private mode: this visit only */ }
  return next;
}

/** Keys a focused control on this page needs (arrow through the sounds, Space/Enter on a button). */
function controlKey(e) {
  return /^Arrow/.test(e.key) || e.key === " " || e.key === "Enter";
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

export function mount(root, ctx = {}) {
  const signal = ctx.signal;
  const storage = (() => { try { return window.localStorage; } catch { return null; } })();
  let saved = loadSos(storage);
  let waiting = !saved.key && !saved.code;
  let presses = 0;
  let lightTimer = 0;

  const $status = el("p", { class: "sos-status", role: "status", "aria-live": "polite" });
  const $refuse = el("p", { class: "sos-refuse", role: "alert", hidden: "" });
  const $warn = el("p", { class: "sos-warn", hidden: "" });
  const $light = el("span", { class: "sos-light", "aria-hidden": "true" });
  const $lightText = el("span", { class: "sos-light-text" });
  const $pad = el("div", { class: "sos-pad", tabindex: "0", "aria-labelledby": "sos-pad-label" },
    $light, el("span", { id: "sos-pad-label", class: "sos-pad-label" }), $lightText);
  const $again = el("button", { type: "button", class: "sos-again" });
  const sounds = SOUNDS.map((s) => {
    const input = el("input", { type: "radio", name: "sos-sound", value: s.id, id: `sos-sound-${s.id}` });
    input.checked = s.id === saved.sound;
    input.addEventListener("change", () => { if (input.checked) saved = saveSos(storage, { sound: s.id }); }, { signal });
    return el("label", { class: "sos-sound", for: `sos-sound-${s.id}` }, input,
      el("span", { class: "sos-sound-name", text: s.name }), el("span", { class: "sos-sound-note", text: s.note }));
  });

  root.append(el("article", { class: "sos", "aria-labelledby": "sos-title" },
    el("p", { class: "eyebrow", text: "SOS" }),
    el("h1", { id: "sos-title", class: "sos-title", text: "Pair the band" }),
    el("p", { class: "sos-lede", text: "One press of the band turns the room off and plays a calm sound. Hold it 1.5 s to come back." }),
    el("p", { class: "sos-honest", text: "In reality this is a band on the wrist with one button. For today a presenter remote stands in for it." }),
    $pad, $status, $refuse, $warn,
    el("p", { class: "sos-actions" }, $again),
    el("fieldset", { class: "sos-sounds" },
      el("legend", { class: "sos-h", text: "Calm sound" }), ...sounds),
    el("p", { class: "sos-note", text: "Plug the remote in before you start: macOS may show Keyboard Setup Assistant the first time (close it). The band only works while this browser tab has focus." }),
    el("p", {}, el("a", { href: "#about", class: "link", text: "Back to About" }))));

  function render() {
    $pad.dataset.state = waiting ? "waiting" : "paired";
    $pad.querySelector(".sos-pad-label").textContent = waiting
      ? "Press the band's button once"
      : `Paired: ${saved.label}`;
    $lightText.textContent = waiting ? "Waiting for a press…" : presses ? `Test presses: ${presses}` : "Press it again to test";
    $status.textContent = waiting
      ? "Listening for one button. Any key on this keyboard counts too, so press the remote."
      : `Saved. The band is ${saved.label}${saved.code && saved.code !== saved.key ? ` (${saved.code})` : ""}.`;
    $again.textContent = waiting ? "Waiting…" : "Pair a different button";
    $again.disabled = waiting;
  }

  function show($node, text) {
    $node.textContent = text || "";
    $node.hidden = !text;
  }

  function flash() {
    clearTimeout(lightTimer);
    $pad.dataset.lit = "true";
    lightTimer = setTimeout(() => { delete $pad.dataset.lit; }, LIGHT_MS);
  }

  function onKey(e) {
    const inControl = e.target?.closest?.("input, button, a, select, textarea") && e.target !== $pad;
    if (inControl && controlKey(e)) return; // the sound picker / buttons keep their keys
    const step = classifyKey(e);
    if (step.kind === "ignore") {
      if (e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey) { e.preventDefault(); e.stopPropagation(); }
      return;
    }
    if (step.kind === "refuse") {
      if (e.key === "Tab") { show($refuse, waiting ? step.reason : ""); return; } // Tab still moves focus
      e.preventDefault();
      e.stopPropagation();
      show($refuse, step.reason);
      return;
    }
    e.preventDefault();
    e.stopPropagation(); // the stage keys (keys.js) never see a pairing press
    if (waiting) {
      saved = saveSos(storage, step.band);
      waiting = false;
      presses = 0;
      show($refuse, "");
      show($warn, step.warning);
      flash();
      render();
      return;
    }
    if (matchesBand(saved, e)) {
      presses += 1;
      show($refuse, "");
      flash();
      render();
    } else {
      show($refuse, `That was ${step.band.label}, not the paired button (${saved.label}). Use "Pair a different button" to change it.`);
    }
  }

  $again.addEventListener("click", () => {
    waiting = true;
    presses = 0;
    show($refuse, "");
    show($warn, "");
    render();
    $pad.focus({ preventScroll: true });
  }, { signal });
  // capture on window: runs before the document-level stage keys and app.js's Escape handler
  window.addEventListener("keydown", onKey, { capture: true, signal });

  if (!waiting) show($warn, pairWarning(saved.key, saved.code));
  render();
  queueMicrotask(() => $pad.focus({ preventScroll: true }));
  return () => clearTimeout(lightTimer);
}
