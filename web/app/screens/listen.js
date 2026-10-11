// Listen (M14, T-347): the stage screen. One huge state word (the /judge pill, same hysteresis), the person, the
// target line, RAW / Selective washing the whole frame amber / green, Focus / Quiet, saved person 1 / 2, Release,
// Quiet's captions, and the banners (keys off, mic, engine lost) and REPLAY badge. Keys are /judge's (keys.js).
//
// Honesty: everything rendered here comes from web/app/store.js (engine telemetry, caption messages, the page's
// own measured mic level). Nothing animates to look alive: the level halo is flat until the mic streams and then
// follows the real peak (24 dB/s release, session.js); a value nobody measured shows "—".
//
// The session is one per page (app.session): the first screen that needs it creates and connects it
// (ensureSession), so Setup/People/Listen share the socket and the mic.

import { gateState, renderGate } from "../../console/gate-state.js";
import { DEGRADED_NOTICE, MODES } from "../../console/judge-modes.js";
import { createSession } from "../session.js";
import { METER_FLOOR_DB, delayText, store as appStore } from "../store.js";
import { KEYS_OFF_TEXT, KEY_HINTS, bindKeys, watchKeyFocus } from "../keys.js";
import { mountPhoneLink } from "../connect.js";
import { captionWords, dialValue, dropAction, haptic, roomLabel } from "../delight.js";
import { pulseFor } from "../transitions.js";
import { BUSY_TEXT, busyState } from "../preflight.js";
import { scenePrompt, touchDevice } from "../platform.js";

export const title = "Listen";

/** The state word for each pill state (judge-pill.js), and its tone (colour). */
export const STATE_WORDS = {
  idle: { word: "Idle", tone: "idle" },
  lost: { word: "Reconnecting", tone: "bad" },
  nomic: { word: "No mic", tone: "bad" },
  loading: { word: "Loading voice", tone: "work" },
  calibrating: { word: "Calibrating", tone: "work" },
  isolated: { word: "Isolated", tone: "focus" },
  look: { word: "Searching", tone: "idle" },
  speak: { word: "Waiting for a voice", tone: "work" },
  reacquiring: { word: "Finding them", tone: "warn" },
  fallback: { word: "Not isolated", tone: "warn" },
};

/**
 * T-463: the word while the engine's own gate doubts the pick. "Isolated" in green is shown only when the gate says
 * SOLO (or reports nothing, an engine without the field); UNSURE and OVERLAP-DUCKED take the warn tone.
 */
export const GATE_WORDS = {
  locked: { word: "Locked", tone: "focus" }, // T-487: the team slot is the person; steady, never flips on a pause
  unsure: { word: "Listening", tone: "warn" },
  ducked: { word: "Overlap", tone: "warn" },
};
/** T-463: the first seconds after Start, before the first healthy window (no "the Mac fell behind" at Start). */
export const WARMUP_MS = 3000;
export const STARTING = { word: "Starting", tone: "work" };
const WARMUP_STATES = new Set(["fallback", "reacquiring", "speak", "look"]);

/** What the pressed mode does, visible on touch screens where a title tooltip never shows (T-365 P1-13). */
export const MODE_NOTES = {
  focus_hq: "Focus: you hear the chosen person over the room.",
  quiet: "Quiet: nothing plays. Their words appear as captions.",
};

/** "Loading voice — aryan" -> "aryan"; the name the pill carries after its dash, if any. */
function pillDetail(text) {
  const i = String(text || "").indexOf(" — ");
  return i < 0 ? "" : text.slice(i + 3);
}

/**
 * The pill's sentence in plain words (T-365 P0-6): engine words ("overrun", "known voice: aryan") never reach the
 * stage. `who` is the display name of the chosen person ("Aryan"), never a profile id. Returns "" when the sentence
 * would only repeat the state word.
 */
export function plainSentence(pill, { word = "", who = "" } = {}) {
  const text = String(pill?.text || "");
  const detail = pillDetail(text).toLowerCase();
  const head = (text.split(" — ")[0] || "").trim();
  if (/overrun/.test(detail) || /overrun/i.test(head)) return "Catching up: the Mac fell behind, so you hear the cleaned room for a moment.";
  if (pill?.state === "loading") return who ? `Loading ${who}'s saved voice…` : "Loading the saved voice…";
  if (pill?.state === "isolated") return ""; // the word and the name already say it
  // T-403: the app is audio-only; the judge pill's camera instructions ("Look at the person") would send a judge to
  // look for a camera. Say what the listener can do instead.
  if (/look at (the person|them)/i.test(text)) {
    if (pill?.state === "reacquiring") return who ? `Finding ${who} again: keep them talking.` : "Finding them again: keep them talking.";
    return who ? `Waiting to hear ${who}.` : "Choose a saved person below.";
  }
  // the ladder holds the saved voice out for another reason (warming, health): say what plays, not the engine's why
  if (pill?.state === "fallback" && /^voice profile paused/i.test(head)) {
    return `You hear the cleaned room until ${who ? `${who}'s` : "the saved"} voice is back.`;
  }
  if (/^known voice/.test(detail)) return head === word ? "" : head;
  if (detail && head) return head === word ? "" : head; // a lowercase id after the dash is dropped
  return text === word ? "" : text;
}

/**
 * Pure: what the Listen screen shows for a store state. `keysOff` comes from the focus watcher.
 * Every field is a string or boolean the DOM writes as is; none is invented when telemetry is missing.
 */
export function listenModel(s, { keysOff = false, keys = true, warming = false } = {}) {
  const pill = s.pill || { state: "idle", text: "Press Start", note: "" };
  const t = s.telemetry;
  const person = s.person || {};
  const quiet = s.mode === "quiet";
  const raw = s.ab === "raw";
  const gate = pill.state === "isolated" && !raw && !quiet ? gateState(t) : null;
  const loudest = gate === "loudest"; // HEARING_PICK=loudest: the person at the mic plays; no profile, never UNSURE
  const doubt = GATE_WORDS[gate] || null; // T-463: never a green "Isolated" next to UNSURE
  // T-463: right after Start the ladder is still climbing (an overrun, no voice yet): say Starting, not "fell behind"
  const starting = Boolean(warming) && !raw && (WARMUP_STATES.has(pill.state) || Boolean(doubt));
  const look = starting ? STARTING : doubt || STATE_WORDS[pill.state] || { word: pill.state, tone: "idle" };
  const connecting = pill.state === "lost" && s.connection === "connecting" && !s.attempt;
  const live = Boolean(s.replaying) || (s.started && !["idle", "nomic", "lost"].includes(pill.state));
  // Raw while live: the room plays, so the word says Raw (amber); the pill sentence keeps the engine's state
  const rawWord = raw && live && !quiet;
  const word = connecting ? "Connecting" : rawWord ? "Raw" : look.word;
  // who the frame is about: the targeted saved person, else a calibrated voice, else nobody
  const who = loudest ? "Person at the mic" : person.currentLabel || (person.calibrated ? "Calibrated voice" : "");
  // a touch screen has no Space key: name the control instead (finish review)
  const back = keys ? "Space to hear" : "Tap Selective to hear";
  const name = who || "the chosen voice";
  let sentence = rawWord ? (who ? `${back} ${who} alone.` : `${back} the chosen voice alone.`)
    : plainSentence(pill, { word: look.word, who: person.currentLabel || person.lastLabel || "" });
  if (starting && !rawWord) sentence = "Warming up the engine…";
  else if (loudest) sentence = "Only the loudest voice, the person at the mic, plays. Everything else is silenced.";
  else if (gate === "locked") sentence = `Locked on ${name}: their voice plays steadily until you pick someone else.`;
  else if (gate === "unsure") sentence = `Waiting to hear ${name} clearly: they may be quiet, or another voice is louder.`;
  else if (gate === "ducked") sentence = "Two voices overlap, so the engine plays its best guess softened.";
  let hear;
  if (!live) hear = s.connection === "open" ? "Nothing playing yet" : "Not connected";
  else if (quiet) hear = "Quiet: nothing plays, captions only";
  else if (raw) hear = "You hear the room, unprocessed (Raw)";
  else if (starting) hear = "Starting: the engine is warming up";
  else if (loudest) hear = "You hear the loudest voice";
  else if (gate === "locked") hear = `You hear ${name}, locked`;
  else if (gate === "unsure") hear = `Not sure it is ${name} yet`;
  else if (gate === "ducked") hear = `You hear ${name}, softened`;
  else if (pill.state === "isolated") hear = who ? `You hear ${who}` : "You hear the isolated voice";
  else hear = "You hear the engine's output (not isolated yet)";
  const truth = s.truth || {};
  const dash = (v, unit = "") => (v == null ? "—" : `${v}${unit}`);
  const retry = s.connection === "reconnecting" && s.retryInMs != null
    ? `Engine lost. Retrying in ${(s.retryInMs / 1000).toFixed(1)} s (attempt ${s.attempt}).` : "";
  const people = (person.names || []).length ? person.labels : [];
  return {
    gate,
    loudest,
    starting,
    word,
    tone: rawWord ? "raw" : look.tone,
    state: connecting ? "connecting" : pill.state,
    sentence,
    note: rawWord ? "" : pill.note || "",
    detail: pillDetail(pill.text),
    who,
    // the hero's name line: "for Joseph" under "Listening" (the name is not what plays yet), else the name
    whoLine: who && gate === "unsure" && !starting ? `for ${who}` : who && gate === "locked" && !starting ? `on ${who}` : who,
    hear,
    live,
    targetLine: loudest ? "Person at the mic" : person.released ? releasedLine(keys) : person.line || "Target: —",
    modeNote: MODE_NOTES[quiet ? "quiet" : "focus_hq"],
    ab: raw ? "raw" : "processed",
    mode: quiet ? "quiet" : "focus_hq",
    effectiveMode: s.effectiveMode || null,
    people: [0, 1].map((i) => ({
      label: people[i] || `Person ${i + 1}`,
      enabled: Boolean(people[i]),
      // T-463: the chosen person is pressed even before Start (a neutral fill: chosen; green only while isolated plays)
      pressed: Boolean(people[i]) && !person.released && person.names[i] === (person.current || person.last),
    })),
    released: Boolean(person.released),
    start: s.replaying ? "hidden" : s.starting ? "starting" : s.started ? "stop" : "start",
    banners: {
      keysOff: keysOff ? KEYS_OFF_TEXT : "",
      mic: s.mic?.banner || "",
      engine: retry,
      // T-485: under the team preset, the server's load or the enrolled tier overrunning (real telemetry only)
      busy: s.telemetry?.preset?.pins && busyState(s.telemetry).busy ? BUSY_TEXT : "",
      busyWhy: s.telemetry?.preset?.pins ? busyState(s.telemetry).why : "",
    },
    replay: s.replaying ? (s.replayClip ? `Replay · ${String(s.replayClip).split("/")[0]}` : "Replay") : "",
    captions: {
      show: quiet || (s.captions || []).length > 0,
      quiet,
      speaker: s.caption?.speaker ? capital(s.caption.speaker) : "Target",
      lines: captionLines(s.captions),
      paused: s.caption?.paused ? pausedCopy(s.caption.paused, who || person.lastLabel) : "",
      stale: !live && (s.captions || []).length > 0,
      degraded: s.caption?.degraded ? DEGRADED_NOTICE : "",
    },
    truth: {
      backend: truth.backend || "—", // "Mac" or "GPU"; the torch device ("mps") lives in Diagnostics (T-388)
      delay: delayText(truth), // T-463: one format on every screen, "settling…" until it holds (store.js)
      rtt: truth.rttMs === 0 ? "<10 ms" : dash(truth.rttMs, " ms"), // the steady RTT moves in 10 ms steps (P2-3)
      gpuRtt: truth.gpuRttMs == null ? "" : `${truth.gpuRttMs} ms`,
      tier: truth.tier && truth.tier !== "—" ? truth.tier : "—",
    },
    confidence: typeof truth.confidence === "number" ? truth.confidence : null,
    hasTelemetry: Boolean(t),
  };
}

/** A touch on the room dial: "drag" once it moves sideways past 8 px, "scroll" once it moves up or down, else "wait". */
export function dialGesture(dx, dy, slop = 8) {
  if (Math.abs(dx) > slop && Math.abs(dx) > Math.abs(dy)) return "drag";
  if (Math.abs(dy) > slop) return "scroll";
  return "wait";
}

/** Quiet's pause notice in plain words: never "Captions paused: … captions paused". */
export function pausedCopy(paused, who) {
  if (!paused) return "";
  return who ? `Captions wait until ${who} is isolated.` : "Captions wait until the chosen voice is isolated.";
}

/** T-463: the Release line in plain words (no "Target: none (Release)"); a touch screen has no 1 / 2 keys. */
export function releasedLine(keys = true) {
  return keys ? "No one chosen. Press 1 or 2, or drag a person into the ring." : "No one chosen. Tap a person to hear them.";
}

const normCaption = (text) => String(text || "").toLowerCase().replace(/[^\p{L}\p{N}\s']/gu, "").replace(/\s+/g, " ").trim();
// words a sentence carries on with; a fragment that starts with one after an unfinished line is not a new sentence
const CONTINUES = new Set(["a", "an", "and", "at", "because", "but", "by", "for", "from", "in", "into", "of", "on", "or", "so",
  "than", "that", "the", "then", "to", "when", "where", "which", "while", "with"]);

/**
 * Caption lines as shown (T-463): Whisper's "//" removed, a final identical to the final before it dropped (the
 * screen never shows one sentence twice in a row), and a fragment that continues an unfinished line ("…early" /
 * "To beat the traffic.") starts lower-case. Only common joining words are lowered, never a name.
 */
export function captionLines(captions = []) {
  const out = [];
  for (const c of captions || []) {
    let text = cleanLine(c.text);
    if (!text) continue;
    const prev = out[out.length - 1];
    if (c.final && prev?.final && normCaption(prev.text) === normCaption(text)) continue;
    const first = text.split(/\s+/)[0];
    if (prev && !/[.?!…]["')\]]?$/.test(prev.text) && CONTINUES.has(first.toLowerCase()) && first[0] !== first[0].toLowerCase()) {
      text = first[0].toLowerCase() + text.slice(1);
    }
    out.push({ text, final: Boolean(c.final) });
  }
  return out;
}

/** A caption line as shown: Whisper's trailing "//" (not yet stripped by the server's clean_caption) removed. */
export function cleanLine(text) {
  return String(text || "").replace(/\s*\/\/\s*$/, "").trim();
}

function capital(s) {
  const v = String(s || "");
  return v ? v[0].toUpperCase() + v.slice(1) : v;
}

/** Mic peak (dBFS) -> 0..1 for the halo: -60 dBFS flat, 0 dBFS full. */
export function levelUnit(db) {
  if (typeof db !== "number" || !Number.isFinite(db)) return 0;
  return Math.min(1, Math.max(0, (db - METER_FLOOR_DB) / -METER_FLOOR_DB));
}

/** The page's one session: created and connected by the first screen that needs it. */
export function ensureSession(app) {
  if (app && !app.session) {
    app.session = createSession({ store: appStore });
    app.session.connect();
  }
  return app?.session || null;
}

// ------------------------------------------------------------------ DOM

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

function kbd(text) {
  return h("kbd", { class: "ls-kbd", text });
}

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

function setAttr(node, name, value) {
  const v = value == null ? null : String(value);
  if (node.getAttribute(name) !== v) {
    if (v == null) node.removeAttribute(name);
    else node.setAttribute(name, v);
  }
}

export function mount(root, ctx = {}) {
  const app = ctx.app || globalThis.hearingApp || null;
  const signal = ctx.signal;
  const store = appStore;
  const session = ensureSession(app);
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = matchMedia("(hover: hover) and (pointer: fine)");
  const touch = touchDevice(); // T-458: tap hints, the dial's horizontal-only drag, the inline scene's prompt
  let keysOff = false;

  // ---- structure
  const banners = h("div", { class: "ls-banners", role: "status", "aria-live": "polite" },
    h("p", { class: "ls-banner", "data-kind": "engine", hidden: true }),
    h("p", { class: "ls-banner", "data-kind": "mic", hidden: true },
      h("span", { class: "ls-banner-text" }),
      h("button", { type: "button", class: "ls-banner-action", "data-act": "start", text: "Start again" })),
    h("p", { class: "ls-banner", "data-kind": "keys", hidden: true }),
    h("p", { class: "ls-banner", "data-kind": "busy", hidden: true }));
  const replayBadge = h("p", { class: "ls-replay", hidden: true, "aria-live": "polite" },
    h("span", { class: "ls-replay-dot", "aria-hidden": "true" }), h("span", { class: "ls-replay-text" }));

  const halo = h("div", { class: "ls-halo", "aria-hidden": "true", "data-live": "off" },
    h("span", { class: "ls-halo-ring", style: "--k:3" }),
    h("span", { class: "ls-halo-ring", style: "--k:2" }),
    h("span", { class: "ls-halo-ring", style: "--k:1" }),
    h("span", { class: "ls-halo-core" }));

  const hero = h("section", { class: "ls-hero", "aria-labelledby": "ls-word" },
    h("p", { class: "ls-hear" }),
    h("h1", { id: "ls-word", class: "ls-word" }),
    h("p", { class: "ls-who" }),
    h("p", { class: "ls-gate", hidden: true, role: "status" }),
    h("p", { class: "ls-sentence" }),
    h("p", { class: "ls-note", hidden: true }),
    h("p", { class: "ls-target" }));

  const captions = h("section", { class: "ls-captions", "aria-label": "Captions", hidden: true },
    h("header", { class: "ls-cap-head" },
      h("span", { class: "ls-cap-speaker" }),
      h("span", { class: "ls-cap-source", text: "captions from the engine" })),
    h("p", { class: "ls-cap-warn", hidden: true }),
    h("div", { class: "ls-cap-lines" }));
  // T-404: a screen reader hears the state word with what plays, once per change, and each finished caption line
  // once; the visible caption list is re-rendered on every partial, so it is not a live region itself
  const srState = h("p", { class: "sr-only", role: "status" });
  const srCaption = h("p", { class: "sr-only", "aria-live": "polite" });

  const seg = (label, hint, attrs) => h("button", { type: "button", class: "ls-seg", "aria-pressed": "false", ...attrs },
    h("span", { class: "ls-seg-label", text: label }), kbd(hint));

  const abGroup = h("div", { class: "ls-group ls-ab", role: "group", "aria-label": "What you hear" },
    seg("Selective", KEY_HINTS.ab, { "data-ab": "processed" }),
    seg("Raw", KEY_HINTS.ab, { "data-ab": "raw" }));
  const modeGroup = h("div", { class: "ls-group ls-mode", role: "group", "aria-label": "Mode" },
    seg(MODES.focus_hq.title, KEY_HINTS.focus_hq, { "data-mode": "focus_hq", title: MODES.focus_hq.copy }),
    seg(MODES.quiet.title, KEY_HINTS.quiet, { "data-mode": "quiet", title: MODES.quiet.copy }));
  const modeNote = h("p", { class: "ls-mode-note" });
  const peopleGroup = h("div", { class: "ls-group ls-people", role: "group", "aria-label": "Saved person" },
    seg("Person 1", KEY_HINTS.person0, { "data-person": "0", disabled: true }),
    seg("Person 2", KEY_HINTS.person1, { "data-person": "1", disabled: true }),
    seg("Release", KEY_HINTS.release, { "data-act": "release", class: "ls-seg ls-release" }));
  // T-427 direct manipulation: the room dial (control.suppression, the T-093 curve) and the saved voices in the
  // sound scene (drag or tap one into the ring = the same choosePerson the Person buttons send)
  const dial = h("div", { class: "ls-dial", role: "slider", tabindex: "0", "aria-label": "Room turned down",
    "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": "100", "aria-valuetext": "silent" },
  h("span", { class: "ls-dial-track", "aria-hidden": "true" }, h("span", { class: "ls-dial-fill" }), h("span", { class: "ls-dial-thumb" })));
  // T-458: the dial's two ends named (phones; right = the room turned down, the opposite of a volume slider)
  const dialEnds = h("p", { class: "ls-dial-ends", "aria-hidden": "true" }, h("span", { text: "room as it is" }), h("span", { text: "room silent" }));
  const dialValueText = h("output", { class: "ls-dial-value", text: "silent" });
  const voiceBtn = (i) => h("button", { type: "button", class: "ls-voice", "data-voice": String(i), "aria-pressed": "false", disabled: true },
    h("span", { class: "ls-voice-dot", "aria-hidden": "true" }), h("span", { class: "ls-voice-name", text: `Person ${i + 1}` }));
  const voices = h("div", { class: "ls-voices", role: "group", "aria-label": "Sound scene: drag a saved person into the ring, or tap them", "data-has": "0" },
    h("span", { class: "ls-voices-ring", "aria-hidden": "true" }), voiceBtn(0), voiceBtn(1),
    h("p", { class: "ls-voices-hint", "aria-hidden": "true", text: scenePrompt(touch) }));
  const startBtn = h("button", { type: "button", class: "ls-start", "data-act": "start" },
    h("span", { class: "ls-start-label", text: "Start listening" }));

  const rail = h("nav", { class: "ls-rail", "aria-label": "Stage controls" },
    h("div", { class: "ls-rail-cell" }, h("span", { class: "ls-rail-label", text: "Hear" }), abGroup),
    h("div", { class: "ls-rail-cell" }, h("span", { class: "ls-rail-label", text: "Mode" }), modeGroup, modeNote),
    h("div", { class: "ls-rail-cell ls-rail-people" }, h("span", { class: "ls-rail-label", text: "Person" }), peopleGroup),
    h("div", { class: "ls-rail-cell ls-rail-room" }, h("span", { class: "ls-rail-label" }, "Room ", dialValueText), dial, dialEnds),
    h("div", { class: "ls-rail-cell ls-rail-start" }, startBtn));

  // the keys once, in a legend under the rail, instead of a chip in every button (T-365 anti-template)
  const legend = h("p", { class: "ls-legend" },
    ...[[KEY_HINTS.ab, "Raw or Selective"], [`${KEY_HINTS.focus_hq} / ${KEY_HINTS.quiet.split(" ").pop()}`, "Focus or Quiet"],
      [`${KEY_HINTS.person0} / ${KEY_HINTS.person1}`, "saved person"], [KEY_HINTS.release, "Release"]]
      .map(([k, what]) => h("span", { class: "ls-legend-item" }, kbd(k), ` ${what}`)));
  const fact = (key, label) => h("div", { class: "ls-fact", "data-fact": key },
    h("dt", { text: label }), h("dd", { text: "—" }));
  const truth = h("dl", { class: "ls-truth", "aria-label": "Measured now" },
    fact("backend", "Backend"), fact("delay", "Engine delay"), fact("rtt", "Network round trip"), fact("tier", "Tier"),
    h("div", { class: "ls-fact ls-fact-diag" }, h("button", { type: "button", class: "ls-diag", "data-act": "diag" },
      h("span", { text: "Diagnostics" }), kbd(KEY_HINTS.diag))));
  // no dead ends (T-365 P0-7): the Diagnostics button and D exist only when the server has the sheet
  const hasDiag = !app?.files?.screens || app.files.screens.includes("diag");
  truth.querySelector(".ls-fact-diag").hidden = !hasDiag;

  const frame = h("div", { class: "listen", "data-ab": "processed", "data-tone": "idle", "data-mode": "focus_hq" },
    h("div", { class: "ls-wash", "aria-hidden": "true" }),
    h("div", { class: "ls-top" }, banners, replayBadge),
    h("div", { class: "ls-body" }, hero, halo, captions), voices,
    rail, legend, truth, srState, srCaption);
  root.classList.add("screen-listen");
  root.replaceChildren(frame);
  // the shared device banner (web/console/devices.js, fixed over the page on /judge) sits in flow here, above the
  // word, so it never covers the state (T-365 P0-5); it goes back to <body> when Listen leaves
  const devBanner = document.querySelector("[data-dev-banner]");
  if (devBanner) frame.querySelector(".ls-top").prepend(devBanner);
  mountPhoneLink(frame.querySelector(".ls-top"), store, { signal }); // T-351: QR to the phone companion

  const q = (sel) => frame.querySelector(sel);
  const el = {
    hear: q(".ls-hear"), word: q(".ls-word"), who: q(".ls-who"), gate: q(".ls-gate"), sentence: q(".ls-sentence"), note: q(".ls-note"),
    target: q(".ls-target"), engine: q('[data-kind="engine"]'), mic: q('[data-kind="mic"]'), micText: q('[data-kind="mic"] .ls-banner-text'),
    keys: q('[data-kind="keys"]'), busy: q('[data-kind="busy"]'), replayText: q(".ls-replay-text"), capSpeaker: q(".ls-cap-speaker"), capWarn: q(".ls-cap-warn"),
    capLines: q(".ls-cap-lines"), startLabel: q(".ls-start-label"),
  };

  // ---- actions (buttons and keys send exactly the same thing)
  const act = {
    start: () => {
      if (!session) return;
      if (store.state.started && !store.state.replaying) session.stop();
      else session.start(); // inside the click/keydown: the AudioContext is created in the gesture (iOS)
    },
    ab: (side) => session?.setAb(side),
    mode: (mode) => session?.setMode(mode),
    person: (i) => session?.choosePerson(i),
    release: () => session?.release(),
    calibrate: () => session?.calibrate(),
    replay: () => session?.replay().catch((e) => store.note(`replay: ${e?.message || e}`)),
    diag: () => {
      if (!app || !hasDiag) return;
      if (app.sheet === "diag") app.closeSheet();
      else app.openSheet("diag");
    },
  };

  frame.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled || !frame.contains(b)) return;
    if (b.dataset.ab) act.ab(b.dataset.ab);
    else if (b.dataset.mode) act.mode(b.dataset.mode);
    else if (b.dataset.person) act.person(Number(b.dataset.person));
    else if (b.dataset.voice) { if (!dragged) chooseVoice(Number(b.dataset.voice)); dragged = false; }
    else if (b.dataset.act) act[b.dataset.act]?.();
  }, { signal });

  bindKeys(document, {
    ab: () => act.ab(),
    mode: (a) => act.mode(a.mode),
    person: (a) => act.person(a.index),
    release: () => act.release(),
    calibrate: () => act.calibrate(),
    replay: () => act.replay(),
    diag: () => act.diag(),
  }, { signal, drawerOpen: () => app?.sheet === "diag" });

  // ---- render
  // T-463: the warm-up after Start lasts WARMUP_MS, or less once a healthy window (Isolated, gate SOLO) arrives
  let streamSince = 0;
  let healthy = false;
  let warmTimer = 0;
  function warmingNow() {
    const st = store.state;
    const streaming = Boolean(st.started && !st.replaying && st.connection === "open");
    if (!streaming) { streamSince = 0; healthy = false; clearTimeout(warmTimer); return false; }
    if (!streamSince) {
      streamSince = performance.now();
      clearTimeout(warmTimer);
      warmTimer = setTimeout(() => { if (!signal?.aborted) render(); }, WARMUP_MS + 50);
    }
    if (st.pill?.state === "isolated" && ["solo", "locked", "loudest", null].includes(gateState(st.telemetry))) healthy = true;
    return !healthy && performance.now() - streamSince < WARMUP_MS;
  }

  function render() {
    const m = listenModel(store.state, { keysOff, keys: finePointer.matches && !touch, warming: warmingNow() });
    frame.dataset.ab = m.ab;
    frame.dataset.tone = m.tone;
    frame.dataset.state = m.state;
    frame.dataset.mode = m.mode;
    frame.dataset.loudest = String(m.loudest);
    frame.dataset.live = String(m.live);
    frame.dataset.replay = String(Boolean(m.replay));
    frame.dataset.captions = String(m.captions.show);
    // T-463: green (wash, scene) only while the gate says SOLO; a doubting gate or the warm-up washes neutral
    frame.dataset.doubt = String(Boolean(m.starting || m.gate === "unsure" || m.gate === "ducked"));
    document.body.dataset.ab = m.ab; // the shell chrome takes the wash too
    setText(el.hear, m.hear);
    setText(el.word, m.word);
    el.word.style.setProperty("--chars", String(Math.max(4, m.word.length))); // the word fills its column, never overflows
    // the name slot holds a person's name or nothing; never a pill detail (T-365 P0-6)
    setText(el.who, m.tone === "raw" || !m.live || m.starting ? "" : m.whoLine);
    el.who.hidden = !el.who.textContent;
    renderGate(el.gate, store.state.telemetry, m.state === "isolated" && m.ab !== "raw" && m.mode !== "quiet");
    // T-458: on a phone an unsure window is not "Isolated in green next to a red UNSURE": the word takes the warn tone
    // (listen.css) and the chip says what it means
    const unsure = el.gate.dataset.gate === "unsure";
    frame.dataset.unsure = String(unsure);
    if (unsure && touch) setText(el.gate, "low confidence: still checking");
    setText(el.sentence, m.sentence);
    el.sentence.hidden = !m.sentence;
    setText(el.note, m.note);
    el.note.hidden = !m.note;
    setText(el.target, m.targetLine);
    el.target.hidden = !el.who.hidden && m.targetLine === `Target: ${m.who}`; // the name says it once

    setText(el.engine, m.banners.engine);
    el.engine.hidden = !m.banners.engine;
    setText(el.micText, m.banners.mic);
    el.mic.hidden = !m.banners.mic;
    el.keys.hidden = !m.banners.keysOff;
    setText(el.keys, m.banners.keysOff);
    setText(el.busy, m.banners.busy);
    el.busy.hidden = !m.banners.busy;
    setAttr(el.busy, "title", m.banners.busyWhy || null);
    setText(el.replayText, m.replay);
    replayBadge.hidden = !m.replay;

    for (const b of abGroup.querySelectorAll("[data-ab]")) setAttr(b, "aria-pressed", b.dataset.ab === m.ab);
    for (const b of modeGroup.querySelectorAll("[data-mode]")) setAttr(b, "aria-pressed", b.dataset.mode === m.mode);
    for (const b of peopleGroup.querySelectorAll("[data-person]")) {
      const p = m.people[Number(b.dataset.person)];
      setText(b.querySelector(".ls-seg-label"), p.label);
      b.disabled = !p.enabled;
      setAttr(b, "aria-pressed", p.pressed);
      setAttr(b, "title", p.enabled ? `Make ${p.label} the target` : store.state.connection === "open"
        ? "No saved voice profile" : "Waiting for the engine");
    }
    setAttr(peopleGroup.querySelector(".ls-release"), "aria-pressed", m.released);
    setText(modeNote, m.modeNote);
    startBtn.hidden = m.start === "hidden";
    startBtn.disabled = m.start === "starting";
    startBtn.dataset.kind = m.start;
    setText(el.startLabel, { start: "Start listening", starting: "Starting…", stop: "Stop", hidden: "" }[m.start]);

    captions.hidden = !m.captions.show;
    setText(el.capSpeaker, m.captions.quiet ? m.captions.speaker : "Captions");
    const warn = m.captions.stale ? "From the last session." : m.captions.quiet ? m.captions.paused || m.captions.degraded : "";
    setText(el.capWarn, warn);
    el.capWarn.hidden = !warn;
    el.capLines.dataset.paused = String(Boolean(m.captions.quiet && m.captions.paused));
    el.capLines.dataset.stale = String(m.captions.stale);
    el.capWarn.dataset.kind = m.captions.paused ? "paused" : "degraded";
    renderCaptions(m.captions.lines);
    setText(srState, m.sentence ? `${m.word}. ${m.hear}. ${m.sentence}` : `${m.word}. ${m.hear}.`);
    const lastFinal = [...m.captions.lines].reverse().find((c) => c.final);
    if (m.live && lastFinal) setText(srCaption, lastFinal.text);

    for (const [key, value] of Object.entries(m.truth)) {
      const dd = truth.querySelector(`[data-fact="${key}"] dd`);
      if (dd) setText(dd, key === "rtt" && m.truth.gpuRtt ? `${value} · GPU ${m.truth.gpuRtt}` : value);
    }
    renderVoices(m);
    renderDial(store.state.suppression);
    const pulse = pulseFor("listen", lastState, m.state);
    if (pulse === "lock") haptic("isolate", { reduced: reducedMotion.matches }); // the engine said Isolated, not a tap
    lastState = m.state;
    halo.dataset.live = m.live ? "on" : "off";
    halo.style.setProperty("--conf", m.confidence == null ? "0" : m.confidence.toFixed(2));
    if (!m.live) setLevel(null);
  }

  let shownCaptions = "";
  let shownTexts = [];
  function renderCaptions(lines) {
    const key = JSON.stringify(lines);
    if (key === shownCaptions) return;
    shownCaptions = key;
    // T-427: a word fades in when the engine's text first carries it (delight.js captionWords); words already on
    // screen stay still, and nothing is drawn ahead of the recogniser
    const prev = shownTexts;
    shownTexts = lines.map((c) => c.text);
    el.capLines.replaceChildren(...(lines.length
      ? lines.map((c) => {
        let fresh = 0;
        const spans = captionWords(c.text, prev).map(({ word, fresh: isNew }) => {
          const w = h("span", { class: isNew ? "ls-w is-new" : "ls-w", text: word });
          if (isNew) w.style.setProperty("--i", String(Math.min(fresh++, 6)));
          return w;
        });
        const line = h("p", { class: c.final ? "ls-cap" : "ls-cap partial" });
        spans.forEach((w, i) => line.append(...(i ? [" ", w] : [w])));
        return line;
      })
      : [h("p", { class: "ls-cap empty", text: "No captions yet. They appear when the target speaks." })]));
  }

  // ---- T-427: the saved voices in the scene. Their place says what the engine reports (aria-pressed = the target
  // the telemetry names); a tap or a drop only asks, and shows "asked" until the engine answers
  let lastState = null;
  let dragged = false;
  let askedTimer = 0;
  function renderVoices(m) {
    const has = m.people.some((p) => p.enabled);
    voices.dataset.has = has ? "1" : "0";
    voices.dataset.live = String(m.live);
    voices.dataset.isolated = String(m.state === "isolated" && m.ab !== "raw" && m.mode !== "quiet" && m.tone === "focus");
    for (const b of voices.querySelectorAll("[data-voice]")) {
      const p = m.people[Number(b.dataset.voice)];
      setText(b.querySelector(".ls-voice-name"), p.label);
      b.disabled = !p.enabled;
      setAttr(b, "aria-pressed", p.pressed);
      setAttr(b, "aria-label", p.pressed ? `${p.label}: chosen. Drag out of the ring to release` : `Hear ${p.label}`);
      if (p.pressed && b.dataset.asked) delete b.dataset.asked;
    }
  }
  function chooseVoice(i) {
    const b = voices.querySelector(`[data-voice="${i}"]`);
    if (!b || b.disabled || b.getAttribute("aria-pressed") === "true") return;
    for (const o of voices.querySelectorAll("[data-asked]")) delete o.dataset.asked;
    b.dataset.asked = "true";
    clearTimeout(askedTimer);
    askedTimer = setTimeout(() => { delete b.dataset.asked; }, 4000); // no answer: drop the "asked" look, never fake one
    haptic("press", { reduced: reducedMotion.matches });
    act.person(i);
  }
  // drag: the voice follows the finger (translate only, compositor); on release it lands by dropAction
  let drag = null;
  voices.addEventListener("pointerdown", (e) => {
    const b = e.target.closest("[data-voice]");
    if (!b || b.disabled || (e.pointerType === "mouse" && e.button !== 0)) return;
    drag = { b, id: e.pointerId, x0: e.clientX, y0: e.clientY, moved: false };
    try { b.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
  }, { signal });
  voices.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const dx = e.clientX - drag.x0;
    const dy = e.clientY - drag.y0;
    if (!drag.moved && Math.hypot(dx, dy) < 6) return;
    drag.moved = true;
    drag.b.dataset.dragging = "true";
    drag.b.style.translate = `${dx}px ${dy}px`;
    const r = ringRect();
    const bb = drag.b.querySelector(".ls-voice-dot").getBoundingClientRect();
    const over = dropAction({ dx: bb.x + bb.width / 2 - r.cx, dy: bb.y + bb.height / 2 - r.cy, r: r.r,
      current: drag.b.getAttribute("aria-pressed") === "true" });
    voices.dataset.over = over;
  }, { signal });
  const endDrag = (e, cancel = false) => {
    if (!drag || e.pointerId !== drag.id) return;
    const { b, moved } = drag;
    drag = null;
    const over = voices.dataset.over;
    delete voices.dataset.over;
    delete b.dataset.dragging;
    b.style.translate = ""; // springs home (CSS transition); home is wherever the engine's answer puts it
    if (!moved) return;
    dragged = true; // the click that follows a drag is not a tap
    if (cancel) return;
    if (over === "choose") chooseVoice(Number(b.dataset.voice));
    else if (over === "release") act.release();
  };
  voices.addEventListener("pointerup", (e) => endDrag(e), { signal });
  voices.addEventListener("pointercancel", (e) => endDrag(e, true), { signal });
  function ringRect() {
    const ring = voices.querySelector(".ls-voices-ring").getBoundingClientRect();
    return { cx: ring.x + ring.width / 2, cy: ring.y + ring.height / 2, r: Math.max(24, ring.width / 2) };
  }

  // ---- T-427: the room dial. Its position is what this page sent (control.suppression); the sound it makes is the
  // engine's (the scene's form follows the played stream)
  function renderDial(v) {
    const value = dialValue(v);
    setAttr(dial, "aria-valuenow", value);
    setAttr(dial, "aria-valuetext", `room ${roomLabel(value)}`);
    dial.style.setProperty("--v", String(value / 100));
    setText(dialValueText, roomLabel(value));
  }
  const setDial = (v) => {
    const value = dialValue(v);
    if (value !== store.state.suppression) session?.setSuppression(value);
  };
  const dialFromX = (x) => {
    const r = dial.querySelector(".ls-dial-track").getBoundingClientRect();
    return ((x - r.x) / Math.max(1, r.width)) * 100;
  };
  // T-458: a finger may only be scrolling past the dial (touch-action: pan-y in listen.css), so a touch moves it only
  // once the drag is horizontal (dialGesture); a tap without a move sets it where it landed. A mouse grabs at once.
  let dialPtr = null;
  let dialTouch = null;
  const grabDial = (e) => {
    dialPtr = e.pointerId;
    dial.dataset.dragging = "true";
    try { dial.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
    setDial(dialFromX(e.clientX));
  };
  dial.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (e.pointerType === "mouse") grabDial(e);
    else dialTouch = { id: e.pointerId, x0: e.clientX, y0: e.clientY };
  }, { signal });
  dial.addEventListener("pointermove", (e) => {
    if (e.pointerId === dialPtr) { setDial(dialFromX(e.clientX)); return; }
    if (!dialTouch || e.pointerId !== dialTouch.id) return;
    const g = dialGesture(e.clientX - dialTouch.x0, e.clientY - dialTouch.y0);
    if (g === "drag") { dialTouch = null; grabDial(e); } else if (g === "scroll") dialTouch = null;
  }, { signal });
  const dialEnd = (e, cancel = false) => {
    if (dialTouch && e.pointerId === dialTouch.id) { // a tap: no move past the slop
      dialTouch = null;
      if (!cancel) setDial(dialFromX(e.clientX));
    }
    if (e.pointerId === dialPtr) { dialPtr = null; delete dial.dataset.dragging; }
  };
  dial.addEventListener("pointerup", (e) => dialEnd(e), { signal });
  dial.addEventListener("pointercancel", (e) => dialEnd(e, true), { signal });
  dial.addEventListener("keydown", (e) => {
    const v = store.state.suppression;
    const next = { ArrowRight: v + 5, ArrowUp: v + 5, ArrowLeft: v - 5, ArrowDown: v - 5, PageUp: v + 20, PageDown: v - 20,
      Home: 0, End: 100 }[e.key];
    if (next == null || e.metaKey || e.ctrlKey || e.altKey) return;
    e.preventDefault();
    setDial(next);
  }, { signal });

  // ---- level halo: the real mic peak (or the replayed room), at most once per frame
  let pendingLevel = null;
  let raf = 0;
  function setLevel(db) {
    pendingLevel = db;
    // the sound scene (scene.js) draws the level when it is mounted; the halo is hidden then (motion.css, T-379)
    if (raf || document.documentElement.classList.contains("has-scene")) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const u = pendingLevel == null ? 0 : levelUnit(pendingLevel);
      halo.style.setProperty("--lvl", (reducedMotion.matches ? Math.round(u * 4) / 4 : u).toFixed(3));
    });
  }

  const offs = [
    store.on("change", () => render()),
    store.on("level", ({ db }) => { if (halo.dataset.live === "on") setLevel(db); }),
  ];
  signal?.addEventListener("abort", () => cleanup(), { once: true });
  function cleanup() {
    offs.splice(0).forEach((off) => off());
    cancelAnimationFrame(raf);
    clearTimeout(askedTimer);
    clearTimeout(warmTimer);
    delete document.body.dataset.ab;
    if (devBanner?.isConnected) document.body.append(devBanner);
  }
  watchKeyFocus(window, (off) => { keysOff = off; render(); }, { signal }); // renders once now
  return cleanup;
}
