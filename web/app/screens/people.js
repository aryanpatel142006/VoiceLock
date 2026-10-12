// People screen (M14 task 5, T-348): whose voice to hear. The saved voice profiles from telemetry
// `enrollment.profiles` as big cards (the first two are keys 1 and 2 on Listen and /judge), Release, and a
// labelled "Calibrate a new voice" kept apart from the stage path (the stage uses saved profiles; live
// calibration needs the person talking alone and is not a demonstrated stage control, PLAN.md).
//
// Every state shown comes from the engine: which profile plays (`enrollment.source == "profile"`), the
// calibration countdown (`calibrate_s - calibrate_elapsed_s`), and the profile versions from `/health`.

import { personLabel } from "../../console/judge-targets.js";
import { gateState } from "../../console/gate-state.js";
import { sharedSession } from "./setup.js";
import { showDevFacts } from "../platform.js";

export const title = "People";
export const CALIBRATE_S = 5;
export const MAX_PEOPLE = 4; // Person 1-4 (T-482)
const NAMES_KEY = "hearing.people.names";
const STAGE_KEYS = 2; // keys 1 and 2 (judge-targets.js savedPeople)

/** The cards: every saved profile, the first two with their stage key; `current` marks the one that plays. */
export function peopleCards(t, { released = false, versions = {} } = {}) {
  const names = t?.enrollment?.profiles || [];
  const current = !released && t?.enrollment?.source === "profile" ? t.enrollment.profile || null : null;
  return names.slice(0, MAX_PEOPLE).map((name, i) => ({
    name,
    label: personLabel(name),
    key: i < STAGE_KEYS ? String(i + 1) : null,
    current: name === current,
    version: versions?.[name] || null,
  }));
}

/** Inline renames, kept in this browser only (the engine's profile name is unchanged). */
export function loadNames(storage = globalThis.localStorage) {
  try { return JSON.parse(storage?.getItem(NAMES_KEY) || "{}") || {}; } catch { return {}; }
}
/** Empty "Person N" slots up to MAX_PEOPLE, so the list reads as four places to fill. */
export function emptySlots(n) { return Array.from({ length: Math.max(0, MAX_PEOPLE - n) }, (_, i) => `Person ${n + i + 1}`); }

/** A card's one line of status, in plain words (no "Key 1." line: the person keys are gone from the stage). */
export function cardMeta(c, { playing = false, streaming = false, held = "" } = {}) {
  const state = !c.current ? "Saved voice." : playing ? "Following now."
    : streaming && held === "raw" ? "Chosen. Raw plays the room for now."
      : streaming && held === "quiet" ? "Chosen. Quiet: captions only."
        : streaming ? "Chosen. Finding them." : "Chosen. Plays when you start.";
  return [state, c.version ? `Profile ${c.version}.` : null].filter(Boolean).join(" ");
}

/** What the Calibrate control shows, from telemetry `enrollment` (as web/console/calibrate.js). */
export function calibrateView(t, { started = false, connected = false } = {}) {
  const e = t?.enrollment;
  if (!connected) return { state: "off", button: `Learn a voice (${CALIBRATE_S} s)`, disabled: true, note: "Needs the engine." };
  if (!started) {
    return { state: "off", button: `Learn a voice (${CALIBRATE_S} s)`, disabled: true,
      note: "Start the mic on Setup first: calibration listens to the live room." };
  }
  if (e?.state === "calibrating") {
    const left = Math.max(0, (e.calibrate_s ?? CALIBRATE_S) - (e.calibrate_elapsed_s ?? 0));
    return { state: "calibrating", button: `Listening… ${left.toFixed(1)} s`, disabled: true, left,
      total: e.calibrate_s ?? CALIBRATE_S, note: "Keep talking alone until it finishes." };
  }
  if (e?.state === "failed") {
    return { state: "failed", button: `Try again (${CALIBRATE_S} s)`, disabled: false,
      note: "Too quiet: under 1 s of speech in the window. Move closer and try again." };
  }
  if (e?.source && e.source !== "profile") {
    return { state: "calibrated", button: `Learn it again (${CALIBRATE_S} s)`, disabled: false,
      note: "A calibrated voice is the target now. Pick a saved person to go back to the stage setup." };
  }
  return { state: "ready", button: `Learn a voice (${CALIBRATE_S} s)`, disabled: false,
    note: "Talk alone for about 5 seconds. The next 5 s become their voice." };
}

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

export function mount(root, { app, signal }) {
  const session = sharedSession(app);
  const store = session.store;
  let versions = {};
  let cardsSig = "";
  let names = loadNames();

  const status = h("p", { class: "pp-status", "aria-live": "polite" });
  const cards = h("div", { class: "pp-cards" });
  const release = h("button", { type: "button", class: "pp-release" },
    h("span", { class: "pp-release-label", text: "Stop following" }), h("span", { class: "pp-release-note", text: "follow no one" }));
  const target = h("p", { class: "pp-target", "aria-live": "polite" });
  const calBtn = h("button", { type: "button", class: "pp-cal-btn" });
  const calNote = h("p", { class: "pp-cal-note" });
  const calBar = h("span", { class: "pp-cal-bar", "aria-hidden": "true" }, h("i"));

  root.append(h("section", { class: "people", "aria-labelledby": "people-title" },
    h("header", { class: "pp-head" },
      h("h1", { id: "people-title", class: "pp-title", text: "Who do you want to hear?" }),
      status),
    cards,
    h("p", { class: "pp-honest", text: "Hears two voices at a time." }),
    h("div", { class: "pp-actions" }, release, target),
    h("aside", { class: "pp-cal", "aria-labelledby": "pp-cal-title" },
      h("div", { class: "pp-cal-copy" },
        h("h2", { id: "pp-cal-title", class: "pp-cal-title", text: "Learn a new voice" }),
        // T-458: stage-crew copy; a visitor's phone sees it only with ?dev=1
        showDevFacts() ? h("p", { class: "pp-cal-off", text: "Off the stage path. The demo uses the saved voices above." }) : null,
        calNote),
      h("div", { class: "pp-cal-do" }, calBtn, calBar)),
    h("footer", { class: "pp-foot" }, h("a", { href: "#listen", class: "pp-next" }, h("span", { text: "Go to Listen" })))));

  function paint() {
    const s = store.state;
    const t = s.telemetry;
    const connected = s.connection === "open";
    const list = peopleCards(t, { released: s.released, versions });
    root.querySelector(".people").dataset.released = String(s.released);

    if (!connected) status.textContent = s.connection === "reconnecting" ? "Engine lost: reconnecting…" : "Connecting to the engine…";
    else if (!t) status.textContent = "Waiting for the engine's first report…";
    else if (!list.length) status.textContent = ""; // the empty state below says it once (T-388)
    else status.textContent = `${list.length} saved ${list.length === 1 ? "voice" : "voices"} on this server. Tap one to follow it.`;

    // rebuild only when what the cards show changes (telemetry is ~10 Hz; keep focus and hover stable)
    // "Following now" only while the person is really heard (T-365 P0-3): streaming, isolated, Selective, not Quiet
    const streaming = connected && (s.started || s.replaying);
    const held = s.mode === "quiet" ? "quiet" : s.ab === "raw" ? "raw" : "";
    // T-463: and the engine's gate is not doubting the pick (UNSURE / OVERLAP-DUCKED is not "Following now")
    const playing = streaming && !held && s.pill?.state === "isolated" && [null, "solo"].includes(gateState(t));
    const sig = JSON.stringify([list, names, connected, Boolean(t), streaming, playing, held]);
    if (sig === cardsSig) { /* unchanged */ } else if (list.length) {
      cardsSig = sig;
      cards.replaceChildren(...list.map((c0, i) => {
        const c = { ...c0, label: names[c0.name] || c0.label };
        const card = h("button", { type: "button", class: "pp-card", "data-person": c.name, "aria-pressed": String(c.current),
          "data-playing": String(c.current && playing),
          disabled: !connected, "aria-describedby": c.key ? null : "pp-extra" },
        h("span", { class: "pp-card-key", "aria-hidden": "true", text: c.key || "·" }),
        h("span", { class: "pp-card-name", text: c.label }),
        h("span", { class: "pp-card-meta", text: cardMeta(c, { playing, streaming, held }) }));
        card.addEventListener("click", () => choose(i, c.name), { signal });
        const ren = h("button", { type: "button", class: "pp-rename", "aria-label": `Rename ${c.label}`, text: "Rename" });
        ren.addEventListener("click", () => {
          const v = globalThis.prompt?.("Name this person", c.label);
          if (v == null) return;
          names = { ...names, [c.name]: v.trim().slice(0, 24) || undefined };
          try { localStorage.setItem(NAMES_KEY, JSON.stringify(names)); } catch { /* private mode */ }
          paint();
        }, { signal });
        return h("div", { class: "pp-slot" }, card, ren);
      }), ...emptySlots(list.length).map((label) => h("div", { class: "pp-slot pp-slot-empty" },
        h("p", { class: "pp-empty-slot", text: `${label}: empty. Talk alone for about 5 seconds (Learn a voice, below).` }))));
      if (list.length > STAGE_KEYS) {
        cards.append(h("p", { id: "pp-extra", class: "pp-extra",
          text: "Keys 1 and 2 on Listen and /judge reach the first two; tap a card for the others." }));
      }
    } else {
      cardsSig = sig;
      cards.replaceChildren(h("div", { class: "pp-empty" },
        h("p", { class: "pp-empty-title", text: t ? "No one saved yet" : "—" }),
        h("p", { class: "pp-empty-note" }, t
          ? "No voice is saved on this laptop. Learn one below, or record the team's voices on the laptop and restart it (scripts/enroll_profiles.sh)."
          : "The saved voices appear once the engine reports them.")));
    }

    release.disabled = !connected;
    release.setAttribute("aria-pressed", String(s.released));
    target.textContent = s.person.line;

    const cal = calibrateView(t, { started: s.started && !s.replaying, connected });
    root.querySelector(".pp-cal").dataset.state = cal.state;
    calBtn.textContent = cal.button;
    calBtn.disabled = cal.disabled;
    calNote.textContent = cal.note;
    calBar.querySelector("i").style.transform = `scaleX(${cal.state === "calibrating" ? (1 - cal.left / cal.total).toFixed(3) : 0})`;
  }

  function choose(i, name) {
    if (i < STAGE_KEYS) session.choosePerson(i);
    else { // beyond keys 1/2: the same control the keys send, for this profile
      store.set({ released: false });
      session.send({ enroll: { action: "profile", name } });
      store.note(`target: saved profile ${name}`);
      session.tick();
    }
  }

  release.addEventListener("click", () => session.release(), { signal });
  calBtn.addEventListener("click", () => session.calibrate(CALIBRATE_S), { signal });

  const off = store.on("change", ({ keys }) => {
    if (keys.some((k) => ["telemetry", "connection", "released", "person", "started", "replaying", "pill"].includes(k))) paint();
  });
  fetch("/health", { cache: "no-store", signal })
    .then((r) => (r.ok ? r.json() : null))
    .then((hj) => { versions = hj?.profile_versions || {}; paint(); })
    .catch(() => undefined);
  paint();
  return off;
}
