// Comfort Mode (PR #402): what the SOS calm frame offers after the press.
//
// Press SOS -> the app's own audio is cut and the engine goes to Quiet (captions only, T-149) -> Play My Sounds opens
// and starts the remembered sound (else gentle rain) inside the press, fading in (Aryan, Oct 11: this replaces the old
// "nothing plays before a choice" rule). Back to choices shows the three large choices (Stay Quiet / Play My Sounds /
// Talk to Me) with the sound still playing; Stay Quiet fades it out; Talk to Me ducks it while the voice speaks.
// Nothing is asked during the overload, Talk to Me never speaks before it is chosen, and a Stop audio control is
// always on screen.
//
// Talk to Me is a calm voice companion: POST /comfort/chat (hearing/server/comfort.py). On open it asks for a greeting
// (no text, no history) and speaks it. Each turn is the person's voice (MediaRecorder on the browser mic, sent as
// base64 and transcribed on the server) or typed text, with the whole transcript as `history`. The reply comes back
// as text plus WAV audio (ElevenLabs) and is played at once; "Stop speaking" / Stop audio / Esc stop it. The source
// is labelled honestly: "Personal · Gemini" only when the server said source:"gemini", else "Default".
// `suggest_remember` is only an offer ("Remember this?"): nothing is saved until Remember posts /comfort/feedback
// confirm:true, after which `comfort:profile-updated` fires on window. "Call my person" is a real tel: link on a
// phone; elsewhere it is a labelled demo overlay ("Demo · no call is placed") that never claims a call connected.
// Asking for it ("Can you call Jordan?") does the same: the reply carries action:"call" and the overlay opens while
// "Okay. Calling Jordan now." plays; on a phone the overlay holds a big tel: button (a tel: link needs a tap there).
//
// Pieces built on their own branches are loaded if present and skipped honestly if not: ./sounds.js (mountSounds,
// stopAllSounds), ./screens/learned.js (mountLearned), ./quiet.js (mountQuiet, enterQuiet, exitQuiet),
// ./controls-legend.js (setControlContext). Without sounds.js the built-in calm loops
// (sos.js: brown/pink/white/music) are the sounds, so at least one real sound always plays.
//
// Pure parts (loadComfort/saveComfort, cmInit/cmReduce, transcriptFor, sourceLabel, voiceLabel, latencyLabel,
// fetchChat, postFeedback, callPlan, telHref, pickMime) run under node --test; mountComfort is
// the browser half, called by sos.js. Support for sensory overload, not medical advice. Quiet here controls only this
// app's audio.

export const COMFORT_KEY = "hearing.comfort";
export const DEFAULT_PROFILE = "alex-demo";
export const DEMO_PROFILES = ["alex-demo", "sam-demo"];
export const CHAT_TIMEOUT_MS = 25000; // speech-to-text (<= 10 s) + Gemini (<= 6 s) + ElevenLabs voice
export const MAX_SHOWN_TURNS = 4;
export const MAX_RECORD_MS = 30000;
export const HOLD_MS = 350; // a press shorter than this is a tap (start / stop), longer is hold-to-talk
export const PHONE_WIDTH = 700; // narrow + coarse pointer = a phone: "Call my person" is a real tel: link there
// 20 ms of silence (8 kHz mono WAV). iOS Safari only lets an <audio> element play after it was started inside a tap, so
// Talk to Me plays this on one element during the tap and reuses that element for every reply (see unlockVoice).
export const SILENT_WAV = "data:audio/wav;base64,UklGRmQBAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";
export const MIME_PREFS = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
export const CHOICES = [
  { id: "quiet", label: "Stay Quiet", hint: "Nothing plays. Their words appear here." },
  { id: "sounds", label: "Play My Sounds", hint: "Only the sound you pick." },
  { id: "talk", label: "Talk to Me", hint: "A calm voice that listens." },
];
export const BUILTIN_SOUNDS = [
  { id: "brown", label: "Brown noise" }, { id: "pink", label: "Pink noise" }, { id: "white", label: "White noise" },
  { id: "music", label: "Chill music" },
];

// ---------------------------------------------------------------- pure helpers

/** The client setting: on unless {on:false} was saved; `profile` is the comfort profile id in use. */
export function loadComfort(storage) {
  let raw = null;
  try { raw = JSON.parse(storage?.getItem(COMFORT_KEY) || "null"); } catch { raw = null; }
  const obj = raw && typeof raw === "object" ? raw : {};
  const profile = typeof obj.profile === "string" && /^[a-z0-9-]{1,48}$/.test(obj.profile) ? obj.profile : DEFAULT_PROFILE;
  return { on: obj.on !== false, profile };
}

export function saveComfort(storage, patch) {
  const next = { ...loadComfort(storage), ...patch };
  try { storage?.setItem(COMFORT_KEY, JSON.stringify(next)); } catch { /* private mode: this visit only */ }
  return next;
}

/**
 * The Comfort state machine. Phases: hidden -> sounds (the press starts the sound) -> menu -> quiet | sounds | talk
 * -> menu ... -> hidden.
 * Inside talk: `turns` [{role:"user"|"assistant", text, pending?, source?, timing?}] is the transcript, `busy` while a
 * reply is awaited (the greeting on entry), `error` when the last request failed, `reply` the last server body,
 * `remember` the server's offer {strategy, helpful, label, ask} with `rememberState` idle | offer | saved | skipped,
 * `offerCall` when the reply suggested the trusted person, `rec` idle | recording, `call` while the call overlay
 * shows (a tap on "Call my person", or a reply with action "call": the person asked the voice to call). Events: {type:"open", on} · {type:"choose", choice} · {type:"send", text?, audio?} · {type:"reply", reply,
 * timing?} · {type:"remember"} · {type:"notnow"} · {type:"rec", on} · {type:"call", on} · {type:"back"} · {type:"close"}.
 */
export function cmInit() {
  return { phase: "hidden", turns: [], busy: false, error: null, reply: null, remember: null, rememberState: "idle",
    offerCall: false, rec: "idle", call: false };
}

function cleanRemember(r) {
  const o = r?.suggest_remember;
  if (!o || typeof o !== "object" || typeof o.strategy !== "string" || typeof o.helpful !== "boolean") return null;
  const label = typeof o.label === "string" && o.label ? o.label : o.strategy;
  return { strategy: o.strategy, helpful: o.helpful, label,
    ask: typeof o.ask === "string" && o.ask ? o.ask : `Remember that ${label.toLowerCase()} ${o.helpful ? "helps" : "doesn't help"}?` };
}

export function cmReduce(s, ev) {
  switch (ev.type) {
    case "open":
      return ev.on === false ? cmInit() : { ...cmInit(), phase: "sounds" };
    case "choose":
      if (s.phase === "hidden" || !CHOICES.some((c) => c.id === ev.choice)) return s;
      return { ...cmInit(), phase: ev.choice }; // talk: mountComfort sends the greeting request at once
    case "send": {
      if (s.phase !== "talk" || s.busy) return s;
      const text = typeof ev.text === "string" ? ev.text.trim() : "";
      const turns = text ? [...s.turns, { role: "user", text }] : ev.audio ? [...s.turns, { role: "user", text: "", pending: true }] : s.turns;
      return { ...s, turns, busy: true, error: null, remember: null, rememberState: "idle", offerCall: false, rec: "idle" };
    }
    case "reply": {
      if (s.phase !== "talk" || !s.busy) return s;
      const r = ev.reply;
      let turns = s.turns;
      const last = turns[turns.length - 1];
      if (last?.pending) { // the voice turn: the server's transcript, or nothing if it heard nothing
        const heard = typeof r?.user_text === "string" ? r.user_text.trim() : "";
        turns = heard ? [...turns.slice(0, -1), { role: "user", text: heard }] : turns.slice(0, -1);
      }
      const good = r && typeof r === "object" && typeof r.reply === "string" && r.reply.trim();
      if (!good) return { ...s, turns, busy: false, error: "server", reply: null };
      const source = r.source === "gemini" ? "gemini" : "default";
      turns = [...turns, { role: "assistant", text: r.reply.trim(), source, timing: ev.timing || null }];
      const remember = cleanRemember(r);
      return { ...s, turns, busy: false, error: null, reply: r, remember, rememberState: remember ? "offer" : "idle",
        offerCall: r.offer_call === true, call: r.action === "call" ? true : s.call };
    }
    case "remember":
      return s.rememberState === "offer" ? { ...s, rememberState: "saved" } : s;
    case "notnow":
      return s.rememberState === "offer" ? { ...s, rememberState: "skipped" } : s;
    case "rec":
      if (s.phase !== "talk" || (ev.on && s.busy)) return s;
      return { ...s, rec: ev.on ? "recording" : "idle" };
    case "call":
      return s.phase === "talk" ? { ...s, call: Boolean(ev.on) } : s;
    case "back":
      return s.phase === "hidden" ? s : { ...cmInit(), phase: "menu" };
    case "close":
      return cmInit();
    default:
      return s;
  }
}

/** The last `n` turns, plus a {thinking:true} row while a reply is awaited. */
export function transcriptFor(s, n = MAX_SHOWN_TURNS) {
  const rows = (s.turns || []).slice(-n);
  return s.busy ? [...rows, { role: "assistant", thinking: true }] : rows;
}

/** Where the reply came from. Never "Gemini" unless the server said source:"gemini". */
export function sourceLabel(reply) {
  if (reply?.source === "gemini") return { title: "Personal · Gemini", note: "", source: "gemini" };
  const why = { no_key: "No Gemini key on this server: a plain default reply.", timeout: "Gemini took too long: a plain default reply.",
    error: "Gemini could not be reached: a plain default reply.", invalid: "Gemini's answer was not usable: a plain default reply.",
    stt_error: "Speech-to-text was unavailable.", no_speech: "Nothing was heard.",
    call_request: "You asked to call your person." }[reply?.reason] || "A plain default reply.";
  return { title: "Default", note: why, source: "default" };
}

/** The honest voice line: ElevenLabs only when audio really came back. */
export function voiceLabel(reply) {
  if (reply?.engine === "elevenlabs" && reply.audio_base64) return { spoken: true, text: "Voice: ElevenLabs" };
  return { spoken: false, text: "Voice unavailable, showing text" };
}

/** "Reply in 1.8 s" from the measured round trip (ms), or "" when nothing was measured. */
export function latencyLabel(timing) {
  const ms = Number(timing?.roundTripMs);
  if (!Number.isFinite(ms) || ms <= 0) return "";
  return `Reply in ${(Math.round(ms / 100) / 10).toFixed(1)} s`;
}

/** A fetch with a hard deadline; resolves {reply, reason} and never rejects. */
async function fetchJson(url, body, { fetchImpl = globalThis.fetch, timeoutMs = CHAT_TIMEOUT_MS } = {}) {
  if (typeof fetchImpl !== "function") return { reply: null, reason: "offline" };
  const ctl = typeof AbortController === "function" ? new AbortController() : null;
  const timer = setTimeout(() => ctl?.abort(), timeoutMs);
  try {
    const r = await fetchImpl(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: ctl?.signal });
    if (!r.ok) return { reply: null, reason: r.status === 404 ? "disabled" : "error" };
    return { reply: await r.json(), reason: null };
  } catch (e) {
    return { reply: null, reason: e?.name === "AbortError" ? "timeout" : "offline" };
  } finally {
    clearTimeout(timer);
  }
}

/**
 * POST /comfort/chat: {profile_id, history, speak} plus `text` or `audio_b64` (+ `audio_mime`). No text and an empty
 * history is the greeting request. `history` is sent as [{role, text}] only (no timings, no pending rows).
 */
export function fetchChat(profileId, { history = [], text, audioB64, audioMime, speak = true, base = "", ...opts } = {}) {
  const turns = (Array.isArray(history) ? history : [])
    .filter((t) => t && (t.role === "user" || t.role === "assistant") && typeof t.text === "string" && t.text.trim() && !t.pending)
    .map((t) => ({ role: t.role, text: t.text }));
  const body = { profile_id: profileId, history: turns, speak: Boolean(speak) };
  if (typeof text === "string" && text.trim()) body.text = text.trim();
  else if (typeof audioB64 === "string" && audioB64) {
    body.audio_b64 = audioB64;
    if (typeof audioMime === "string" && audioMime) body.audio_mime = audioMime.split(";")[0];
  }
  return fetchJson(`${base}/comfort/chat`, body, opts);
}

/** POST /comfort/feedback. `confirm` false never saves (the server returns saved:false). */
export function postFeedback({ profileId, strategy, suggestion, helpful, confirm }, { base = "", ...opts } = {}) {
  const id = strategy || suggestion;
  return fetchJson(`${base}/comfort/feedback`, { profile_id: profileId, strategy: id, suggestion: id, helpful: Boolean(helpful), confirm: Boolean(confirm) }, opts);
}

/** "555-0100" -> "tel:5550100"; a leading + is kept. */
export function telHref(phone) {
  const digits = String(phone || "").replace(/[^\d+]/g, "").replace(/(?!^)\+/g, "");
  return digits ? `tel:${digits}` : "";
}

/**
 * How "Call my person" behaves here: null without a contact; a real tel: link on a phone (coarse pointer and a narrow
 * screen); otherwise the labelled demo overlay (a laptop cannot place a call, and the demo must never pretend it did).
 */
export function callPlan(contact, { coarse = false, width = 1024 } = {}) {
  if (!contact || typeof contact !== "object" || typeof contact.name !== "string" || !contact.name.trim()) return null;
  const name = contact.name.trim();
  const first = name.split(/[\s(]/)[0] || name;
  const href = telHref(contact.phone);
  if (coarse && width < PHONE_WIDTH && href) return { kind: "tel", href, name, first };
  return { kind: "overlay", href: "", name, first };
}

/**
 * What the call overlay shows for a callPlan. Laptop: "Calling Jordan…" with the demo label (no call is placed).
 * Phone: the overlay opens when the voice was asked to call; a page cannot dial a tel: link without a tap, so it shows
 * one big "Call Jordan" tel: button and never pretends the call started. Null without a plan.
 */
export function callScreenFor(plan) {
  if (!plan) return null;
  if (plan.kind === "tel") return { calling: `Tap to call ${plan.first}.`, demo: false, href: plan.href, telText: `Call ${plan.first}` };
  return { calling: `Calling ${plan.first}…`, demo: true, href: "", telText: "" };
}

/** The first recorder MIME type the browser supports (`isSupported(type)`), or "" for the browser's default. */
export function pickMime(isSupported) {
  if (typeof isSupported !== "function") return "";
  return MIME_PREFS.find((t) => { try { return isSupported(t) === true; } catch { return false; } }) || "";
}

// ---------------------------------------------------------------- browser half

async function optional(spec) {
  try { return await import(spec); } catch { return null; }
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result || "").split(",", 2)[1] || "");
    fr.onerror = () => reject(fr.error || new Error("read failed"));
    fr.readAsDataURL(blob);
  });
}

/**
 * Mount Comfort into the SOS calm frame. `host` (sos.js) gives: doc, win, body (.sos-calm-body), before (insert
 * point), setSound(id) (brown|pink|white|music|silence), onShow(fn(on)), focusHold(), setQuiet(on) (engine Quiet /
 * Focus), modeNow() ("quiet"|"focus_hq"|null), captions() (the store's caption lines), onCaptions(fn).
 */
/**
 * Stay Quiet's captions: ask the engine for them under SOS (`sos_captions`), and make sure the page is listening.
 * Oct 11 (session 20261011T141338Z-5): SOS was pressed before Start listening, the page streamed no audio at all, so
 * the card waited on "Waiting for them to speak." for minutes. Called from the Stay Quiet tap, so the AudioContext
 * that start() creates is inside a user gesture. Returns {sent, started}: started = this call started the mic.
 */
export function stayQuietCaptions(session, on) {
  if (!session) return { sent: false, started: false };
  let sent = false;
  try { session.send?.({ sos_captions: on }); sent = true; } catch { /* no live session */ }
  const st = session.store?.state;
  if (!on || !st || st.started || st.starting || typeof session.start !== "function") return { sent, started: false };
  try { Promise.resolve(session.start()).catch(() => {}); } catch { return { sent, started: false }; }
  return { sent, started: true };
}

export function mountComfort(host) {
  const { doc, win } = host;
  const css = (name) => {
    if (doc.querySelector(`link[data-cf-css="${name}"]`)) return;
    const link = doc.createElement("link");
    link.rel = "stylesheet";
    link.setAttribute("data-cf-css", name);
    link.href = new URL(`./${name}`, import.meta.url).href;
    doc.head.append(link);
  };
  css("comfort.css");
  const storage = (() => { try { return win.localStorage; } catch { return null; } })();

  const el = (tag, attrs = {}, ...kids) => {
    const n = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") n.textContent = v;
      else n.setAttribute(k, v);
    }
    n.append(...kids.filter(Boolean));
    return n;
  };
  const btn = (cls, text, attrs = {}) => el("button", { type: "button", class: cls, text, ...attrs });

  // always visible while SOS is on: stop + captions; the paired band is a single button (sos.js)
  const stopBtn = btn("cf-stop", "Stop audio", { "aria-label": "Stop audio: stop any sound or voice this app plays" });
  const capList = el("ol", { class: "cf-caps", "aria-label": "Captions" });
  const capHead = el("p", { class: "cf-caps-head", text: "Captions" });
  const caps = el("section", { class: "cf-caps-wrap" }, capHead, capList);
  const quietSlot = el("div", { class: "cf-quiet-slot" });

  // menu
  const choiceBtns = CHOICES.map((c) => el("button", { type: "button", class: "cf-choice", "data-choice": c.id },
    el("span", { class: "cf-choice-word", text: c.label }), el("span", { class: "cf-choice-hint", text: c.hint })));
  const menu = el("section", { class: "cf-menu", "aria-labelledby": "cf-q" },
    el("h2", { class: "cf-q", id: "cf-q", text: "What would help?" }),
    el("div", { class: "cf-choices" }, ...choiceBtns),
    el("p", { class: "cf-fine", text: "Nothing is required. You can just stay here." }));

  // quiet
  // ONE calm card: the label, one sentence, the captions of the person you're talking to (quiet.js, embedded: no
  // second header, no Back to Focus: ↓ / the hold already go back)
  const quietCard = el("section", { class: "cf-card cf-quiet", "aria-label": "Stay Quiet" },
    el("h2", { class: "cf-card-label", text: "Stay Quiet" }),
    el("p", { class: "cf-big", text: "Nothing plays. Their words appear here." }),
    quietSlot, caps);

  // sounds
  const soundsSlot = el("div", { class: "cf-sounds-slot" });
  // no name of its own: the player inside is the "Play My Sounds" region (sounds.js snd-title), so it is announced once
  const soundsCard = el("section", { class: "cf-card cf-sounds" },
    el("h2", { class: "cf-card-label", text: "Play My Sounds" }), soundsSlot);

  // talk: the chat panel
  const srcTag = el("span", { class: "cf-tag cf-src-tag", hidden: "" });
  const voiceTag = el("span", { class: "cf-tag cf-voice", hidden: "" });
  const latTag = el("span", { class: "cf-tag cf-lat", hidden: "" });
  const srcNote = el("p", { class: "cf-src-note", hidden: "" });
  const talkHead = el("header", { class: "cf-talk-head" },
    el("h2", { class: "cf-talk-title", id: "cf-talk-h", text: "Talk to Me" }),
    el("p", { class: "cf-talk-sub", text: "A calm voice that listens" }),
    el("p", { class: "cf-tags" }, srcTag, voiceTag, latTag), srcNote);
  const chat = el("ol", { class: "cf-chat", "aria-label": "Conversation", "aria-live": "polite", "aria-relevant": "additions text" });
  const talkStatus = el("p", { class: "cf-talk-status", role: "status", hidden: "" });
  // remember (the server's offer; saved only on Remember)
  const confirmQ = el("p", { class: "cf-confirm-q" });
  const rememberBtn = btn("cf-text-btn cf-remember", "Remember", { "aria-keyshortcuts": "ArrowRight PageDown" });
  const notNowBtn = btn("cf-text-btn cf-notnow", "Not now", { "aria-keyshortcuts": "ArrowLeft PageUp" });
  const savedLine = el("p", { class: "cf-saved", role: "status", hidden: "" });
  const rememberBox = el("div", { class: "cf-confirm", hidden: "" },
    el("p", { class: "cf-card-label", text: "Remember this?" }), confirmQ,
    el("div", { class: "cf-rate-row" }, rememberBtn, notNowBtn), savedLine);
  // input: hold to talk (tap also starts / stops), or type
  const micDot = el("span", { class: "cf-mic-dot", "aria-hidden": "true" });
  const micWord = el("span", { class: "cf-mic-word", text: "Hold to talk" });
  const micHint = el("span", { class: "cf-mic-hint", text: "or tap to start and stop" });
  const micBtn = el("button", { type: "button", class: "cf-mic", "aria-pressed": "false", "aria-describedby": "cf-mic-hint" },
    micDot, micWord, micHint);
  micHint.id = "cf-mic-hint";
  const typeIn = el("input", { type: "text", class: "cf-type-in", autocomplete: "off", autocapitalize: "sentences",
    enterkeyhint: "send", placeholder: "Or type here", "aria-label": "Type a message instead" });
  const sendBtn = el("button", { type: "submit", class: "cf-text-btn cf-send", text: "Send" });
  const typeForm = el("form", { class: "cf-type", novalidate: "" }, typeIn, sendBtn);
  const stopSpeakBtn = btn("cf-text-btn cf-stop-speak", "Stop speaking", { "aria-keyshortcuts": "Escape" });
  const callBtn = btn("cf-text-btn cf-call", "Call my person");
  const callLink = el("a", { class: "cf-text-btn cf-call", href: "#", hidden: "", text: "Call my person" });
  const callOffer = el("p", { class: "cf-call-offer", hidden: "" });
  const talkRow = el("div", { class: "cf-talk-row" }, stopSpeakBtn, callBtn, callLink);
  const talkCard = el("section", { class: "cf-card cf-talk", "aria-labelledby": "cf-talk-h" },
    talkHead, chat, talkStatus, rememberBox, callOffer, micBtn, typeForm, talkRow);

  // the demo call overlay (laptop): never says "connected"
  const callName = el("p", { class: "cf-call-name", id: "cf-call-name" });
  const calling = el("p", { class: "cf-calling", role: "status" });
  const endBtn = btn("cf-call-end", "End");
  const callDemo = el("p", { class: "cf-call-demo", text: "Demo · no call is placed" });
  const callTel = el("a", { class: "cf-call-tel", href: "#", hidden: "" }); // phone: the one tap that dials
  const callScreen = el("div", { class: "cf-callscreen", role: "dialog", "aria-modal": "true", "aria-labelledby": "cf-call-name", hidden: "" },
    el("div", { class: "cf-callscreen-body" },
      el("p", { class: "cf-card-label", text: "Call my person" }), callName, calling,
      callDemo, callTel, endBtn));

  // learned (entry point from Comfort)
  const learnedToggle = btn("cf-text-btn cf-learned-toggle", "What I've learned about you", { "aria-expanded": "false" });
  const learnedSlot = el("div", { class: "cf-learned-slot", hidden: "" });

  const profileLine = el("p", { class: "cf-profile" });
  const profileBtn = btn("cf-text-btn cf-profile-btn", "Switch profile");
  const backBtn = btn("cf-text-btn cf-back", "Back to choices");
  const profileRow = el("div", { class: "cf-profile-row" }, profileLine, profileBtn);
  const foot = el("p", { class: "cf-foot", text: "Support for sensory overload, not medical advice." });
  const status = el("p", { class: "cf-status", role: "status", "aria-live": "polite" });
  const panel = el("div", { class: "sos-comfort", hidden: "" },
    el("div", { class: "cf-top" }, stopBtn),
    menu, quietCard, soundsCard, talkCard,
    el("div", { class: "cf-nav" }, backBtn, learnedToggle),
    el("div", { class: "cf-learned" }, learnedSlot),
    profileRow,
    foot, status);
  host.body.insertBefore(panel, host.before);
  // the overlay sits beside the calm body, not in it, so the body (and the hold) can go inert while it is open
  (host.body.parentNode || host.body).append(callScreen);
  function showCall(open) {
    callScreen.hidden = !open;
    for (const n of [host.body, host.body.querySelector?.(".sos-calm-hold")]) if (n) n.inert = open;
  }

  let s = cmInit();
  let seq = 0;
  let setting = loadComfort(storage);
  let audio = null; // the ElevenLabs clip playing (the voiceEl while a reply plays, else null)
  let voiceEl = null; // the one <audio> element every reply uses, unlocked by a tap (iOS autoplay rules)
  let sounds = null; // {stop()} from mountSounds
  let mods = { sounds: null, learned: null, quiet: null, legend: null };
  let learnedUi = null; // mountLearned's handle (unmounted before a remount, so no stale listener renders into the slot)
  let profiles = [];
  let quietUi = null; // mountQuiet's handle
  let micByQuiet = false; // Stay Quiet started the page's microphone (it was not listening): stopped when the card closes
  let capSub = null;
  // recording
  let stream = null; // the mic, kept while Talk to Me is open
  let recorder = null;
  let chunks = [];
  let recTimer = 0;
  let pressT0 = 0;
  let stopOnUp = false;
  let pressing = false;
  let lastFocus = null;

  const api = {
    get state() { return s; },
    get on() { return loadComfort(storage).on; },
    get profile() { return loadComfort(storage).profile; },
    get speaking() { return Boolean(audio); },
    get soundsPlaying() { return Boolean(sounds?.controller?.state?.playing); },
    enable(on = true) { setting = saveComfort(storage, { on }); return setting.on; },
    setProfile(id) {
      setting = saveComfort(storage, { profile: id });
      renderProfile();
      // nothing from the old profile stays on screen: its conversation (and voice), its sounds panel, its learned view
      if (s.phase === "talk") dispatch({ type: "back" });
      if (s.phase === "sounds") { stopSounds(); renderSounds(); }
      else stopSounds(); // the old profile's sound, still playing under the choices
      if (learnedUi) { try { learnedUi.unmount(); } catch { /* gone */ } learnedUi = null; }
      if (!learnedSlot.hidden) renderLearned(true);
      return setting.profile;
    },
    choose: (c) => { dispatch({ type: "choose", choice: c }); if (c === "talk") sendTurn({}); },
    send: (text) => sendTurn({ text }),
    stop: stopAll,
    stopSpeaking: stopSpeech,
    endCall,
    modules: () => Object.fromEntries(Object.entries(mods).map(([k, v]) => [k, Boolean(v)])),
  };
  win.hearingComfort = api;

  // the optional pieces, loaded once; each is simply absent when its branch is not merged
  const ready = (async () => {
    const [sm, lm, qm, gm] = await Promise.all([
      optional("./sounds.js"), optional("./screens/learned.js"), optional("./quiet.js"), optional("./controls-legend.js"),
    ]);
    mods = {
      sounds: typeof sm?.mountSounds === "function" ? sm : null,
      learned: typeof lm?.mountLearned === "function" ? lm : null,
      quiet: typeof qm?.mountQuiet === "function" ? qm : null,
      legend: typeof gm?.setControlContext === "function" ? gm : null,
    };
    caps.hidden = Boolean(mods.quiet);
    if (mods.learned) css("screens/learned.css");
    render();
    // Perf: the Quiet card is built now, while the page is idle, not on the first press. It needs the session (its
    // store), which the first screen that uses one creates before app.js emits "screen".
    let offScreen = null;
    const mountIdle = () => {
      if (!win.hearingApp?.session) return false;
      mountQuietUi();
      offScreen?.();
      return true;
    };
    if (!mountIdle()) offScreen = win.hearingApp?.on?.("screen", mountIdle) || null;
  })();

  /** Quiet's caption surface (quiet.js), mounted once a session exists so it can follow the store. */
  function mountQuietUi() {
    if (!mods.quiet || quietUi) return;
    const session = win.hearingApp?.session || null;
    try {
      quietUi = mods.quiet.mountQuiet(quietSlot, { session, autoEnter: false, embedded: true, paused: () => s.phase === "hidden" });
    } catch { quietUi = null; }
  }

  /** The engine's Quiet / Focus: quiet.js's own control when present, else the host's (both end in session.setMode). */
  function setQuiet(on) {
    if (mods.quiet) {
      try { (on ? mods.quiet.enterQuiet : mods.quiet.exitQuiet)(win.hearingApp?.session || null); return; } catch { /* fall through */ }
    }
    host.setQuiet?.(on);
  }

  /** The engine sends no captions under SOS (T-462) unless Stay Quiet asks for them: on while the card shows. A mic
   * that Stay Quiet had to start goes off again when the card closes (the page was not listening before). */
  function sosCaptions(on) {
    const session = win.hearingApp?.session || null;
    if (on) micByQuiet = stayQuietCaptions(session, true).started;
    else {
      stayQuietCaptions(session, false);
      if (micByQuiet) { micByQuiet = false; try { session?.stop?.(); } catch { /* already stopped */ } }
    }
  }

  function profileName(id) {
    const p = profiles.find((x) => x.id === id);
    const name = p?.preferred_name || (id === "sam-demo" ? "Sam" : id === "alex-demo" ? "Alex" : id);
    const demo = p ? Boolean(p.fictional || p.demo) : DEMO_PROFILES.includes(id);
    return demo ? `${name} · Demo profile` : name;
  }

  function contactNow() {
    return profiles.find((x) => x.id === setting.profile)?.support_contact || null;
  }

  async function loadProfiles() {
    try {
      const r = await win.fetch("/comfort/profiles");
      if (r.ok) profiles = await r.json();
    } catch { /* offline: the names fall back to the ids */ }
    renderProfile();
    if (s.phase === "talk") renderTalk();
  }

  function renderProfile() {
    profileLine.textContent = `Profile: ${profileName(setting.profile)}`;
  }

  // The only keys left are the Remember offer's (arrows / PageUp / PageDown). The one paired band key is handled by
  // sos.js first; there is no keyboard / remote toolbar and no global arrow binding for Quiet, Focus or Stop.
  function renderLegend() {
    if (!mods.legend) return;
    const offer = s.phase === "talk" && s.rememberState === "offer";
    try {
      if (!offer) { mods.legend.setControlContext("focus", {}); return; }
      mods.legend.setControlContext("remember", {
        remember: () => { if (s.rememberState === "offer") remember(); }, notNow: () => dispatch({ type: "notnow" }),
      });
    } catch { /* the keys module is optional */ }
  }

  function renderCaptions() {
    if (mods.quiet) return;
    const lines = (host.captions?.() || []).slice(-3);
    capList.replaceChildren(...lines.map((c) => el("li", { class: "cf-cap", "data-final": String(Boolean(c.final)), text: c.text })));
    capHead.textContent = lines.length ? "Captions" : "Captions show here when someone near you speaks.";
  }

  function render() {
    panel.hidden = s.phase === "hidden";
    doc.body.dataset.comfort = s.phase === "hidden" ? "" : s.phase;
    menu.hidden = s.phase !== "menu";
    quietCard.hidden = s.phase !== "quiet";
    soundsCard.hidden = s.phase !== "sounds";
    talkCard.hidden = s.phase !== "talk";
    backBtn.hidden = s.phase === "menu" || s.phase === "hidden";
    profileRow.hidden = s.phase === "talk"; // Talk to Me stays clean: the conversation, its controls, Back
    foot.hidden = s.phase === "talk";
    // Quiet's captions panel shows on Stay Quiet only; on the menu, Sounds and Talk it was a second, competing panel
    quietSlot.hidden = s.phase !== "quiet";
    caps.hidden = Boolean(mods.quiet) || s.phase !== "quiet";
    if (s.phase === "talk") renderTalk();
    else showCall(false);
    // the player lives until the frame closes: Back to choices (and Talk to Me) keep the sound playing
    if (s.phase === "sounds") renderSounds();
    else if (s.phase === "hidden") stopSounds();
    renderLegend();
    renderCaptions();
  }

  function renderSounds() {
    if (soundsSlot.childElementCount) return;
    if (mods.sounds) {
      try { sounds = mods.sounds.mountSounds(soundsSlot, { profileId: setting.profile, onState: soundsState }) || null; return; } catch { /* fall through */ }
    }
    // built-in: the calm loops sos.js already synthesises; nothing plays until a tap
    const row = el("div", { class: "cf-sound-row", role: "group", "aria-label": "Sounds" });
    const note = el("p", { class: "cf-fine", text: "Tap one to play it. Only that sound plays." });
    for (const snd of BUILTIN_SOUNDS) {
      const b = btn("cf-sound", snd.label, { "data-sound": snd.id, "aria-pressed": "false" });
      b.addEventListener("click", () => {
        host.setSound?.(snd.id);
        for (const o of row.querySelectorAll(".cf-sound")) o.setAttribute("aria-pressed", String(o === b));
        note.textContent = `${snd.label} is playing.`;
        status.textContent = `Playing ${snd.label} · tap Stay Quiet for silence`;
      });
      row.append(b);
    }
    soundsSlot.append(row, note);
  }

  /** The SOS press: start Play My Sounds at once, inside the press gesture (so the browser lets it play). */
  function autoplaySounds() {
    if (sounds?.autoplay) { try { sounds.autoplay(); } catch { /* no audio: the player still shows */ } return; }
    // built-in loops (no sounds.js): sos.js only starts a loop once its frame is on, which is right after this hook
    const first = soundsSlot.querySelector?.(".cf-sound");
    if (first) (win.queueMicrotask || ((f) => Promise.resolve().then(f)))(() => { if (s.phase !== "hidden") first.click(); });
  }

  /** The player's state: the status line says what plays and how to silence it (said once per sound, not per change). */
  let soundsSaid = "";
  function soundsState(st) {
    const now = st?.playing && st.current ? st.current : "";
    if (now === soundsSaid) return;
    soundsSaid = now;
    if (now) status.textContent = `Playing ${mods.sounds?.soundName?.(now) || now} · tap Stay Quiet for silence`;
  }

  /** Stay Quiet: the sound fades out (short), the player stays mounted with its selection. */
  function quietSounds() {
    if (sounds?.stop) { try { sounds.stop(); } catch { /* already stopped */ } }
    if (soundsSlot.querySelector?.(".cf-sound[aria-pressed='true']")) {
      host.setSound?.("silence");
      for (const o of soundsSlot.querySelectorAll(".cf-sound")) o.setAttribute("aria-pressed", "false");
    }
  }

  function stopSounds() {
    if (sounds?.stop) { try { sounds.stop(); } catch { /* already stopped */ } }
    if (sounds?.destroy) { try { sounds.destroy(); } catch { /* already gone */ } }
    sounds = null;
    soundsSaid = "";
    if (soundsSlot.childElementCount) soundsSlot.replaceChildren();
  }

  const seenTurns = new WeakSet(); // turn objects already drawn once: only a new one fades in
  function turnRow(t) {
    if (t.thinking) return el("li", { class: "cf-turn cf-thinking", "data-role": "assistant", text: "Thinking…" });
    const who = el("span", { class: "cf-who", text: t.role === "user" ? "You" : "" });
    const words = el("span", { class: "cf-words", text: t.pending ? "…" : t.text });
    const li = el("li", { class: "cf-turn", "data-role": t.role }, who, words);
    if (t.pending) li.dataset.pending = "1";
    if (t.source) li.dataset.source = t.source;
    return li;
  }

  function renderTalk() {
    const r = s.reply;
    const label = sourceLabel(r);
    talkCard.dataset.source = r ? label.source : "";
    talkCard.dataset.busy = String(s.busy);
    talkCard.dataset.rec = s.rec;
    // the tags describe the last reply; nothing is claimed before one arrived
    srcTag.hidden = !r;
    srcTag.textContent = label.title;
    srcNote.textContent = r ? label.note : "";
    srcNote.hidden = !srcNote.textContent;
    const v = voiceLabel(r);
    voiceTag.hidden = !r;
    voiceTag.textContent = v.text;
    const last = s.turns[s.turns.length - 1];
    // the reply time is shown only for a personal (Gemini) reply; the default lines are instant and need no figure
    latTag.textContent = r && label.source === "gemini" ? latencyLabel(last?.role === "assistant" ? last.timing : null) : "";
    latTag.hidden = !latTag.textContent;
    // the transcript: the last few turns, large; "Thinking…" while a reply is awaited. A turn that was not on screen
    // before fades in (comfort.css .cf-turn[data-enter]); the ones already shown are rebuilt without the fade.
    const shown = transcriptFor(s);
    const rows = shown.map((t) => {
      const row = turnRow(t);
      if (!seenTurns.has(t)) { seenTurns.add(t); row.dataset.enter = "1"; }
      return row;
    });
    chat.replaceChildren(...rows);
    talkStatus.textContent = s.error === "server" ? "Could not reach the voice right now. You can try again, or just stay here."
      : s.error === "mic" ? "The microphone is not available here. You can type instead." : "";
    talkStatus.hidden = !talkStatus.textContent;
    // remember: only the server's offer, saved only on Remember
    rememberBox.hidden = s.rememberState === "idle" || !s.remember;
    if (s.remember) confirmQ.textContent = s.remember.ask;
    const answered = s.rememberState === "saved" || s.rememberState === "skipped";
    rememberBtn.hidden = answered;
    notNowBtn.hidden = answered;
    confirmQ.hidden = answered;
    savedLine.hidden = !answered;
    if (s.rememberState === "skipped") savedLine.textContent = "Not saved. Nothing changed.";
    // input
    micBtn.setAttribute("aria-pressed", String(s.rec === "recording"));
    micBtn.setAttribute("aria-disabled", String(s.busy));
    micWord.textContent = s.rec === "recording" ? "Listening…" : s.busy ? "One moment…" : "Hold to talk";
    micHint.textContent = s.rec === "recording" ? (pressing ? "let go to send" : "tap again to send") : "or tap to start and stop";
    stopSpeakBtn.setAttribute("aria-disabled", String(!audio));
    // call my person: a tel: link on a phone, the demo overlay elsewhere, nothing without a contact
    const plan = callPlan(contactNow(), { coarse: Boolean(win.matchMedia?.("(pointer: coarse)")?.matches), width: win.innerWidth });
    callBtn.hidden = !plan || plan.kind !== "overlay";
    callLink.hidden = !plan || plan.kind !== "tel";
    if (plan) {
      const text = `Call my person · ${plan.first}`;
      callBtn.textContent = text;
      callLink.textContent = text;
      callLink.href = plan.href || "#";
      callBtn.setAttribute("aria-label", `Call my person: ${plan.name} (demo, no call is placed)`);
      callLink.setAttribute("aria-label", `Call my person: ${plan.name}`);
      callName.textContent = plan.name;
      const cs = callScreenFor(plan);
      calling.textContent = cs.calling;
      callDemo.hidden = !cs.demo;
      callTel.hidden = !cs.href;
      callTel.href = cs.href || "#";
      callTel.textContent = cs.telText;
      if (cs.href) callTel.setAttribute("aria-label", `Call ${plan.name}`);
    }
    callBtn.dataset.offer = s.offerCall ? "1" : "";
    callLink.dataset.offer = s.offerCall ? "1" : "";
    callOffer.hidden = !(s.offerCall && plan);
    if (plan) callOffer.textContent = `If it would help, you can call ${plan.first} now.`;
    const wasOpen = !callScreen.hidden;
    showCall(Boolean(s.call && plan));
    if (!callScreen.hidden && !wasOpen) { lastFocus = doc.activeElement; (callTel.hidden ? endBtn : callTel).focus({ preventScroll: true }); }
  }

  function dispatch(ev) {
    const prev = s;
    s = cmReduce(s, ev);
    if (s === prev) return;
    if (prev.phase === "talk" && s.phase !== "talk") { seq += 1; stopSpeech(); endRecording(false); releaseMic(); } // nothing keeps talking or listening off the card
    if ((prev.phase === "quiet") !== (s.phase === "quiet")) sosCaptions(s.phase === "quiet");
    if (s.phase === "quiet" && prev.phase !== "quiet") quietSounds(); // Stay Quiet means silence
    render();
    if (prev.phase !== s.phase) {
      // on open the hold keeps focus (sos.js show()), so Space held right after the press still brings the room back
      // instead of clicking "Stay Quiet"; Tab reaches the choices. Back from a card lands on the first choice.
      // Perf: the focus moves in the next frame, with the card's own style pass, not as a forced one of its own.
      const phase = s.phase;
      const raf = win.requestAnimationFrame || ((f) => f());
      raf(() => {
        if (s.phase !== phase) return;
        if (prev.phase === "hidden") return; // the press: the hold keeps focus (Play My Sounds opens under it)
        if (phase === "menu") choiceBtns[0].focus({ preventScroll: true });
        else if (phase === "hidden") host.focusHold?.();
        else if (cardHeading(phase)) focusHeading(cardHeading(phase)); // a screen reader reads where it landed
        else backBtn.focus({ preventScroll: true });
      });
    }
  }

  /** The heading of the card a choice opened (Play My Sounds: the player's own title once it is mounted). */
  function cardHeading(phase) {
    if (phase === "quiet") return quietCard.querySelector(".cf-card-label");
    if (phase === "sounds") return soundsCard.querySelector(".snd-title") || soundsCard.querySelector(".cf-card-label");
    if (phase === "talk") return talkHead.querySelector(".cf-talk-title");
    return null;
  }

  function focusHeading(node) {
    node.setAttribute("tabindex", "-1"); // focusable from script only, never a Tab stop
    node.focus({ preventScroll: true });
  }

  // ---- talk: one turn (text, voice, or the greeting when both are empty)
  async function sendTurn({ text, audioB64, audioMime } = {}) {
    if (s.phase !== "talk" || s.busy) return;
    const my = ++seq;
    stopSpeech();
    const history = s.turns; // fetchChat keeps only finished turns
    const isAudio = Boolean(audioB64);
    dispatch({ type: "send", text, audio: isAudio });
    const t0 = win.performance.now();
    const { reply } = await fetchChat(setting.profile, { history, text, audioB64, audioMime, speak: true, fetchImpl: win.fetch.bind(win) });
    if (my !== seq) return;
    const roundTripMs = Math.round(win.performance.now() - t0);
    const timing = { roundTripMs, latency_ms: reply?.latency_ms ?? null, stt_ms: reply?.stt_ms ?? null, tts_ms: reply?.tts_ms ?? null, source: reply?.source || null };
    dispatch({ type: "reply", reply, timing });
    if (reply?.engine === "elevenlabs" && reply.audio_base64) speak(reply.audio_base64);
  }

  // iOS Safari (and WebKit generally) blocks audio.play() outside a tap, and a reply arrives seconds after the tap. One
  // element is created and started with a silent clip inside the tap; later replies only change its src, which stays allowed.
  function unlockVoice() {
    if (voiceEl) return;
    try {
      voiceEl = new win.Audio();
      voiceEl.preload = "auto";
      voiceEl.src = SILENT_WAV;
      const p = voiceEl.play();
      if (p?.catch) p.catch(() => { voiceEl = null; }); // refused (not a tap): try again on the next one
    } catch { voiceEl = null; }
  }

  function speak(b64) {
    stopSpeech();
    try {
      voiceEl ||= new win.Audio();
      const el = voiceEl;
      audio = el;
      const done = () => {
        if (audio !== el) return;
        audio = null; el.onended = el.onerror = null;
        talkCard.dataset.speaking = ""; stopSpeakBtn.setAttribute("aria-disabled", "true");
        duckSounds(false);
      };
      el.onended = done;
      el.onerror = done;
      el.src = `data:audio/wav;base64,${b64}`;
      talkCard.dataset.speaking = "1";
      stopSpeakBtn.setAttribute("aria-disabled", "false");
      duckSounds(true); // the sound sits low under the voice
      el.play().catch(() => { voiceTag.textContent = "Voice blocked, showing text"; done(); });
    } catch { audio = null; }
  }

  function stopSpeech() {
    if (audio) { const el = audio; audio = null; el.onended = el.onerror = null; try { el.pause(); } catch { /* gone */ } }
    talkCard.dataset.speaking = "";
    stopSpeakBtn.setAttribute("aria-disabled", "true");
    duckSounds(false);
  }

  /** Play My Sounds keeps playing under Talk to Me, ramped to a quarter while the voice speaks. */
  function duckSounds(on) {
    if (sounds?.duck) { try { sounds.duck(on); } catch { /* optional */ } }
  }

  function stopAll() {
    stopSpeech();
    if (mods.sounds?.stopAllSounds) { try { mods.sounds.stopAllSounds(); } catch { /* optional */ } }
    host.setSound?.("silence");
    for (const o of soundsSlot.querySelectorAll(".cf-sound")) o.setAttribute("aria-pressed", "false");
    status.textContent = "Stopped. Nothing from this app is playing.";
  }

  // ---- recording: hold (or tap) the big button; the clip goes to the server as base64
  async function startRecording() {
    if (s.phase !== "talk" || s.busy || s.rec === "recording") return false;
    if (typeof win.MediaRecorder !== "function" || !win.navigator?.mediaDevices?.getUserMedia) { micFailed(); return false; }
    try {
      stream ||= await win.navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
    } catch { micFailed(); return false; }
    if (s.phase !== "talk" || s.busy) return false;
    const mimeType = pickMime((t) => win.MediaRecorder.isTypeSupported?.(t));
    try { recorder = new win.MediaRecorder(stream, mimeType ? { mimeType } : undefined); } catch { micFailed(); return false; }
    chunks = [];
    const rec = recorder;
    rec.addEventListener("dataavailable", (e) => { if (e.data?.size) chunks.push(e.data); });
    rec.addEventListener("stop", async () => {
      if (rec !== recorder) return;
      recorder = null;
      const parts = chunks; chunks = [];
      if (!rec.sendWhenDone || !parts.length) return;
      const blob = new win.Blob(parts, { type: rec.mimeType || mimeType || "audio/webm" });
      let b64 = "";
      try { b64 = await blobToBase64(blob); } catch { b64 = ""; }
      if (b64) sendTurn({ audioB64: b64, audioMime: blob.type });
    });
    rec.start();
    dispatch({ type: "rec", on: true });
    win.clearTimeout(recTimer);
    recTimer = win.setTimeout(() => endRecording(true), MAX_RECORD_MS);
    status.textContent = "Listening.";
    return true;
  }

  function endRecording(send) {
    win.clearTimeout(recTimer);
    pressing = false; stopOnUp = false;
    const rec = recorder;
    if (!rec) return;
    rec.sendWhenDone = Boolean(send);
    try { if (rec.state !== "inactive") rec.stop(); else recorder = null; } catch { recorder = null; }
    dispatch({ type: "rec", on: false });
    if (send) status.textContent = "Sending what you said.";
  }

  function releaseMic() {
    if (stream) { try { for (const t of stream.getTracks()) t.stop(); } catch { /* gone */ } stream = null; }
  }

  function micFailed() {
    s = { ...s, error: "mic" };
    render();
    typeIn.focus({ preventScroll: true });
  }

  // pointer: down starts (or, while listening, marks a stop); up after a hold sends, up after a tap keeps listening
  micBtn.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    unlockVoice(); // this tap is what lets replies play on iOS
    if (s.rec === "recording") { stopOnUp = true; return; }
    pressT0 = win.performance.now();
    pressing = true;
    try { micBtn.setPointerCapture?.(e.pointerId); } catch { /* fine */ }
    startRecording().then((ok) => { if (ok && pressing) renderTalk(); });
  });
  const micUp = (e) => {
    if (e.type === "pointerup" && e.button !== 0) return;
    if (e.type === "pointerup") unlockVoice(); // touch only counts as a gesture on release in WebKit
    if (stopOnUp) { endRecording(true); return; }
    if (!pressing) return;
    pressing = false;
    if (win.performance.now() - pressT0 >= HOLD_MS) endRecording(true); // it was a hold
    else renderTalk(); // a tap: keep listening until the next tap
  };
  micBtn.addEventListener("pointerup", micUp);
  micBtn.addEventListener("pointercancel", micUp);
  micBtn.addEventListener("click", (e) => { // keyboard (Space / Enter): a tap
    if (e.detail !== 0) return;
    unlockVoice();
    if (s.rec === "recording") endRecording(true);
    else startRecording();
  });
  micBtn.addEventListener("contextmenu", (e) => e.preventDefault());

  typeForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = typeIn.value.trim();
    if (!text || s.busy) return;
    unlockVoice();
    typeIn.value = "";
    if (s.rec === "recording") endRecording(false);
    sendTurn({ text });
  });
  stopSpeakBtn.addEventListener("click", () => { stopSpeech(); status.textContent = "Stopped speaking."; });

  // call my person
  callBtn.addEventListener("click", () => dispatch({ type: "call", on: true }));
  callLink.addEventListener("click", (e) => { if (callLink.getAttribute("href") === "#") e.preventDefault(); });
  function endCall() {
    if (!s.call) return;
    dispatch({ type: "call", on: false });
    const back = lastFocus && doc.contains(lastFocus) ? lastFocus : callBtn;
    lastFocus = null;
    try { back.focus({ preventScroll: true }); } catch { /* fine */ }
  }
  endBtn.addEventListener("click", endCall);
  callScreen.addEventListener("keydown", (e) => { // focus stays on the overlay's controls (End, and Call on a phone)
    if (e.key !== "Tab") return;
    e.preventDefault();
    (callTel.hidden || doc.activeElement === callTel ? endBtn : callTel).focus();
  });

  async function remember() {
    const rm = s.remember;
    if (!rm || s.rememberState !== "offer") return;
    dispatch({ type: "remember" });
    savedLine.textContent = "Saving…";
    const { reply } = await postFeedback({ profileId: setting.profile, strategy: rm.strategy, helpful: rm.helpful, confirm: true }, { fetchImpl: win.fetch.bind(win) });
    if (!reply?.saved) {
      savedLine.textContent = "Could not save. Nothing changed.";
      return;
    }
    savedLine.textContent = typeof reply.change === "string" && reply.change ? reply.change
      : `Saved. Next time I'll ${rm.helpful ? "offer" : "skip"} ${rm.label.toLowerCase()}.`;
    const change = { suggestion: rm.strategy, strategy: rm.strategy, helpful: rm.helpful, label: rm.label.toLowerCase(), text: reply.change || null };
    win.dispatchEvent(new win.CustomEvent("comfort:profile-updated", { detail: { profile_id: setting.profile, change, profile: reply.profile } }));
    if (reply.profile) profiles = profiles.map((p) => (p.id === reply.profile.id ? reply.profile : p));
    renderLearned(true);
  }

  // ---- learned
  function renderLearned(force = false) {
    if (learnedSlot.hidden && !force) return;
    if (mods.learned) {
      if (!learnedUi) {
        try { learnedUi = mods.learned.mountLearned(learnedSlot, { profileId: setting.profile, switcher: false }); } catch { mods.learned = null; }
      }
      if (learnedUi) return;
    }
    builtinLearned();
  }

  function plainLabel(t) {
    const str = String(t || "");
    if (str.startsWith("activity:")) return str.slice("activity:".length);
    return { breathing: "a slow breath", movement: "moving or a fidget", quiet: "staying quiet" }[str] || str;
  }

  // built-in: the saved lists, straight from the profile
  async function builtinLearned() {
    let p = profiles.find((x) => x.id === setting.profile);
    try {
      const r = await win.fetch("/comfort/profiles");
      if (r.ok) { profiles = await r.json(); p = profiles.find((x) => x.id === setting.profile); }
    } catch { /* offline */ }
    const list = (title, items, empty) => el("div", { class: "cf-learned-group" },
      el("p", { class: "cf-card-label", text: title }),
      items.length ? el("ul", { class: "cf-learned-list" }, ...items.map((t) => el("li", { class: "cf-learned-item", text: plainLabel(t) })))
        : el("p", { class: "cf-fine", text: empty }));
    learnedSlot.replaceChildren(
      el("p", { class: "cf-fine", text: p ? `${profileName(p.id)}. Saved only when you tap Remember.` : "No profile loaded." }),
      list("Helps", p?.helps || p?.worked || [], "Nothing saved yet."),
      list("Skip next time", p?.avoid || p?.dislikes || [], "Nothing saved yet."),
      list("Remembered from feedback", (p?.history || []).map((h) => h.text || h.label), "Nothing confirmed yet."));
    learnedSlot.dataset.flash = "1";
    win.setTimeout(() => { delete learnedSlot.dataset.flash; }, 1200);
  }

  // ---- inputs
  for (const b of choiceBtns) {
    b.addEventListener("click", () => {
      const c = b.dataset.choice;
      if (c === "talk") unlockVoice(); // the tap that opens Talk to Me unlocks audio for the greeting and every reply
      dispatch({ type: "choose", choice: c });
      if (c === "quiet") { stopAll(); setQuiet(true); status.textContent = "Stay Quiet. Nothing plays. Their words appear here."; }
      if (c === "talk") sendTurn({}); // the greeting
    });
  }
  backBtn.addEventListener("click", () => dispatch({ type: "back" }));
  stopBtn.addEventListener("click", stopAll);
  rememberBtn.addEventListener("click", remember);
  notNowBtn.addEventListener("click", () => dispatch({ type: "notnow" }));
  learnedToggle.addEventListener("click", () => {
    learnedSlot.hidden = !learnedSlot.hidden;
    learnedToggle.setAttribute("aria-expanded", String(!learnedSlot.hidden));
    if (!learnedSlot.hidden) renderLearned(true);
  });
  profileBtn.addEventListener("click", () => {
    const ids = profiles.length ? profiles.map((p) => p.id) : DEMO_PROFILES;
    const next = ids[(ids.indexOf(setting.profile) + 1) % ids.length];
    api.setProfile(next);
    status.textContent = `Profile: ${profileName(next)}`;
  });

  // keys: Escape ends the demo call. Window capture, so sos.js's hold handler (document) never sees it.
  win.addEventListener("keydown", (e) => {
    if (s.call && e.key === "Escape" && !e.repeat) { e.preventDefault(); e.stopImmediatePropagation(); endCall(); }
  }, true);
  win.addEventListener("comfort:profile-updated", () => { if (!learnedSlot.hidden) renderLearned(true); });
  // the screen changed (app.js sets body[data-screen]): the keys follow it
  try { new win.MutationObserver(() => renderLegend()).observe(doc.body, { attributes: true, attributeFilter: ["data-screen"] }); } catch { /* old browser */ }
  host.onShow?.((on) => {
    setting = loadComfort(storage);
    if (on) {
      mountQuietUi();
      quietUi?.clear?.(); // a new SOS: no words from the last one
      const st = host.store?.();
      if (st && !capSub) capSub = st.on("change", () => renderCaptions());
      dispatch({ type: "open", on: setting.on });
      // sos.js press() already put the engine in Quiet (before anything was drawn). This hook runs inside the press
      // (pointerdown / keydown), so Play My Sounds starts now, fading in; pressing the band again stops it.
      if (s.phase === "sounds") autoplaySounds();
    } else {
      seq += 1;
      stopSpeech();
      capSub?.(); // the store subscription goes with the frame (it leaked one per open)
      capSub = null;
      dispatch({ type: "close" });
    }
  });
  loadProfiles();
  render();
  void ready;
  return api;
}
