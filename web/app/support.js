// Trusted Support (concept card, PR #402). A small, optional card: pick someone you trust, pick what (if anything) to
// share, and see an illustrative phone-notification preview of what they WOULD get.
//
// Honesty line: this is a Supporter Preview. Nothing is ever sent. The card makes no network call of its own (no fetch,
// no socket, no beacon), reads no audio, transcript or sensory data, and detects nothing automatically: the person taps
// "Request support" themselves. "Copy message" puts the text on their clipboard so they can send it in their own app.
// The supporter name stays in this page's memory (never stored, never sent).
//
// Pure parts (SUPPORT_MESSAGES, cleanName, previewHtml, cardHtml) run under node --test; mountSupport(el) is the thin
// browser half: it renders once, then delegates events on `el`.

export const TITLE = "Trusted Support";
export const BADGE = "Supporter Preview · Concept";
export const INTRO = "Choose someone you trust and what, if anything, to share.";
export const DEFAULT_NAME = "Your supporter";
export const NOT_SENT = "Preview only, nothing was sent.";
export const MAX_NAME = 40;
export const SUPPORT_MESSAGES = [
  { id: "quiet", text: "I prefer quiet right now." },
  { id: "calmer", text: "Could you help me find somewhere calmer?" },
  { id: "space", text: "Please give me some space." },
  { id: "help", text: "I'd like your help." },
];

const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ESC[c]);

/** The name as typed, trimmed and capped; blank falls back to the default. */
export function cleanName(raw) {
  const n = String(raw ?? "").replace(/\s+/g, " ").trim().slice(0, MAX_NAME);
  return n || DEFAULT_NAME;
}

export function messageById(id) {
  return SUPPORT_MESSAGES.find((m) => m.id === id) || null;
}

/** The illustrative notification the supporter WOULD receive. Pure markup; nothing here is sent anywhere. */
export function previewHtml(name, message) {
  return `<div class="sup-phone" role="group" aria-label="Preview of the notification ${esc(cleanName(name))} would see">
  <p class="sup-notif-to">For ${esc(cleanName(name))}</p>
  <div class="sup-notif">
    <p class="sup-notif-app"><span class="sup-notif-dot" aria-hidden="true"></span>VoiceLock · Preview</p>
    <p class="sup-notif-title">A message from your friend</p>
    <p class="sup-notif-body" data-role="preview-text">${esc(message.text)}</p>
  </div>
  <p class="sup-notsent" data-role="not-sent">${NOT_SENT}</p>
  <button type="button" class="sup-btn sup-btn-ghost" data-act="copy">Copy message</button>
</div>`;
}

export function cardHtml(uid = "sup") {
  const choices = SUPPORT_MESSAGES.map((m) => `
      <label class="sup-choice">
        <input type="radio" name="${uid}-msg" value="${m.id}" data-act="choose">
        <span class="sup-choice-text">${esc(m.text)}</span>
      </label>`).join("");
  return `<section class="sup-card" aria-labelledby="${uid}-title">
  <header class="sup-head">
    <h2 class="sup-title" id="${uid}-title">${TITLE}</h2>
    <p class="sup-badge">${BADGE}</p>
  </header>
  <p class="sup-intro">${INTRO}</p>
  <div class="sup-field">
    <label class="sup-label" for="${uid}-name">Supporter name</label>
    <input class="sup-input" id="${uid}-name" data-act="name" type="text" maxlength="${MAX_NAME}" autocomplete="off"
      placeholder="${DEFAULT_NAME}" aria-describedby="${uid}-name-hint">
    <p class="sup-hint" id="${uid}-name-hint">Stays on this device. Leave blank to use “${DEFAULT_NAME}”.</p>
  </div>
  <fieldset class="sup-choices">
    <legend class="sup-label">What to share</legend>${choices}
  </fieldset>
  <button type="button" class="sup-btn sup-btn-main" data-act="request">Request support</button>
  <div class="sup-slot" data-slot="preview" aria-live="polite"></div>
  <p class="sup-status" data-slot="status" role="status"></p>
  <p class="sup-fine">Optional. Nothing is sent from here. No audio, transcript or sensory data is ever shared, and the app never decides for you.</p>
</section>`;
}

/**
 * Mount the card into `el`. `opts.clipboard` (default navigator.clipboard) is injectable for tests.
 * Returns {getState, destroy}.
 */
export function mountSupport(el, opts = {}) {
  const doc = el.ownerDocument || (typeof document !== "undefined" ? document : null);
  const uid = opts.uid || "sup";
  if (doc?.head && !doc.querySelector?.('link[data-support-css]')) {
    const link = doc.createElement("link");
    link.rel = "stylesheet";
    link.href = new URL("./support.css", import.meta.url).href;
    link.setAttribute("data-support-css", "");
    doc.head.append(link);
  }
  el.innerHTML = cardHtml(uid);
  const slot = (n) => el.querySelector(`[data-slot="${n}"]`);
  const state = { name: "", choice: null, shown: false };
  const say = (t) => { slot("status").textContent = t; };
  const draw = () => {
    const m = messageById(state.choice);
    slot("preview").innerHTML = state.shown && m ? previewHtml(state.name, m) : "";
  };

  async function copy() {
    const m = messageById(state.choice);
    if (!m) return;
    const cb = opts.clipboard || (typeof navigator !== "undefined" ? navigator.clipboard : null);
    try {
      if (!cb?.writeText) throw new Error("no clipboard");
      await cb.writeText(m.text);
      say("Copied. Paste it into your own messages app to send it.");
    } catch {
      say("Could not copy. Select the message and copy it yourself.");
    }
  }

  const onInput = (e) => {
    if (e.target?.dataset?.act === "name") { state.name = e.target.value; if (state.shown) draw(); }
  };
  const onChange = (e) => {
    if (e.target?.dataset?.act === "choose") {
      state.choice = e.target.value;
      if (state.shown) { draw(); say(""); }
    }
  };
  const onClick = (e) => {
    const b = e.target?.closest?.("[data-act]");
    const act = b?.dataset?.act;
    if (act === "request") {
      if (!messageById(state.choice)) { state.shown = false; draw(); say("Pick a message first, or just close this card."); return; }
      state.shown = true; draw(); say("");
    } else if (act === "copy") copy();
  };
  el.addEventListener("input", onInput);
  el.addEventListener("change", onChange);
  el.addEventListener("click", onClick);
  return {
    getState: () => ({ name: cleanName(state.name), choice: state.choice, previewShown: state.shown }),
    destroy() {
      el.removeEventListener("input", onInput);
      el.removeEventListener("change", onChange);
      el.removeEventListener("click", onClick);
    },
  };
}
