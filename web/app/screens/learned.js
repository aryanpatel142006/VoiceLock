// "What I've Learned About You": a calm, editable view of one comfort profile (Comfort Companion).
//
// Shows only what the server saved (GET /comfort/profiles): nothing here is invented or inferred. Four sections:
// what helps, what to avoid, sound preferences, and what was remembered from confirmed check-in feedback (the
// server writes `worked` / `dislikes` only when the person confirms, POST /comfort/feedback with confirm:true).
// Every edit or removal is a PUT /comfort/profiles/{id} of the whole profile; the server's reply is what we show.
// Alex and Sam are fictional demo profiles (fictional:true) and carry a "Demo profile" badge.
//
// When another part of the app saves something it fires window "comfort:profile-updated" {profile_id, change};
// this screen re-fetches and marks what is new or changed with a soft highlight and a "New" tag (with
// prefers-reduced-motion: the tag only, no motion).
//
//   mountLearned(el, { profileId, fetch, window }) -> { ready: Promise, refresh(), unmount() }
//   mount(root, ctx)  // app shell contract (#learned, ?profile=<id>)

export const title = "What I've Learned";
export const UPDATE_EVENT = "comfort:profile-updated";
const API = "/comfort/profiles";
const FEEDBACK_KINDS = new Set(["breathing", "movement", "quiet"]);

const SOUND_MODES = [
  { id: "silence", label: "Quiet" },
  { id: "music", label: "Soft music" },
  { id: "speech", label: "A calm voice" },
];
const TALK = [
  { id: "little", label: "Just a few words" },
  { id: "some", label: "A little more" },
];

// ---------------------------------------------------------------- pure helpers (node --test)

const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

/** A saved id ("breathing", "activity:drawing", "loud music") in plain words. */
export function plainLabel(id) {
  const raw = String(id || "").trim();
  if (raw === "breathing") return "Breathing exercises";
  if (raw === "movement") return "Moving around or a fidget";
  if (raw === "quiet") return "Sitting in quiet";
  if (raw === "lofi") return "Lo-fi music";
  if (raw.startsWith("activity:")) return cap(raw.slice("activity:".length).trim());
  return cap(raw.replace(/-/g, " ")); // catalog ids ("brown-noise", "step-away") in words
}

/** The four sections of one profile, every item keyed so an update can be diffed and a removal can be undone. */
export function sectionsFor(profile) {
  const p = profile || {};
  // the server's `helps` / `avoid` are the source of truth (worked / dislikes / the flags are its mirrors); a profile
  // without them (older clients, the tests) is read from the mirrors
  const helpsList = Array.isArray(p.helps) ? p.helps : null;
  const avoidList = Array.isArray(p.avoid) ? p.avoid : (p.dislikes || []);
  const inHelps = (x) => (helpsList || []).some((h) => String(h).toLowerCase() === String(x).toLowerCase());
  const extra = (helpsList || []).filter((h) => !["breathing", "movement"].includes(String(h).toLowerCase()));
  const helps = [
    ...(p.activities || []).map((a) => ({ key: `activity:${a}`, text: cap(a), remove: { list: "activities", value: a } })),
    ...((helpsList ? inHelps("breathing") : p.breathing_helps) ? [{ key: "flag:breathing", text: "Familiar breathing", remove: { flag: "breathing_helps" } }] : []),
    ...((helpsList ? inHelps("movement") : p.movement_helps) ? [{ key: "flag:movement", text: "Moving, or holding a fidget", remove: { flag: "movement_helps" } }] : []),
    ...extra.map((h) => ({ key: `helps:${h}`, text: plainLabel(h), remove: { list: "helps", value: h } })),
  ];
  const avoid = avoidList.map((d) => ({ key: `dislike:${d}`, text: plainLabel(d), remove: { list: "dislikes", value: d } }));
  const inAvoid = (x) => avoidList.some((a) => String(a).toLowerCase() === String(x).toLowerCase());
  const mode = SOUND_MODES.find((m) => m.id === p.preferred_mode);
  const talk = TALK.find((t) => t.id === p.talk_amount);
  const sound = [
    { key: `mode:${p.preferred_mode}`, field: "preferred_mode", label: "Favorite sound", text: mode ? mode.label : "Not saved yet" },
    { key: `volume:${Number.isFinite(p.volume) ? p.volume : "-"}`, field: "volume", label: "Comfortable volume",
      text: Number.isFinite(p.volume) ? `${Math.round(p.volume)}%` : "Not saved yet" },
    { key: `talk:${p.talk_amount}`, field: "talk_amount", label: "Spoken guidance",
      text: talk ? `Only when you ask · ${talk.label.toLowerCase()}` : "Only when you ask" },
  ];
  // the server's confirmed-feedback history when it exists (written only by POST /comfort/feedback confirm:true);
  // older profiles without it fall back to worked / dislikes
  // a confirmed check-in shows while it still holds (removing the item it saved, here or above, drops the line too)
  const remembered = Array.isArray(p.history) && p.history.length ? p.history
    .filter((h, i, all) => !all.slice(i + 1).some((later) => later.strategy === h.strategy)) // the latest check-in per idea
    .filter((h) => (helpsList ? (h.helpful ? inHelps(h.strategy) : inAvoid(h.strategy)) : true))
    .map((h, i) => ({
    key: `history:${h.ts || i}:${h.strategy}`,
    text: h.text || `${cap(String(h.label || h.strategy))} ${h.helpful ? "helped" : "wasn't helpful"} last time.`,
    remove: h.helpful ? { list: "helps", value: h.strategy } : { list: "avoid", value: h.strategy },
  })) : [
    ...(p.worked || []).map((w) => ({ key: `worked:${w}`, text: `${plainLabel(w)} helped last time.`, remove: { list: "worked", value: w } })),
    ...(p.dislikes || []).filter((d) => FEEDBACK_KINDS.has(d) || String(d).startsWith("activity:")).map((d) => {
      const label = plainLabel(d);
      const verb = /s$/i.test(label) ? "weren't" : "wasn't";
      return { key: `notworked:${d}`, text: `${label} ${verb} helpful last time.`, remove: { list: "dislikes", value: d } };
    }),
  ];
  return { helps, avoid, sound, remembered };
}

/** Keys present in `next` but not in `prev` (new or changed items). */
export function freshKeys(prev, next) {
  const keys = (s) => new Set(s ? [...s.helps, ...s.avoid, ...s.sound, ...s.remembered].map((i) => i.key) : []);
  if (!prev) return new Set();
  const before = keys(prev);
  return new Set([...keys(next)].filter((k) => !before.has(k)));
}

/** The profile without one item: a list entry dropped or a "helps" flag turned off. */
export function withoutItem(profile, remove) {
  const next = { ...profile };
  const drop = (list, value) => {
    if (Array.isArray(profile[list])) next[list] = profile[list].filter((x) => String(x).toLowerCase() !== String(value).toLowerCase());
  };
  // the server keeps `helps` / `avoid` over their mirrors, so a removal edits both or it would come straight back
  const MIRROR = { dislikes: "avoid", avoid: "dislikes", worked: "helps", helps: "worked" };
  if (remove.flag) {
    next[remove.flag] = false;
    const id = remove.flag === "breathing_helps" ? "breathing" : "movement";
    drop("helps", id); drop("worked", id);
  }
  if (remove.list) {
    next[remove.list] = (profile[remove.list] || []).filter((x) => x !== remove.value);
    if (MIRROR[remove.list]) drop(MIRROR[remove.list], remove.value);
  }
  return next;
}

export function newMeProfile() {
  return { preferred_name: "Me", fictional: false, preferred_mode: "silence", talk_amount: "little",
    activities: [], breathing_helps: false, movement_helps: false, dislikes: [], worked: [] };
}

// ---------------------------------------------------------------- the screen

export function mountLearned(el, opts = {}) {
  const win = opts.window || globalThis.window;
  const doFetch = opts.fetch || ((...a) => globalThis.fetch(...a));
  const doc = el.ownerDocument || globalThis.document;
  const reduced = () => Boolean(win?.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches);
  const state = { profiles: [], id: opts.profileId || null, sections: null, fresh: new Set(), status: "", undo: null,
    loading: true, error: null, editing: null, alive: true };

  function h(tag, attrs = {}, ...children) {
    const node = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : String(v));
    }
    for (const c of children.flat()) if (c != null && c !== false) node.append(c);
    return node;
  }

  const current = () => state.profiles.find((p) => p.id === state.id) || null;

  async function load(highlight = false) {
    try {
      const res = await doFetch(API, { headers: { Accept: "application/json" } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const list = await res.json();
      state.profiles = Array.isArray(list) ? list : [];
      state.error = null;
    } catch (err) {
      state.error = "I couldn't reach your saved profile just now. Nothing was changed.";
    }
    if (!current()) state.id = (state.profiles.find((p) => !p.fictional) || state.profiles[0] || {}).id || null;
    const next = sectionsFor(current());
    state.fresh = highlight ? freshKeys(state.sections, next) : new Set();
    state.sections = next;
    state.loading = false;
    render();
  }

  async function save(next, message, undoProfile) {
    const p = current();
    if (!p) return;
    try {
      const res = await doFetch(`${API}/${encodeURIComponent(p.id)}`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(next),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const saved = await res.json();
      state.profiles = state.profiles.map((x) => (x.id === saved.id ? saved : x));
      state.sections = sectionsFor(saved);
      state.fresh = new Set();
      state.status = message;
      state.undo = undoProfile || null;
      if (next.volume != null && saved.volume == null) {
        state.status = "This version can't keep a volume yet, so I didn't save it.";
        state.undo = null;
      }
    } catch {
      state.status = "That didn't save. Your profile is unchanged.";
      state.undo = null;
    }
    state.editing = null;
    render();
  }

  async function createMe() {
    try {
      const res = await doFetch(`${API}/me`, { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newMeProfile()) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const saved = await res.json();
      state.profiles = [...state.profiles.filter((x) => x.id !== saved.id), saved];
      state.id = saved.id;
      state.sections = sectionsFor(saved);
      state.status = "Your own profile is ready. It starts empty.";
    } catch {
      state.status = "I couldn't make your profile just now.";
    }
    render();
  }

  // ---- rendering
  function itemRow(item, section) {
    const isNew = state.fresh.has(item.key);
    const p = current();
    return h("li", { class: `lr-item${isNew ? " is-new" : ""}${isNew && !reduced() ? " is-glow" : ""}`, "data-key": item.key },
      h("span", { class: "lr-item-text", text: item.text }),
      isNew ? h("span", { class: "lr-new", text: "New" }) : null,
      item.remove ? h("button", {
        type: "button", class: "lr-remove", "data-action": "remove", "aria-label": `Remove “${item.text}” from ${section}`,
        onclick: () => save(withoutItem(p, item.remove), `Removed “${item.text}”.`, p),
      }, h("span", { "aria-hidden": "true", text: "Remove" })) : null);
  }

  function section(id, heading, lede, items, empty) {
    return h("section", { class: "lr-section", "aria-labelledby": `lr-h-${id}`, "data-section": id },
      h("h2", { class: "lr-h", id: `lr-h-${id}`, tabindex: "-1", text: heading }),
      lede ? h("p", { class: "lr-lede", text: lede }) : null,
      items.length ? h("ul", { class: "lr-list", role: "list" }, items.map((i) => itemRow(i, heading)))
        : h("p", { class: "lr-empty", text: empty }));
  }

  function choiceGroup(field, label, options, value) {
    const p = current();
    return h("div", { class: "lr-choices", role: "group", "aria-label": label },
      options.map((o) => h("button", {
        type: "button", class: "lr-choice", "data-choice": `${field}:${o.id}`, "aria-pressed": String(o.id === value),
        onclick: () => save({ ...p, [field]: o.id }, `${label}: ${o.label}. Saved.`, p),
      }, o.label)));
  }

  function soundRow(item) {
    const p = current();
    const isNew = state.fresh.has(item.key);
    const editing = state.editing === item.field;
    let editor = null;
    if (editing && item.field === "preferred_mode") editor = choiceGroup("preferred_mode", "Favorite sound", SOUND_MODES, p.preferred_mode);
    if (editing && item.field === "talk_amount") editor = choiceGroup("talk_amount", "Spoken guidance", TALK, p.talk_amount);
    if (editing && item.field === "volume") {
      const start = Number.isFinite(p.volume) ? Math.round(p.volume) : 40;
      const out = h("output", { class: "lr-vol-out", text: `${start}%` });
      const input = h("input", { type: "range", min: "0", max: "100", step: "5", value: String(start), class: "lr-range",
        "aria-label": "Comfortable volume", oninput: () => { out.textContent = `${input.value}%`; } });
      editor = h("div", { class: "lr-vol" }, input, out,
        h("button", { type: "button", class: "lr-btn", "data-action": "save-volume",
          onclick: () => save({ ...p, volume: Number(input.value) }, `Comfortable volume: ${input.value}%. Saved.`, p) }, "Save"));
    }
    return h("li", { class: `lr-item lr-pref${isNew ? " is-new" : ""}${isNew && !reduced() ? " is-glow" : ""}`, "data-key": item.key },
      h("div", { class: "lr-pref-head" },
        h("span", { class: "lr-pref-label", text: item.label }),
        h("span", { class: "lr-pref-value", text: item.text }),
        isNew ? h("span", { class: "lr-new", text: "New" }) : null,
        h("button", { type: "button", class: "lr-edit", "data-action": `edit-${item.field}`, "aria-expanded": String(editing),
          "aria-label": `${editing ? "Close" : "Change"} ${item.label.toLowerCase()}`,
          onclick: () => { state.editing = editing ? null : item.field; render(); } }, editing ? "Close" : "Change")),
      editor);
  }

  function switcher() {
    const hasMe = state.profiles.some((p) => !p.fictional);
    return h("div", { class: "lr-switch", role: "group", "aria-label": "Whose profile" },
      state.profiles.map((p) => h("button", {
        type: "button", class: "lr-person", "data-profile": p.id, "aria-pressed": String(p.id === state.id),
        onclick: () => { state.id = p.id; state.sections = sectionsFor(p); state.fresh = new Set(); state.status = ""; state.undo = null; state.editing = null; render(); },
      }, h("span", { class: "lr-person-name", text: p.preferred_name }), p.fictional ? h("span", { class: "lr-demo", text: "Demo" }) : null)),
      hasMe ? null : h("button", { type: "button", class: "lr-person lr-person-new", "data-action": "create-me", onclick: createMe }, "+ Me"));
  }

  function render() {
    if (!state.alive) return;
    const p = current();
    const s = state.sections || sectionsFor(null);
    const head = h("header", { class: "lr-head" },
      h("h1", { class: "lr-title", id: "lr-title", text: "What I've learned about you" }),
      h("p", { class: "lr-intro", text: "Only what you've told me or confirmed. You can change or remove anything." }));
    const body = [];
    if (state.loading) body.push(h("p", { class: "lr-empty", text: "Opening your profile…" }));
    else if (state.error && !p) body.push(h("p", { class: "lr-empty", role: "alert", text: state.error }));
    else if (!p) body.push(h("p", { class: "lr-empty", text: "No profile is saved yet." }));
    else {
      body.push(h("div", { class: "lr-who" },
        h("p", { class: "lr-name", text: p.preferred_name }),
        p.fictional ? h("span", { class: "lr-badge", "data-badge": "demo", text: "Demo profile" }) : null,
        p.fictional ? h("p", { class: "lr-demo-note", text: "A made-up person for showing the app. Not a real user." }) : null));
      body.push(h("div", { class: "lr-grid" },
        section("helps", "What helps me", null, s.helps, "Nothing saved yet."),
        section("avoid", "What I prefer to avoid", null, s.avoid, "Nothing to avoid saved yet."),
        h("section", { class: "lr-section", "aria-labelledby": "lr-h-sound", "data-section": "sound" },
          h("h2", { class: "lr-h", id: "lr-h-sound", tabindex: "-1", text: "My sound preferences" }),
          h("ul", { class: "lr-list", role: "list" }, s.sound.map(soundRow))),
        section("remembered", "What you've remembered", "From check-ins you confirmed.", s.remembered,
          "Nothing yet. When you tell me something helped, and say save it, it shows up here.")));
    }
    const status = h("div", { class: "lr-status", role: "status", "aria-live": "polite" },
      state.status ? h("span", { text: state.status }) : null,
      state.undo ? h("button", { type: "button", class: "lr-btn lr-undo", "data-action": "undo",
        onclick: () => { const back = state.undo; save(back, "Put back.", null); } }, "Undo") : null);
    el.replaceChildren(h("article", { class: "learned", "aria-labelledby": "lr-title" },
      head, state.profiles.length && opts.switcher !== false ? switcher() : null, status, ...body));
  }

  const onUpdate = (ev) => {
    const detail = ev?.detail || {};
    if (detail.profile_id && !state.profiles.some((p) => p.id === detail.profile_id)) state.id = state.id || detail.profile_id;
    load(detail.profile_id ? detail.profile_id === state.id : true);
  };
  win?.addEventListener?.(UPDATE_EVENT, onUpdate);
  render();
  const ready = load(false);
  return {
    ready,
    refresh: (highlight = true) => load(highlight),
    get state() { return state; },
    unmount() { state.alive = false; win?.removeEventListener?.(UPDATE_EVENT, onUpdate); },
  };
}

/** App shell contract: #learned (optionally ?profile=<id>). */
export function mount(root, ctx = {}) {
  const view = mountLearned(root, { profileId: ctx.params?.get?.("profile") || null });
  return () => view.unmount();
}
