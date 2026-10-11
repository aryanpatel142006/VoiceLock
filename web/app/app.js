// Selective Hearing app shell (T-345, M14): screen registry, hash routing, cinematic cuts, PWA registration.
//
// Screen modules live in ./screens/<id>.js and are imported only when shown. The server lists the files
// present (GET /app/_files.json, hearing/server/static.py), so a screen that has not landed shows a
// placeholder instead of a 404. A module that fails to import shows the placeholder with the reason.
//
// Screen module contract (default export or named exports, both work):
//   export function mount(root, ctx) { ...; return cleanup }   // cleanup: function, {unmount()} or nothing
//   export const title = "Listen";                              // optional, overrides the registry title
// ctx = {
//   app,        // this shell: app.go(id), app.openSheet(id), app.closeSheet(), app.on(event, fn), app.files
//   id,         // the screen id
//   params,     // URLSearchParams of location.search (e.g. ?watch=<session>)
//   signal,     // AbortSignal aborted when the screen leaves: addEventListener(..., {signal})
//   store,      // ./store.js module namespace once it is on the server (T-3 of M14), else null
// }
// A screen's ./screens/<id>.css is linked before mount when the server lists it. Top-level stylesheets
// (tokens.css, motion.css) are linked at boot. When ./motion.js exports `cut(update, {from, to})`, it runs
// every screen change; otherwise the shell uses document.startViewTransition (cross-fade).
//
// Events (app.on): "screen" {id, previous}; "start-gesture" (fired synchronously inside the pointerdown /
// keydown on Start: create or resume the AudioContext here, iOS wants it inside the gesture); "start"
// (the hold completed, from rAF ~0.7 s later, still inside the transient user activation).

export const SCREENS = [
  { id: "welcome", title: "Welcome", nav: true },
  { id: "setup", title: "Setup", nav: true },
  { id: "people", title: "People", nav: true },
  { id: "listen", title: "Listen", nav: true },
  { id: "about", title: "About", nav: true },
  { id: "phone", title: "Phone", nav: false },
  { id: "diag", title: "Diagnostics", nav: false, sheet: true },
  { id: "sos", title: "Pair the band", nav: false }, // T-461: SOS remote pairing, linked from About
];
const DEFAULT = "welcome";
const byId = new Map(SCREENS.map((s) => [s.id, s]));
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

const listeners = new Map();
const state = { screen: null, cleanup: null, controller: null, sheet: null, sheetCleanup: null, token: 0 };

export const app = {
  screens: SCREENS,
  files: { screens: [], styles: [], modules: [], css: [] },
  store: null,
  ready: false,
  get screen() { return state.screen; },
  get sheet() { return state.sheet; },
  go(id) { location.hash = id; },
  openSheet(id = "diag") { location.hash = id; },
  // T-463: closing the sheet goes back in history when the sheet pushed the entry, so Back never reopens it
  closeSheet() {
    if (sheetPushed && history.state?.sheet) { sheetPushed = false; history.back(); return; }
    history.replaceState(null, "", `#${state.screen || DEFAULT}`);
    show(route());
  },
  on(event, fn) {
    if (!listeners.has(event)) listeners.set(event, new Set());
    listeners.get(event).add(fn);
    return () => listeners.get(event).delete(fn);
  },
  emit(event, detail) {
    for (const fn of listeners.get(event) || []) {
      try { fn(detail); } catch (err) { console.warn(`[app] ${event} listener failed`, err); }
    }
  },
};
window.hearingApp = app; // tests and the console poke at it

const $stage = document.getElementById("stage");
const $nav = document.getElementById("nav");
const $sheet = document.getElementById("sheet");
const $sheetBody = document.getElementById("sheet-body");
const $chrome = document.getElementById("chrome");
// T-404: below 900 px the Diagnostics sheet covers the whole screen (diag.css), so the page under it leaves the
// focus order (inert); wider, it is a side panel over a visible stage and the stage stays usable.
const sheetCovers = matchMedia("(max-width: 899.98px)");
let sheetOpener = null;
let sheetPushed = false; // the open sheet has its own history entry (opened by a hash change from a screen)
function coverUnder(on) {
  for (const node of [document.querySelector(".skip"), $chrome, $stage]) if (node) node.inert = on && sheetCovers.matches;
}
sheetCovers.addEventListener("change", () => coverUnder(Boolean(state.sheet)));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && state.sheet) app.closeSheet(); });

// ---------- files the server has ----------
async function loadFiles() {
  try {
    const res = await fetch("_files.json", { cache: "no-store" });
    if (res.ok) return await res.json();
  } catch { /* offline and not cached: everything is a placeholder except the built-in welcome */ }
  return { screens: [], styles: [], modules: [], css: [] };
}

function linkStyle(href) {
  const existing = document.querySelector(`link[data-href="${href}"]`);
  if (existing) return existing.ready;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = href;
  link.dataset.href = href;
  link.ready = new Promise((resolve) => { link.onload = link.onerror = () => resolve(); });
  document.head.append(link);
  return link.ready;
}

const modules = new Map();
function importScreen(id) {
  if (!modules.has(id)) {
    modules.set(id, (async () => {
      if (!app.files.screens.includes(id)) return { missing: true };
      try {
        const styled = app.files.styles.includes(id) ? linkStyle(`screens/${id}.css`) : null;
        const mod = await import(`./screens/${id}.js`);
        await styled;
        const mount = mod.mount || mod.default?.mount;
        if (typeof mount !== "function") return { error: `screens/${id}.js exports no mount(root, ctx)` };
        return { mount: mount.bind(mod.default || mod), title: mod.title || mod.default?.title };
      } catch (err) {
        console.warn(`[app] screens/${id}.js failed to load`, err);
        modules.delete(id); // retry on the next visit (a fixed file, a flaky network)
        return { error: String(err?.message || err) };
      }
    })());
  }
  return modules.get(id);
}

// ---------- routing ----------
function route() {
  const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
  const params = new URLSearchParams(location.search);
  if (byId.has(hash)) return hash;
  if (params.has("watch")) return "phone";
  return DEFAULT;
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

/** Nav: only screens this server has (no "Not in this build" dead ends, T-365 P0-7); welcome is built in. */
function renderNav() {
  const have = (s) => s.id === "welcome" || app.files.screens.includes(s.id);
  $nav.replaceChildren(...SCREENS.filter((s) => s.nav && have(s)).map((s, i) => el("a", {
    href: `#${s.id}`, "data-screen": s.id, class: "nav-item",
  }, el("span", { class: "nav-index", text: String(i + 1).padStart(2, "0") }), el("span", { class: "nav-label", text: s.title }))));
}

function markNav(id) {
  for (const a of $nav.querySelectorAll("a")) {
    if (a.dataset.screen === id) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
}

function placeholder(screen, reason) {
  const file = `screens/${screen.id}.js`;
  return el("section", { class: "placeholder", "data-placeholder": screen.id, "aria-labelledby": `ph-${screen.id}` },
    el("p", { class: "eyebrow", text: reason ? "Screen failed to load" : "Not in this build" }),
    el("h1", { id: `ph-${screen.id}`, class: "placeholder-title", text: screen.title }),
    el("p", { class: "placeholder-note" },
      reason ? `${file}: ${reason}` : "This screen's module is not on the server yet (",
      reason ? null : el("code", { text: file }), reason ? null : "). Nothing here is simulated."),
    el("p", { class: "placeholder-links" },
      el("a", { href: "#welcome", class: "link", text: "Back to the start" }), " · ",
      el("a", { href: "/judge", class: "link", text: "Stage page (/judge)" })));
}

async function render(screen, root, signal) {
  if (screen.id === "welcome" && !app.files.screens.includes("welcome")) return mountWelcome(root, signal);
  const loaded = await importScreen(screen.id);
  if (signal.aborted) return null;
  if (!loaded.mount) {
    root.replaceChildren(placeholder(screen, loaded.error));
    return null;
  }
  const ctx = { app, id: screen.id, params: new URLSearchParams(location.search), signal, store: app.store };
  try {
    const out = await loaded.mount(root, ctx);
    return typeof out === "function" ? out : out?.unmount ? () => out.unmount() : null;
  } catch (err) {
    console.warn(`[app] screens/${screen.id}.js mount failed`, err);
    root.replaceChildren(placeholder(screen, String(err?.message || err)));
    return null;
  }
}

function runCleanup(fn) {
  try { fn?.(); } catch (err) { console.warn("[app] screen cleanup failed", err); }
}

async function cut(update, from, to) {
  if (motionModule?.cut) {
    try { return await motionModule.cut(update, { from, to }); } catch (err) { console.warn("[app] motion cut failed", err); }
  }
  if (!document.startViewTransition || reducedMotion.matches || from === null) return update();
  const vt = document.startViewTransition(update);
  vt.ready.catch(() => undefined); // skipped transitions (tab hidden) are not errors
  return vt.updateCallbackDone;
}

async function show(id) {
  const screen = byId.get(id);
  if (screen.sheet) return showSheet(screen);
  if (state.sheet) hideSheet();
  if (state.screen === id) return;
  const token = ++state.token;
  const previous = state.screen;
  // import before the cut so the transition captures the finished new screen, not a blank frame
  if (screen.id !== "welcome" || app.files.screens.includes("welcome")) await importScreen(id);
  if (token !== state.token) return;
  await cut(async () => {
    state.controller?.abort();
    runCleanup(state.cleanup);
    state.cleanup = null;
    const controller = new AbortController();
    state.controller = controller;
    state.screen = id;
    document.body.dataset.screen = id;
    markNav(id);
    const root = el("div", { class: "screen", "data-screen": id });
    rescueBanner();
    $stage.replaceChildren(root);
    const cleanup = await render(screen, root, controller.signal);
    if (token === state.token) state.cleanup = cleanup;
    else runCleanup(cleanup);
    placeBanner();
    document.title = `${screen.title} · Selective Hearing`;
    // T-463: every screen opens at its top (Setup's scroll used to carry over to People)
    if (previous !== null) window.scrollTo({ top: 0, left: 0, behavior: "instant" });
  }, previous, id);
  if (token !== state.token) return;
  if (previous !== null && scrollY) window.scrollTo({ top: 0, left: 0, behavior: "instant" }); // nothing scrolled it during the cut
  if (previous !== null) $stage.focus({ preventScroll: true });
  app.emit("screen", { id, previous });
}

async function showSheet(screen) {
  const pushed = state.screen !== null; // a hash change from a screen pushed an entry; a deep link did not
  if (state.screen === null) await show("listen"); // a deep link to #diag opens over the stage screen
  if (state.sheet === screen.id) return;
  hideSheet();
  state.sheet = screen.id;
  sheetPushed = pushed;
  if (pushed) history.replaceState({ sheet: screen.id }, "", location.hash);
  const controller = new AbortController();
  sheetOpener = document.activeElement && document.activeElement !== document.body ? document.activeElement : null;
  $sheet.hidden = false;
  coverUnder(true);
  document.body.dataset.sheet = screen.id;
  document.title = `${screen.title} · Selective Hearing`; // T-403: the tab named the screen under the sheet
  const root = el("div", { class: "screen", "data-screen": screen.id });
  $sheetBody.replaceChildren(el("div", { class: "sheet-head" },
    el("button", { type: "button", class: "sheet-close", "aria-label": "Close diagnostics", text: "Close" })), root);
  $sheetBody.querySelector(".sheet-close").addEventListener("click", () => app.closeSheet());
  const cleanup = await render(screen, root, controller.signal);
  state.sheetCleanup = () => { controller.abort(); runCleanup(cleanup); };
  // keyboard and screen-reader users land in the sheet they opened (T-404), not on the page under it
  if (state.sheet === screen.id) $sheetBody.querySelector(".sheet-close")?.focus({ preventScroll: true });
  app.emit("screen", { id: screen.id, previous: state.screen });
}

function hideSheet() {
  if (!state.sheet) return;
  sheetPushed = false;
  runCleanup(state.sheetCleanup);
  state.sheetCleanup = null;
  state.sheet = null;
  $sheet.hidden = true;
  $sheetBody.replaceChildren();
  delete document.body.dataset.sheet;
  coverUnder(false);
  // focus goes back where it was before the sheet opened (the Diagnostics button, or the page for the D key)
  const back = sheetOpener?.isConnected ? sheetOpener : $stage;
  sheetOpener = null;
  back.focus({ preventScroll: true });
  const under = byId.get(state.screen);
  if (under) document.title = `${under.title} · Selective Hearing`;
}

// ---------- the shared device banner (web/console/devices.js) ----------
// T-463: devices.js appends it to <body> as a fixed red bar over the nav (it is that on /judge). In the app it sits in
// the screen, in flow and sticky at its top (app.css), so the nav stays clickable and the heading is never under it.
// Listen places it itself (its .ls-top); a screen change parks it on <body> first so replacing the stage keeps it.
const devBanner = () => document.querySelector("[data-dev-banner]");
function rescueBanner() {
  const b = devBanner();
  if (b && $stage.contains(b)) document.body.append(b);
}
function placeBanner() {
  const b = devBanner();
  const root = $stage.querySelector(":scope > .screen");
  if (b && root && !$stage.contains(b)) root.prepend(b);
}
new MutationObserver(() => { if (state.screen && devBanner()?.parentElement === document.body) placeBanner(); })
  .observe(document.body, { childList: true });

// ---------- built-in Welcome (until a screens/welcome.js lands) ----------
const HOLD_MS = 700;

function mountWelcome(root, signal) {
  const ring = el("span", { class: "hold-ring", "aria-hidden": "true" });
  const start = el("button", { type: "button", class: "hold", id: "start", "aria-describedby": "start-note" },
    ring, el("span", { class: "hold-label", text: "Start" }), el("span", { class: "hold-hint", text: "press and hold" }));
  const field = el("div", { class: "field", "data-live": "idle", "aria-hidden": "true" },
    ...[1, 2, 3, 4].map((i) => el("span", { class: "field-ring", style: `--i:${i}` })),
    el("span", { class: "field-core" }));
  root.append(el("section", { class: "welcome", "aria-labelledby": "welcome-title" },
    field,
    el("p", { class: "eyebrow welcome-eyebrow" }, el("span", { class: "dot", "aria-hidden": "true" }), "idle · microphone off"),
    el("h1", { id: "welcome-title", class: "display" },
      el("span", { class: "line", text: "Turn down the world" }),
      el("span", { class: "line dim", text: "without disconnecting from it." })),
    el("div", { class: "welcome-actions" }, start,
      el("p", { id: "start-note", class: "note", text: "A short delay behind the room, like live captions. Headphones on before you start." })),
    installHint()));

  let t0 = 0;
  let raf = 0;
  const set = (p) => start.style.setProperty("--p", p.toFixed(3));
  const cancel = () => {
    cancelAnimationFrame(raf);
    raf = 0;
    start.classList.remove("holding");
    set(0);
  };
  const done = () => {
    cancel();
    start.classList.add("done");
    app.emit("start");
    app.go("setup");
  };
  const tick = (now) => {
    const p = Math.min(1, (now - t0) / (reducedMotion.matches ? 1 : HOLD_MS));
    set(p);
    if (p >= 1) return done();
    raf = requestAnimationFrame(tick);
  };
  const begin = (e) => {
    if (e.type === "pointerdown" && e.button !== 0) return;
    if (e.type === "keydown" && (e.repeat || (e.key !== "Enter" && e.key !== " "))) return;
    if (e.type === "keydown") e.preventDefault();
    if (raf) return;
    app.emit("start-gesture");
    start.classList.add("holding");
    t0 = performance.now();
    raf = requestAnimationFrame(tick);
  };
  start.addEventListener("pointerdown", begin, { signal });
  start.addEventListener("keydown", begin, { signal });
  for (const ev of ["pointerup", "pointerleave", "pointercancel", "blur"]) start.addEventListener(ev, cancel, { signal });
  start.addEventListener("keyup", (e) => { if (e.key === "Enter" || e.key === " ") cancel(); }, { signal });
  start.addEventListener("click", (e) => e.preventDefault(), { signal });
  start.addEventListener("contextmenu", (e) => e.preventDefault(), { signal });
  return cancel;
}

function installHint() {
  const standalone = matchMedia("(display-mode: standalone)").matches || navigator.standalone === true;
  if (standalone || !matchMedia("(pointer: coarse)").matches) return null; // the stage laptop needs no install line
  const ios = /iP(hone|ad|od)/.test(navigator.userAgent);
  return el("p", { class: "install note", text: ios
    ? "On iPhone: Share, then Add to Home Screen, for a full-screen app."
    : "Install: use the browser's Install app button for a full-screen window." });
}

// ---------- PWA ----------
// app.sw: the registration's outcome, which is what says whether the certificate is trusted (T-365 P1-12);
// `controller` is null after a hard reload or before clients.claim(), so it says nothing about trust.
//   {state: "ok"} | {state: "untrusted"} (an SSL error fetching sw.js) | {state: "failed", message} | {state: "unsupported"}
app.sw = { state: "pending" };
export function workerOutcome(err) {
  if (!err) return { state: "ok" };
  const message = String(err?.message || err);
  if (err?.name === "SecurityError" && /certificate|ssl/i.test(message)) return { state: "untrusted" };
  return { state: "failed", message };
}
async function registerWorker() {
  if (!("serviceWorker" in navigator) || !window.isSecureContext) {
    app.sw = { state: "unsupported" };
  } else {
    try {
      await navigator.serviceWorker.register("sw.js", { scope: "./", updateViaCache: "none" });
      app.sw = workerOutcome(null);
    } catch (err) {
      app.sw = workerOutcome(err);
      if (app.sw.state !== "untrusted") console.warn("[app] service worker not registered", err);
    }
  }
  app.emit("sw", app.sw);
}

// Warm the other screen modules once the first screen has painted, so the first cut to each one is not a
// 55-475 ms long frame inside the View Transition (T-379 audit).
function warmScreens() {
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 400));
  const queue = app.files.screens.filter((id) => byId.has(id) && !modules.has(id));
  const next = () => {
    const id = queue.shift();
    if (!id) return;
    importScreen(id).finally(() => idle(next, { timeout: 2000 }));
  };
  idle(next, { timeout: 2000 });
}

// ---------- boot ----------
let motionModule = null;

async function boot() {
  app.files = await loadFiles();
  renderNav();
  await Promise.all(app.files.css.map((name) => linkStyle(name)));
  const optional = async (name) => {
    if (!app.files.modules.includes(name)) return null;
    try { return await import(`./${name}`); } catch (err) { console.warn(`[app] ${name} failed to load`, err); return null; }
  };
  let platform;
  [app.store, motionModule, platform] = await Promise.all([optional("store.js"), optional("motion.js"), optional("platform.js")]);
  // T-458: a touch-first device (CSS keys phone-only fixes on it; some Android browsers report a fine pointer)
  document.documentElement.toggleAttribute("data-touch", Boolean(platform?.touchDevice()));
  addEventListener("hashchange", () => show(route()));
  await show(route());
  app.ready = true;
  document.documentElement.classList.add("ready");
  registerWorker();
  warmScreens();
}

boot();
