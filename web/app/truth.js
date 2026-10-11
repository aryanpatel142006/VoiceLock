// Truth strip (M14, T-353): the always-visible honesty line on Listen: Backend (Mac / GPU), Delay (Steady EMA),
// RTT, Session / recording, and the REPLAY badge. Diagnostics (screens/diag.js) reuses the formatters.
//
// Every value is read from `store.state.truth` (store.js truthStrip): engine telemetry or the page's own
// measurements. A value nobody measured yet is null and renders as "—"; the number ticker only ever moves to a
// measured value, it never invents a path between two of them.
//
// Listen (screens/listen.js) mounts it: `const off = mountTruth(el, { store, signal })`, where `store` is the
// store.js namespace (`ctx.store`) or the Store itself. Styles live in screens/diag.css; mountTruth links it.

export const DASH = "—";
const STYLE_HREF = "screens/diag.css"; // same data-href app.js uses, so the sheet and the strip never link it twice

/** "—" until measured; 1480 -> "1.48 s", 52 -> "52 ms". */
export function fmtMs(ms) {
  if (!Number.isFinite(ms)) return DASH;
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

/** The first 8 characters of a session id, "—" without one. */
export function shortId(id) {
  return id ? String(id).slice(0, 8) : DASH;
}

/** "Mac" / "GPU" with the reason the engine gave; fallback = the GPU was asked for and the Mac plays. */
export function backendText(truth) {
  if (!truth?.backend) return { value: DASH, sub: "", tone: "" };
  const sub = [truth.device, truth.hqActive ? `hq_active ${truth.hqActive}` : ""].filter(Boolean).join(" · ");
  return { value: truth.backend, sub, tone: truth.fallback ? "warn" : "" };
}

/**
 * The strip's items from `store.state.truth`: {key, label, num, fmt, value, sub, tone}. `num` is the number the
 * ticker moves to (null = "—"); `value` is what is shown once settled.
 */
export function truthItems(truth) {
  const t = truth || {};
  const backend = backendText(t);
  const rec = t.recording === true ? "recording" : t.recording === false ? "not recording" : "";
  return [
    { key: "backend", label: "Backend", num: null, value: backend.value, sub: backend.sub, tone: backend.tone,
      title: t.hqReason || "" },
    { key: "delay", label: "Delay", num: Number.isFinite(t.delayMs) ? t.delayMs : null, fmt: fmtMs,
      value: fmtMs(t.delayMs), sub: t.latencyPreset || "", tone: "" },
    { key: "rtt", label: "RTT", num: Number.isFinite(t.rttMs) ? t.rttMs : null, fmt: fmtMs, value: fmtMs(t.rttMs),
      sub: Number.isFinite(t.gpuRttMs) ? `GPU ${fmtMs(t.gpuRttMs)}` : "", tone: "" },
    { key: "session", label: "Session", num: null, value: shortId(t.sessionId), sub: rec,
      tone: t.recording === true ? "rec" : "" },
  ];
}

// ---------- number ticker (Kalkbrenner's year counter; ours only ticks to measured values) ----------
const reduced = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Show `num` formatted by `fmt` in `el`. The first value, a return from "—" and reduced motion jump straight to
 * it; a change between two measured values counts across `ms` (never longer than a telemetry tick or two).
 */
export function setNumber(el, num, fmt = fmtMs, ms = 220) {
  const state = (el._ticker ||= { shown: null, raf: 0 });
  cancelAnimationFrame(state.raf);
  if (!Number.isFinite(num)) {
    state.shown = null;
    el.textContent = DASH;
    return;
  }
  const from = state.shown;
  if (from == null || from === num || reduced()) {
    state.shown = num;
    el.textContent = fmt(num);
    return;
  }
  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min(1, (now - t0) / ms);
    const eased = 1 - (1 - p) ** 3;
    state.shown = p >= 1 ? num : from + (num - from) * eased;
    el.textContent = fmt(state.shown);
    state.raf = p >= 1 ? 0 : requestAnimationFrame(step);
  };
  state.raf = requestAnimationFrame(step);
}

// ---------- DOM ----------
function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

/** Link screens/diag.css once (the strip is on Listen, before the sheet is ever opened). */
export function ensureStyles() {
  if (typeof document === "undefined" || document.querySelector(`link[data-href="${STYLE_HREF}"]`)) return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = STYLE_HREF;
  link.dataset.href = STYLE_HREF;
  document.head.append(link);
}

/** `store` as passed by the shell is the store.js namespace; accept either form. */
export function storeOf(store) {
  return store?.store || store || null;
}

/**
 * Mount the strip into `host`. Returns an unmount function. Safe to call twice on one host (the first wins).
 * Re-renders only when the store's truth, replay or connection state changed, coalesced to one frame.
 */
export function mountTruth(host, { store, signal } = {}) {
  const s = storeOf(store);
  const existing = host.querySelector(":scope > [data-truth-strip]");
  if (existing) return () => {};
  ensureStyles();
  const strip = node("div", "truth-strip");
  strip.dataset.truthStrip = "";
  const badge = node("span", "truth-replay", "Replay");
  badge.setAttribute("role", "status");
  badge.hidden = true;
  const list = node("dl", "truth");
  list.setAttribute("aria-label", "Engine truth: backend, delay, round trip, session");
  const cells = new Map();
  for (const item of truthItems(null)) {
    const wrap = node("div", "truth-item");
    wrap.dataset.key = item.key;
    const val = node("span", "truth-val", DASH);
    const sub = node("span", "truth-sub");
    const dd = node("dd", "truth-dd");
    dd.append(val, sub);
    wrap.append(node("dt", "truth-dt", item.label), dd);
    list.append(wrap);
    cells.set(item.key, { wrap, val, sub });
  }
  strip.append(badge, list);
  host.append(strip);

  const render = () => {
    const st = s?.state;
    const truth = st?.truth;
    strip.dataset.stale = st && st.connection !== "open" ? "true" : "false";
    badge.hidden = !truth?.replay;
    for (const item of truthItems(truth)) {
      const cell = cells.get(item.key);
      cell.wrap.dataset.tone = item.tone || "";
      cell.wrap.title = item.title || "";
      cell.sub.textContent = item.sub;
      cell.sub.hidden = !item.sub;
      if (item.fmt) setNumber(cell.val, item.num, item.fmt);
      else cell.val.textContent = item.value;
    }
  };
  let raf = 0;
  const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; render(); }); };
  const off = s?.on?.("change", ({ keys }) => {
    if (keys.some((k) => k === "truth" || k === "connection" || k === "replaying")) schedule();
  });
  render();
  let gone = false;
  const unmount = () => {
    if (gone) return;
    gone = true;
    off?.();
    cancelAnimationFrame(raf);
    for (const { val } of cells.values()) cancelAnimationFrame(val._ticker?.raf || 0);
    strip.remove();
  };
  signal?.addEventListener("abort", unmount, { once: true });
  return unmount;
}

export default { mountTruth };
