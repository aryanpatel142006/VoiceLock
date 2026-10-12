// Diagnostics sheet (#diag, key D; M14, T-353). Opens over Listen. Rows: backend (Mac / GPU + hq_active), tier,
// delay EMA with "Steady", RTT, confidence meter, preset pins, server rev warning, fallback log, scene frame time.
//
// Honesty: every row is "—" until its field arrives in telemetry (or the page measures it). Nothing is estimated
// here. Sources: `store.state.truth` (store.js truthStrip), `store.state.telemetry`, `store.state.log` and, for the
// frame time, `app.scene.stats()` -> {p95Ms, lastMs, frames} (scene.js measures it; the row stays "—" until then).

import { DASH, backendText, fmtMs, setNumber, storeOf } from "../truth.js";
import { SETTLING, SteadyWatch, fmtDelay } from "../store.js";

export const title = "Diagnostics";

const SAMPLE_MS = 500;
const SCENE_BUDGET_MS = 4; // tokens.css --scene-budget-ms; DESIGN_SPEC section 5.5

// T-463: the steadiness rule moved to store.js (one watch the store runs, read by every screen as truth.delaySteady)
export { SteadyWatch };

const PIN_LABELS = [
  ["hq_backend", "Focus HQ backend"],
  ["same_bucket_guard", "Same-voice guard"],
  ["lock_on_start", "Lock on start"],
  ["hold_lock_s", "Hold lock", (v) => `${v} s`],
  ["synth_feed", "Synthetic feed"],
  ["asr", "Captions ASR"],
  ["audio_only", "Audio only"],
  ["remote_git_rev", "GPU rev"],
];

function pinValue(v, fmt) {
  if (typeof v === "boolean") return v ? "on" : "off";
  return fmt ? fmt(v) : String(v);
}

/** The preset pins as [label, value] rows; the session's live latency preset and isolation come first. */
export function pinRows(telemetry, truth) {
  const rows = [];
  if (truth?.latencyPreset) rows.push(["Latency preset", truth.latencyPreset]);
  if (truth?.isolation) rows.push(["Isolation", truth.isolation]);
  const pins = telemetry?.preset?.pins;
  if (pins) {
    for (const [key, label, fmt] of PIN_LABELS) {
      if (pins[key] != null && pins[key] !== "") rows.push([label, pinValue(pins[key], fmt)]);
    }
    if (telemetry.preset.name) rows.unshift(["Preset", telemetry.preset.name]);
  }
  return rows;
}

/** The server rev row: the engine's own warning, else "No warning" once telemetry flows, else "—". */
export function revInfo(telemetry, truth) {
  if (!telemetry) return { value: DASH, sub: "", tone: "" };
  const rev = telemetry.remote?.git_rev || telemetry.preset?.pins?.remote_git_rev || truth?.gitRev || null;
  const sub = rev ? `GPU rev ${String(rev).slice(0, 7)}` : "";
  if (truth?.revWarning) return { value: truth.revWarning, sub, tone: "warn" };
  return { value: "No warning", sub, tone: "" };
}

/** The tier row: label + the ladder's own reason, "same voice" note when the engine says so. */
export function tierInfoRow(telemetry, truth) {
  if (!truth?.tier || truth.tier === DASH) return { value: DASH, sub: "", tone: "" };
  const reason = telemetry?.tiers?.reason || "";
  const sub = [reason, truth.sameBucket ? "similar voices, limited isolation" : ""].filter(Boolean).join(" · ");
  return { value: truth.tier, sub, tone: truth.sameBucket ? "warn" : "" };
}

/** Scene frame time (measured by scene.js): p95 against the 4 ms budget. */
export function sceneInfo(stats) {
  if (!stats || !Number.isFinite(stats.p95Ms)) return { value: DASH, sub: "scene not reporting", tone: "" };
  const last = Number.isFinite(stats.lastMs) ? ` · last ${stats.lastMs.toFixed(1)} ms` : "";
  const frames = Number.isFinite(stats.frames) ? ` · ${stats.frames} frames` : "";
  return { value: `${stats.p95Ms.toFixed(1)} ms`, sub: `p95${last}${frames}`, tone: stats.p95Ms > SCENE_BUDGET_MS ? "warn" : "" };
}

// ---------- DOM ----------
function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

const SECTIONS = [
  { title: "Engine", rows: ["backend", "tier", "rev"] },
  { title: "Timing", rows: ["delay", "rtt", "scene"] },
  { title: "Target", rows: ["confidence"] },
];
const LABELS = {
  backend: "Backend", tier: "Tier", rev: "Server rev", delay: "Engine delay", rtt: "Network round trip", scene: "Scene frame time",
  confidence: "Voice match",
};

function row(key) {
  const wrap = node("div", "diag-row");
  wrap.dataset.key = key;
  const dt = node("dt", "diag-dt", LABELS[key]);
  const dd = node("dd", "diag-dd");
  const val = node("span", "diag-val", DASH);
  const sub = node("span", "diag-sub");
  const flag = node("span", "diag-flag");
  flag.hidden = true;
  dd.append(val, flag, sub);
  wrap.append(dt, dd);
  return { wrap, val, sub, flag, dd };
}

export function mount(root, ctx = {}) {
  const store = storeOf(ctx.store);
  const signal = ctx.signal;
  const section = node("section", "diag");
  section.setAttribute("aria-labelledby", "diag-title");
  const head = node("header", "diag-head");
  head.append(Object.assign(node("h2", "diag-title", "What the engine says"), { id: "diag-title" }),
    node("p", "diag-note", store ? "Every value is from engine telemetry or measured on this page. — means not reported yet."
      : "The app store is not on this server, so there is nothing to show. Nothing here is simulated."));
  section.append(head);

  const cells = {};
  let index = 0;
  for (const sec of SECTIONS) {
    const block = node("section", "diag-block");
    block.append(node("h3", "diag-h", `${String(++index).padStart(2, "0")} ${sec.title}`));
    const dl = node("dl", "diag-list");
    for (const key of sec.rows) {
      cells[key] = row(key);
      dl.append(cells[key].wrap);
    }
    block.append(dl);
    section.append(block);
  }
  // confidence meter: a hairline track, the fill is a transform (no layout)
  const meter = node("div", "diag-meter");
  meter.setAttribute("role", "meter");
  meter.setAttribute("aria-label", "Target confidence");
  meter.setAttribute("aria-valuemin", "0");
  meter.setAttribute("aria-valuemax", "1");
  const fill = node("span", "diag-meter-fill");
  meter.append(fill);
  cells.confidence.dd.prepend(meter);

  const pinsBlock = node("section", "diag-block");
  pinsBlock.append(node("h3", "diag-h", `${String(++index).padStart(2, "0")} Preset pins`));
  const pinList = node("dl", "diag-list diag-pins");
  pinsBlock.append(pinList);
  const logBlock = node("section", "diag-block");
  logBlock.append(node("h3", "diag-h", `${String(++index).padStart(2, "0")} Fallback log`));
  const logList = node("ol", "diag-log");
  logList.setAttribute("aria-label", "Fallback log, newest first");
  logBlock.append(logList);
  section.append(pinsBlock, logBlock);
  root.append(section);

  const setRow = (key, info) => {
    const c = cells[key];
    c.wrap.dataset.tone = info.tone || "";
    c.val.textContent = info.value;
    c.sub.textContent = info.sub || "";
    c.sub.hidden = !info.sub;
  };

  let pinKey = "";
  let logKey = null;
  const render = () => {
    const st = store?.state;
    const truth = st?.truth;
    const t = st?.telemetry || null;
    section.dataset.stale = st && st.connection !== "open" ? "true" : "false";

    setRow("backend", { ...backendText(truth), sub: [backendText(truth).sub, truth?.hqReason].filter(Boolean).join(" · ") });
    setRow("tier", tierInfoRow(t, truth));
    setRow("rev", revInfo(t, truth));

    // T-463: the same number and unit as Listen and Setup (store.js fmtDelay), "settling…" until it holds
    const steady = truth?.delaySteady || null;
    if (steady === "steady") setNumber(cells.delay.val, truth.delayMs, fmtDelay);
    else {
      setNumber(cells.delay.val, null);
      if (steady) cells.delay.val.textContent = SETTLING;
    }
    cells.delay.wrap.dataset.tone = "";
    cells.delay.sub.textContent = truth?.latencyPreset ? `${truth.latencyPreset} preset · engine, before device buffers` : "";
    cells.delay.sub.hidden = !truth?.latencyPreset;
    cells.delay.flag.hidden = steady !== "steady";
    cells.delay.flag.textContent = "Steady";
    cells.delay.flag.dataset.state = steady || "";

    // T-463: the steady RTT moves in 10 ms steps; under one step reads "<10 ms", as on Listen
    setNumber(cells.rtt.val, Number.isFinite(truth?.rttMs) ? truth.rttMs : null, (v) => (v < 10 ? "<10 ms" : fmtMs(v)));
    cells.rtt.sub.textContent = Number.isFinite(truth?.gpuRttMs) ? `page ↔ engine · laptop ↔ GPU ${fmtMs(truth.gpuRttMs)}` : "";
    cells.rtt.sub.hidden = !cells.rtt.sub.textContent;

    const conf = Number.isFinite(truth?.confidence) ? Math.min(1, Math.max(0, truth.confidence)) : null;
    cells.confidence.val.textContent = conf == null ? DASH : conf.toFixed(2);
    cells.confidence.sub.textContent = conf == null ? "" : t?.tse?.source === "remote" ? "TSE on the GPU" : "TSE";
    cells.confidence.sub.hidden = !cells.confidence.sub.textContent;
    fill.style.transform = `scaleX(${conf ?? 0})`;
    if (conf == null) meter.removeAttribute("aria-valuenow");
    else meter.setAttribute("aria-valuenow", conf.toFixed(2));
    meter.dataset.empty = conf == null ? "true" : "false";

    setRow("scene", sceneInfo(ctx.app?.scene?.stats?.()));

    const pins = pinRows(t, truth);
    const nextPins = JSON.stringify(pins) + (t ? "t" : "");
    if (nextPins !== pinKey) {
      pinKey = nextPins;
      const items = pins.length ? pins : [[t ? "Team preset" : "Preset", t ? "none on this session" : DASH]];
      pinList.replaceChildren(...items.map(([k, v]) => {
        const w = node("div", "diag-row");
        w.append(node("dt", "diag-dt", k), node("dd", "diag-dd", v));
        return w;
      }));
    }

    const log = st?.log || [];
    const nextLog = log.length ? `${log.length}:${log[log.length - 1].at}:${log[log.length - 1].text}` : "";
    if (nextLog !== logKey) {
      logKey = nextLog;
      logList.replaceChildren(...(log.length ? [...log].reverse().map((l) => {
        const li = node("li", "diag-log-line");
        li.append(node("time", "diag-log-at", l.at || DASH), node("span", "diag-log-text", l.text));
        return li;
      }) : [Object.assign(node("li", "diag-log-empty", DASH), { title: "No fallback or error events this session" })]));
    }
  };

  let raf = 0;
  const schedule = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; render(); }); };
  const offs = [store?.on?.("change", schedule), store?.on?.("note", schedule)].filter(Boolean);
  const timer = setInterval(render, SAMPLE_MS); // Steady needs time to pass; the scene reports without events
  render();

  const onKey = (e) => {
    if (e.key === "Escape" && !e.defaultPrevented) ctx.app?.closeSheet?.();
  };
  addEventListener("keydown", onKey, signal ? { signal } : undefined);

  return () => {
    clearInterval(timer);
    cancelAnimationFrame(raf);
    for (const off of offs) off();
    removeEventListener("keydown", onKey);
    for (const c of Object.values(cells)) cancelAnimationFrame(c.val._ticker?.raf || 0);
  };
}

export default { mount, title };
