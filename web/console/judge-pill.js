// Plain-language acquisition pill for the judge page (T-088): one prompt the presenter can read aloud.
// judgePill(telemetry, {cameraLost, connected}) -> {state, text, note}; `note` is the same-voice caveat.
const LIP_TIERS = new Set(["focus_hq", "av"]); // AV-TSE: separates similar voices by the lips

/** The Enrolled tier (voice only, no camera) is playing or ready to (T-135), locally or on the GPU. */
export function enrolledHealthy(t) {
  const tiers = t?.tiers;
  if (!tiers) return false;
  return tiers.active === "enrolled" || Boolean(tiers.tiers?.some((x) => x.name === "enrolled" && x.eligible));
}

/** The remote ladder's playing tier while the GPU's audio plays (T-156), else null. */
export function gpuTier(t) {
  if (t?.hq_active !== "remote" || !t.tiers?.remote || t.remote?.source === "local") return null;
  return t.tiers.active || null;
}

export function judgePill(t, { cameraLost = false, connected = true } = {}) {
  if (!connected) return { state: "lost", text: "Reconnecting to the engine…", note: "" };
  if (cameraLost && enrolledHealthy(t)) { // T-135: the voice profile needs no camera
    return { state: "isolated", text: "Camera lost — voice profile still isolating", note: "" };
  }
  if (cameraLost) return { state: "fallback", text: "Camera lost — noise suppression only", note: "" };
  if (!t?.acquisition) return { state: "idle", text: "Press Start", note: "" };
  const { state, detail = "" } = t.acquisition;
  const cal = /^calibrating ([\d.]+)\/([\d.]+) s/.exec(detail);
  let out;
  if (cal) {
    out = { state: "calibrating", text: `Calibrating ${Math.min(Number(cal[1]), Number(cal[2])).toFixed(0)}/${Number(cal[2]).toFixed(0)} s` };
  } else if (state === "locked" && detail.startsWith("known voice: ")) {
    // T-125: a pre-recorded voice profile is playing from the first window
    out = { state: "isolated", text: `Isolated — ${detail}` };
  } else if (state === "acquiring" && detail.startsWith("loading saved voice: ")) {
    // T-329 (audio_only): the saved voice is chosen, the Enrolled tier is still starting (~4 s cold)
    out = { state: "loading", text: `Loading voice — ${detail.slice("loading saved voice: ".length)}` };
  } else if (state === "acquiring" && /^saved voice .+ not playing: /.test(detail)) {
    // T-329: the ladder holds Enrolled out (e.g. overrun): what plays is not isolated
    out = { state: "fallback", text: `Voice profile paused — ${detail.replace(/^saved voice .+ not playing: /, "")}` };
  } else if (state === "locked") {
    out = { state: "isolated", text: "Isolated — now let the other person talk" };
  } else if (state === "searching" && detail.startsWith("lost the person you locked")) {
    out = { state: "reacquiring", text: "Lost the person you locked — look at them again" };
  } else if (state === "reacquiring") {
    out = { state: "reacquiring", text: "Look at the person again" };
  } else if (state === "locked-visual" && t.auto_calibrate?.state === "done") {
    // T-135: the clip is in, the ladder is on its way to Enrolled: no step back to "speak"
    out = { state: "calibrating", text: "Calibrated — locking on…" };
  } else if (state === "acquiring" || state === "locked-visual") {
    out = { state: "speak", text: "Ask them to say one sentence" };
  } else {
    out = { state: "look", text: "Look at the person" };
  }
  const gpu = gpuTier(t);
  if (gpu) out.text += ` · iLab GPU · ${gpu}`; // T-156: what the GPU plays, as for a Mac session
  out.note = "";
  if (t.tiers?.same_bucket) {
    // Enrolled (voice-only) is skipped for two similar voices; the GPU's AV tier still separates them.
    out.note = LIP_TIERS.has(t.tiers.active) || t.hq_active === "remote"
      ? "similar voices — using lips" : "similar voices — limited isolation";
  }
  return out;
}

export const MIC_STALL_MS = 2000; // T-158: Start pressed, connected, no audio frame sent for this long
export const MIC_START_TIMEOUT_MS = 8000; // T-158: getUserMedia / addModule still pending -> banner, Start back

/**
 * T-158: the pill must never look live while the mic is not streaming. Server telemetry flows within
 * milliseconds of connecting, so the acquisition text alone ("Look at the person") reads as live.
 * Returns the pill to show instead of judgePill's, or null when the mic is streaming (or replaying).
 * `sinceFrameMs`: ms since the last audio frame went out (or since Start finished, before the first).
 */
export function micPill({ connected = true, started = false, replaying = false, sinceFrameMs = 0 } = {}) {
  if (!connected || replaying) return null;
  if (!started) return { state: "idle", text: "Press Start", note: "" };
  if (!(sinceFrameMs <= MIC_STALL_MS)) {
    return { state: "nomic", text: "Mic not streaming — check the mic, then press Start again", note: "" };
  }
  return null;
}

/** T-158: the red banner line for a Start that failed (getUserMedia rejected, or hung past the timeout). */
export function micProblem(err) {
  const name = err?.name || "";
  const msg = err?.message || String(err || "");
  if (name === "TimeoutError") {
    return `Microphone did not start within ${MIC_START_TIMEOUT_MS / 1000} s (permission prompt or audio engine stuck): `
      + "allow the mic for this page, then press Start again.";
  }
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Microphone blocked: allow it for this page (address bar → site settings → Microphone), then press Start.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No microphone found: plug in or pick MacBook Pro Microphone, then press Start.";
  }
  if (name === "NotReadableError") return "Microphone busy or unreadable (another app has it?): free it, then press Start.";
  return `Microphone unavailable: ${msg || name || "unknown error"}. Press Start to try again.`;
}

/** Reject with a TimeoutError if `promise` has not settled after `ms`. */
export function withTimeout(promise, ms, setTimer = setTimeout, clearTimer = clearTimeout) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimer(() => {
      const e = new Error(`still pending after ${ms} ms`);
      e.name = "TimeoutError";
      reject(e);
    }, ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimer(timer));
}

export const PILL_HOLD_MS = 500; // T-135: a non-isolated state shows only after this long

/**
 * Pill hysteresis (T-135): a change to `isolated` shows at once; any other state must be what judgePill
 * says for PILL_HOLD_MS before it replaces the shown one, so one dropped window never flickers the
 * pill. Text updates inside the shown state (Calibrating 1/2 -> 2/2) apply at once. `changes` counts
 * displayed state changes (scripts/judge_check.py asserts <= 3 from Start to Isolated).
 */
export class PillHysteresis {
  constructor(holdMs = PILL_HOLD_MS) {
    this.holdMs = holdMs;
    this.shown = null;
    this.pending = null; // {state, since}
    this.changes = 0;
    this.history = [];
  }

  update(pill, nowMs) {
    if (this.shown === null || pill.state === this.shown.state) {
      this.pending = null;
      if (this.shown === null) this.history.push(pill.state);
      this.shown = pill;
      return pill;
    }
    if (pill.state !== "isolated") {
      if (this.pending?.state !== pill.state) this.pending = { state: pill.state, since: nowMs };
      if (nowMs - this.pending.since < this.holdMs) return this.shown;
    }
    this.pending = null;
    this.shown = pill;
    this.changes++;
    this.history.push(pill.state);
    if (this.history.length > 50) this.history.shift();
    return pill;
  }
}

/** Drawer numbers (T-135): an EMA (`alpha`) rounded to `step`, so they stop flickering every 100 ms. */
export class Steady {
  constructor(step = 50, alpha = 0.2) {
    this.step = step;
    this.alpha = alpha;
    this.value = null;
  }

  update(x) {
    if (!Number.isFinite(x)) return this.value == null ? null : Math.round(this.value / this.step) * this.step;
    this.value = this.value == null ? x : this.value + this.alpha * (x - this.value);
    return Math.round(this.value / this.step) * this.step;
  }

  reset() { this.value = null; }
}
