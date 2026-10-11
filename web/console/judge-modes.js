// T-150: the two big /judge modes. Focus plays the isolated voice; Quiet plays nothing and captions
// the target. Pure helpers (no DOM) so node --test covers them; judge.js wires them to buttons and F/Q.

export const MODES = {
  focus_hq: { key: "f", title: "Focus", copy: "hear who matters" },
  quiet: { key: "q", title: "Quiet", copy: "stay connected without listening" },
};

// T-149: the engine's `quiet` mode is digital silence + captions from the isolated target only.
export const QUIET_CONTROL = { mode: "quiet" };

export const DEGRADED_NOTICE = "Captions are hearing the whole room — the target voice is not isolated";

/** The control fields for `mode`; Focus after a camera loss without a healthy Enrolled tier plays Natural. */
export function modeControl(mode, { cameraNatural = false } = {}) {
  if (mode === "quiet") return { ...QUIET_CONTROL };
  return { mode: cameraNatural ? "natural" : "focus_hq" };
}

/** The mode a key press asks for, or null. */
export function modeForKey(key) {
  const k = String(key || "").toLowerCase();
  return Object.keys(MODES).find((m) => MODES[m].key === k) || null;
}

/** Quiet's caption header: who is captioned and whether the feed fell back to the mixture. */
export function captionFeed(t) {
  return {
    speaker: t?.enrollment?.profile || "Target",
    degraded: t?.synth_feed === "mixture",
    paused: t?.quiet?.feed === "paused" ? t.quiet.reason || "captions paused" : "",
  };
}
