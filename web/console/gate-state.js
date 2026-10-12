// T-382: the live gate state, read from telemetry.tse (confidence, gated, soft_gated). Nothing is invented: with no
// field (no enrolled pick playing, older engine) the state is null and the pill stays hidden.
//   SOLO            the enrolled pick is confident and plays at full level
//   OVERLAP-DUCKED  an uncertain window played its best stream softened (HEARING_GATED_SOFT, T-370)
//   UNSURE          a gated window played silence, or confidence is below UNSURE_BELOW
//   LOCKED          T-487: a saved teammate's team slot plays every window (tse.locked), no per-window identity gate
export const UNSURE_BELOW = 0.5;
export const GATE_LABELS = { solo: "SOLO", ducked: "OVERLAP-DUCKED", unsure: "UNSURE", locked: "LOCKED", loudest: "ISOLATED \u00b7 closest voice" };

/**
 * HEARING_PICK=loudest: the engine plays only the loudest separated voice (the voice closest to the listener), no voice profile, no
 * gate. The field that says so is not fixed, so any string field equal to "loudest" in the telemetry (top level, tse,
 * extras, the tier health list, up to five levels deep) counts. Never true for an engine that reports nothing of the kind.
 */
export function loudestMode(t, depth = 0) {
  if (!t || typeof t !== "object" || depth > 5) return false;
  // the live engine reports it only per tier health (telemetry.tiers.tiers[].health[].pick, depth 5, inside arrays);
  // the session's top-level tse drops it, so arrays are searched too (else /judge read LOCKED / UNSURE)
  for (const v of Array.isArray(t) ? t : Object.values(t)) {
    if (typeof v === "string" ? v.toLowerCase() === "loudest" : loudestMode(v, depth + 1)) return true;
  }
  return false;
}

/** @returns {"solo"|"ducked"|"unsure"|"locked"|"loudest"|null} loudest mode never reports UNSURE */
export function gateState(t) {
  if (loudestMode(t)) return "loudest";
  const tse = t?.tse;
  if (!tse || tse.gated == null) return null;
  if (tse.locked) return "locked";
  if (tse.soft_gated) return "ducked";
  if (tse.gated) return "unsure";
  return tse.confidence != null && tse.confidence < UNSURE_BELOW ? "unsure" : "solo";
}

/** Writes the pill element: hidden unless `show` and a state is known. */
export function renderGate(el, t, show = true) {
  const s = show ? gateState(t) : null;
  el.hidden = !s;
  el.dataset.gate = s || "";
  const text = s ? GATE_LABELS[s] : "";
  if (el.textContent !== text) el.textContent = text; // 10 Hz telemetry: no DOM write when nothing changed
}
