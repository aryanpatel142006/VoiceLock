// T-382: the live gate state, read from telemetry.tse (confidence, gated, soft_gated). Nothing is invented: with no
// field (no enrolled pick playing, older engine) the state is null and the pill stays hidden.
//   SOLO            the enrolled pick is confident and plays at full level
//   OVERLAP-DUCKED  an uncertain window played its best stream softened (HEARING_GATED_SOFT, T-370)
//   UNSURE          a gated window played silence, or confidence is below UNSURE_BELOW
export const UNSURE_BELOW = 0.5;
export const GATE_LABELS = { solo: "SOLO", ducked: "OVERLAP-DUCKED", unsure: "UNSURE" };

/** @returns {"solo"|"ducked"|"unsure"|null} */
export function gateState(t) {
  const tse = t?.tse;
  if (!tse || tse.gated == null) return null;
  if (tse.soft_gated) return "ducked";
  if (tse.gated) return "unsure";
  return tse.confidence != null && tse.confidence < UNSURE_BELOW ? "unsure" : "solo";
}

/** Writes the pill element: hidden unless `show` and a state is known. */
export function renderGate(el, t, show = true) {
  const s = show ? gateState(t) : null;
  el.hidden = !s;
  el.dataset.gate = s || "";
  el.textContent = s ? GATE_LABELS[s] : "";
}
