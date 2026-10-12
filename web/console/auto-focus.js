// Auto voice focus (board Q-2010): the one control left on the stage. Nobody chooses a name; the engine
// (HEARING_PICK=loudest) isolates the loudest voice, which is the voice closest to the listener, and silences the rest.
//   ON  = the processed side of the existing A/B switch (`ab: "processed"`)
//   OFF = the raw side (`ab: "raw"`): the whole room, unprocessed
// No new message: the switch sends the same `{ab}` control as Space always did. Its state is read back from the
// engine's telemetry (`t.ab`) so the switch never claims what is not playing; until telemetry names a side, the
// page's own last request stands in, flagged `pending`.

export const AUTO_FOCUS_LABEL = "Auto voice focus";
export const AUTO_FOCUS_ON = "Isolated · closest voice";
export const AUTO_FOCUS_ON_DETAIL = "The voice closest to you comes through, everything else goes quiet.";
export const AUTO_FOCUS_OFF = "Off · hearing the whole room";

/**
 * @param {string|undefined} serverAb  telemetry `ab` ("raw" | "processed"), or nothing yet
 * @param {string|undefined} localAb   what this page last asked for
 * @returns {{on: boolean, ab: "raw"|"processed", pending: boolean, line: string, next: "raw"|"processed"}}
 */
export function autoFocusState(serverAb, localAb) {
  const known = serverAb === "raw" || serverAb === "processed";
  const ab = known ? serverAb : localAb === "raw" ? "raw" : "processed";
  const on = ab === "processed";
  const asked = localAb === "raw" || localAb === "processed" ? localAb : ab;
  return { on, ab, pending: asked !== ab, line: on ? AUTO_FOCUS_ON : AUTO_FOCUS_OFF, next: on ? "raw" : "processed" };
}
