// T-485: the live-test pre-flight on Start under the team preset (stage rig only). Live test, Oct 11: at 4:28 the
// output was the MacBook speakers and the iPhone mic heard them (feedback); at 4:11 the iPhone mic had dropped; at
// 4:19 agent jobs overloaded the Mac (42/64 windows overran). Start is refused for the first two; the third is a
// warning from real telemetry (the server's load average, the enrolled tier's overruns). Pure; session.js runs it.
import { isBluetoothName } from "../console/devices.js";

export const MIC_LIVE_DB = -45; // the mic must peak above this within MIC_WAIT_MS
export const MIC_WAIT_MS = 3000;
export const SPEAKERS_TEXT = "Connect your AirPods: playing through the laptop speakers causes feedback";
export const DEAD_MIC_TEXT = `We can't hear the mic: nothing above ${MIC_LIVE_DB} dBFS in ${MIC_WAIT_MS / 1000} s. Check it, then Start again.`;
export const BUSY_TEXT = "The laptop is busy: close other jobs";

const HEADPHONES = /air ?pods|head ?(phone|set)|ear ?(phone|bud)|buds|beats/i;
const labelHas = (label, want) => String(label).toLowerCase().includes(String(want).toLowerCase());

/** The pre-flight runs under the team preset (telemetry carries its pins) on the stage rig, never on a phone. */
export function preflightApplies(telemetry, stageRig = true) {
  return stageRig !== false && Boolean(telemetry?.preset?.pins);
}

/**
 * Why this output may not play the demo, or "" when it may. Speakers (the laptop's, a display's) are refused unless
 * the label also names AirPods / Bluetooth / headphones or the preset's pinned output. Any other label passes, and
 * so does an unknown one ("": the browser cannot name its output): devices.js still flags an output other than the
 * pinned one, and refusing every unrecognised name would block a check browser's fake output.
 */
export function outputProblem(label, expected = null) {
  const out = String(label || "").trim();
  if (!out || isBluetoothName(out) || HEADPHONES.test(out) || (expected && labelHas(out, expected))) return "";
  return /speaker/i.test(out) ? SPEAKERS_TEXT : "";
}

/**
 * One pre-flight step. `elapsedMs` since the mic opened, `level` the meter's peak (dBFS), `output` the output label.
 * Returns {done, problem}: done with "" passes, done with a problem refuses Start, not done keeps waiting.
 */
export function preflightStep({ elapsedMs, level, output, expectedOutput = null }) {
  const out = outputProblem(output, expectedOutput);
  if (out) return { done: true, problem: out };
  if (level > MIC_LIVE_DB) return { done: true, problem: "" };
  if (elapsedMs >= MIC_WAIT_MS) return { done: true, problem: DEAD_MIC_TEXT };
  return { done: false, problem: "" };
}

/**
 * Whether the engine's machine is too busy, from telemetry only: `host_load.busy` (the server's 1-minute load
 * average per core >= 1) or a tier's stage unhealthy for overrunning / late windows, or computing slower than its
 * hop. Returns {busy, why} (`why` names the measurement for the banner's title and Diagnostics).
 */
export function busyState(telemetry) {
  const load = telemetry?.host_load;
  if (load?.busy) return { busy: true, why: `load ${load.load1} on ${load.cpus} cores` };
  for (const tier of telemetry?.tiers?.tiers || []) {
    for (const h of tier.health || []) {
      if (h && (h.reason === "overrun" || h.reason === "late")) {
        return { busy: true, why: `${tier.name}: ${h.reason} (${h.overruns ?? 0} overruns in ${h.windows ?? 0} windows)` };
      }
      if (h && h.utilisation >= 1) return { busy: true, why: `${tier.name}: compute ${Math.round(100 * h.utilisation)} % of its hop` };
    }
  }
  return { busy: false, why: "" };
}
