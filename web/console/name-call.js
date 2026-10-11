// Name Call mode (T-120): whoever says the wearer's name becomes the target, no camera needed.
// Shared by /judge and /simple: the control message and the plain-language pill.
const KEY = "hearing.nameCall.name";

export function savedName() {
  try { return localStorage.getItem(KEY) || ""; } catch { return ""; }
}

export function saveName(text) {
  try { localStorage.setItem(KEY, text); } catch { /* private mode: keep it for this page only */ }
}

const GUARDS = "hearing.nameCall.guards";
const MATCHES = ["strict", "normal", "loose"];

/** False-trigger guards (T-123): `match` strict|normal|loose and `exclude` words that never count as the name. */
export function savedGuards() {
  try {
    const g = JSON.parse(localStorage.getItem(GUARDS) || "{}");
    return { match: MATCHES.includes(g.match) ? g.match : "normal", exclude: Array.isArray(g.exclude) ? g.exclude : [] };
  } catch { return { match: "normal", exclude: [] }; }
}

export function saveGuards(guards) {
  try { localStorage.setItem(GUARDS, JSON.stringify(guards)); } catch { /* private mode */ }
}

/** Wire a match `<select>` and an exclude `<input>` ("Ryan, Brian"); `onChange()` re-sends the control. */
export function mountNameGuards(select, input, onChange) {
  const g = savedGuards();
  select.value = g.match;
  input.value = g.exclude.join(", ");
  const update = () => {
    saveGuards({ match: select.value, exclude: input.value.split(",").map((s) => s.trim()).filter(Boolean).slice(0, 16) });
    onChange();
  };
  select.addEventListener("change", update);
  input.addEventListener("change", update);
}

/** `control` fields that start Name Call for `text` ("Aryan, Ary" -> name Aryan, alias Ary), with the saved guards. */
export function nameCallControl(text, suppression = 100, guards = savedGuards()) {
  const [name, ...aliases] = String(text || "").split(",").map((s) => s.trim()).filter(Boolean);
  return { mode: "name_call", suppression, name: { text: name || "Aryan", aliases, match: guards.match, exclude: guards.exclude } };
}

/** telemetry.name_call -> {state, text}: Listening for “X” → Heard “X” from a new voice — locking → Isolated → Target quiet. */
export function nameCallPill(t, fallbackName = "") {
  const nc = t?.name_call;
  const name = nc?.name || fallbackName || "your name";
  switch (nc?.state) {
    case "locking":
      return { state: "calibrating", text: nc.target_profile
        ? `Heard “${name}” — saved profile ${nc.target_profile}`
        : `Heard “${name}” from a new voice — locking` };
    case "isolated":
      return { state: "isolated", text: nc.target_profile ? `Isolated · saved profile ${nc.target_profile}` : "Isolated" };
    case "quiet":
      return { state: "speak", text: "Target quiet" };
    default:
      return { state: "look", text: `Listening for “${name}”` };
  }
}
