// Manual voice calibration + strict isolation (T-085), shared by the simple page and the console.
// "Calibrate voice (3 s)" sends control.enroll {action: "start", seconds: 3}: ask the person to say
// one sentence alone; every voiced chunk of the next 3 s becomes their enrollment. "Strict" sends
// control.isolation: fewer seconds of the other person, a few more short dropouts.
const SECONDS = 3;
const KEY = "hearing.isolation";

export function mountCalibration(anchor, sendControl) {
  const box = document.createElement("span");
  box.className = "calibrate";
  box.innerHTML =
    `<button type="button" class="cal-btn" title="Ask them to say one sentence, then press (key C on the console)">Calibrate voice (${SECONDS} s)</button>` +
    `<button type="button" class="cal-clear" title="Forget the calibrated voice" hidden>Clear</button>` +
    `<label class="cal-strict" title="Strict: fewer seconds of the other person, slightly more dropouts">` +
    `<input type="checkbox"> Strict</label>`;
  anchor.after(box);
  const btn = box.querySelector(".cal-btn");
  const clear = box.querySelector(".cal-clear");
  const strict = box.querySelector("input");
  let isolation = "standard";
  try { isolation = localStorage.getItem(KEY) === "strict" ? "strict" : "standard"; } catch { /* private mode */ }
  strict.checked = isolation === "strict";
  const start = () => sendControl({ enroll: { action: "start", seconds: SECONDS } });
  btn.addEventListener("click", start);
  clear.addEventListener("click", () => sendControl({ enroll: { action: "clear" } }));
  strict.addEventListener("change", () => {
    isolation = strict.checked ? "strict" : "standard";
    try { localStorage.setItem(KEY, isolation); } catch { /* private mode */ }
    sendControl({ isolation });
  });
  return {
    start,
    initialFields: () => ({ isolation }),
    render(t) {
      const e = t?.enrollment;
      if (!e) return;
      box.dataset.state = e.state;
      if (e.state === "calibrating") {
        btn.disabled = true;
        btn.textContent = `Listening… ${Math.max(0, e.calibrate_s - e.calibrate_elapsed_s).toFixed(1)} s`;
      } else {
        btn.disabled = false;
        btn.textContent = e.state === "calibrated" ? "Calibrated ✓ (again)"
          : e.state === "failed" ? `Too quiet, try again (${SECONDS} s)` : `Calibrate voice (${SECONDS} s)`;
      }
      clear.hidden = e.state !== "calibrated";
      if (t.isolation && (t.isolation === "strict") !== strict.checked) strict.checked = t.isolation === "strict";
    },
  };
}
