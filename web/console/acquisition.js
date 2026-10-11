// One acquisition vocabulary for the simple page and debug console.
const LABELS = {
  searching: "Searching", candidate: "Candidate", acquiring: "Acquiring voice",
  "locked-visual": "Target locked", locked: "Isolated", reacquiring: "Reacquiring",
};

export function renderAcquisition(element, acquisition) {
  const label = LABELS[acquisition?.state];
  element.dataset.state = label ? acquisition.state : "";
  element.textContent = label
    ? `${label}${acquisition.detail ? ` · ${acquisition.detail}` : ""}`
    : "Waiting for engine";
}
