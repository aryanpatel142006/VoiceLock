// The shared renderer must expose delay details and distinguish visual lock from isolation.
import assert from "node:assert/strict";
import { test } from "node:test";
import { renderAcquisition } from "../acquisition.js";

test("both page pills show the acquisition state and delay explanation", () => {
  const element = { textContent: "", dataset: {}, title: "" };
  for (const [state, label] of [
    ["searching", "Searching"], ["candidate", "Candidate"], ["acquiring", "Acquiring voice"],
    ["locked-visual", "Target locked"], ["locked", "Isolated"], ["reacquiring", "Reacquiring"],
  ]) {
    renderAcquisition(element, { state, detail: "enrolling 1.8/3.0 s", since_s: 0.2 });
    assert.equal(element.dataset.state, state);
    assert.equal(element.textContent, `${label} · enrolling 1.8/3.0 s`);
  }
  renderAcquisition(element, null);
  assert.equal(element.textContent, "Waiting for engine");
  assert.equal(element.dataset.state, "");
});
