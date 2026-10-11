// T-404: the live-region markup the a11y audit fixed. The behaviour (focus into the Diagnostics sheet, inert, Escape)
// runs in a browser: web/app/tests/e2e/test_a11y.py (slow).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const read = (path) => readFileSync(new URL(path, import.meta.url), "utf8");

test("the stage is not one live region (it re-rendered at the 10 Hz telemetry rate)", () => {
  const main = read("../index.html").match(/<main\b[^>]*>/)[0];
  assert.match(main, /id="stage"/);
  assert.doesNotMatch(main, /aria-live/);
});

test("Listen announces through one status and one caption region, not the visible caption list", () => {
  const src = read("../screens/listen.js");
  assert.match(src, /h\("div", \{ class: "ls-cap-lines" \}\)/);
  assert.equal(src.match(/class: "sr-only", role: "status"/g)?.length, 1);
  assert.equal(src.match(/class: "sr-only", "aria-live": "polite"/g)?.length, 1);
});
