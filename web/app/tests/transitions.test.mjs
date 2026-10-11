// T-389: the pure half of the motion choreography (web/app/transitions.js).
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { cardDelays, cutDirection, pulseFor, pulseKeyframes, SPRINGS, tickPlan } from "../transitions.js";

test("cuts follow the scene order: forward pushes in, back pulls out, off-axis dissolves", () => {
  assert.equal(cutDirection("welcome", "setup"), "forward");
  assert.equal(cutDirection("setup", "listen"), "forward");
  assert.equal(cutDirection("listen", "people"), "back");
  assert.equal(cutDirection("listen", "about"), "dissolve");
  assert.equal(cutDirection("phone", "welcome"), "dissolve");
  assert.equal(cutDirection(null, "welcome"), "dissolve");
  assert.equal(cutDirection("welcome", "setup", { reduced: true }), "fade");
});

test("pulses only on a real engine change, never on a first render or a repeat", () => {
  assert.equal(pulseFor("listen", "reacquiring", "isolated"), "lock");
  assert.equal(pulseFor("listen", "loading", "isolated"), "lock");
  assert.equal(pulseFor("listen", null, "isolated"), null); // the screen mounted already isolated
  assert.equal(pulseFor("listen", "isolated", "isolated"), null); // 10 Hz re-render of the same state
  assert.equal(pulseFor("listen", "isolated", "reacquiring"), "drop");
  assert.equal(pulseFor("listen", "isolated", "fallback"), "drop");
  assert.equal(pulseFor("listen", "isolated", "calibrating"), null); // a deliberate step, not a loss
  assert.equal(pulseFor("listen", "speak", "loading"), null);
  assert.equal(pulseFor("card", "false", "true"), "lock");
  assert.equal(pulseFor("card", null, "true"), null); // a card that just appeared playing
  assert.equal(pulseFor("card", "true", "false"), null);
  assert.equal(pulseFor("hold", "holding", "full"), "commit");
  assert.equal(pulseFor("hold", "starting", "failed"), "drop");
  assert.equal(pulseFor("hold", "holding", "idle"), null);
  assert.equal(pulseFor("nope", "a", "b"), null);
});

test("pulse keyframes are compositor-only, and opacity-only under reduced motion", () => {
  for (const name of ["lock", "commit", "drop"]) {
    for (const reduced of [false, true]) {
      const k = pulseKeyframes(name, { reduced });
      const props = Object.keys(k.keyframes);
      assert.ok(props.every((p) => p === "scale" || p === "opacity"), `${name}: ${props}`);
      // T-403: numeric keyframes only; a spring between transform strings ("scale(0.97)" -> "none") ended at scale(0)
      assert.ok(Object.values(k.keyframes).flat().every((v) => typeof v === "number"), `${name}: ${JSON.stringify(k.keyframes)}`);
      if (reduced) assert.deepEqual(props, ["opacity"]);
      assert.ok(k.ms <= 500, `${name} ${k.ms} ms`);
    }
  }
  assert.equal(pulseKeyframes("bogus"), null);
});

test("card stagger is capped and collapses under reduced motion", () => {
  assert.deepEqual(cardDelays(0), []);
  assert.deepEqual(cardDelays(2, { start: 0.2 }), [0.2, 0.27]);
  const many = cardDelays(12);
  assert.ok(many.at(-1) <= 0.35 + 1e-9, `last card waits ${many.at(-1)} s`);
  assert.deepEqual(cardDelays(3, { start: 0.1, reduced: true }), [0.1, 0.1, 0.1]);
});

test("the ticker only moves between two measured values and never past them", () => {
  assert.deepEqual(tickPlan(null, 1500), { jump: true }); // first value: written as is, never counted up from 0
  assert.deepEqual(tickPlan(1500, NaN), { jump: true, dash: true });
  assert.deepEqual(tickPlan(1500, 1500), { jump: true });
  assert.deepEqual(tickPlan(1500, 750, { reduced: true }), { jump: true });
  assert.deepEqual(tickPlan(1500, 750), { jump: false, lo: 750, hi: 1500 });
});

test("springs: same easings in transitions.js and motion.css, at most 2.5 % overshoot", () => {
  const css = readFileSync(new URL("../motion.css", import.meta.url), "utf8");
  for (const [name, s] of Object.entries(SPRINGS)) {
    assert.ok(css.includes(`--spring-${name}: ${s.ease};`), `motion.css --spring-${name} differs`);
    assert.ok(css.includes(`--spring-${name}-dur: ${s.ms}ms;`), `motion.css --spring-${name}-dur differs`);
    const pts = s.ease.replace(/^linear\(|\)$/g, "").split(",").map(Number);
    assert.equal(pts[0], 0);
    assert.equal(pts.at(-1), 1);
    assert.ok(Math.max(...pts) <= 1.025, `${name} overshoots ${Math.max(...pts)}`);
  }
});

test("motion.css keeps every animated property on the compositor (no layout properties in keyframes)", () => {
  const css = readFileSync(new URL("../motion.css", import.meta.url), "utf8");
  const frames = css.match(/@keyframes[^{]+\{(?:[^{}]*\{[^}]*\})+\s*\}/g) || [];
  assert.ok(frames.length >= 10);
  for (const block of frames) {
    for (const [, prop] of block.matchAll(/([a-z-]+)\s*:/g)) {
      assert.ok(["opacity", "transform", "filter"].includes(prop), `${block.split("{")[0].trim()} animates ${prop}`);
    }
  }
  // reduced motion: the cut is a cross-fade that wins over every direction rule
  assert.match(css, /prefers-reduced-motion: reduce[\s\S]*mo-fade-out var\(--dur-cut\) linear both !important/);
});
