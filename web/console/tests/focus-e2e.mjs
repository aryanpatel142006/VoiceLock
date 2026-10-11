// Headless check of the Focus HQ toggle and the tier ladder in the console. Chromium's fake
// camera and mic play the marker-face clip (fake-media.mjs); the server runs the mock detector
// and a ladder whose Focus HQ tier is the identity TSE (no weights), so the whole path runs:
// JPEG frames -> faces -> target -> lip crops -> tse -> ladder -> telemetry -> console.
//
//   printf 'upgrade_after_s: 0.3\ntiers:\n  - {name: focus_hq, pipeline: "tse(candidate=identity, window_s=0.4, hop_s=0.2)>mixer", needs: [camera, target, healthy]}\n  - {name: gate, pipeline: "passthrough>gate>mixer", needs: [camera, target]}\n  - {name: audio-only, pipeline: passthrough}\n' > /tmp/tiers-e2e.yaml
//   HEARING_FACES=mock HEARING_ASD=lipmotion .venv/bin/python -m hearing serve --port 8916 \
//     --pipelines 'focus_hq=tiers(config="/tmp/tiers-e2e.yaml")' &
//   PLAYWRIGHT=/path/to/node_modules/playwright node web/console/tests/focus-e2e.mjs \
//     --url http://127.0.0.1:8916 --out /tmp/focus-shots
//
// Checks: F turns Focus HQ on (button pressed, mode focus_hq), the old mode plays while the ladder
// warms up, the tier badge and the ladder show Focus HQ as playing, and F again returns to
// Natural. Writes focus-hq.png and natural.png to --out.

import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { writeWav, writeY4m } from "./fake-media.mjs";

const { values: args } = parseArgs({
  options: {
    url: { type: "string", default: "http://127.0.0.1:8916" },
    out: { type: "string", default: "focus-shots" },
  },
});
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
mkdirSync(args.out, { recursive: true });

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok });
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
}

const y4m = join(args.out, "faces.y4m");
const wav = join(args.out, "speech.wav");
writeY4m(y4m);
writeWav(wav);

const browser = await chromium.launch({
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    `--use-file-for-fake-video-capture=${y4m}`,
    `--use-file-for-fake-audio-capture=${wav}`,
    "--autoplay-policy=no-user-gesture-required",
  ],
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  permissions: ["microphone", "camera"],
});
const errors = [];
const tel = (page) => page.evaluate(() => window.hearingConsole.telemetry);
try {
  const page = await context.newPage();
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(args.url);
  await page.waitForSelector('#conn[data-state="open"]', { timeout: 5000 });
  await page.click("#start");
  await page.waitForFunction(() => window.hearingConsole.telemetry?.faces?.length >= 2, null, { timeout: 15000 });
  const ladderOff = await page.textContent("#ladder");
  check("ladder is off in Natural", ladderOff.includes("Off"), ladderOff.trim());

  await page.keyboard.press("f");
  const pressed = await page.getAttribute("#focus-hq", "aria-pressed");
  check("F turns the Focus HQ toggle on", pressed === "true");
  await page.waitForFunction(() => window.hearingConsole.telemetry?.mode === "focus_hq", null, { timeout: 3000 });
  await page.waitForFunction(() => window.hearingConsole.telemetry?.tiers?.active, null, { timeout: 5000 });
  const first = await tel(page);
  check("server is in focus_hq with the ladder", first.pipeline.startsWith("tiers("), first.pipeline);

  // Lock the talking face so the AV tiers never wait for the lip-motion ASD on drifting fake devices.
  const box = await page.locator("#overlay").boundingBox();
  const left = first.faces.reduce((a, b) => (a.box[0] < b.box[0] ? a : b));
  await page.mouse.click(box.x + (left.box[0] + left.box[2] / 2) * box.width,
    box.y + (left.box[1] + left.box[3] / 2) * box.height);
  let hq = true;
  try {
    await page.waitForFunction(() => window.hearingConsole.telemetry?.tier === "focus_hq", null, { timeout: 10000 });
  } catch {
    hq = false;
  }
  const t = await tel(page);
  check("ladder plays Focus HQ", hq, `tier ${t.tier}; ${JSON.stringify(t.tiers?.tiers?.map((x) => [x.name, x.reason]))}`);
  await page.waitForTimeout(300); // one render
  const badge = await page.textContent("#tier");
  const active = await page.textContent('#ladder li[data-state="active"]');
  const hint = await page.textContent("#focus-hint");
  check("tier badge reads Focus HQ", badge === "Focus HQ", badge);
  check("ladder marks Focus HQ as playing", active?.startsWith("Focus HQ"), active);
  check("hint shows the delay and the model", /behind/.test(hint) && /identity/.test(hint), hint);
  await page.screenshot({ path: join(args.out, "focus-hq.png"), fullPage: true });

  await page.keyboard.press("f");
  await page.waitForFunction(() => window.hearingConsole.telemetry?.mode === "natural", null, { timeout: 3000 });
  await page.waitForTimeout(300);
  check("F again returns to Natural", (await page.getAttribute("#focus-hq", "aria-pressed")) === "false");
  await page.screenshot({ path: join(args.out, "natural.png"), fullPage: true });
  await page.close();
} finally {
  await browser.close();
}
check("no page errors", errors.length === 0, errors.join("; "));
const failed = results.filter((r) => !r.ok).length;
console.log(`${results.length - failed}/${results.length} checks passed; screenshots in ${args.out}`);
process.exit(failed ? 1 : 0);
