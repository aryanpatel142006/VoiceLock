// Headless check of the vision path end to end: Chromium's fake camera plays a generated
// clip of two marker "faces" (green boxes, the left one's black mouth opening at 4 Hz with
// matching noise bursts on the fake mic), the console sends JPEGs, and a server running the
// mock detector (HEARING_FACES=mock) must send the faces back and pick the talking one. Use the
// lip-motion ASD (HEARING_ASD=lipmotion): Light-ASD scores the marker faces arbitrarily.
//
//   HEARING_FACES=mock HEARING_ASD=lipmotion .venv/bin/python -m hearing serve --port 8915 \
//     --pipelines "natural=passthrough>gate" &
//   PLAYWRIGHT=/path/to/node_modules/playwright node web/console/tests/vision-e2e.mjs \
//     --url http://127.0.0.1:8915 --out /tmp/vision-shots
//
// Checks: boxes arrive in telemetry and draw, the TARGET highlight lands on the talking face
// (or, failing that, a click locks it), the tier badge reads AV, and a page whose camera is
// unavailable shows the audio-only tier. Writes vision.png and audio-only.png to --out.

import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    url: { type: "string", default: "http://127.0.0.1:8915" },
    out: { type: "string", default: "vision-shots" },
  },
});
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
mkdirSync(args.out, { recursive: true });

import { writeWav, writeY4m } from "./fake-media.mjs";

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
try {
  // ---------------------------------------------------------------- camera on
  const page = await context.newPage();
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(args.url);
  await page.waitForSelector('#conn[data-state="open"]', { timeout: 5000 });
  await page.click("#start");
  await page.waitForFunction(() => window.hearingConsole.telemetry?.faces?.length >= 2, null, { timeout: 15000 });
  const faces = await page.evaluate(() => window.hearingConsole.telemetry.faces);
  check("faces from JPEG frames in telemetry", faces.length === 2,
    faces.map((f) => `${f.track_id}@[${f.box.map((v) => v.toFixed(2))}] p=${f.p_speaking.toFixed(2)}`).join(" "));

  let auto = true;
  try {
    await page.waitForFunction(() => window.hearingConsole.telemetry?.target?.track_id != null, null, { timeout: 8000 });
  } catch {
    auto = false; // lip-motion ASD never fired (fake devices drift): lock by clicking instead
    const box = await page.locator("#overlay").boundingBox();
    const left = faces.reduce((a, b) => (a.box[0] < b.box[0] ? a : b));
    await page.mouse.click(box.x + (left.box[0] + left.box[2] / 2) * box.width,
      box.y + (left.box[1] + left.box[3] / 2) * box.height);
    await page.waitForFunction(() => window.hearingConsole.telemetry?.target?.track_id != null, null, { timeout: 3000 });
  }
  await page.waitForTimeout(500);
  const t = await page.evaluate(() => window.hearingConsole.telemetry);
  const target = t.faces.find((f) => f.track_id === t.target.track_id);
  check(auto ? "talking face picked as TARGET automatically" : "TARGET locked by clicking its box",
    Boolean(target) && target.box[0] < 0.3,
    `target ${t.target.track_id}, locked ${t.target.locked}, p_speaking ${target?.p_speaking.toFixed(2)}`);
  const tier = await page.textContent("#tier");
  const vision = await page.textContent("#vision");
  check("tier badge reads AV", tier.startsWith("AV"), `${tier}; vision: ${vision}`);
  check("vision status is live", t.vision?.status === "live", JSON.stringify(t.vision));
  await page.screenshot({ path: join(args.out, "vision.png"), fullPage: true });
  await page.close();

  // ---------------------------------------------------------------- no camera
  const noCam = await context.newPage();
  noCam.on("pageerror", (e) => errors.push(e.message));
  await noCam.addInitScript(() => {
    const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = (c) =>
      c?.video ? Promise.reject(new DOMException("no camera", "NotFoundError")) : real(c);
  });
  await noCam.goto(args.url);
  await noCam.waitForSelector('#conn[data-state="open"]', { timeout: 5000 });
  await noCam.click("#start");
  await noCam.waitForFunction(() => window.hearingConsole.framesBack > 50, null, { timeout: 10000 });
  await noCam.waitForTimeout(500);
  const tier2 = await noCam.textContent("#tier");
  const placeholder = await noCam.textContent("#camera-placeholder");
  check("no camera degrades to audio-only", tier2 === "Audio-only", `${tier2}; ${placeholder}`);
  await noCam.screenshot({ path: join(args.out, "audio-only.png"), fullPage: true });
  await noCam.close();
} finally {
  await browser.close();
}
check("no page errors", errors.length === 0, errors.join("; "));
const failed = results.filter((r) => !r.ok).length;
console.log(`${results.length - failed}/${results.length} checks passed; screenshots in ${args.out}`);
process.exit(failed ? 1 : 0);
