// Headless check of the Focus HQ backend selector (Local / GPU / Auto). Two servers on free
// ports: a "GPU" session server and the laptop server pointing its remote Focus HQ at it.
//
//   .venv/bin/python -m hearing serve --port $GPU --pipelines 'focus_hq=passthrough' &
//   HEARING_REMOTE_URL=ws://127.0.0.1:$GPU/ws/session HEARING_HQ_BACKEND=local \
//     .venv/bin/python -m hearing serve --port $LAPTOP --pipelines 'focus_hq=gain(db=-6)' &
//   PLAYWRIGHT=/path/to/node_modules/playwright node web/console/tests/hq-backend-e2e.mjs \
//     --url http://127.0.0.1:$LAPTOP --gpu-pid <pid of the GPU server> --out /tmp/hq-shots
//
// Checks: the selector starts on Local; GPU moves Focus HQ to the remote (telemetry hq_active,
// the RTT readout); with --gpu-pid, killing the GPU server falls back to Local within 1 s while
// frames keep coming back; Local switches back. Writes gpu.png and fallback.png to --out.

import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { writeWav, writeY4m } from "./fake-media.mjs";

const { values: args } = parseArgs({
  options: {
    url: { type: "string", default: "http://127.0.0.1:8917" },
    out: { type: "string", default: "hq-shots" },
    "gpu-pid": { type: "string" },
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
const until = (page, fn, timeout) => page.waitForFunction(fn, null, { timeout, polling: 20 });
try {
  const page = await context.newPage();
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(args.url);
  await page.waitForSelector('#conn[data-state="open"]', { timeout: 5000 });
  await page.click("#start");
  await page.keyboard.press("f");
  await until(page, () => window.hearingConsole.telemetry?.mode === "focus_hq", 3000);
  await page.waitForTimeout(300);
  check("selector starts on Local", await page.isChecked('input[name=hq-backend][value=local]'));
  check("readout says Local", (await page.textContent("#hq-active")).includes("Local"),
    await page.textContent("#hq-active"));

  await page.click('input[name=hq-backend][value=remote]', { force: true });
  await until(page, () => window.hearingConsole.telemetry?.hq_active === "remote", 5000);
  await page.waitForTimeout(500);
  const t = await tel(page);
  check("GPU plays Focus HQ remotely", t.hq_backend === "remote" && t.pipeline.startsWith("remote("), t.pipeline);
  check("telemetry carries the RTT", typeof t.remote_rtt_ms === "number", String(t.remote_rtt_ms));
  const readout = await page.textContent("#hq-active");
  check("readout shows GPU and RTT", /GPU · RTT/.test(readout), readout);
  await page.screenshot({ path: join(args.out, "gpu.png"), fullPage: true });

  const rate = async (ms) => {
    const a = await page.evaluate(() => window.hearingConsole.framesBack);
    await page.waitForTimeout(ms);
    return ((await page.evaluate(() => window.hearingConsole.framesBack)) - a) / (ms / 1000);
  };
  const gpuRate = await rate(1000);
  check("audio frames come back on the GPU path", gpuRate > 30, `${gpuRate} frames/s`);

  if (args["gpu-pid"]) {
    const back = await page.evaluate(() => window.hearingConsole.framesBack);
    const t0 = Date.now();
    process.kill(Number(args["gpu-pid"]), "SIGKILL");
    await until(page, () => window.hearingConsole.telemetry?.hq_active === "local", 3000);
    const ms = Date.now() - t0;
    check("GPU gone: Local within 1 s", ms < 1000, `${ms} ms`);
    const after = await page.evaluate(() => window.hearingConsole.framesBack);
    check("audio kept coming back while it fell back", after - back > ms / 40, `${after - back} frames in ${ms} ms`);
    const localRate = await rate(1000);
    check("audio frames come back on the local fallback", localRate > 30, `${localRate} frames/s`);
    const hint = await page.textContent("#focus-hint");
    check("hint explains the fallback", /GPU unavailable/.test(hint), hint);
    await page.screenshot({ path: join(args.out, "fallback.png"), fullPage: true });
  }

  await page.click('input[name=hq-backend][value=local]', { force: true });
  await until(page, () => window.hearingConsole.telemetry?.hq_backend === "local", 3000);
  await until(page, () => window.hearingConsole.telemetry?.pipeline === "gain(db=-6)", 3000);
  check("Local switches back", true);
  await page.close();
} finally {
  await browser.close();
}
check("no page errors", errors.length === 0, errors.join("; "));
const failed = results.filter((r) => !r.ok).length;
console.log(`${results.length - failed}/${results.length} checks passed; screenshots in ${args.out}`);
process.exit(failed ? 1 : 0);
