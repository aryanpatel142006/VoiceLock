// Headless check of the console against a running `hearing serve`.
//
//   .venv/bin/python -m hearing serve --port 8765 &
//   PLAYWRIGHT=/path/to/node_modules/playwright node web/console/tests/e2e.mjs \
//     --url http://127.0.0.1:8765 --out /tmp/console-shots
//
// Chromium's fake devices stand in for the mic (a beep) and camera (a test pattern).
// Checks: the page connects, audio streams both ways, telemetry numbers render, mode and
// slider controls reach the server, injected face boxes draw, and a dead WebSocket URL
// shows the error state. Context numbers render, and an injected recommendation shows a
// toast whose Switch button changes the server mode. Writes live.png, faces.png and dead.png to --out.

import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({
  options: {
    url: { type: "string", default: "http://127.0.0.1:8765" },
    out: { type: "string", default: "console-shots" },
    dead: { type: "string", default: "ws://127.0.0.1:9/ws/session" },
  },
});
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");

mkdirSync(args.out, { recursive: true });
const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok, detail });
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
}

const browser = await chromium.launch({
  args: [
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--autoplay-policy=no-user-gesture-required",
  ],
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  permissions: ["microphone", "camera"],
});
const consoleErrors = [];

try {
  // ---------------------------------------------------------------- live session
  const page = await context.newPage();
  page.on("pageerror", (e) => consoleErrors.push(e.message));
  page.on("console", (m) => m.type() === "error" && consoleErrors.push(m.text()));
  const index = await page.goto(args.url);
  check("console served with no-cache", (index.headers()["cache-control"] || "").includes("no-cache"));
  await page.waitForSelector('#conn[data-state="open"]', { timeout: 5000 });
  check("connects to /ws/session", true);

  await page.waitForFunction(() => window.hearingConsole.telemetry?.latency_ms, null, { timeout: 3000 });
  check("telemetry before the mic starts", true);

  await page.click("#start");
  await page.waitForFunction(() => window.hearingConsole.framesBack > 100, null, { timeout: 10000 });
  await page.waitForTimeout(1500);
  const live = await page.evaluate(() => {
    const s = window.hearingConsole;
    return {
      framesOut: s.framesOut, framesBack: s.framesBack, rate: s.audio?.rate, rttMs: s.rttMs,
      buffered: s.playback.buffered, underruns: s.playback.underruns, dropped: s.playback.dropped,
      levelIn: s.levelIn, video: Boolean(s.video), total: document.querySelector("#total").textContent,
      label: document.querySelector("#total-label").textContent,
      tier: document.querySelector("#tier").textContent,
      device: document.querySelector("#device").textContent,
      legend: document.querySelector("#legend").innerText.replace(/\s+/g, " "),
      noise: document.querySelector("#noise").textContent,
      snr: document.querySelector("#snr").textContent,
    };
  });
  check("audio streams up", live.framesOut > 100, `${live.framesOut} frames at ${live.rate} Hz`);
  check("audio streams back", live.framesBack > 100, `${live.framesBack} frames, rtt ${live.rttMs?.toFixed(2)} ms`);
  check("camera streams", live.video);
  check("latency number shown", /^\d+$/.test(live.total), `${live.total} ms (${live.label})`);
  check("tier and device shown", live.tier !== "—" && live.device !== "—", `${live.tier}, ${live.device}`);
  check("context numbers from the server", live.noise.endsWith("dB"), `noise ${live.noise}, SNR ${live.snr}`);
  console.log(`     legend: ${live.legend}`);
  console.log(`     playback: buffered ${live.buffered} samples, underruns ${live.underruns}, dropped ${live.dropped}`);

  await page.keyboard.press("5");
  await page.waitForFunction(() => window.hearingConsole.telemetry?.mode === "quiet", null, { timeout: 2000 });
  check("key 5 switches the server to quiet", true);
  await page.waitForFunction(() => document.querySelector("#suppression").disabled);
  check("slider disabled in quiet", true);
  await page.click("label.mode:has(input[value=natural])");
  await page.waitForFunction(() => window.hearingConsole.telemetry?.mode === "natural", null, { timeout: 2000 });
  check("clicking Natural switches back", true);
  await page.locator("#suppression").fill("80");
  await page.waitForTimeout(300);
  check("slider shows its value", (await page.textContent("#suppression-out")) === "80");
  await page.selectOption("#voice-override", "female_adult");
  await page.waitForFunction(
    () => document.querySelector("#voice-chip").textContent === "Voice: adult female (your choice)",
    null, { timeout: 2000 });
  check("voice override reaches the server and the chip", true);
  await page.screenshot({ path: join(args.out, "live.png"), fullPage: true });

  // ---------------------------------------------------------------- face overlay (boxes arrive in M2)
  const facesPage = await context.newPage();
  let sentRecommendation = false;
  await facesPage.routeWebSocket(/\/ws\/session/, (ws) => {
    const server = ws.connectToServer();
    server.onMessage((message) => {
      if (typeof message === "string" && message.includes('"telemetry"')) {
        const t = JSON.parse(message);
        t.faces = [
          { track_id: 17, box: [0.18, 0.2, 0.2, 0.42], p_speaking: 0.83 },
          { track_id: 4, box: [0.62, 0.26, 0.16, 0.34], p_speaking: 0.07 },
        ];
        t.target = { track_id: 17, confidence: 0.91, locked: false };
        t.voice = { bucket: "male_adult", source: "face+voice", override: null };
        message = JSON.stringify(t);
      }
      ws.send(message);
      if (typeof message === "string" && message.includes('"context"') && !sentRecommendation) {
        sentRecommendation = true;
        ws.send(JSON.stringify({
          type: "recommendation", mode: "synthetic", confidence: 0.78, reason: "noisy, 3+ speakers", auto: false,
        }));
      }
    });
  });
  await facesPage.goto(args.url);
  await facesPage.waitForSelector('#conn[data-state="open"]');
  await facesPage.click("#start");
  await facesPage.waitForFunction(() => window.hearingConsole.framesBack > 50, null, { timeout: 10000 });
  await facesPage.waitForTimeout(800);
  const target = await facesPage.textContent("#target");
  check("target readout from telemetry", target.includes("face 17"), target);
  const chip = await facesPage.textContent("#voice-chip");
  check("voice chip shows estimate as apparent", chip.includes("from face + voice"), chip);
  await facesPage.waitForSelector("#toast:not([hidden])", { timeout: 3000 });
  const toast = await facesPage.textContent("#toast-text");
  check("recommendation toast shown", toast.includes("Synthetic") && toast.includes("78%"), toast);
  await facesPage.screenshot({ path: join(args.out, "faces.png"), fullPage: true });
  await facesPage.click("#toast-accept");
  await facesPage.waitForFunction(() => window.hearingConsole.telemetry?.mode === "synthetic", null, { timeout: 2000 });
  check("accepting the toast switches the server", await facesPage.isHidden("#toast"));
  await facesPage.close();

  // ---------------------------------------------------------------- dead socket
  const dead = await context.newPage();
  await dead.goto(`${args.url}/?ws=${encodeURIComponent(args.dead)}`);
  await dead.waitForSelector("#banner:not([hidden])", { timeout: 5000 });
  const banner = await dead.textContent("#banner-text");
  const conn = await dead.getAttribute("#conn", "data-state");
  check("dead URL shows the error state", conn === "error" && banner.includes(args.dead), banner);
  check("numbers blank without telemetry", (await dead.textContent("#total")) === "—");
  await dead.screenshot({ path: join(args.out, "dead.png"), fullPage: true });
} finally {
  await browser.close();
}

check("no page errors", consoleErrors.length === 0, consoleErrors.join(" | "));
const failed = results.filter((r) => !r.ok);
console.log(`${results.length - failed.length}/${results.length} checks passed; screenshots in ${args.out}`);
process.exit(failed.length ? 1 : 0);
