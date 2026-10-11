// Headless check of the simple test page (/simple) against `hearing serve` with mock stages.
//
//   PLAYWRIGHT=/path/to/node_modules/playwright node web/console/tests/simple-e2e.mjs --out /tmp/simple-shots
//
// Without --url it starts its own server on a free port (weight-free pipelines, mock ASR/TTS/voice,
// the marker face detector) and stops it at the end; --url http://127.0.0.1:PORT uses a running one.
// Chromium's fake mic and camera (fake-media.mjs) stand in for the devices.
//
// Checks: /simple is served (no-cache) and / is still the console; the six mode buttons, the
// suppression slider, Runs on (Local / GPU / Auto) and the voice override send the right control
// messages (recorded on the socket) and the server applies the modes; the status line's tier and
// latency follow telemetry (injected values, then the measured mic-to-ear number once audio flows);
// captions show in Text/Synthetic; the page fits 390 px with no horizontal scroll and big targets.
// Writes phone.png, phone-text.png and wide.png to --out.

import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import { createServer } from "node:net";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { writeWav, writeY4m } from "./fake-media.mjs";

const { values: args } = parseArgs({
  options: {
    url: { type: "string" },
    out: { type: "string", default: "simple-shots" },
    python: { type: "string" },
  },
});
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
mkdirSync(args.out, { recursive: true });
const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok });
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
}

async function freePort() {
  return new Promise((done, fail) => {
    const srv = createServer();
    srv.once("error", fail);
    srv.listen(0, "127.0.0.1", () => {
      const { port } = srv.address();
      srv.close(() => done(port));
    });
  });
}

async function startServer() {
  const port = await freePort();
  const python = args.python || join(repo, ".venv/bin/python");
  const env = {
    ...process.env, HEARING_DEVICE: "cpu", HEARING_ASR: "mock", HEARING_TTS: "mock",
    HEARING_VOICE_MATCH: "mock", HEARING_FACES: "mock", HEARING_ASD: "lipmotion", HEARING_HQ_BACKEND: "local",
  };
  delete env.HEARING_REMOTE_URL;
  delete env.HEARING_PREFS_LOG;
  const proc = spawn(python, [
    "-m", "hearing", "serve", "--port", String(port), "--log-level", "warning", "--pipelines",
    "natural=passthrough; focus_hq=gain(db=-6); synthetic=passthrough>synth(speak=true); " +
    "text=passthrough>synth(speak=false)",
  ], { cwd: repo, env, stdio: ["ignore", "inherit", "inherit"] });
  const url = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 120; i++) {
    try {
      if ((await fetch(`${url}/health`)).ok) return { url, proc };
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  proc.kill();
  throw new Error(`hearing serve did not come up on ${url}`);
}

const server = args.url ? { url: args.url.replace(/\/$/, ""), proc: null } : await startServer();
const base = server.url;
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
const errors = [];

/** A page whose socket is proxied: client JSON is recorded, telemetry can be patched, server messages injected. */
async function openPage(context) {
  const page = await context.newPage();
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  const wire = { sent: [], patch: null, client: null };
  await page.routeWebSocket(/\/ws\/session/, (ws) => {
    wire.client = ws;
    const upstream = ws.connectToServer();
    ws.onMessage((m) => {
      if (typeof m === "string") wire.sent.push(JSON.parse(m));
      upstream.send(m);
    });
    upstream.onMessage((m) => {
      if (typeof m === "string" && wire.patch && m.includes('"telemetry"')) {
        m = JSON.stringify({ ...JSON.parse(m), ...wire.patch });
      }
      ws.send(m);
    });
  });
  return { page, wire };
}

const lastControl = (wire, key) => [...wire.sent].reverse().find((m) => m.type === "control" && key in m);
const until = (page, fn, arg, timeout = 3000) => page.waitForFunction(fn, arg, { timeout, polling: 20 });
const serverMode = (page, mode) => until(page, (m) => window.hearingSimple.telemetry?.mode === m, mode);

try {
  // ---------------------------------------------------------------- routes
  const simple = await fetch(`${base}/simple`);
  const html = await simple.text();
  check("GET /simple serves the page", simple.ok && html.includes('src="simple.js"'), String(simple.status));
  check("/simple is no-cache", (simple.headers.get("cache-control") || "").includes("no-cache"));
  check("/ is still the console", (await (await fetch(`${base}/`)).text()).includes("<title>Hearing console</title>"));

  // ---------------------------------------------------------------- phone: controls
  const phone = await browser.newContext({
    viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
    permissions: ["microphone", "camera"],
  });
  const { page, wire } = await openPage(phone);
  await page.goto(`${base}/simple`);
  await page.waitForSelector('#conn[data-state="open"]', { timeout: 5000 });
  await until(page, () => window.hearingSimple.telemetry?.latency_ms);
  const hello = wire.sent.find((m) => m.type === "hello");
  check("hello from the simple client", hello?.client === "simple" && hello.sample_rate === 16000, JSON.stringify(hello));
  const first = wire.sent.find((m) => m.type === "control");
  check("first control carries mode and suppression", first?.mode === "natural" && first.suppression === 50,
    JSON.stringify(first));

  for (const mode of ["focus_hq", "synthetic", "text", "quiet", "auto", "natural"]) {
    await page.tap(`#modes button[data-mode=${mode}]`);
    await serverMode(page, mode);
    const sent = lastControl(wire, "mode");
    const pressed = await page.getAttribute(`#modes button[data-mode=${mode}]`, "aria-pressed");
    check(`${mode} button sends {mode: ${mode}} and the server applies it`,
      sent.mode === mode && Object.keys(sent).length === 2 && pressed === "true", JSON.stringify(sent));
  }

  await page.locator("#suppression").fill("80");
  await page.waitForTimeout(150);
  check("slider sends suppression 80", lastControl(wire, "suppression")?.suppression === 80,
    JSON.stringify(lastControl(wire, "suppression")));
  check("slider shows 80", (await page.textContent("#suppression-out")) === "80");
  await page.tap('#modes button[data-mode="quiet"]');
  check("slider disabled in Quiet", await page.isDisabled("#suppression"));
  await page.tap('#modes button[data-mode="natural"]');

  for (const [value, label] of [["remote", "GPU"], ["auto", "Auto"], ["local", "Local"]]) {
    await page.tap(`.seg label:has(input[value=${value}])`);
    await until(page, (v) => window.hearingSimple.telemetry?.hq_backend === v, value);
    check(`Runs on ${label} sends {hq_backend: ${value}}`, lastControl(wire, "hq_backend")?.hq_backend === value &&
      (await page.isChecked(`input[name=hq-backend][value=${value}]`)));
  }
  await until(page, () => document.querySelector("#hq-active").textContent.startsWith("Playing: Local"));
  const readout = await page.textContent("#hq-active");
  check("Runs-on readout shows the active backend", readout.startsWith("Playing: Local"), readout);
  await page.tap('.seg label:has(input[value="remote"])');
  await until(page, () => document.querySelector("#hq-active").dataset.fallback === "true");
  const fallback = await page.textContent("#hq-active");
  check("GPU with no remote shows the local fallback", fallback.includes("Local"), fallback);
  wire.patch = { remote_rtt_ms: 12.3, hq_active: "remote", hq_backend: "auto" };
  await until(page, () => document.querySelector("#hq-active").textContent.includes("RTT"));
  const rtt = await page.textContent("#hq-active");
  check("Runs-on readout shows active backend + RTT", rtt === "Playing: Auto → GPU · RTT 12 ms", rtt);

  // ---------------------------------------------------------------- A/B toggle (raw vs selective hearing)
  check("A/B starts on Selective hearing", (await page.getAttribute('#ab button[data-ab="processed"]', "aria-pressed")) === "true");
  await page.tap('#ab button[data-ab="raw"]');
  await until(page, () => window.hearingSimple.telemetry?.ab === "raw");
  check("Raw sends {ab: raw} and the server echoes it", lastControl(wire, "ab")?.ab === "raw" &&
    (await page.getAttribute('#ab button[data-ab="raw"]', "aria-pressed")) === "true");
  await page.keyboard.press("Space");
  await until(page, () => window.hearingSimple.telemetry?.ab === "processed");
  check("Space toggles back to processed", lastControl(wire, "ab")?.ab === "processed");

  // ---------------------------------------------------------------- telemetry drives the status line
  wire.patch = {
    tier: "gate", latency_ms: { passthrough: 0.1, algorithmic: 0, resample: 0, server: 0.4, total: 63.4 },
    faces: [
      { track_id: 17, box: [0.06, 0.2, 0.2, 0.5], p_speaking: 0.83 },
      { track_id: 4, box: [0.4, 0.22, 0.2, 0.48], p_speaking: 0.07 },
    ],
    target: { track_id: 17, confidence: 0.91, locked: false },
    voice: { bucket: "male_adult", source: "face+voice", override: null },
  };
  await until(page, () => document.querySelector("#st-latency").textContent === "63");
  check("tier from telemetry", (await page.textContent("#st-tier")) === "AV · gate");
  check("latency from telemetry (server estimate before the mic)",
    (await page.textContent("#st-latency-kind")) === "server estimate");
  wire.patch = { ...wire.patch, tier: "audio-only", latency_ms: { ...wire.patch.latency_ms, total: 88.0 } };
  await until(page, () => document.querySelector("#st-latency").textContent === "88");
  check("tier and latency update with the next telemetry", (await page.textContent("#st-tier")) === "Audio-only");

  // Acquisition is driven by wire telemetry, including delay explanations.
  for (const [state, label, detail] of [
    ["searching", "Searching", "looking for a speaking face"],
    ["candidate", "Candidate", "waiting for face dwell"],
    ["acquiring", "Acquiring voice", "enrolling 1.8/3.0 s"],
    ["locked-visual", "Target locked", "face selected; waiting for isolation"],
    ["locked", "Isolated", "target isolation active"],
    ["reacquiring", "Reacquiring", "target lost 0.4 s"],
  ]) {
    wire.patch = { ...wire.patch, acquisition: { state, since_s: 0.4, detail } };
    await until(page, (s) => document.querySelector("#acquisition")?.dataset.state === s, state);
    const text = await page.textContent("#acquisition");
    check(`simple acquisition ${state}`, text.includes(label) && text.includes(detail), text);
    if (state === "acquiring") {
      await page.screenshot({ path: join(args.out, "phone-acquiring.png"), fullPage: true });
      check("acquisition fits 390 px", await page.evaluate(() => document.documentElement.scrollWidth <= 390));
    }
  }

  // ---------------------------------------------------------------- captions + voice
  check("captions hidden in Natural", await page.isHidden("#captions-box"));
  await page.tap('#modes button[data-mode="synthetic"]');
  await serverMode(page, "synthetic");
  await until(page, () => !document.querySelector("#voice-box").hidden);
  const chip = await page.textContent("#voice-chip");
  check("voice chip in Synthetic shows the apparent voice", chip === "Voice: adult male (from face + voice)", chip);
  await page.selectOption("#voice-override", "neutral");
  check("voice override sends {voice_override: neutral}", lastControl(wire, "voice_override")?.voice_override === "neutral");
  await page.tap('#modes button[data-mode="text"]');
  await serverMode(page, "text");
  await until(page, () => document.querySelector("#voice-box").hidden);
  check("captions shown in Text, voice chip hidden", await page.isVisible("#captions-box") &&
    await page.isHidden("#voice-box"));
  wire.client.send(JSON.stringify({ type: "caption", text: "see you at noon", final: true, t0: 1, t1: 2 }));
  await until(page, () => document.querySelector("#captions").textContent.includes("see you at noon"));
  check("caption from the server shows", true);
  const textHeight = await page.evaluate(() => document.documentElement.scrollHeight);
  check("Text mode with captions still one screen at 390 x 844", textHeight <= 844, `scrollHeight ${textHeight}`);
  await page.screenshot({ path: join(args.out, "phone-text.png"), fullPage: true });

  // ---------------------------------------------------------------- live audio: mic-to-ear
  await page.tap('#modes button[data-mode="natural"]');
  wire.patch = { faces: wire.patch.faces, target: wire.patch.target };
  await page.tap("#start");
  await until(page, () => window.hearingSimple.framesBack > 100, null, 10000);
  await page.waitForTimeout(800);
  const kind = await page.textContent("#st-latency-kind");
  const ms = await page.textContent("#st-latency");
  check("mic-to-ear latency once audio flows", kind === "mic to ear" && /^\d+$/.test(ms), `${ms} ms (${kind})`);
  check("camera preview live", await page.evaluate(() => document.querySelector("#view").classList.contains("live")));

  // ---------------------------------------------------------------- 390 px layout
  const layout = await page.evaluate(() => ({
    scroll: document.documentElement.scrollWidth,
    height: document.documentElement.scrollHeight,
    minButton: Math.min(...[...document.querySelectorAll("#modes button")].map((b) => b.getBoundingClientRect().height)),
    slider: document.querySelector("#suppression").getBoundingClientRect().height,
  }));
  check("fits 390 px with no horizontal scroll", layout.scroll <= 390, `scrollWidth ${layout.scroll}`);
  check("one screen at 390 x 844", layout.height <= 844, `scrollHeight ${layout.height}`);
  check("mode buttons and slider are big touch targets", layout.minButton >= 44 && layout.slider >= 44,
    `buttons ${layout.minButton} px, slider ${layout.slider} px`);
  await page.screenshot({ path: join(args.out, "phone.png"), fullPage: true });
  console.log(`     phone: measured ${ms} ms, ${await page.evaluate(() => window.hearingSimple.framesBack)} frames back`);
  await phone.close();

  // ---------------------------------------------------------------- wide screen
  const wide = await browser.newContext({ viewport: { width: 1280, height: 860 }, permissions: ["microphone", "camera"] });
  const { page: wpage } = await openPage(wide);
  await wpage.goto(`${base}/simple`);
  await wpage.waitForSelector('#conn[data-state="open"]', { timeout: 5000 });
  await wpage.click("#start");
  await until(wpage, () => window.hearingSimple.framesBack > 50, null, 10000);
  await wpage.keyboard.press("2");
  await serverMode(wpage, "focus_hq");
  check("key 2 picks Focus HQ", true);
  await wpage.waitForTimeout(500);
  await wpage.screenshot({ path: join(args.out, "wide.png") });
  await wide.close();

  const consoleContext = await browser.newContext({ viewport: { width: 390, height: 844 } });
  const { page: cpage, wire: cwire } = await openPage(consoleContext);
  cwire.patch = { acquisition: { state: "reacquiring", since_s: 0.4, detail: "target lost 0.4 s" } };
  await cpage.goto(`${base}/`);
  await until(cpage, () => document.querySelector("#acquisition")?.dataset.state === "reacquiring");
  check("console acquisition pill", (await cpage.textContent("#acquisition")).includes("target lost 0.4 s"));
  await cpage.screenshot({ path: join(args.out, "console-reacquiring.png"), fullPage: true });
  await consoleContext.close();

  check("no page errors", errors.length === 0, errors.join(" | "));
} catch (e) {
  check("run", false, e.stack || String(e));
} finally {
  await browser.close();
  server.proc?.kill();
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
