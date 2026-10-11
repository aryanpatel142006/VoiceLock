// Headless layout check of the simple test page (/simple): nothing wider than the viewport, every control on screen.
//
//   PLAYWRIGHT=/path/to/node_modules/playwright node web/console/tests/simple-layout.mjs [--out /tmp/simple-layout]
//
// No Python server: web/console is served statically on a free port, so the socket fails and the page shows
// its error state (banner + "Can't reach the engine"). A second pass also opens the captions and voice chip
// with long text. Each width runs as a phone (isMobile, touch) and as a plain desktop window.
//
// Do not check phone widths with the Chrome CLI (`chrome --headless --window-size=390,844 --screenshot`):
// headless Chrome will not make a window narrower than 500 px, so the page lays out at 500 px (desktop
// rules, "Start mic and camera" in full) and the 390 px screenshot crops the right 110 px. That was T-058.
// Playwright's viewport (CDP device metrics) has no such floor; this script asserts innerWidth to be sure.

import { createReadStream, mkdirSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({ options: { out: { type: "string" } } });
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || "playwright");
if (args.out) mkdirSync(args.out, { recursive: true });
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const results = [];
function check(name, ok, detail = "") {
  results.push({ name, ok });
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? `: ${detail}` : ""}`);
}

const TYPES = { ".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".mjs": "text/javascript" };
const server = createServer((req, res) => {
  const path = new URL(req.url, "http://x").pathname;
  const file = normalize(join(root, path === "/simple" ? "simple.html" : path));
  if (!file.startsWith(root) || !statSync(file, { throwIfNoEntry: false })?.isFile()) {
    res.writeHead(404).end();
    return;
  }
  res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" });
  createReadStream(file).pipe(res);
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const base = `http://127.0.0.1:${server.address().port}`;

const WIDTHS = [320, 360, 375, 390, 414, 480, 600, 899, 900, 1280];
const CONTROLS = "button, a, input, select, output, .seg label, .slider, .runs-on, .view, .status";

/** Page metrics: scrollWidth plus every visible control whose box leaves the viewport horizontally. */
const measure = (page) => page.evaluate((sel) => {
  const width = document.documentElement.clientWidth;
  const out = [...document.querySelectorAll(sel)]
    .filter((e) => e.getClientRects().length && getComputedStyle(e).visibility !== "hidden" && !e.closest("[hidden]"))
    .map((e) => ({ e, r: e.getBoundingClientRect() }))
    .filter(({ e, r }) => r.left < -0.5 || r.right > width + 0.5 || (e.type !== "radio" && r.width < 1))
    .map(({ e, r }) => `${e.id || e.dataset.mode || e.className || e.tagName}[${Math.round(r.left)}..${Math.round(r.right)}]`);
  return { inner: innerWidth, scroll: document.documentElement.scrollWidth, out };
}, CONTROLS);

const browser = await chromium.launch();
const errors = [];
try {
  for (const width of WIDTHS) {
    for (const mobile of [true, false]) {
      const kind = mobile ? "phone" : "window";
      const context = await browser.newContext({
        viewport: { width, height: 844 }, isMobile: mobile, hasTouch: mobile, deviceScaleFactor: mobile ? 2 : 1,
      });
      const page = await context.newPage();
      page.on("pageerror", (e) => errors.push(e.message));
      await page.goto(`${base}/simple`);
      await page.waitForSelector('#conn[data-state="error"]', { timeout: 5000 });
      await page.waitForSelector("#banner:not([hidden])", { timeout: 5000 });

      const idle = await measure(page);
      check(`${width} ${kind}: viewport really is ${width} px`, idle.inner === width, `innerWidth ${idle.inner}`);
      check(`${width} ${kind}: scrollWidth <= ${width} (error banner up)`, idle.scroll <= width, `scrollWidth ${idle.scroll}`);
      check(`${width} ${kind}: every control inside the viewport`, idle.out.length === 0, idle.out.join(", "));

      await page.evaluate(() => {
        document.querySelector("#captions-box").hidden = false;
        document.querySelector("#voice-box").hidden = false;
        document.querySelector("#voice-chip").textContent = "Voice: older female (from face + voice)";
        document.querySelector("#captions").innerHTML =
          "<p>Thanks for coming, the keynote starts at noon in the main hall downstairs.</p>" +
          '<p class="partial">Supercalifragilisticexpialidocious-and-a-very-long-unbroken-word</p>';
        document.querySelector("#hq-active").textContent = "Playing: Auto → GPU · RTT 123 ms · fallback Local";
      });
      const busy = await measure(page);
      check(`${width} ${kind}: scrollWidth <= ${width} (captions + voice chip)`, busy.scroll <= width,
        `scrollWidth ${busy.scroll}`);
      check(`${width} ${kind}: every control inside the viewport (captions + voice chip)`, busy.out.length === 0,
        busy.out.join(", "));
      if (args.out && width === 390) await page.screenshot({ path: join(args.out, `${width}-${kind}.png`), fullPage: true });
      await context.close();
    }
  }
  check("no page errors", errors.length === 0, errors.join(" | "));
} catch (e) {
  check("run", false, e.stack || String(e));
} finally {
  await browser.close();
  server.close();
}

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
