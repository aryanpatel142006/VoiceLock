// Usage: PLAYWRIGHT=/path/to/playwright node web/console/tests/replay-e2e.mjs http://127.0.0.1:PORT
// The test server must have a local grid-two-face.mp4 and mock Text ASR (no device permission).
import assert from 'node:assert/strict';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT || 'playwright');
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROMIUM, args: ['--autoplay-policy=no-user-gesture-required'] });
try {
  for (const path of ['/', '/simple']) {
    const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.addInitScript(() => {
      navigator.mediaDevices.getUserMedia = () => { throw new Error('Replay requested a device'); };
    });
    await page.goto(new URL(path, process.argv[2]).href);
    await page.waitForFunction(() => (window.hearingConsole || window.hearingSimple)?.ws?.readyState === 1);
    await page.getByRole('button', { name: 'Replay clip', exact: true }).click();
    await page.locator('#replay-badge').waitFor({ state: 'visible' });
    await page.waitForFunction(() => {
      const s = window.hearingConsole || window.hearingSimple;
      return s.framesBack > 20 && document.querySelector('#replay-preview').naturalWidth > 0;
    });
    await page.locator('#modes button[data-mode=text], #modes input[value=text]').click();
    await page.waitForFunction(() => document.querySelector('#captions').textContent.includes('Replay captions demonstrate'));
    await page.screenshot({ path: join(tmpdir(), `t066-${path === '/' ? 'console' : 'simple'}.png`) });
    await page.getByRole('button', { name: 'Stop replay', exact: true }).click();
    await page.locator('#replay-badge').waitFor({ state: 'hidden' });
    assert.deepEqual(errors, []);
    console.log(`${path}: replay preview, processed audio, Text captions, stop; no getUserMedia`);
    await page.close();
  }
} finally { await browser.close(); }
