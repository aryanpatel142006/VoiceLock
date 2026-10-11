// Full page modules against a real local session server, with device-free DOM/audio adapters.
// JSDOM=/path/to/jsdom node --experimental-vm-modules web/console/tests/replay-dom.mjs http://127.0.0.1:PORT
// Browser visual/audio QA remains the separate replay-e2e.mjs check.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { SourceTextModule } from 'node:vm';
const require = createRequire(import.meta.url);
const { JSDOM } = require(process.env.JSDOM || 'jsdom');
const WebSocket = require(require.resolve('ws', { paths: [process.env.JSDOM || 'jsdom'] }));
const root = resolve('web/console');
const wait = async (check) => {
  const until = Date.now() + 20000;
  while (!check()) {
    if (Date.now() > until) throw new Error('Timed out waiting for replay page');
    await new Promise(r => setTimeout(r, 20));
  }
};
const failures = [];
for (const [path, html, entry, stateKey] of [
  ['/', 'index.html', 'app.js', 'hearingConsole'],
  ['/simple', 'simple.html', 'simple.js', 'hearingSimple'],
]) {
  const dom = new JSDOM(await readFile(resolve(root, html), 'utf8'), {
    url: new URL(path, process.argv[2]).href, runScripts: 'outside-only', pretendToBeVisual: true,
  });
  const w = dom.window;
  let deviceCalls = 0, played = 0, raw = 0, allowDevices = false, stopped = 0;
  w.WebSocket = WebSocket;
  w.fetch = fetch;
  w.Blob = Blob;
  w.URL.createObjectURL = URL.createObjectURL;
  w.URL.revokeObjectURL = URL.revokeObjectURL;
  w.navigator.mediaDevices = { async getUserMedia() {
    deviceCalls++;
    if (!allowDevices) throw new Error('device access');
    return { getTracks: () => [{ stop() { stopped++; } }] };
  } };
  w.HTMLMediaElement.prototype.play = async () => {};
  Object.defineProperty(w.HTMLVideoElement.prototype, 'videoWidth', { get: () => 640 });
  Object.defineProperty(w.HTMLVideoElement.prototype, 'videoHeight', { get: () => 360 });
  w.MediaStream = class { getTracks() { return []; } };
  w.AudioContext = class {
    sampleRate = 16000;
    destination = {};
    audioWorklet = { addModule: async () => {} };
    resume() { return Promise.resolve(); }
    close() { return Promise.resolve(); }
    createMediaStreamSource() { return { connect() {} }; }
  };
  w.AudioWorkletNode = class {
    port = { postMessage(message) {
      if (message.type === 'play') played += message.samples.length;
      if (message.type === 'replay-raw') raw += message.samples.length;
    }};
    connect() {}
  };
  w.HTMLCanvasElement.prototype.getContext = () => ({
    clearRect() {}, strokeRect() {}, fillRect() {}, fillText() {}, drawImage() {},
    measureText: () => ({ width: 100 }),
  });
  w.HTMLCanvasElement.prototype.toBlob = (callback) => callback(null);
  const cache = new Map();
  async function module(file) {
    if (!cache.has(file)) {
      const m = new SourceTextModule(await readFile(file, 'utf8'), { context: dom.getInternalVMContext(), identifier: file });
      cache.set(file, m);
      await m.link((specifier) => module(resolve(root, specifier)));
    }
    return cache.get(file);
  }
  try {
    await (await module(resolve(root, entry))).evaluate();
    await wait(() => w[stateKey]?.sessionId);
    w.document.querySelector('#replay').click();
    await wait(() => !w.document.querySelector('#replay-badge').hidden && played > 6400 && raw > 6400);
    assert.ok(w.document.querySelector('#replay-preview').src.startsWith('blob:'));
    assert.equal(w[stateKey].video.stream.getTracks().length, 0);
    w.document.querySelector('#modes button[data-mode=text], #modes input[value=text]').click();
    await wait(() => w.document.querySelector('#captions').textContent.includes('Replay captions demonstrate'));
    w.document.querySelector('#replay').click();
    await wait(() => w.document.querySelector('#replay-badge').hidden);
    assert.equal(w[stateKey].audio, null);
    assert.equal(deviceCalls, 0);
    console.log(`${path}: preview, ${played} processed / ${raw} raw samples, captions, stop; zero device calls`);
    // Catch camera initialization regressions in the real page's Start handler,
    // while replacing only hardware access and browser media/canvas primitives.
    allowDevices = true;
    w.document.querySelector('#start').click();
    await wait(() => !w.document.querySelector('#start').disabled);
    assert.ok(w[stateKey].video, `${path}: live Start must initialize camera capture`);
    assert.equal(w.document.querySelector('#video').srcObject, w[stateKey].video.stream);
    assert.equal(w.document.querySelector('#view').classList.contains('live'), true);
    assert.equal(w.document.querySelector('#start').getAttribute('aria-pressed'), 'true');
    assert.equal(w.document.querySelector('#replay-badge').hidden, true);
    assert.equal(deviceCalls, 2);
    w.document.querySelector('#start').click();
    assert.equal(w[stateKey].video, null);
    assert.equal(w[stateKey].audio, null);
    assert.equal(w.document.querySelector('#video').srcObject, null);
    assert.equal(stopped, 2);
    console.log(`${path}: live Start initializes camera and audio; Stop releases both tracks`);
  } catch (error) {
    failures.push(error);
  } finally {
    const state = w[stateKey];
    if (state?.video?.timer) w.clearInterval(state.video.timer);
    const ws = w[stateKey]?.ws;
    if (ws) { ws.onclose = ws.onmessage = ws.onerror = null; ws.close(); }
    dom.window.close();
  }
}
if (failures.length) throw new AggregateError(failures, 'Replay/live page regressions');
