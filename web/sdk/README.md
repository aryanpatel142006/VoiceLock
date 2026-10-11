# hearing-client.js — the browser SDK

`web/sdk/hearing-client.js` is the event-day phone app's client for `hearing serve`. It re-exports the
console's wire module (`web/console/protocol.js`: frame layout, `hello`, `control`, `sessionUrl`) and adds
`HearingClient`, an `EventTarget` that opens `/ws/session`, sends `hello`/`control`/audio/video and emits
every server message as a `CustomEvent` (`telemetry`, `caption`, `context`, `recommendation`, `error`,
`session`, `replay`, plus `message` for all of them and `audio` for processed output samples).

The server serves it at `/sdk/hearing-client.js` (and the console files at `/console/`, so its
`../console/protocol.js` import resolves over http the same as in the repo). Contract: `docs/CONTRACTS.md#events`.

## 20-line example page

Save as any `.html` served by the laptop (or paste into `web/console/`), open
`https://<laptop-ip>:8000/<page>` on the phone, tap Start:

```html
<!doctype html><meta name="viewport" content="width=device-width, initial-scale=1">
<button id="start">Start</button> <input id="s" type="range" min="0" max="100" value="70">
<select id="mode"><option>natural</option><option>focus_hq</option><option>synthetic</option>
  <option>text</option><option>quiet</option><option>auto</option></select>
<p id="status">idle</p><p id="caption"></p>
<script type="module">
  import { HearingClient } from "/sdk/hearing-client.js";
  const hc = new HearingClient({ client: "phone" });   // wss://<this host>/ws/session from location
  const $ = (id) => document.getElementById(id);
  hc.addEventListener("telemetry", ({ detail: t }) => {
    $("status").textContent = `${t.mode} · ${t.tier} · ${Math.round(t.latency_ms.total)} ms`;
  });
  hc.addEventListener("caption", ({ detail: c }) => { $("caption").textContent = c.text; });
  hc.addEventListener("error", ({ detail: e }) => { $("status").textContent = `error: ${e.message}`; });
  $("start").onclick = async () => {                  // the AudioContext must start inside this tap (iOS)
    await hc.start();                                  // mic -> engine -> speakers, hello carries the rate
    hc.control({ mode: $("mode").value, suppression: +$("s").value });
  };
  $("s").oninput = () => hc.control({ suppression: +$("s").value });
  $("mode").onchange = () => hc.control({ mode: $("mode").value });
</script>
```

Video: draw the camera into a canvas (≤ 640×360), `canvas.toBlob(..., "image/jpeg", 0.7)` and
`hc.sendVideo(await blob.arrayBuffer())` at up to 25 fps, then `hc.hello(hc.sampleRate, 20, true)` once
so the server knows video follows (see `web/console/simple.js` `startVideo`).

## Phone constraints the SDK already handles
- **Secure context**: `getUserMedia` only works on `https://` (or localhost). Start the laptop with
  `bash scripts/demo.sh --lan` (`hearing serve --host 0.0.0.0 --tls`); `sessionUrl` turns `https:` into `wss:`.
- **iOS Safari**: the `AudioContext` is created and resumed synchronously in the Start tap; it runs at the
  native rate (48 kHz on iPhones) and `hello.sample_rate` says so, the server resamples to 16 kHz and back.
  AudioWorklet needs iOS 14.5+. No background audio: the page resumes the context when it becomes
  visible again; keep the screen on during a demo.
- **Android Chrome**: `echoCancellation`, `noiseSuppression`, `autoGainControl` are off (the engine wants
  the raw room); the context runs at 16 kHz and Chrome resamples the mic.
- Checked by `scripts/phone_check.py` (Playwright iPhone 14 + Pixel 7 emulation over https).
