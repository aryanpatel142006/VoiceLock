// VoiceLock web demo: a stand-in for the Python server, running in the visitor's browser.
//
// The hosted demo has no audio engine behind it, so this script (loaded before the app) answers the app in the
// server's own protocol (docs/CONTRACTS.md):
//   - the listening "mic" is a recorded cafe clip (deploy/media/room.wav: Ahmad and Joseph talking over each other);
//   - /ws/session plays back the same clip after our fine-tuned separator kept only Ahmad (voice.wav), processed
//     offline earlier, frame-aligned to the mic and 1.5 s behind like the real engine; the room dial mixes the room
//     back in, Quiet sends captions instead of sound, SOS is acknowledged as the server does;
//   - /ws/who replays a recorded run of the fine-tuned speaker recognizer on clips of Aryan, then Joseph;
//   - /comfort/* goes to a Vercel function that runs hearing/server/comfort.py (real Gemini and ElevenLabs), with
//     this visitor's profiles kept in their own browser and sent along with each request.
// Telemetry, health and the Who's speaking messages are recordings from the real server (deploy/demo/data/).
// Talk to Me's hold-to-talk asks for the visitor's real microphone; only the listening path is replaced.
(() => {
  "use strict";
  const BASE = "/demo/";
  const SR = 16000;
  const HEADER = 13;
  const KIND_IN = 1;
  const KIND_OUT = 3;
  const PROFILES_KEY = "voicelock-demo-profiles";
  const page = location.pathname.startsWith("/who") ? "who" : "app";

  // ------------------------------------------------------------------ recorded data
  const json = (name) => fetchReal(`${BASE}data/${name}.json`).then((r) => r.json());
  const realFetch = window.fetch.bind(window);
  function fetchReal(...a) { return realFetch(...a); }
  const data = {};
  const ready = Promise.all(["health", "telemetry_idle", "telemetry_live", "context", "captions", "who_ready", "who_status", "who_log"]
    .map((n) => json(n).then((v) => { data[n] = v; }))).catch((e) => console.error("demo data failed to load", e));

  // ------------------------------------------------------------------ audio clips
  let decodeCtx = null;
  const clips = {};
  function clip(name) {
    clips[name] ||= fetchReal(`${BASE}media/${name}.wav`).then((r) => r.arrayBuffer()).then((buf) => {
      decodeCtx ||= new OfflineAudioContext(1, 1, SR);
      return decodeCtx.decodeAudioData(buf);
    }).then((ab) => (ab.sampleRate === SR ? ab.getChannelData(0) : resample(ab.getChannelData(0), ab.sampleRate)));
    return clips[name];
  }
  function resample(x, from) {
    const n = Math.round((x.length * SR) / from);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const p = (i * from) / SR;
      const j = Math.floor(p);
      const f = p - j;
      out[i] = (x[j] || 0) * (1 - f) + (x[j + 1] || 0) * f;
    }
    return out;
  }

  // ------------------------------------------------------------------ the fake listening mic
  // `mic` is the clip playing into the app: its start time decides which sample of the processed clip goes back.
  const mic = { startedAt: 0, name: "room", length: 0 };
  let micCtx = null;
  const realGum = navigator.mediaDevices?.getUserMedia?.bind(navigator.mediaDevices);
  async function demoMic() {
    const name = page === "who" ? "who" : "room";
    const samples = await clip(name);
    micCtx ||= new AudioContext({ sampleRate: SR });
    await micCtx.resume().catch(() => {});
    const buffer = micCtx.createBuffer(1, samples.length, SR);
    buffer.copyToChannel(samples, 0);
    const src = micCtx.createBufferSource();
    src.buffer = buffer;
    src.loop = true;
    const dest = micCtx.createMediaStreamDestination();
    src.connect(dest);
    src.start();
    Object.assign(mic, { startedAt: performance.now(), name, length: samples.length });
    const track = dest.stream.getAudioTracks()[0];
    const stop = track.stop.bind(track);
    track.stop = () => { try { src.stop(); } catch { /* already stopped */ } stop(); };
    Object.defineProperty(track, "label", { value: "Demo: recorded cafe clip" });
    return dest.stream;
  }
  if (realGum) {
    navigator.mediaDevices.getUserMedia = (constraints = {}) => {
      // The listening path asks with echoCancellation off (web/console/devices.js audioConstraints);
      // Talk to Me's hold-to-talk asks with it on and gets the visitor's real mic.
      const a = constraints.audio;
      if (a && typeof a === "object" && a.echoCancellation === false) return demoMic();
      return realGum(constraints);
    };
  }
  const micPos = () => Math.floor(((performance.now() - mic.startedAt) / 1000) * SR);

  // ------------------------------------------------------------------ frames
  function pack(kind, seq, ts, samples) {
    const out = new ArrayBuffer(HEADER + samples.length * 2);
    const v = new DataView(out);
    v.setUint8(0, kind);
    v.setUint32(1, seq >>> 0, true);
    v.setFloat64(5, ts, true);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(HEADER + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return out;
  }
  function unpack(buf) {
    const v = new DataView(buf);
    return { kind: v.getUint8(0), seq: v.getUint32(1, true), ts: v.getFloat64(5, true), n: (buf.byteLength - HEADER) / 2 };
  }

  // ------------------------------------------------------------------ a socket that never leaves the browser
  class DemoSocket extends EventTarget {
    constructor(url) {
      super();
      this.url = url;
      this.readyState = 0;
      this.binaryType = "blob";
      this.protocol = "";
      this.extensions = "";
      this.bufferedAmount = 0;
      ["onopen", "onmessage", "onclose", "onerror"].forEach((k) => { this[k] = null; });
      setTimeout(() => ready.then(() => {
        if (this.readyState !== 0) return;
        this.readyState = 1;
        this._fire("open", new Event("open"));
        this.opened?.();
      }), 30);
    }
    _fire(type, ev) {
      this.dispatchEvent(ev);
      this[`on${type}`]?.(ev);
    }
    deliver(payload) {
      if (this.readyState !== 1) return;
      this._fire("message", new MessageEvent("message", { data: typeof payload === "string" ? payload : payload }));
    }
    sendJson(obj) { this.deliver(JSON.stringify(obj)); }
    send(msg) {
      if (this.readyState !== 1) return;
      if (typeof msg === "string") {
        let m;
        try { m = JSON.parse(msg); } catch { return; }
        this.onText?.(m);
      } else {
        const buf = msg instanceof ArrayBuffer ? msg : msg.buffer;
        this.onFrame?.(buf);
      }
    }
    close() {
      if (this.readyState >= 2) return;
      this.readyState = 3;
      this.closed?.();
      this._fire("close", new CloseEvent("close", { code: 1000, wasClean: true }));
    }
  }
  DemoSocket.CONNECTING = 0;
  DemoSocket.OPEN = 1;
  DemoSocket.CLOSING = 2;
  DemoSocket.CLOSED = 3;

  // ------------------------------------------------------------------ /ws/session
  function sessionSocket(ws) {
    const st = { mode: "focus_hq", ab: "processed", suppression: 100, sos: false, seq: 0, outPos: null, streaming: false,
      lastFrameAt: 0, since: performance.now(), captionAt: -1, timers: [] };
    const delay = Math.round(((data.telemetry_live?.tiers?.delay_ms || 1500) / 1000) * SR);
    const sessionId = `demo-${Math.random().toString(16).slice(2, 10)}`;
    let room = null;
    let voice = null;
    Promise.all([clip("room"), clip("voice")]).then(([r, v]) => { room = r; voice = v; });

    function telemetry() {
      const live = st.streaming && performance.now() - st.lastFrameAt < 1000;
      const t = structuredClone(live ? data.telemetry_live : data.telemetry_idle);
      Object.assign(t, { seq: ++st.seq, mode: st.mode, effective_mode: st.mode, ab: st.ab, preset: null, sos: { on: st.sos },
        recording: false, started_at: new Date(Date.now() - (performance.now() - st.since)).toISOString(), pid: 0 });
      if (live && t.acquisition) t.acquisition.since_s = Math.round((performance.now() - st.lastStart) / 100) / 10;
      return t;
    }
    ws.opened = () => {
      ws.sendJson({ type: "session", session_id: sessionId });
      st.timers.push(setInterval(() => ws.sendJson(telemetry()), 100));
      st.timers.push(setInterval(() => ws.sendJson({ ...data.context, mode: st.mode }), 1000));
    };
    ws.closed = () => st.timers.forEach(clearInterval);
    ws.onText = (m) => {
      if (m.type === "hello") return;
      if (m.type !== "control") return;
      if (m.mode) st.mode = m.mode === "quiet" ? "quiet" : "focus_hq";
      if (m.ab) st.ab = m.ab;
      if (typeof m.suppression === "number") st.suppression = m.suppression;
      if (m.profile === "judge" && m.suppression == null) st.suppression = 100;
      if (typeof m.sos === "boolean") {
        st.sos = m.sos;
        ws.sendJson({ type: "sos", on: m.sos, mode: st.mode, t: Date.now() / 1000 });
      }
    };
    ws.onFrame = (buf) => {
      const f = unpack(buf);
      if (f.kind !== KIND_IN || !room || !voice) return;
      const now = performance.now();
      if (!st.streaming || now - st.lastFrameAt > 500) { // a (re)start: line the output up with the clip again
        st.outPos = micPos() - f.n - delay;
        st.lastStart = now;
        st.captionAt = -1;
      }
      st.streaming = true;
      st.lastFrameAt = now;
      const out = new Float32Array(f.n);
      if (st.mode !== "quiet") {
        const keepRoom = 1 - Math.max(0, Math.min(100, st.suppression)) / 100;
        const len = voice.length;
        for (let i = 0; i < f.n; i++) {
          const p = st.outPos + i;
          if (p < 0) continue; // the first 1.5 s: the engine has not caught up yet
          const k = p % len;
          out[i] = voice[k] + keepRoom * (room[k] - voice[k]);
        }
      }
      st.outPos += f.n;
      ws.deliver(pack(KIND_OUT, f.seq, f.ts, out));
      if (st.mode === "quiet") captions();
    };
    function captions() {
      const len = voice.length;
      if (st.outPos < 0) return;
      const t = (st.outPos % len) / SR;
      const lap = Math.floor(st.outPos / len);
      for (const [i, c] of data.captions.entries()) {
        const key = lap * 1000 + i;
        if (key <= st.captionAt || t < c.t1) continue;
        if (t - c.t1 > 1) { st.captionAt = key; continue; } // passed while not in Quiet
        st.captionAt = key;
        const nowS = Date.now() / 1000;
        ws.sendJson({ type: "caption", text: c.text, final: true, t0: nowS - (c.t1 - c.t0), t1: nowS });
      }
    }
  }

  // ------------------------------------------------------------------ /ws/who
  function whoSocket(ws) {
    const st = { timers: [], start: null, next: 0, lap: 0, lastFrameAt: 0 };
    ws.opened = () => {
      ws.sendJson({ type: "who_status", state: "loading" });
      setTimeout(() => ws.sendJson({ ...data.who_ready, sessions_active: 0 }), 600);
      st.timers.push(setInterval(tick, 50));
    };
    ws.closed = () => st.timers.forEach(clearInterval);
    ws.onText = () => {};
    ws.onFrame = () => {
      const now = performance.now();
      if (st.start === null || now - st.lastFrameAt > 500) Object.assign(st, { start: mic.startedAt || now, next: 0, lap: 0 });
      st.lastFrameAt = now;
    };
    function tick() {
      if (st.start === null || performance.now() - st.lastFrameAt > 500) return;
      const lapS = (mic.length || 28 * SR) / SR;
      const t = (performance.now() - st.start) / 1000 - st.lap * lapS;
      if (t >= lapS) { st.lap += 1; st.next = 0; return; }
      const log = data.who_log;
      while (st.next < log.length && log[st.next][0] <= t) {
        ws.sendJson({ ...log[st.next][1], t: Math.round((st.lap * lapS + log[st.next][0]) * 1000) / 1000 });
        st.next += 1;
      }
    }
  }

  // ------------------------------------------------------------------ the phone companion's live list: nothing to follow
  function watchSocket(ws) { ws.opened = () => {}; ws.onText = () => {}; ws.onFrame = () => {}; }

  const RealWebSocket = window.WebSocket;
  function DemoWebSocket(url, protocols) {
    const path = new URL(url, location.href).pathname;
    if (path === "/ws/session") { const ws = new DemoSocket(url); sessionSocket(ws); return ws; }
    if (path === "/ws/who") { const ws = new DemoSocket(url); whoSocket(ws); return ws; }
    if (path === "/ws/watch") { const ws = new DemoSocket(url); watchSocket(ws); return ws; }
    return new RealWebSocket(url, protocols);
  }
  Object.assign(DemoWebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3, prototype: RealWebSocket.prototype });
  window.WebSocket = DemoWebSocket;

  // ------------------------------------------------------------------ HTTP routes
  const jsonResponse = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  function loadProfiles() {
    try { return JSON.parse(localStorage.getItem(PROFILES_KEY) || "{}") || {}; } catch { return {}; }
  }
  function saveProfiles(p) {
    try { localStorage.setItem(PROFILES_KEY, JSON.stringify(p)); } catch { /* private window: this visit only */ }
  }
  function remember(body) {
    const p = loadProfiles();
    const add = (x) => { if (x && typeof x === "object" && typeof x.id === "string") p[x.id] = x; };
    if (Array.isArray(body)) body.forEach(add);
    else if (body && typeof body === "object") { add(body.profile); if ("preferred_name" in body) add(body); }
    saveProfiles(p);
  }
  function profilesHeader() {
    const text = JSON.stringify(loadProfiles());
    const bytes = new TextEncoder().encode(text);
    if (bytes.length > 48000) return null; // never send more than the function accepts; it falls back to the seeds
    let bin = "";
    bytes.forEach((b) => { bin += String.fromCharCode(b); });
    return btoa(bin);
  }
  async function comfort(url, init = {}) {
    const headers = new Headers(init.headers || {});
    const h = profilesHeader();
    if (h) headers.set("x-voicelock-profiles", h);
    const res = await realFetch(`/api/comfort?path=${encodeURIComponent(url.pathname)}`, { ...init, headers });
    const copy = res.clone();
    if (copy.ok) copy.json().then(remember).catch(() => {});
    return res;
  }
  window.fetch = async (input, init) => {
    const raw = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const url = new URL(raw, location.href);
    if (url.origin === location.origin) {
      await ready;
      const p = url.pathname;
      if (p === "/health") return jsonResponse(data.health);
      if (p === "/sessions") return jsonResponse([]);
      if (p === "/who/status") return jsonResponse(data.who_status);
      if (p === "/replay/clips") return jsonResponse({ default: null, clips: [] });
      if (p.startsWith("/comfort/")) {
        const req = typeof input === "object" && !(input instanceof URL) ? input : null;
        return comfort(url, { method: init?.method || req?.method || "GET", headers: init?.headers || req?.headers,
          body: init?.body ?? (req && req.method !== "GET" ? await req.text() : undefined), signal: init?.signal });
      }
    }
    return realFetch(input, init);
  };

  // ------------------------------------------------------------------ "What is this?"
  const SEEN_KEY = "voicelock-demo-intro-seen";
  function el(tag, attrs = {}, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    n.append(...kids);
    return n;
  }
  function intro() {
    const close = () => { dialog.close(); try { localStorage.setItem(SEEN_KEY, "1"); } catch { /* ignore */ } };
    const dialog = el("dialog", { class: "vl-demo-intro", "aria-labelledby": "vl-demo-title" },
      el("p", { class: "vl-demo-kicker", text: "HackRU Fall 2026 · web demo" }),
      el("h2", { id: "vl-demo-title", text: "VoiceLock helps you hear one person in a loud room." }),
      el("p", { text: "It's built for people who get overwhelmed by noise, like many autistic people in a café or a crowded hall. VoiceLock keeps the voice closest to you and turns everyone else down. When it's still too much, one press of SOS cuts the room and offers a calm sound, captions, or a voice to talk to." }),
      el("h3", { text: "What's real in this demo" }),
      el("ul", {},
        el("li", { text: "The microphone is a recording: 30 seconds of Ahmad and Joseph talking over each other in café noise. After Start you hear the same 30 seconds after our model kept only Ahmad. That part was processed earlier on a laptop, not live in your browser, and Quiet's captions are a transcript of it." }),
        el("li", { text: "Talk to Me is live. Gemini writes the replies and ElevenLabs speaks them. Hold to talk uses your own microphone." }),
        el("li", { text: "Who's speaking replays a real run of our fine-tuned model on recordings of Aryan, then Joseph." }),
        el("li", { text: "Headphones help. Space switches between the kept voice and the whole room." })),
      el("div", { class: "vl-demo-actions" },
        el("button", { type: "button", class: "vl-demo-go", text: "Try it", onclick: close }),
        el("a", { href: "/app/#about", onclick: close, text: "Results and limits" }),
        el("a", { href: "https://github.com/aryanpatel142006/VoiceLock", target: "_blank", rel: "noopener", text: "Code on GitHub" })));
    dialog.addEventListener("cancel", close);
    return dialog;
  }
  function chrome() {
    const style = el("style", {}, `
      .vl-demo-intro{max-width:36rem;width:calc(100% - 2rem);border:1px solid var(--color-line,#333);border-radius:16px;
        background:var(--color-bg-raise,#111);color:var(--color-fg,#eee);padding:1.75rem;font-family:var(--font-ui,system-ui);line-height:1.5}
      .vl-demo-intro::backdrop{background:rgb(0 0 0 / .7)}
      .vl-demo-intro h2{font-size:1.5rem;line-height:1.2;margin:.25rem 0 .75rem}
      .vl-demo-intro h3{font-size:1rem;margin:1.25rem 0 .5rem}
      .vl-demo-intro p,.vl-demo-intro li{color:var(--color-dim,#bbb);font-size:.95rem}
      .vl-demo-intro ul{padding-left:1.1rem;margin:0}.vl-demo-intro li{margin:.35rem 0}
      .vl-demo-kicker{font-family:var(--font-mono,monospace);font-size:.75rem!important;letter-spacing:.04em;margin:0}
      .vl-demo-actions{display:flex;flex-wrap:wrap;gap:.75rem 1.25rem;align-items:center;margin-top:1.5rem}
      .vl-demo-actions a{color:var(--color-fg,#eee)}
      .vl-demo-go{font:inherit;font-weight:600;background:var(--color-fg,#eee);color:var(--color-bg,#000);border:0;border-radius:999px;padding:.6rem 1.4rem;cursor:pointer}
      .vl-demo-chip{position:fixed;left:1rem;bottom:1rem;z-index:2147483000;font:500 .8rem var(--font-ui,system-ui);
        background:var(--color-bg-raise,#111);color:var(--color-dim,#bbb);border:1px solid var(--color-line,#333);border-radius:999px;padding:.45rem .9rem;cursor:pointer}
      .vl-demo-chip:hover{color:var(--color-fg,#eee)}`);
    const dialog = intro();
    const chip = el("button", { type: "button", class: "vl-demo-chip", text: "Demo · recorded audio · What is this?", onclick: () => dialog.showModal() });
    document.head.append(style);
    document.body.append(dialog, chip);
    let seen = false;
    try { seen = localStorage.getItem(SEEN_KEY) === "1"; } catch { /* ignore */ }
    if (!seen && page === "app") dialog.showModal();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", chrome);
  else chrome();
})();
