// hearing-client.js: the browser SDK for the selective-hearing engine (docs/CONTRACTS.md#events).
// One wire implementation: the frame layout and message builders come from the console's protocol.js.
//
//   import { HearingClient } from "https://<laptop>:8000/sdk/hearing-client.js";
//   const hc = new HearingClient();                 // ws/wss + host derived from location
//   hc.addEventListener("telemetry", (e) => show(e.detail));
//   startButton.onclick = () => hc.start();         // inside the tap: iOS needs that for audio
//
// Events (CustomEvent, payload in .detail): open, close, telemetry, caption, context,
// recommendation, error, session, replay, message (every JSON message), audio (Float32Array of
// processed output at the context rate; it is also played unless {play: false}).

import {
  KIND_AUDIO_IN, KIND_AUDIO_OUT, KIND_VIDEO_IN, MODES, audioContextOptions, control, floatToPcm16, hello,
  isIOS, packFrame, pcm16ToFloat, sessionUrl, unpackFrame,
} from "../console/protocol.js";

export {
  KIND_AUDIO_IN, KIND_AUDIO_OUT, KIND_VIDEO_IN, MODES, audioContextOptions, control, hello, isIOS, packFrame,
  sessionUrl, unpackFrame,
};

const FRAME_MS = 20;

/** getUserMedia audio constraints: the raw room (the engine does its own enhancement). */
export const AUDIO_CONSTRAINTS = {
  channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false,
};

export class HearingClient extends EventTarget {
  /** @param {{url?: string, client?: string, play?: boolean, worklet?: string}} opts */
  constructor(opts = {}) {
    super();
    this.url = opts.url || sessionUrl(globalThis.location);
    this.client = opts.client || "sdk";
    this.play = opts.play !== false;
    this.workletUrl = opts.worklet || new URL("../console/audio-worklet.js", import.meta.url).href;
    this.ws = null;
    this.audio = null; // {ctx, node, stream, rate}
    this.seq = { audio: 0, video: 0 };
    this.sampleRate = 16000;
    this.video = false;
  }

  _emit(type, detail) {
    this.dispatchEvent(new CustomEvent(type, { detail }));
  }

  /** Open the session socket; resolves once it is open. */
  connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      ws.onopen = () => { this.hello(); this._emit("open", { url: this.url }); resolve(this); };
      ws.onerror = () => reject(new Error(`cannot reach ${this.url}`));
      ws.onclose = (e) => this._emit("close", { code: e.code });
      ws.onmessage = (e) => (typeof e.data === "string" ? this._onText(e.data) : this._onBinary(e.data));
    });
  }

  _send(data) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(data);
  }

  /** Tell the server the audio rate/frame and whether video follows (sent on connect and on start). */
  hello(sampleRate = this.sampleRate, frameMs = FRAME_MS, video = this.video) {
    this._send(JSON.stringify({ ...hello(sampleRate, frameMs, video), client: this.client }));
  }

  /** Partial control update, e.g. {mode: "natural", suppression: 70}. */
  control(fields) {
    this._send(JSON.stringify(control(fields)));
  }

  /** Send mic samples (Float32Array at this.sampleRate, 10-40 ms) as one audio-in frame. */
  sendAudio(samples, captureTs = performance.timeOrigin / 1000 + performance.now() / 1000) {
    this._send(packFrame(KIND_AUDIO_IN, this.seq.audio++, captureTs, floatToPcm16(samples)));
  }

  /** Send one JPEG (ArrayBuffer or Blob bytes, <= 640x360) as a video-in frame. */
  sendVideo(jpeg, captureTs = performance.timeOrigin / 1000 + performance.now() / 1000) {
    this._send(packFrame(KIND_VIDEO_IN, this.seq.video++, captureTs, jpeg));
  }

  /**
   * Mic -> engine -> speakers. Call it from the Start tap: iOS only lets an AudioContext run when it
   * is created in a user gesture, so the context is made before anything is awaited.
   */
  async start() {
    const ctx = new AudioContext(audioContextOptions());
    const resumed = ctx.resume(); // still inside the gesture
    try {
      if (!this.ws) await this.connect();
      const stream = await navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS });
      await ctx.audioWorklet.addModule(this.workletUrl);
      const rate = ctx.sampleRate;
      const node = new AudioWorkletNode(ctx, "hearing-io", {
        numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
        processorOptions: {
          frameSamples: Math.round((rate * FRAME_MS) / 1000),
          targetSamples: Math.round(rate * 0.08), maxSamples: Math.round(rate * 0.3),
        },
      });
      node.port.onmessage = (e) => { if (e.data.type === "frame") this.sendAudio(e.data.samples); };
      ctx.createMediaStreamSource(stream).connect(node);
      node.connect(ctx.destination);
      await resumed;
      this.audio = { ctx, node, stream, rate };
      this.sampleRate = rate;
      this.hello();
      // iOS suspends audio in the background; resume when the page is shown again.
      document.addEventListener("visibilitychange", this._onVisible ??= () => {
        if (document.visibilityState === "visible") this.audio?.ctx.resume();
      });
      return this;
    } catch (e) {
      ctx.close();
      throw e;
    }
  }

  stop() {
    if (this.audio) {
      this.audio.stream.getTracks().forEach((t) => t.stop());
      this.audio.ctx.close();
      this.audio = null;
    }
  }

  close() {
    this.stop();
    this.ws?.close();
    this.ws = null;
  }

  _onText(text) {
    let msg;
    try { msg = JSON.parse(text); } catch { return; }
    this._emit("message", msg);
    if (msg.type) this._emit(msg.type, msg);
  }

  _onBinary(buffer) {
    let frame;
    try { frame = unpackFrame(buffer); } catch { return; }
    if (frame.kind !== KIND_AUDIO_OUT) return;
    const samples = pcm16ToFloat(frame.payload);
    this._emit("audio", samples);
    if (this.play && this.audio) this.audio.node.port.postMessage({ type: "play", samples }, [samples.buffer]);
  }
}
