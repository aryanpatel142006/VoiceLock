// AudioWorklet for the console: mic capture in frames, and a jitter buffer for playback.
//
// Capture: input channel 0 is cut into frames of `frameSamples` and posted to the main
// thread as {type: "frame", samples} (the buffer is transferred).
// Playback: the main thread posts {type: "play", samples}; the processor waits until
// `targetSamples` are buffered, plays from the ring, and on underrun goes silent and
// re-buffers. If the queue grows past `maxSamples` (server burst, clock drift) the
// oldest audio is dropped back to the target so latency cannot creep up.
// A/B: the raw mic is always kept in `rawRing`. {type: "ab", state: "raw"|"processed", delayMs}
// crossfades (AB_FADE_MS) between the engine output and the raw mic delayed by `delayMs`
// (the engine's declared latency, so both are time-aligned). The engine ring is consumed in
// both states, so it never backs up; the main thread keeps sending audio in both.
// Every ~100 ms it posts {type: "stats", buffered, underruns, dropped}.
// AGC (T-135, opt-in with {type: "agc", on: true}; the judge page): the raw mic is levelled by the
// same level rule as the server's output AGC (hearing/audio/agc.py: -20 dBFS, max +12 / -20 dB, 50 / 800 ms
// envelope, hold below -50 dBFS, soft limit above 0.8) before it enters `rawRing`, so RAW and SELECTIVE play
// at the same loudness. T-179 floor rule ({type: "agc", on: true, floorRule: true}, sent when the server's
// telemetry `agc.floor_rule` is true): the floor is max(-50 dBFS, noise floor + 10 dB) (noise floor = 10th
// percentile of the last 1.5 s of block levels), under it the gain only moves back toward 0 dB, never up,
// and the gain is capped at +9 dB. Neither mode ports the server's T-169 noise hold or pause relax.

const AGC = {
  targetDb: -20, maxGainDb: 12, maxCutDb: 20, attackMs: 50, releaseMs: 800, floorDb: -50, holdDb: 12, blockMs: 10, limit: 0.8,
  noiseWindowMs: 1500, noiseMarginDb: 10, noisePercentile: 10, floorRuleMaxGainDb: 9,
};

/** numpy.percentile (linear interpolation) of an unsorted array. */
function percentile(values, p) {
  const v = values.slice().sort((a, b) => a - b);
  const r = (p / 100) * (v.length - 1);
  const i = Math.floor(r);
  return i + 1 < v.length ? v[i] + (v[i + 1] - v[i]) * (r - i) : v[i];
}

/** Streaming AGC + soft limiter: a port of the level and floor rules of hearing.audio.agc.Agc (no pause relax). */
class AgcState {
  constructor(rate, floorRule = false) {
    this.floorRule = floorRule; // T-179: noise-aware floor, never up under it, max +9 dB
    this.maxGainDb = floorRule ? AGC.floorRuleMaxGainDb : AGC.maxGainDb;
    this.block = Math.max(1, Math.round((rate * AGC.blockMs) / 1000));
    this.attack = Math.exp(-AGC.blockMs / AGC.attackMs);
    this.release = Math.exp(-AGC.blockMs / AGC.releaseMs);
    this.envDb = null;
    this.gainDb = 0; // the gain applied this block
    this.levels = []; // the last noiseWindowMs of block levels
    this.noiseDb = null;
    this.g0 = 1;
    this.g1 = 1;
    this.pos = 0;
    this.acc = 0;
  }

  /** One sample in, one levelled and soft-limited sample out. */
  step(x) {
    this.pos++;
    let y = x * (this.g0 + ((this.g1 - this.g0) * this.pos) / this.block);
    this.acc += x * x;
    if (this.pos === this.block) {
      const level = 10 * Math.log10(this.acc / this.block + 1e-10);
      let floor = AGC.floorDb;
      if (this.floorRule) {
        this.levels.push(level);
        if (this.levels.length > Math.round(AGC.noiseWindowMs / AGC.blockMs)) this.levels.shift();
        this.noiseDb = percentile(this.levels, AGC.noisePercentile);
        floor = Math.max(AGC.floorDb, this.noiseDb + AGC.noiseMarginDb);
      }
      const below = level < floor;
      // blocks under the effective floor or holdDb under the envelope (pauses) hold it (hearing/audio/agc.py)
      if (!below && (this.envDb == null || level >= this.envDb - AGC.holdDb)) {
        if (this.envDb == null) this.envDb = level;
        else {
          const c = level > this.envDb ? this.attack : this.release;
          this.envDb = c * this.envDb + (1 - c) * level;
        }
      }
      let g = this.envDb == null ? 0 : Math.max(-AGC.maxCutDb, Math.min(this.maxGainDb, AGC.targetDb - this.envDb));
      if (below && this.floorRule) g = Math.min(g, Math.max(this.gainDb, 0)); // under the floor: toward 0 dB, never up
      this.gainDb = g;
      this.g0 = this.g1;
      this.g1 = 10 ** (this.gainDb / 20);
      this.pos = 0;
      this.acc = 0;
    }
    const a = Math.abs(y);
    if (a > AGC.limit) {
      const head = 1 - AGC.limit;
      y = Math.sign(y) * (AGC.limit + head * Math.tanh((a - AGC.limit) / head));
    }
    return y;
  }
}

class HearingIO extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const o = options.processorOptions || {};
    this.frameSamples = o.frameSamples || 320;
    this.targetSamples = o.targetSamples || Math.round(sampleRate * 0.04);
    this.maxSamples = o.maxSamples || Math.round(sampleRate * 0.16);
    this.replayRaw = [];
    this.replayOffset = 0;
    this.frame = new Float32Array(this.frameSamples);
    this.fill = 0;

    this.ring = new Float32Array(Math.max(sampleRate * 2, this.maxSamples * 2));
    this.read = 0;
    this.write = 0;
    this.buffered = 0;
    this.playing = false;
    this.underruns = 0;
    this.dropped = 0;
    this.statsEvery = Math.round(sampleRate / 10);
    this.sinceStats = 0;

    this.rawRing = new Float32Array(Math.round(sampleRate * 2));
    this.rawWrite = 0; // total samples written (never wrapped; index modulo the ring)
    this.rawGain = 0; // 0 = engine output, 1 = delayed raw mic
    this.rawTarget = 0;
    this.rawStep = 1 / Math.round(sampleRate * 0.1); // 100 ms crossfade
    this.rawDelay = 0; // samples, slews toward rawDelayTarget
    this.rawDelayTarget = 0;
    this.maxDelay = this.rawRing.length - 2;
    this.rawAgc = null; // AgcState while {type: "agc", on: true}

    this.port.onmessage = (e) => this.onMessage(e.data);
  }

  onMessage(msg) {
    if (msg.type === "replay-raw") {
      this.replayRaw.push(msg.samples);
      if (this.replayRaw.length > 8) { this.replayRaw.shift(); this.replayOffset = 0; }
    }
    else if (msg.type === "play") this.enqueue(msg.samples);
    else if (msg.type === "agc") {
      const floorRule = !!msg.floorRule;
      if (!msg.on) this.rawAgc = null;
      else if (!this.rawAgc || this.rawAgc.floorRule !== floorRule) this.rawAgc = new AgcState(sampleRate, floorRule);
    }
    else if (msg.type === "ab") {
      if (msg.state) this.rawTarget = msg.state === "raw" ? 1 : 0;
      if (msg.delayMs != null) {
        this.rawDelayTarget = Math.max(0, Math.min(this.maxDelay, (msg.delayMs * sampleRate) / 1000));
        if (!this.rawSeeded) this.rawDelay = this.rawDelayTarget;
        this.rawSeeded = true;
      }
    }
    else if (msg.type === "flush") {
      this.read = this.write = this.buffered = 0;
      this.playing = false;
    } else if (msg.type === "config") {
      if (msg.targetSamples) this.targetSamples = msg.targetSamples;
      if (msg.maxSamples) this.maxSamples = msg.maxSamples;
    }
  }

  enqueue(samples) {
    const cap = this.ring.length;
    for (let i = 0; i < samples.length; i++) {
      this.ring[this.write] = samples[i];
      this.write = (this.write + 1) % cap;
    }
    this.buffered += samples.length;
    if (this.buffered > cap) { // overwrote unread audio: resync the read head
      this.dropped += this.buffered - cap;
      this.buffered = cap;
      this.read = this.write;
    }
    if (this.buffered > this.maxSamples) {
      const drop = this.buffered - this.targetSamples;
      this.read = (this.read + drop) % cap;
      this.buffered -= drop;
      this.dropped += drop;
    }
  }

  /** Crossfade `buf` (engine output, silence while buffering) with the delayed raw mic. */
  mixRaw(buf, rawStart) {
    const rl = this.rawRing.length;
    for (let k = 0; k < buf.length; k++) {
      const d = this.rawDelayTarget - this.rawDelay;
      this.rawDelay += Math.max(-0.02, Math.min(0.02, d)); // slew: at most 2% speed change
      if (this.rawGain < this.rawTarget) this.rawGain = Math.min(this.rawTarget, this.rawGain + this.rawStep);
      else if (this.rawGain > this.rawTarget) this.rawGain = Math.max(this.rawTarget, this.rawGain - this.rawStep);
      const pos = rawStart + k - this.rawDelay;
      const i0 = Math.floor(pos);
      const frac = pos - i0;
      const a = pos < 0 ? 0 : this.rawRing[((i0 % rl) + rl) % rl];
      const b = pos + 1 < 0 ? 0 : this.rawRing[(((i0 + 1) % rl) + rl) % rl];
      buf[k] = buf[k] * (1 - this.rawGain) + (a + (b - a) * frac) * this.rawGain;
    }
  }

  process(inputs, outputs) {
    let input = inputs[0] && inputs[0][0];
    const qn = outputs[0] && outputs[0][0] ? outputs[0][0].length : 128;
    if (!input && this.replayRaw.length) {
      input = new Float32Array(qn);
      for (let k = 0; k < qn && this.replayRaw.length; k++) {
        input[k] = this.replayRaw[0][this.replayOffset++];
        if (this.replayOffset === this.replayRaw[0].length) {
          this.replayRaw.shift();
          this.replayOffset = 0;
        }
      }
    }
    const rawStart = this.rawWrite;
    if (input) {
      const rl = this.rawRing.length;
      const agc = this.rawAgc;
      for (let k = 0; k < input.length; k++) this.rawRing[(this.rawWrite + k) % rl] = agc ? agc.step(input[k]) : input[k];
      this.rawWrite += input.length;
    } else {
      this.rawWrite += qn; // no mic: keep the timeline moving (the ring holds zeros)
    }
    if (input) {
      let i = 0;
      while (i < input.length) {
        const n = Math.min(input.length - i, this.frameSamples - this.fill);
        this.frame.set(input.subarray(i, i + n), this.fill);
        this.fill += n;
        i += n;
        if (this.fill === this.frameSamples) {
          this.port.postMessage({ type: "frame", samples: this.frame }, [this.frame.buffer]);
          this.frame = new Float32Array(this.frameSamples);
          this.fill = 0;
        }
      }
    }

    const out = outputs[0];
    const n = out[0] ? out[0].length : 128;
    if (!this.playing && this.buffered >= this.targetSamples) this.playing = true;
    if (this.playing && out[0]) {
      const cap = this.ring.length;
      const take = Math.min(n, this.buffered);
      for (let k = 0; k < take; k++) {
        out[0][k] = this.ring[this.read];
        this.read = (this.read + 1) % cap;
      }
      this.buffered -= take;
      if (take < n) { // underrun: rest of the quantum is silence, then re-buffer
        out[0].fill(0, take);
        this.underruns++;
        this.playing = false;
      }
    }
    if (out[0] && (this.rawGain > 0 || this.rawTarget > 0)) this.mixRaw(out[0], rawStart);
    for (let c = 1; c < out.length; c++) out[c].set(out[0]);

    this.sinceStats += n;
    if (this.sinceStats >= this.statsEvery) {
      this.sinceStats = 0;
      this.port.postMessage({
        type: "stats", buffered: this.buffered, underruns: this.underruns, dropped: this.dropped,
        rawGainDb: this.rawAgc ? this.rawAgc.gainDb : null,
      });
    }
    return true;
  }
}

registerProcessor("hearing-io", HearingIO);
