// The synth worker (synth.js renderSamples): one loop per message, the samples transferred back.
import { samplesSync } from "./synth.js";

self.onmessage = (e) => {
  const { id, kind, rate } = e.data || {};
  try {
    const x = samplesSync(kind, rate);
    self.postMessage({ id, samples: x }, [x.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String(err?.message || err) });
  }
};
