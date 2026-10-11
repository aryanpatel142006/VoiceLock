"""Caption Whisper compute per window on the T-120 babble scene: temperature fallback vs ``temperature=0.0`` (T-131).

Cuts the Name Call scene (``eval.namecall_eval.build_scene``) into 2.5 s windows every 1 s and times one decode per window,
two passes each, in one process: ``old_fallback`` is faster-whisper's default temperature ladder, ``new_t0`` is the live
``WhisperASR._transcribe`` (beam 1, temperature 0). Wall-clock: run under ``swarm-lock --exclusive``.

    .venv/bin/python -m eval.caption_latency
"""

import json
import os
import time

import numpy as np

from eval.namecall_eval import build_scene
from hearing import models
from hearing.synth import asr

SR = 16000
WIN_S = 2.5


def main() -> int:
    mix = build_scene()["mix"]
    n = round(WIN_S * SR)
    wins = [mix[i:i + n] for i in range(0, len(mix) - n, SR)]
    eng = asr.WhisperASR()
    model = models.load(eng._key)
    model.transcribe(np.zeros(SR, np.float32), language="en")  # warm-up
    res: dict[str, list[float]] = {}
    for label in ("old_fallback", "new_t0", "old_fallback", "new_t0"):
        for w in wins:
            t = time.perf_counter()
            if label == "new_t0":
                eng._frames, eng._t0, eng._t1 = [w], 0.0, WIN_S
                eng._transcribe()
            else:
                segs, _ = model.transcribe(w, language="en", beam_size=1, vad_filter=False,
                                           condition_on_previous_text=False)
                " ".join(s.text for s in segs)  # the segments are lazy: decode them
            res.setdefault(label, []).append(time.perf_counter() - t)
    out = {k: {"n": len(v), "p50_s": round(float(np.median(v)), 3), "p90_s": round(float(np.percentile(v, 90)), 3),
               "max_s": round(max(v), 3)} for k, v in res.items()}
    out["load"] = os.getloadavg()
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
