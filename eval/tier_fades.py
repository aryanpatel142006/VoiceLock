"""Compare tier-switch transitions across a delay jump on GRID speech (T-062).

    python -m eval.tier_fades eval/fixtures/generated/av_full $HEARING_RESULTS_DIR/t062_tier_fades

Two "tiers" are the clean target delayed by their delays; the switch is a splice (equal-power
crossfade over 150 ms), a dip (fade out, fade in), a crossfade over |delta| ("long") or a hold
(silence for |delta|). Prints max step, two-voice overlap and added quiet ms; writes WAVs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from hearing.fusion.tiers import fade_gains

SR = 16000
CF = round(0.15 * SR)
root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
clips = sorted(root.glob("0*/target.wav"))[:12]
x = np.concatenate([sf.read(c, dtype="float32")[0] for c in clips])
x = x / (np.max(np.abs(x)) + 1e-9) * 0.5

def env_db(y, w=80):
    return 10 * np.log10(np.convolve(y ** 2, np.ones(w) / w, mode="same") + 1e-12)

def delayed(ms):
    d = round(ms * SR / 1000); return np.concatenate([np.zeros(d, np.float32), x[: len(x) - d]])

def render(variant, a_ms, b_ms, at):
    """old tier (delay a) until `at`, then the new tier (delay b), transition per variant."""
    A, B = delayed(a_ms), delayed(b_ms)
    n = len(x); g_old = np.ones(n); g_new = np.zeros(n)
    if variant == "hold":  # silence for |delta| (upgrade) then the new tier
        L = round(abs(b_ms - a_ms) * SR / 1000); g_old[at:] = 0; g_new[at + L:] = 1
    else:
        L = CF if variant in ("splice", "dip") else max(CF, round(abs(b_ms - a_ms) * SR / 1000))
        kind = "dip" if variant == "dip" else "splice"
        o, nw = fade_gains(kind, np.arange(L), L)
        g_old[at:at + L] = o; g_old[at + L:] = 0; g_new[at:at + L] = nw; g_new[at + L:] = 1
    return g_old * A + g_new * B, g_old, g_new, A, B

speech_floor = np.percentile(env_db(x)[env_db(x) > -60], 50) - 20  # -20 dB under median speech
rows = []
for a_ms, b_ms in ((13, 1500), (1500, 13), (13, 250), (1500, 3000)):
    for variant in ("splice", "dip", "long", "hold"):
        clicks, gaps, overl = [], [], []
        for at_s in np.arange(4.0, len(x) / SR - 4.0, 1.37):
            at = round(at_s * SR)
            y, go, gn, A, B = render(variant, a_ms, b_ms, at)
            seg = slice(at - CF, at + max(CF, round(abs(b_ms - a_ms) * SR / 1000)) + CF)
            ref_step = np.percentile(np.abs(np.diff(x)), 99.9)
            clicks.append(float(np.max(np.abs(np.diff(y[seg]))) / ref_step))
            # extra silence the switch adds: quiet ms in y minus quiet ms in the sources over the same span
            quiet = [int(np.sum(env_db(s)[seg] < speech_floor)) for s in (y, A, B)]
            gaps.append((quiet[0] - min(quiet[1:])) * 1000 / SR)
            both = (env_db(go * A)[seg] > speech_floor) & (env_db(gn * B)[seg] > speech_floor)
            overl.append(float(np.sum(both)) * 1000 / SR)
        rows.append({"from_ms": a_ms, "to_ms": b_ms, "variant": variant,
                     "max_step_vs_speech_p99.9": round(float(np.max(clicks)), 2),
                     "added_quiet_ms_mean": round(float(np.mean(gaps)), 1),
                     "two_voice_overlap_ms_mean": round(float(np.mean(overl)), 1), "n": len(clicks)})
        y, *_ = render(variant, a_ms, b_ms, round(6.0 * SR))
        sf.write(out / f"{a_ms}to{b_ms}_{variant}.wav", y[round(3 * SR):round(10 * SR)], SR)
for r in rows: print(r)
(out / "grid_fades.json").write_text(json.dumps({"clips": [str(c) for c in clips], "rows": rows}, indent=1))
