"""Mac inference speed of the causal students (T-424): RTF on CPU and MPS, streaming and whole-pass.

Speed does not depend on the weights, so this runs on random-init nets of each variant (or a
checkpoint with ``--ckpt``). Streaming = ``student.Streamer`` fed 10 ms chunks (160 samples, the
pipeline's hop), one thread of work per chunk as the live loop would do it; reports RTF (compute
over audio time) and per-chunk p50/p95 ms. Whole = one causal pass over the clip.
Run under ``swarm-lock --exclusive`` (wall-clock numbers).

    python experiments/gpu/T-424/rtf_mac.py --seconds 10 --out $HEARING_RESULTS_DIR/T-424_rtf.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import student


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def measure(net, dev, seconds, chunk):
    net = net.to(dev).eval()
    x = torch.from_numpy(np.random.default_rng(0).standard_normal(int(seconds * student.SR)).astype(np.float32) * 0.1)
    st = student.Streamer(net, dev)
    for i in range(0, 50 * chunk, chunk):  # warm-up
        st.process(x[i : i + chunk])
    st.reset()
    times = []
    t0 = time.perf_counter()
    for i in range(0, len(x), chunk):
        t = time.perf_counter()
        st.process(x[i : i + chunk])
        sync(dev)
        times.append(time.perf_counter() - t)
    stream = time.perf_counter() - t0
    with torch.inference_mode():
        net(x[None, : student.SR].to(dev))
        sync(dev)
        t = time.perf_counter()
        net(x[None].to(dev)).cpu()
        sync(dev)
        whole = time.perf_counter() - t
    ms = np.array(times) * 1000
    return {
        "stream_rtf": round(stream / seconds, 4),
        "chunk_ms_p50": round(float(np.percentile(ms, 50)), 3),
        "chunk_ms_p95": round(float(np.percentile(ms, 95)), 3),
        "chunk_ms_max": round(float(ms.max()), 3),
        "whole_rtf": round(whole / seconds, 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--chunk", type=int, default=160)
    ap.add_argument("--devices", default="cpu1,cpu,mps", help="cpu1 = CPU with one torch thread")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    result = {"host": platform.node(), "machine": platform.machine(), "torch": torch.__version__,
              "threads": torch.get_num_threads(), "chunk": args.chunk, "seconds": args.seconds, "rows": {}}
    for variant in student.VARIANTS:
        for d in args.devices.split(","):
            if d == "mps" and not torch.backends.mps.is_available():
                continue
            torch.set_num_threads(1 if d == "cpu1" else result["threads"])
            torch.manual_seed(0)
            net = student.build(variant)
            r = measure(net, torch.device(d.removesuffix("1")), args.seconds, args.chunk)
            r.update(params=student.count_params(net), latency_ms=net.latency_ms)
            result["rows"][f"{variant}/{d}"] = r
            print(variant, d, json.dumps(r), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
