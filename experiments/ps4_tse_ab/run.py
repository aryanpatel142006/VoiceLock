"""T-373: offline A/B of PS4 and the REAL-TSE ``spk_emb_100`` baseline against the current demo path.

    .venv/bin/python -m experiments.ps4_tse_ab.run --ps4-dir DIR --real-dir DIR --real-code DIR [--device cpu|mps]

Scenes are the four T-323/T-326 real-pair rows (``$HEARING_FIXTURES_DIR/t141/<target>/demo-team-{easy,hard}``), enrollment is
the saved profile (``eval.contrast_sweep.load_profile``), cut to its first ``--enroll-s`` seconds (PS4 trains on <= 10 s).
Per engine and mode (full file, 1 s window / 0.5 s hop, 2 s window / 1 s hop; chunked output is stitched like the live
enrolled tier: the first hop of window 0, then the last hop of every window) it reports the metrics below on the engine's
own stream and on stream + residual at -20 dB (``+res20``). Nothing here touches the runtime.

wrong % = other-dominant 200 ms frames (``eval.wrong_speaker_audit.other_dominant``) / scene seconds; SI-SDRi vs the target
stem; leak dB = other-stem LS energy in the output vs in the mix (``session_score`` rule, lower is better); ms/1s = wall time
per second of audio fed to the model (chunked: window time / window length).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

SR = 16000
MODES = (("full", None, None), ("1s", 1.0, 0.5), ("2s", 2.0, 1.0))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est, ref = np.asarray(est, np.float64), np.asarray(ref, np.float64)
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    p = a * ref
    return float(10 * np.log10((np.sum(p**2) + 1e-12) / (np.sum((est - p) ** 2) + 1e-12)))


def score(y: np.ndarray, mix: np.ndarray, t: np.ndarray, o: np.ndarray, nz: np.ndarray) -> dict:
    from eval.wrong_speaker_audit import (
        LS_FRAME,
        _ls_gains,
        other_dominant,
        target_atten,
        true_dropout_s,
    )

    n = min(len(y), len(mix))
    y, mix, t, o, nz = (a[:n] for a in (y, mix, t, o, nz))
    gy, gx = _ls_gains(y, t, o, nz), _ls_gains(mix, t, o, nz)
    from eval.wrong_speaker_audit import _frames

    eo = np.sum(_frames(o) ** 2, axis=1)
    leak = 10 * np.log10((np.sum(gy[:, 1] ** 2 * eo) + 1e-12) / (np.sum(gx[:, 1] ** 2 * eo) + 1e-12))
    dom = other_dominant(y, mix, t, o, nz)
    sec = len(dom) * LS_FRAME / SR
    ta = target_atten(y, mix, t, o, nz)
    return {"wrong_pct": round(100 * float(dom.sum()) * LS_FRAME / SR / sec, 1),
            "si_sdri": round(si_sdr(y, t) - si_sdr(mix, t), 2), "leak_db": round(float(leak), 1),
            "dropout_s": round(true_dropout_s(y, t), 2), "target_down_10dB_s": ta["target_down_10dB_s"]}


class Ps4:
    name = "ps4"

    def __init__(self, ckpt_dir: str, device: str):
        sys.path.insert(0, ckpt_dir)
        import inference as P
        import torch

        self.torch, self.dev = torch, torch.device(device)
        self.m = P.load_checkpoint(str(Path(ckpt_dir) / "checkpoint_epoch037.pt"), P.build_model(self.dev), self.dev)
        self.enroll = None

    def set_enroll(self, e: np.ndarray):
        self.enroll = e

    def run(self, mix: np.ndarray, scale: float) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(mix / scale).float()[None].to(self.dev)
            e = torch.from_numpy(self.enroll / max(float(np.abs(self.enroll).max()), 1e-9)).float()[None].to(self.dev)
            out = self.m(x, e)[0]
            if self.dev.type == "mps":
                torch.mps.synchronize()
        return out.squeeze().float().cpu().numpy() * scale


class Real:
    name = "real_tse"

    def __init__(self, model_dir: str, code_dir: str, ps4_dir: str, device: str):
        sys.modules.setdefault("silero_vad", types.SimpleNamespace(load_silero_vad=lambda: None, get_speech_timestamps=None))
        # wespeaker is not installed: its ECAPA_TDNN_GLOB_c512 is the same network as PS4's inference.py copy
        sys.path.insert(0, ps4_dir)
        import inference as P

        for n in ("wespeaker", "wespeaker.models", "wespeaker.models.speaker_model"):
            sys.modules.setdefault(n, types.ModuleType(n))
        sys.modules["wespeaker.models.speaker_model"].get_speaker_model = lambda name: P.ECAPA_TDNN_GLOB_c512
        sys.path.insert(0, code_dir)
        import torch
        import yaml
        from torchaudio.compliance import kaldi
        from wesep.cli.extractor import disable_pretrained
        from wesep.models import get_model
        from wesep.utils.checkpoint import load_pretrained_model

        self.torch, self.kaldi, self.dev = torch, kaldi, torch.device(device)
        cfg = disable_pretrained(yaml.load((Path(model_dir) / "config.yaml").read_text(), Loader=yaml.FullLoader))
        self.m = get_model(cfg["model"]["tse_model"])(cfg["model_args"]["tse_model"])
        load_pretrained_model(self.m, str(Path(model_dir) / "avg_model.pt"))
        self.m.eval().to(self.dev)
        self.spk_feat = cfg["model_args"]["tse_model"].get("spk_feat", False)

    def set_enroll(self, e: np.ndarray):
        pcm = self.torch.from_numpy(e).float()[None]
        if self.spk_feat:
            f = self.kaldi.fbank(pcm, num_mel_bins=80, frame_length=25, frame_shift=10, sample_frequency=SR)
            pcm = (f - f.mean(0))[None]
        self.enroll = pcm.to(self.dev)

    def run(self, mix: np.ndarray, scale: float) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            out = self.m(torch.from_numpy(mix).float()[None].to(self.dev), self.enroll)
            out = out[0] if isinstance(out, (list, tuple)) else out
            if self.dev.type == "mps":
                torch.mps.synchronize()
        return out.squeeze().float().cpu().numpy()


def extract(eng, mix: np.ndarray, win_s, hop_s) -> tuple[np.ndarray, list[float]]:
    """(stitched output, per-call seconds). The model sees raw windows; scale = whole-file peak (PS4's peak norm)."""
    scale = float(np.abs(mix).max()) + 1e-9
    if win_s is None:
        t0 = time.perf_counter()
        y = eng.run(mix, scale)
        return y[: len(mix)], [time.perf_counter() - t0]
    win, hop = round(win_s * SR), round(hop_s * SR)
    out, times = np.zeros_like(mix), []
    for k, i in enumerate(range(0, len(mix) - win + 1, hop)):
        t0 = time.perf_counter()
        y = eng.run(mix[i : i + win], scale)[:win]
        times.append(time.perf_counter() - t0)
        if k == 0:
            out[:hop] = y[:hop]
        out[i + hop : i + win] = y[hop:]
    return out, times


def demo_path(scene: Path, enroll: np.ndarray, other: np.ndarray) -> tuple[dict, list[float]]:
    """The live enrolled tier (EnrolledCandidate, 1 s / 0.5 s, saved profiles) plus the confidence-floor residual."""
    from eval.contrast_sweep import load_scene
    from hearing.fusion.mixer import confidence_floor_db
    from hearing.tse.enrolled import EnrolledCandidate

    sc = load_scene(scene)
    mix = sc["mix"]
    cand = EnrolledCandidate()
    cand.prepare(enroll, other, SR)
    win, hop = SR, SR // 2
    stream, played, played20, times = np.zeros_like(mix), np.zeros_like(mix), np.zeros_like(mix), []
    for k, i in enumerate(range(0, len(mix) - win + 1, hop)):
        w = mix[i : i + win]
        t0 = time.perf_counter()
        y = cand.extract(w, SR, enroll=enroll, other=other)
        times.append(time.perf_counter() - t0)
        z = y + 10.0 ** (-confidence_floor_db(cand.confidence) / 20.0) * (w - y)
        floor20 = 20.0 if cand.confidence == 0.0 else confidence_floor_db(cand.confidence)  # HEARING_GATED_FLOOR_DB=20
        z20 = y + 10.0 ** (-floor20 / 20.0) * (w - y)
        a, b = (0, hop) if k == 0 else (hop, win)
        stream[i + a : i + b], played[i + a : i + b], played20[i + a : i + b] = y[a:b], z[a:b], z20[a:b]
    return {"stream": stream, "played": played, "played_gated20": played20}, times


def main(argv: list[str] | None = None) -> int:
    from eval.contrast_sweep import load_profile, load_scene
    from hearing import device as hdev
    from hearing.audio import io as aio

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ps4-dir", required=True)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--real-code", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--enroll-s", type=float, default=10.0)
    ap.add_argument("--engines", default="demo,ps4,real")
    ap.add_argument("--modes", default="full,1s,2s")
    ap.add_argument("--max-rows", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    F = Path(os.environ.get("HEARING_FIXTURES_DIR", "eval/fixtures/generated"))
    rows = [("joseph", "aryan", "easy"), ("aryan", "joseph", "easy"), ("joseph", "aryan", "hard"), ("aryan", "joseph", "hard")]
    engines, rows = args.engines.split(","), rows[: args.max_rows]
    mods = [m for m in MODES if m[0] in args.modes.split(",")]
    eng = {}
    if "ps4" in engines:
        eng["ps4"] = Ps4(args.ps4_dir, args.device)
    if "real" in engines:
        eng["real_tse"] = Real(args.real_dir, args.real_code, args.ps4_dir, args.device)
    results = []
    for target, other, level in rows:
        scene = F / "t141" / target / f"demo-team-{level}"
        sc = load_scene(scene)
        mix, t, o = sc["mix"], sc["A"], sc["B"]
        nz = aio.load(scene / "stems" / "noise.wav", SR)[0][: len(mix)]
        nz = np.pad(nz, (0, len(mix) - len(nz)))
        enroll = load_profile(F / "profiles", target)
        other_clip = load_profile(F / "profiles", other)
        row = {"row": f"{target}:{other} {level}", "seconds": round(len(mix) / SR, 1), "arms": {}}
        if "demo" in engines:
            outs, times = demo_path(scene, enroll, other_clip)
            for nm, y in outs.items():
                row["arms"][f"demo/{nm}"] = {**score(y, mix, t, o, nz), "ms_per_1s_audio": round(1000 * float(np.sum(times)) / (len(mix) / SR), 0),
                                              "p95_ms_per_window": round(1000 * float(np.percentile(times, 95)), 0)}
        for name, e in eng.items():
            e.set_enroll(enroll[: round(args.enroll_s * SR)])
            for mode, w, h in mods:
                y, times = extract(e, mix, w, h)
                ms = 1000 * float(np.sum(times)) / (len(mix) / SR)
                for variant, z in (("stream", y), ("+res20", y + 0.1 * (mix - y))):
                    row["arms"][f"{name}/{mode}/{variant}"] = {**score(z, mix, t, o, nz), "ms_per_1s_audio": round(ms, 0),
                                                              "p95_ms_per_window": round(1000 * float(np.percentile(times, 95)), 0)}
            print(row["row"], "done", flush=True)
        results.append(row)
        for k, v in row["arms"].items():
            print(f"  {row['row']:20s} {k:22s} {json.dumps(v)}", flush=True)
    out = args.out or Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / f"ps4_tse_ab_{args.device}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": args.device, "hearing_device": hdev.pick(), "enroll_s": args.enroll_s, "rows": results}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
