"""Run the selector gate on the team stream bank and export Arena rows (T-334).

    .venv/bin/python -m eval.selector_gate.run --out <dir> [--final]

The bank pairs window-aligned T-291 slices of recordings as two "streams". These are
CLEAN team recordings standing in for ideal separator outputs, NOT real separated streams
(none are cached; see docs/research/DISTILL_DOWNSTREAM.md). Final cases stay reserved
unless --final is passed; the dev-frozen config is hashed before anything else runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.selector_gate import gate

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "eval/embeddings/T-291/team_slices.npz"
WIN = "1.0s"


def _fixtures() -> Path:
    env = os.environ.get("HEARING_FIXTURES_DIR")
    return (Path(env) if env else ROOT / "eval/fixtures/generated") / "team"


def _read(path: str, start: float, n: int) -> np.ndarray:
    a, _ = sf.read(path, start=int(start * gate.SR), frames=n, dtype="float32")
    return a


def build_team_bank(npz: Path = NPZ) -> list[gate.Case]:
    z = np.load(npz, allow_pickle=True)
    emb, head = z[f"embeddings/{WIN}"], z[f"head_out/{WIN}"]
    spk, rec = z[f"meta/{WIN}/speaker_id"], z[f"meta/{WIN}/recording"]
    st = z[f"meta/{WIN}/start_s"]

    def sel(s, r):
        m = np.where((spk == s) & (rec == r))[0]
        return m[np.argsort(st[m])]

    def stream(s, r):
        return sel(s, r)

    prof = {s: sorted(str(r) for r in set(rec[spk == s]) if str(r).startswith("profile")) for s in ("aryan", "joseph")}
    gal = {s: sel(s, prof[s][0]) for s in prof}
    recs = {s: sorted({str(r) for r in rec[spk == s]} - set(prof[s])) for s in gal}
    cases: list[gate.Case] = []

    def mk(cid, split, group, idx, target, gspk):
        n = min(len(i) for i in idx)
        gi = gal[gspk]
        return gate.Case(
            cid, split, group, np.stack([emb[i[:n]] for i in idx]), np.stack([head[i[:n]] for i in idx]),
            target, emb[gi], 1.0, gallery_head=head[gi],
            audio=[str(_fixtures() / str(spk[i[0]]) / str(rec[i[0]])) for i in idx],
            start_s=[st[i[:n]] for i in idx])

    a_fx = recs["aryan"][0]
    j_recs = recs["joseph"]  # dev uses first, final reserved gets the rest
    j_dev, j_fin = j_recs[0], j_recs[1:2]
    for tgt, other, o_dev, o_fin, tag in (("aryan", "joseph", [j_dev], j_fin, "A"), ("joseph", "aryan", [a_fx], [], "J")):
        t_rec = a_fx if tgt == "aryan" else j_dev
        for o in o_dev:
            cases.append(mk(f"{tgt}-vs-{other}-{o}", "dev", f"{tgt}:{t_rec}", [stream(tgt, t_rec), stream(other, o)], 0, tgt))
        for o in o_fin:
            cases.append(mk(f"{tgt}-vs-{other}-{o}", "final", f"{tgt}:{t_rec}|{o}", [stream(tgt, t_rec), stream(other, o)], 0, tgt))
    # target-absent: gallery speaker never speaks; two other-speaker recordings
    cases.append(mk("absent-aryan-joseph-dev", "dev", "absent:joseph", [stream("joseph", j_dev), stream("joseph", j_dev)], None, "aryan"))
    cases.append(mk("absent-joseph-aryan", "dev", "absent:aryan", [stream("aryan", a_fx), stream("aryan", a_fx)], None, "joseph"))
    return cases


def make_arms(cases: list[gate.Case]) -> list[gate.Arm]:
    models = Path(os.environ.get("HEARING_MODELS_DIR", ROOT / "models"))
    ckpt = models / "team_projection_head.pt"
    arms: list[gate.Arm] = [gate.RawArm(), gate.IdentityArm()]
    m = gate.MahalanobisArm()
    gx = np.concatenate([c.gallery for c in {id(c.gallery): c for c in cases}.values()])
    gl = np.concatenate([np.full(len(c.gallery), i) for i, c in enumerate({id(c.gallery): c for c in cases}.values())])
    m.fit(gx, gl)
    arms.append(m)
    if ckpt.is_file():
        arms.append(gate.HeadArm(str(ckpt)))
    else:
        arms.append(gate.BlockedArm("projection_head", f"checkpoint missing: {ckpt}"))
    arms.append(gate.BlockedArm("lda", "two-speaker gallery gives a 1-D discriminant; cosine in it is degenerate, no checkpoint"))
    dist = os.environ.get("HEARING_DISTILL_CKPT", "")
    arms.append(gate.BlockedArm("distillation", f"no distillation checkpoint (HEARING_DISTILL_CKPT={dist or 'unset'}); R36/distill-student unpushed"))
    return arms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    cases = build_team_bank()
    dev = [c for c in cases if c.split == "dev"]
    final = [c for c in cases if c.split == "final"]
    arms = make_arms(dev)
    frozen = {"ema_alpha": gate.EMA_ALPHA, "absent_fa_target": gate.ABSENT_FA_TARGET, "shrinkage": gate.SHRINKAGE,
              "window": WIN, "thresholds": {}, "dev_cases": [c.case_id for c in dev],
              "final_cases_reserved": [c.case_id for c in final]}
    live = [x for x in arms if not isinstance(x, gate.BlockedArm)]
    for arm in live:
        frozen["thresholds"][arm.name] = gate.calibrate_threshold(arm, dev)
    frozen["sha256"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    (out / "frozen_config.json").write_text(gate.dumps(frozen))
    split = final if a.final else dev
    result = {"created_at": datetime.now(UTC).isoformat(), "split": "final" if a.final else "dev",
              "source_kind": "clean_team_recordings_standin_NOT_real_separated",
              "frozen_config_sha256": frozen["sha256"], "host": platform.node(), "arms": {}}
    per_arm = {}
    for arm in arms:
        if isinstance(arm, gate.BlockedArm):
            result["arms"][arm.name] = {"status": "blocked", "reason": arm.reason}
            continue
        rows = [gate.eval_case(arm, c, frozen["thresholds"][arm.name]) for c in split]
        per_arm[arm.name] = rows
        res = {"status": "ok", "threshold": frozen["thresholds"][arm.name], "summary": gate.summarize(rows),
               "checkpoint": arm.checkpoint, "checkpoint_sha256": gate.sha256(arm.checkpoint) if arm.checkpoint else None,
               "dev_contaminated": arm.name == "projection_head"}
        res["delivered"] = []
        for c, r in zip(split, rows, strict=True):
            if c.audio and all(Path(p).is_file() for p in c.audio):
                wav = out / "audio" / arm.name / f"{c.case_id}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav, gate.delivered_audio(c, np.array(r["selected"]), _read), gate.SR)
                res["delivered"].append({"case_id": c.case_id, "path": str(wav), "sha256": gate.sha256(wav),
                                         "sources": c.audio, "selected": r["selected"]})
        result["arms"][arm.name] = res
    result["raw_identity_parity"] = gate.parity(per_arm["raw_ecapa"], per_arm["identity_head"])
    result["generalization_claim_allowed"] = False
    result["generalization_note"] = "independent speakers/utterances too few; dev-only, projection head saw these speakers"
    (out / "result.json").write_text(gate.dumps(result))
    with (out / "results.jsonl").open("w") as f:  # Arena-shaped rows: candidate / dimensions / compute
        for name, res in result["arms"].items():
            f.write(json.dumps({
                "baseline_id": "raw_ecapa", "created_at": result["created_at"],
                "candidate": {"id": name, "kind": "selector", "checkpoint": res.get("checkpoint", ""),
                              "checkpoint_sha256": res.get("checkpoint_sha256"), "spec": name,
                              "preprocessing": "T-291 ECAPA 1.0 s windows, frozen gallery/EMA/threshold"},
                "status": res["status"], "reason": res.get("reason"),
                "dimensions": {"selection": res.get("summary")}, "delivered": res.get("delivered", []),
                "frozen_config_sha256": frozen["sha256"], "source_kind": result["source_kind"],
                "compute": {"host": platform.node(), "device": "cpu", "unit": "ms_per_window"}}, default=str) + "\n")
    print(json.dumps({n: (r.get("summary") or r) for n, r in result["arms"].items()}, indent=1, default=str)[:3000])
    print("parity", result["raw_identity_parity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
