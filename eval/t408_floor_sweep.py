"""T-408: HEARING_GATED_FLOOR_DB sweep on the T-387 fine-tuned separator (demo_regress batch logs -> table, pick, A/B wavs).

    .venv/bin/python -m eval.t408_floor_sweep .swarm-run/laneA.log .swarm-run/laneB.log \
        --out $HEARING_RESULTS_DIR/T-408_rows.json [--export ~/Desktop/selective-hearing-finetune]

Batch logs are T-386's format (``ROW <arm> <target>:<other> <scene> load ...``, one ``demo_regress`` run, ``EXIT <rc>``);
arms are ``<ckpt>+floor<N>`` (``t387`` = ``$HEARING_MODELS_DIR/T-374/v2_t387.pt``, ``t374`` = ``T-374/fixed_best.pt``),
every row with ``HEARING_PICK=fixed``. Pick rule: among the ``t387`` floors, the least target-muted time (sum over rows)
whose wrong-speaker mean is within ``--tol`` points of ``t387+floor20``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import soundfile as sf

from eval.knob_sweep import live_atten
from eval.namecall_eval import results_dir
from eval.t386_rows import parse_log, scene_manifest, score

ARM_RE = re.compile(r"^(\w+)\+floor(\d+(?:\.\d+)?)$")
REF = "t387+floor20"
DEEP_DB = (15, 20)  # target_down_10dB_s counts a gated window at every floor >= 10 dB; these show how deep it goes


def deep_atten(rows: list[dict]) -> list[dict]:
    """Add ``target_down_<N>dB_s`` (N in DEEP_DB) per row: T-326's target-muted count at stricter thresholds."""
    for r in rows:
        if r.get("session") and (Path(r["session"]) / "events.jsonl").is_file():
            for n in DEEP_DB:
                a = live_atten(Path(r["session"]), scene_manifest(r["target"], r["scene"]), atten_db=float(n))
                r[f"target_down_{n}dB_s"] = a["target_down_10dB_s"]
    return rows


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _sum(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals), 2) if vals else None


def group(scene: str) -> str:
    return "t141" if scene.startswith("t141-") else "holdout"


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Per arm (and per arm x row group): wrong-speaker mean %, leak mean dB, target-muted sum s (all rows / joseph rows)."""
    out: dict[str, dict] = {}
    for arm in sorted({r["label"] for r in rows}):
        for grp in ("all", "t141", "holdout"):
            mine = [r for r in rows if r["label"] == arm and (grp == "all" or group(r["scene"]) == grp)]
            if not mine:
                continue
            out.setdefault(arm, {})[grp] = {
                "n": len(mine),
                "wrong_pct": _mean([r.get("od_pct") for r in mine]),
                "leak_db": _mean([r.get("leak") for r in mine]),
                "muted_s": _sum([r.get("target_down_10dB_s") for r in mine]),
                "muted_joseph_s": _sum([r.get("target_down_10dB_s") for r in mine if r["target"] == "joseph"]),
                **{f"muted{n}_s": _sum([r.get(f"target_down_{n}dB_s") for r in mine]) for n in DEEP_DB},
                "sdri_pre": _mean([r.get("sdri_pre") for r in mine]),
            }
    return out


def pick(summary: dict[str, dict], ckpt: str = "t387", ref: str = REF, tol: float = 0.5) -> str | None:
    """The ``ckpt`` floor with the least target-muted time whose wrong-speaker mean is <= ref's + tol.

    Ties on the 10 dB count go to the shallower muting (fewer s >= 15, then >= 20 dB down), then lower wrong-speaker."""
    if ref not in summary:
        return None
    bar = summary[ref]["all"]["wrong_pct"] + tol
    ok = [a for a, s in summary.items()
          if (m := ARM_RE.match(a)) and m[1] == ckpt and s["all"]["wrong_pct"] is not None and s["all"]["wrong_pct"] <= bar]
    def key(a):
        s = summary[a]["all"]
        return (s["muted_s"], *(s.get(f"muted{n}_s") or 0.0 for n in DEEP_DB), s["wrong_pct"])

    return min(ok, key=key) if ok else None


def _arm_key(arm: str) -> tuple:
    m = ARM_RE.match(arm)
    return (m[1] != "t387", float(m[2])) if m else (True, 0.0)


def means_table(summary: dict[str, dict]) -> str:
    head = ("| rows | arm | n | wrong-speaker (mean) | leak dB (mean) | target-muted s (sum) | of it joseph-target s | "
            "target >= 15 dB down s | target >= 20 dB down s | SI-SDRi pre-AGC (mean) |")
    out = [head, "|" + "---|" * 10]
    for grp in ("all", "t141", "holdout"):
        for arm in sorted(summary, key=_arm_key):
            s = summary[arm].get(grp)
            if s:
                out.append(f"| {grp} | {arm} | {s['n']} | {s['wrong_pct']:.1f} % | {s['leak_db']:+.1f} | {s['muted_s']:.1f} | "
                           f"{s['muted_joseph_s'] or 0:.1f} | {s.get('muted15_s') or 0:.1f} | {s.get('muted20_s') or 0:.1f} | "
                           f"{s['sdri_pre']:+.1f} |")
    return "\n".join(out)


def rows_table(rows: list[dict]) -> str:
    def f(v, fmt):
        return "-" if v is None else fmt.format(v)

    rows = sorted(rows, key=lambda r: (r["scene"], r["target"], _arm_key(r["label"])))
    head = ("| scene | target:other | arm | load (1/5/15) | wrong-speaker | leak dB | target-muted s | p95 window ms | "
            "deadline misses | SI-SDRi pre-AGC | enrolled share |")
    out = [head, "|" + "---|" * 11]
    for r in rows:
        out.append(f"| {r['scene']} | {r['target']}:{r['other']} | {r['label']} | {r.get('load', '-')} | "
                   f"{f(r.get('od_pct'), '{:.1f} %')} | {f(r.get('leak'), '{:+.1f}')} | "
                   f"{f(r.get('target_down_10dB_s'), '{:.1f}')} | {f(r.get('p95'), '{:.0f}')} | {r.get('misses', '-')} | "
                   f"{f(r.get('sdri_pre'), '{:+.1f}')} | {f(r.get('enrolled_share'), '{:.0f} %')} |")
    return "\n".join(out)


def export(rows: list[dict], arm: str, out: Path, scene: str = "holdout-sir0-cafe") -> list[Path]:
    """``<name>-6-t387-floor<N>.wav``, level-matched like T-386's arms 0-4 (same function)."""
    from scripts.t386_export_ab import level_match

    floor = ARM_RE.match(arm)[2]
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for r in rows:
        if r["label"] == arm and r["scene"] == scene and r.get("session"):
            y, sr = sf.read(Path(r["session"]) / "out.wav", dtype="float32")
            p = out / f"{r['target']}-6-t387-floor{floor}.wav"
            sf.write(p, level_match(y), sr)
            written.append(p)
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--sessions", type=Path, default=results_dir().parent.parent / "results" / "sessions")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--export", type=Path, help="write the chosen arm's level-matched wavs here")
    args = ap.parse_args(argv)
    rows = [r for log in args.logs for r in parse_log(log.read_text()) if r.get("exit") == 0]
    rows = deep_atten(score(rows, args.sessions))
    summary = summarize(rows)
    chosen = pick(summary, tol=args.tol)
    if args.out:
        args.out.write_text(json.dumps({"rows": rows, "summary": summary, "chosen": chosen}, indent=1))
    print(means_table(summary), "", rows_table(rows), "", f"chosen: {chosen}", sep="\n")
    if args.export and chosen:
        for p in export(rows, chosen, args.export.expanduser()):
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
