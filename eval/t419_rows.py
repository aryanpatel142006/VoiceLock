"""T-419: rows of the shorter-window A/B (demo_regress batch logs -> JSON + markdown table), with the measured delay.

    .venv/bin/python -m eval.t419_rows .swarm-run/batch_t.log .swarm-run/batch_q.log \
        --extra $HEARING_RESULTS_DIR/T-408_rows.json --out $HEARING_RESULTS_DIR/T-419_rows.json

Logs as :mod:`eval.t386_rows` (``ROW <arm> <target>:<other> <scene> load ...`` / the run's json name / ``EXIT <rc>``).
``--extra`` takes T-408's ``t387+floor20`` holdout rows (the same checkpoint, pick and floor at 1.0 / 0.5) as the
reference arm of the held-out scenes. Delay columns, both read from the run, never computed from the windows:

* ``delay_ms``: the enrolled tier's delay in the recorded stream, measured by ``session_score`` (cross-correlation of
  out.wav against in.wav per run of one playing delay, refined +-150 ms around the telemetry prior and checked by an
  open 0-4 s search); time-weighted over the enrolled runs, so a widened hop shows up.
* ``total_ms``: the median telemetry ``latency_ms.total`` while enrolled plays (the engine's own end-to-end figure:
  stream delay plus the server's compute and capture buffering; no browser or device buffers).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import soundfile as sf

from eval.namecall_eval import results_dir
from eval.t386_rows import parse_log, score

ARMS = ("ft+1.0/0.5", "ft+0.75/0.375", "ft+0.5/0.25", "base+0.5/0.25")
EXTRA_LABELS = {"t387+floor20": "ft+1.0/0.5"}  # T-408's arm = this task's reference (same env, 1.0 / 0.5)


def delay_columns(run_json: Path, events: Path | None = None) -> dict:
    s = json.loads(run_json.read_text())["scores"][0]
    segs = [d for d in s.get("delays") or [] if d.get("tier") == "enrolled" and d["end_s"] > d["start_s"]]
    out: dict = {"delay_ms": None, "delay_max_ms": None, "total_ms": None, "alignment": s.get("alignment")}
    if segs:
        dur = sum(d["end_s"] - d["start_s"] for d in segs)
        out["delay_ms"] = round(sum(d["measured_ms"] * (d["end_s"] - d["start_s"]) for d in segs) / dur, 1)
        out["delay_max_ms"] = max(d["measured_ms"] for d in segs)
        out["corr"] = round(min(d.get("corr", 0.0) for d in segs), 3)
    if events is not None and events.is_file():
        totals = []
        for line in filter(str.strip, events.read_text().splitlines()):
            e = json.loads(line)
            ev = e.get("event") or {}
            if e.get("dir") == "out" and ev.get("type") == "telemetry" and ev.get("tier") == "enrolled":
                t = (ev.get("latency_ms") or {}).get("total")
                if isinstance(t, (int, float)):
                    totals.append(t)
        out["total_ms"] = round(statistics.median(totals), 1) if totals else None
    return out


def add_delays(rows: list[dict]) -> list[dict]:
    for r in rows:
        if r.get("run") and (results_dir() / f"{r['run']}.json").is_file():
            events = Path(r["session"]) / "events.jsonl" if r.get("session") else None
            r.update(delay_columns(results_dir() / f"{r['run']}.json", events))
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def summary(rows: list[dict]) -> str:
    out = [("| rows | arm | n | measured delay ms (mean, max) | engine total ms | wrong-speaker | leak dB | target-muted s (sum) | "
            "p95 window ms (max) | deadline misses | SI-SDRi pre-AGC |"), "|" + "---|" * 11]
    for group, pick in (("t141", lambda r: r["scene"].startswith("t141-")), ("holdout", lambda r: r["scene"].startswith("holdout"))):
        for arm in ARMS:
            rs = [r for r in rows if pick(r) and r["label"] == arm and r.get("od_pct") is not None]
            if not rs:
                continue
            miss = [tuple(map(int, r["misses"].split(" / "))) for r in rs if r.get("misses")]
            muted = [r.get("target_down_10dB_s") for r in rs if r.get("target_down_10dB_s") is not None]
            p95 = [r["p95"] for r in rs if r.get("p95") is not None]
            dmax = [r["delay_max_ms"] for r in rs if r.get("delay_max_ms") is not None]
            out.append(f"| {group} | {arm} | {len(rs)} | {_mean([r.get('delay_ms') for r in rs])}, {max(dmax) if dmax else '-'} | "
                       f"{_mean([r.get('total_ms') for r in rs])} | {_mean([r['od_pct'] for r in rs])} % | "
                       f"{_mean([r.get('leak') for r in rs]):+.1f} | {round(sum(muted), 1) if muted else '-'} | "
                       f"{round(max(p95)) if p95 else '-'} | {sum(m for m, _ in miss)} / {sum(n for _, n in miss)} | "
                       f"{_mean([r.get('sdri_pre') for r in rs]):+.1f} |")
    return "\n".join(out)


def table(rows: list[dict]) -> str:
    def f(v, fmt):
        return "-" if v is None else fmt.format(v)

    key = {a: i for i, a in enumerate(ARMS)}
    rows = sorted(rows, key=lambda r: (r["scene"], r["target"], key.get(r["label"], 9)))
    out = [("| scene | target:other | arm | load (1/5/15) | measured delay ms | engine total ms | wrong-speaker | leak dB | "
            "target-muted s | p95 window ms | deadline misses | SI-SDRi pre-AGC | enrolled share | git |"), "|" + "---|" * 14]
    for r in rows:
        out.append(f"| {r['scene']} | {r['target']}:{r['other']} | {r['label']} | {r.get('load', '-')} | "
                   f"{f(r.get('delay_ms'), '{:.0f}')} | {f(r.get('total_ms'), '{:.0f}')} | "
                   f"{f(r.get('od_pct'), '{:.1f} %')} | {f(r.get('leak'), '{:+.1f}')} | "
                   f"{f(r.get('target_down_10dB_s'), '{:.1f}')} | {f(r.get('p95'), '{:.0f}')} | {r.get('misses', '-')} | "
                   f"{f(r.get('sdri_pre'), '{:+.1f}')} | {f(r.get('enrolled_share'), '{:.0f} %')} | {r.get('git', '-')} |")
    return "\n".join(out)


def export(rows: list[dict], arm: str, scene: str, out: Path) -> list[Path]:
    """``<target>-7-lowdelay-<window>.wav``: the arm's out.wav on ``scene``, level-matched like T-386's A/B folder."""
    spec = importlib.util.spec_from_file_location("t386_export_ab", Path(__file__).parents[1] / "scripts" / "t386_export_ab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    window = arm.split("+", 1)[1].split("/")[0]
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for r in rows:
        if r["label"] == arm and r["scene"] == scene and r.get("session"):
            y, sr = sf.read(Path(r["session"]) / "out.wav", dtype="float32")
            path = out / f"{r['target']}-7-lowdelay-{window}.wav"
            sf.write(path, mod.level_match(y), sr)
            written.append(path)
            print(f"{path.name}: delay {r.get('delay_ms')} ms, wrong-speaker {r.get('od_pct')} %, leak {r.get('leak')} dB, "
                  f"target-muted {r.get('target_down_10dB_s')} s")
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--extra", type=Path, help="T-408 rows JSON: its t387+floor20 holdout rows become the 1.0 / 0.5 arm")
    ap.add_argument("--sessions", type=Path, default=results_dir().parent.parent / "results" / "sessions")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--export", type=Path, help="listening folder for the --arm wavs (e.g. ~/Desktop/selective-hearing-finetune)")
    ap.add_argument("--arm", default="ft+0.5/0.25")
    ap.add_argument("--scene", default="holdout-sir0-cafe")
    args = ap.parse_args(argv)
    rows = score([r for log in args.logs if log.is_file() for r in parse_log(log.read_text())], args.sessions)
    if args.extra:
        rows += [{**r, "label": EXTRA_LABELS[r["label"]], "source": "T-408"} for r in json.loads(args.extra.read_text())["rows"]
                 if r.get("label") in EXTRA_LABELS and r["scene"].startswith("holdout")]
    rows = add_delays(rows)
    if args.out:
        args.out.write_text(json.dumps({"rows": rows}, indent=1))
    print(summary(rows))
    print()
    print(table(rows))
    if args.export:
        export(rows, args.arm, args.scene, args.export.expanduser())


if __name__ == "__main__":
    main()
