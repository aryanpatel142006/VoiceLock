"""T-386: rows of the fine-tuned fixed-order separator A/B (demo_regress batch logs -> JSON + markdown table).

    .venv/bin/python -m eval.t386_rows .swarm-run/batch.log .swarm-run/batch_q.log \
        --extra $HEARING_RESULTS_DIR/T-368_rows_20261010-1712.json --out $HEARING_RESULTS_DIR/T-386_rows.json

A batch log holds ``ROW <arm> <target>:<other> <scene> load <1/5/15>`` lines, each followed by one ``demo_regress``
run (its ``demo_regress_<stamp>.json`` name) and ``EXIT <rc>``. ``--extra`` takes T-368's old/old+floor20 rows as
the base arms of the holdout scenes (same profiles dir, same command; relabelled base / base+floor20).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from eval.knob_sweep import fixtures_dir, live_atten
from eval.namecall_eval import results_dir

ROW_RE = re.compile(r"^ROW (\S+) (\w+):(\w+) (\S+) load (.*)$")
RUN_RE = re.compile(r"(demo_regress_\d{8}-\d{6}(?:-\d+)?)\.json")  # T-402: newer names carry -<pid>
EXTRA_LABELS = {"old": "base", "old+floor20": "base+floor20"}  # T-368's arms on the same command
ARMS = ("base", "base+floor20", "ft+fixed", "ft+fixed+floor20", "ft+fixed+floor20+gate")  # T-402: arm 5


def parse_log(text: str) -> list[dict]:
    rows, cur = [], None
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            cur = {"label": m[1], "target": m[2], "other": m[3], "scene": m[4],
                   "load": " / ".join(m[5].replace(",", " ").split()[:3]), "run": None, "exit": None}
            rows.append(cur)
            continue
        if cur is None:
            continue
        r = RUN_RE.search(line)
        if r:
            cur["run"] = r[1]
        if line.startswith("EXIT "):
            cur["exit"] = int(line.split()[1])
    return rows


def telemetry_columns(events: Path) -> dict:
    """Enrolled share, deadline misses (overruns + late of the enrolled health at the last tick), p95 compute."""
    tele = [e["event"] for e in map(json.loads, filter(str.strip, events.read_text().splitlines()))
            if e.get("dir") == "out" and e["event"].get("type") == "telemetry"]
    tiers = [e.get("tier") for e in tele if e.get("tier")]
    enrolled = [e for e in tele if e.get("tier") == "enrolled"]
    gated = sum(1 for e in enrolled if (e.get("tse") or {}).get("confidence") == 0)
    out = {"enrolled_share": round(100 * tiers.count("enrolled") / len(tiers), 1) if tiers else None,
           "gated": f"{gated}/{len(enrolled)}"}
    for e in reversed(tele):
        tier = next((t for t in (e.get("tiers") or {}).get("tiers") or [] if t.get("name") == "enrolled"), None)
        h = next((h for h in (tier or {}).get("health") or [] if "windows" in h), None)
        if h:
            out["misses"] = f"{int(h.get('overruns', 0)) + int(h.get('late', 0))} / {h.get('windows')}"
            out["p95"] = h.get("compute_p95_ms")
            break
    return out


def scene_manifest(target: str, scene: str) -> Path:
    if scene.startswith("t141-"):
        return fixtures_dir() / "t141" / target / f"demo-team-{scene[5:]}" / "manifest.json"
    return fixtures_dir() / "t368" / target / scene / "manifest.json"


def score(rows: list[dict], sessions_root: Path) -> list[dict]:
    for r in rows:
        if not r.get("run") or r.get("od_pct") is not None:
            continue
        j = json.loads((results_dir() / f"{r['run']}.json").read_text())
        s = j["scores"][0]
        d = sessions_root / s["session"]
        r.update({"session": str(d), "sdri": s.get("si_sdri_db"), "sdri_pre": s.get("si_sdri_pre_agc_db"),
                  "leak": s.get("leak_db"), "od_s": s.get("other_dominant_s"), "seconds": s.get("seconds"),
                  "od_pct": round(100 * s["other_dominant_s"] / s["seconds"], 1) if s.get("seconds") else None,
                  "drop": s.get("true_dropout_s"), "git": j.get("git", "")[:8]})
        if (d / "events.jsonl").is_file():
            r.update(telemetry_columns(d / "events.jsonl"))
            r.update(live_atten(d, scene_manifest(r["target"], r["scene"])))
    return rows


def table(rows: list[dict]) -> str:
    def f(v, fmt):
        return "-" if v is None else fmt.format(v)

    key = {a: i for i, a in enumerate(ARMS)}
    rows = sorted(rows, key=lambda r: (r["scene"], r["target"], key.get(r["label"], 9)))
    head = ("| scene | target:other | arm | load (1/5/15) | wrong-speaker | leak dB | target-muted s | p95 window ms | "
            "deadline misses | SI-SDRi pre-AGC | enrolled share | git |")
    out = [head, "|" + "---|" * 12]
    for r in rows:
        out.append(f"| {r['scene']} | {r['target']}:{r['other']} | {r['label']} | {r.get('load', '-')} | "
                   f"{f(r.get('od_pct'), '{:.1f} %')} | {f(r.get('leak'), '{:+.1f}')} | "
                   f"{f(r.get('target_down_10dB_s'), '{:.1f}')} | {f(r.get('p95'), '{:.0f}')} | {r.get('misses', '-')} | "
                   f"{f(r.get('sdri_pre'), '{:+.1f}')} | {f(r.get('enrolled_share'), '{:.0f} %')} | {r.get('git', '-')} |")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--extra", type=Path, help="T-368 rows JSON: its old / old+floor20 rows become base arms")
    ap.add_argument("--sessions", type=Path, default=results_dir().parent.parent / "results" / "sessions")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rows = [r for log in args.logs for r in parse_log(log.read_text())]
    rows = score(rows, args.sessions)
    if args.extra:
        for r in json.loads(args.extra.read_text())["rows"]:
            if r.get("label") in EXTRA_LABELS:
                rows.append({**r, "label": EXTRA_LABELS[r["label"]], "source": "T-368"})
    if args.out:
        args.out.write_text(json.dumps({"rows": rows}, indent=1))
    print(table(rows))


if __name__ == "__main__":
    main()
