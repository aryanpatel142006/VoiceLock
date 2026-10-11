"""Rank the T-253 research portfolio deterministically (offline, stdlib only, no audio).

    .venv/bin/python experiments/theory/T-253/portfolio.py --out-dir <dir>

score = impact * confidence / (cost * risk); candidates needing hardware or GPU
are tagged blocked_on and ranked after unblocked ones at equal score is NOT
applied: the score alone orders, the tag is reported. Writes ranked.json and
proposals.json (machine-readable task proposals) plus a sha256 of inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = ("id", "title", "origin", "impact", "confidence", "cost", "risk", "needs_hardware",
            "needs_gpu", "independent_path", "evidence", "assumptions", "falsifier", "smallest_experiment")


def load(path: Path) -> dict:
    data = json.loads(path.read_text())
    seen = set()
    for c in data["candidates"]:
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            raise ValueError(f"{c.get('id')}: missing {missing}")
        if c["id"] in seen:
            raise ValueError(f"duplicate id {c['id']}")
        seen.add(c["id"])
        for k in ("impact", "cost", "risk"):
            if not 1 <= c[k] <= 5:
                raise ValueError(f"{c['id']}: {k} out of 1-5")
        if not 0.1 <= c["confidence"] <= 0.9:
            raise ValueError(f"{c['id']}: confidence out of 0.1-0.9")
        if not c["falsifier"].strip():
            raise ValueError(f"{c['id']}: empty falsifier")
    return data


def score(c: dict) -> float:
    return round(c["impact"] * c["confidence"] / (c["cost"] * c["risk"]), 4)


def rank(data: dict) -> list[dict]:
    rows = []
    for c in data["candidates"]:
        blocked = [n for n, f in (("hardware", "needs_hardware"), ("gpu", "needs_gpu")) if c[f]]
        rows.append({**c, "score": score(c), "blocked_on": blocked})
    # branches already measured or owned elsewhere ("none" experiment) are reused, not ranked
    for r in rows:
        r["reused"] = r["smallest_experiment"].lower().startswith("none")
    rows.sort(key=lambda r: (r["reused"], -r["score"], r["id"]))
    n = 0
    for r in rows:
        n += 0 if r["reused"] else 1
        r["rank"] = None if r["reused"] else n
    return rows


def proposals(rows: list[dict]) -> list[dict]:
    """Task proposals for new candidates that can run on CPU with no hardware."""
    out = []
    for r in rows:
        if r["origin"] != "new" or r["blocked_on"]:
            continue
        out.append({"proposal_id": f"PROP-{r['id']}", "title": r["title"], "scope": "CPU offline, existing embeddings/renders only",
                    "falsifier": r["falsifier"], "first_step": r["smallest_experiment"], "rank": r["rank"],
                    "caps": {"cpu_minutes": 30, "new_bytes_mib": 32, "gpu_cases": 0},
                    "production_adoption": False})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=HERE / "candidates.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    raw = a.candidates.read_bytes()
    rows = rank(load(a.candidates))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"input_sha256": hashlib.sha256(raw).hexdigest(), "formula": "impact*confidence/(cost*risk)", "seed": None}
    (a.out_dir / "ranked.json").write_text(json.dumps({"meta": meta, "ranked": rows}, indent=2) + "\n")
    (a.out_dir / "proposals.json").write_text(json.dumps({"meta": meta, "proposals": proposals(rows)}, indent=2) + "\n")
    for r in rows:
        print(f"{r['rank'] or '--':>2} {r['id']:<3} {r['score']:.3f} {','.join(r['blocked_on']) or '-':<12} {r['title'][:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
