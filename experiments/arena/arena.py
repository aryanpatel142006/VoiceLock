"""Arena batches: blind A/B listening trials built from one evaluated rung, and the verdict tally.

``batch.json`` is what the UI reads (trials with blinded ``a``/``b`` audio URLs, the unprocessed
mix as reference). ``key.json`` maps A/B back to candidates and is never served. Verdicts are
appended to ``verdicts.jsonl``; ``tally`` joins them with the key. Only a tally declares a
preference, and only between candidates a human actually listened to.
"""
from __future__ import annotations

import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

from experiments.arena import result

BATCH_SCHEMA = "hearing.arena.batch/1"
CHOICES = ("a", "b", "tie", "both_bad")


def export_batch(run_dir: Path, rung: int, trials_per_candidate: int = 4, seed: int = 0) -> Path:
    """Write ``run_dir/arena/rung<r>/{batch,key}.json`` comparing each candidate with the baseline."""
    records = [r for r in result.load(run_dir / "results.jsonl") if r["rung"] == rung]
    if not records:
        raise ValueError(f"no results for rung {rung} in {run_dir}")
    base_id = records[0]["baseline_id"]
    base = next(r for r in records if r["candidate"]["id"] == base_id)
    base_out = {row["mixture_id"]: row["output"] for row in base["per_mixture"]}
    batch_dir = run_dir / "arena" / f"rung{rung}"
    (run_dir / "audio" / "_mix").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    trials, key = [], {}
    for rec in records:
        cid = rec["candidate"]["id"]
        if cid == base_id or not rec["per_mixture"]:
            continue
        for row in rec["per_mixture"][:trials_per_candidate]:
            mid = row["mixture_id"]
            src = Path(rec["inputs"]["set"]) / mid / "mix.wav"
            mix_rel = f"audio/_mix/{mid}.wav"
            if src.exists() and not (run_dir / mix_rel).exists():
                shutil.copyfile(src, run_dir / mix_rel)
            pair = [(cid, row["output"]), (base_id, base_out[mid])]
            rng.shuffle(pair)
            tid = f"t{len(trials):03d}"
            trials.append({"trial_id": tid, "mixture_id": mid, "reference": mix_rel,
                           "a": pair[0][1], "b": pair[1][1]})
            key[tid] = {"a": pair[0][0], "b": pair[1][0], "candidate": cid, "baseline": base_id}
    rng.shuffle(trials)
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch = {"schema": BATCH_SCHEMA, "batch_id": f"{run_dir.name}-rung{rung}", "run_id": run_dir.name,
             "rung": rung, "choices": list(CHOICES),
             "questions": ["Which one lets you follow the chosen talker more comfortably?"],
             "trials": trials, "created_at": datetime.now(UTC).isoformat(timespec="seconds")}
    (batch_dir / "batch.json").write_text(json.dumps(batch, indent=2))
    (batch_dir / "key.json").write_text(json.dumps(key, indent=2, sort_keys=True))
    return batch_dir


def record_verdict(batch_dir: Path, verdict: dict) -> dict:
    batch = json.loads((batch_dir / "batch.json").read_text())
    if verdict.get("trial_id") not in {t["trial_id"] for t in batch["trials"]}:
        raise ValueError("unknown trial_id")
    if verdict.get("choice") not in CHOICES:
        raise ValueError(f"choice must be one of {CHOICES}")
    clean = {"trial_id": verdict["trial_id"], "choice": verdict["choice"],
             "listener": str(verdict.get("listener", ""))[:40], "note": str(verdict.get("note", ""))[:500],
             "at": datetime.now(UTC).isoformat(timespec="seconds")}
    with (batch_dir / "verdicts.jsonl").open("a") as fh:
        fh.write(json.dumps(clean) + "\n")
    return clean


def tally(batch_dir: Path) -> dict:
    """Per candidate vs the baseline: wins, losses, ties, both-bad (latest verdict per trial+listener)."""
    key = json.loads((batch_dir / "key.json").read_text())
    path = batch_dir / "verdicts.jsonl"
    latest: dict[tuple, dict] = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        v = json.loads(line)
        latest[(v["trial_id"], v["listener"])] = v
    out: dict[str, dict] = {}
    for (tid, _), v in latest.items():
        k = key[tid]
        t = out.setdefault(k["candidate"], {"baseline": k["baseline"], "wins": 0, "losses": 0, "ties": 0,
                                            "both_bad": 0, "n": 0})
        t["n"] += 1
        if v["choice"] in ("a", "b"):
            t["wins" if k[v["choice"]] == k["candidate"] else "losses"] += 1
        else:
            t["ties" if v["choice"] == "tie" else "both_bad"] += 1
    for t in out.values():
        decided = t["wins"] + t["losses"]
        t["verdict"] = ("candidate preferred" if decided and t["wins"] > t["losses"]
                        else "baseline preferred" if decided and t["losses"] > t["wins"] else "no preference")
    (batch_dir / "tally.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    return out
