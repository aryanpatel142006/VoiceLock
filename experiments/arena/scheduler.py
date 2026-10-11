"""ASHA-style successive halving over fixed mixtures, with multi-objective safeguards.

Rung r evaluates every surviving candidate on the first ``n_r = n0 * eta**r`` mixtures of one
fixed, cluster-interleaved order (nested: rung r+1 re-uses rung r's renders and adds new ones).

Safeguards (each one is a line below, not a weight):
- a pilot measures each candidate's throughput (compute-s per audio-s) before the rungs are sized
  to the compute budget;
- the baseline is evaluated first on every rung from a reserved budget share, is never pruned, and
  is re-rendered on reference mixtures each rung and compared against its pinned reference scores;
  any drift halts promotion (every decision becomes HUMAN_REVIEW);
- decisions come from paired, cluster-bootstrapped CIs against the baseline on the same mixtures;
- STOP needs a quality metric confidently worse AND no quality or perceptual metric better or
  uncertain (so SI-SDR alone never prunes, and DNSMOS can veto a STOP unless the output has
  dropped the target, which DNSMOS scores as clean background); uncertain -> CONTINUE;
- the ASHA quota only cuts a candidate that another promoted candidate confidently beats without
  losing on anything;
- the last rung ends in HUMAN_REVIEW, never in a winner: only an Arena verdict declares one.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from experiments.arena import result, stats
from experiments.arena.runner import Candidate, Mixture, Scorers, environment, evaluate

DEFAULT_MARGINS = {  # equivalence margins: a CI inside +-margin is "tied", not better or worse
    "si_sdri": 0.5, "estoi": 0.01, "stoi": 0.01, "wer": 0.02, "wrong_speaker_frac": 0.02,
    "dnsmos_sig": 0.1, "dnsmos_bak": 0.1, "dnsmos_ovrl": 0.1,
}
MISS_VETO = 0.2  # perceptual metrics stop counting when the output drops >20 % more target blocks
REFERENCE_KEYS = ("si_sdri", "estoi", "stoi")
REFERENCE_TOL = 1e-3  # baseline reproducibility tolerance (dB / STOI units)


@dataclass
class Config:
    eta: int = 4
    n0: int = 4  # mixtures in rung 0 (the pilot sizes it down if the budget is short)
    rungs: int = 2
    budget_s: float = 3600.0  # compute-seconds (GPU-seconds when the device is cuda) for this study
    baseline_share: float = 0.25  # reserved for the baseline; candidates never draw on it
    reference_mixtures: int = 2
    margins: dict = field(default_factory=lambda: dict(DEFAULT_MARGINS))
    max_algo_latency_ms: float | None = None  # hard live constraint, if any
    n_boot: int = 2000
    seed: int = 0
    unit: str = "compute_s"  # extraction; distillation studies count speakers per rung instead


def order_mixtures(mixtures: list[Mixture], seed: int) -> list[Mixture]:
    """Fixed evaluation order: clusters interleaved round-robin so every prefix spans clusters."""
    import random

    groups: dict[str, list[Mixture]] = {}
    for m in sorted(mixtures, key=lambda m: m.id):
        groups.setdefault(m.cluster, []).append(m)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    out = []
    while any(groups.values()):
        for k in keys:
            if groups[k]:
                out.append(groups[k].pop(0))
    return out


def compare(rows: list[dict], base_rows: list[dict], cfg: Config, seed: int) -> dict[str, dict]:
    """Paired CI of (candidate - baseline) per metric over the shared mixtures."""
    base = {r["mixture_id"]: r for r in base_rows}
    out = {}
    for metric in (*result.QUALITY, *result.PERCEPTUAL, "present_miss_frac", "rtf", "algo_latency_ms"):
        deltas, clusters = [], []
        for r in rows:
            b = base.get(r["mixture_id"])
            if b is None or r.get(metric) is None or b.get(metric) is None:
                continue
            deltas.append(r[metric] - b[metric])
            clusters.append(r["cluster"])
        ci = stats.paired_cluster_bootstrap(deltas, clusters, cfg.n_boot, seed)
        ci["verdict"] = stats.verdict(ci, result.METRICS[metric][1], cfg.margins.get(metric, 0.0))
        out[metric] = ci
    return out


def decide(cmp: dict[str, dict], final: bool, rows: list[dict], cfg: Config) -> tuple[str, str]:
    q = {m: cmp[m]["verdict"] for m in result.QUALITY if cmp[m]["verdict"] != "missing"}
    p = {m: cmp[m]["verdict"] for m in result.PERCEPTUAL if cmp[m]["verdict"] != "missing"}
    miss = cmp.get("present_miss_frac", {}).get("delta")
    if miss is not None and miss > MISS_VETO:  # DNSMOS rates silence as clean: no veto once the target is gone
        p = {}
    lat = max((r["algo_latency_ms"] for r in rows), default=0.0)
    if cfg.max_algo_latency_ms is not None and lat > cfg.max_algo_latency_ms:
        return "STOP", f"algorithmic latency {lat:.0f} ms > hard limit {cfg.max_algo_latency_ms:.0f} ms"
    worse = sorted(m for m, v in q.items() if v == "worse")
    hopeful = sorted(m for m, v in {**q, **p}.items() if v in ("better", "uncertain"))
    better = sorted(m for m, v in {**q, **p}.items() if v == "better")
    if worse and not hopeful:
        return "STOP", f"confidently worse on {worse}; nothing better or uncertain (perceptual included)"
    if final:
        what = (f"trade-off: better {better}, worse {worse}" if better and worse
                else f"better on {better}" if better else f"unresolved: uncertain on {hopeful}" if hopeful
                else "tied with baseline")
        return "HUMAN_REVIEW", f"{what}; objective metrics never declare a winner - Arena verdict needed"
    if hopeful:
        return "CONTINUE", f"better {better}, uncertain {sorted(set(hopeful) - set(better))}: more evaluation"
    return "CONTINUE", "tied with baseline so far: more evaluation"


def _summary(rows: list[dict]) -> dict:
    means = {}
    for metric in (*result.METRICS, "si_sdr", "compute_p50_ms"):
        vals = [r[metric] for r in rows if r.get(metric) is not None]
        means[metric] = sum(vals) / len(vals) if vals else None
    return means


def _reference(baseline: Candidate, mixtures: list[Mixture], run_dir: Path, rung: int,
               scorers: Scorers) -> dict:
    """Re-render the baseline from scratch on the reference mixtures (no cache) and return its scores."""
    rows = evaluate(baseline, mixtures, run_dir / "reference" / f"rung{rung}", Scorers())
    return {r["mixture_id"]: {k: r[k] for k in REFERENCE_KEYS} for r in rows}


def check_reference(scores: dict, pinned_path: Path) -> tuple[bool, float]:
    """Compare with the pinned reference (pinning it on first use); returns (ok, max abs drift)."""
    if not pinned_path.exists():
        pinned_path.parent.mkdir(parents=True, exist_ok=True)
        pinned_path.write_text(json.dumps(scores, indent=2, sort_keys=True))
        return True, 0.0
    pinned = json.loads(pinned_path.read_text())
    drift = 0.0
    for mid, vals in scores.items():
        if mid not in pinned:
            return False, math.inf
        for k, v in vals.items():
            drift = max(drift, abs(v - pinned[mid][k]))
    return drift <= REFERENCE_TOL, drift


class Study:
    """One scheduler run; state is checkpointed after every rung and resumed from ``state.json``."""

    def __init__(self, experiment_id: str, baseline: Candidate, candidates: list[Candidate],
                 mixtures: list[Mixture], run_dir: Path, cfg: Config, scorers: Scorers,
                 pinned_reference: Path | None = None):
        if baseline.id in {c.id for c in candidates}:
            raise ValueError("the baseline is not a candidate")
        self.experiment_id, self.baseline, self.cfg, self.scorers = experiment_id, baseline, cfg, scorers
        self.candidates = {c.id: c for c in candidates}
        self.mixtures = order_mixtures(mixtures, cfg.seed)
        self.run_dir = run_dir
        self.pinned = pinned_reference or run_dir / f"reference_{baseline.id}.json"
        self.state_path = run_dir / "state.json"
        self.results_path = run_dir / "results.jsonl"
        self.state = self._load_state()

    # --- state -------------------------------------------------------------------------------
    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"experiment_id": self.experiment_id, "run_id": self.run_dir.name, "rungs_done": 0,
                "active": sorted(self.candidates), "decisions": {}, "plan": None,
                "ledger": {"budget_s": self.cfg.budget_s,
                           "baseline_reserved_s": self.cfg.budget_s * self.cfg.baseline_share,
                           "baseline_consumed_s": 0.0, "candidates": {c: 0.0 for c in self.candidates}},
                "checkpoints": {}, "baseline_reference": [], "config": asdict(self.cfg)}

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True))
        tmp.replace(self.state_path)

    def candidate_pool_left(self) -> float:
        led = self.state["ledger"]
        return led["budget_s"] - led["baseline_reserved_s"] - sum(led["candidates"].values())

    # --- pilot -------------------------------------------------------------------------------
    def pilot(self) -> dict:
        """Throughput on one mixture per candidate, then rung sizes that fit the budget."""
        if self.state["plan"]:
            return self.state["plan"]
        first = self.mixtures[:1]
        cost = {}
        for c in [self.baseline, *self.candidates.values()]:
            rows = evaluate(c, first, self.run_dir, self.scorers)
            spent = sum(r["compute_s"] for r in rows if not r["cached"])
            self._charge(c.id, spent)
            cost[c.id] = spent / max(sum(r["audio_s"] for r in rows), 1e-9)
        audio_per_mix = sum(r["audio_s"] for r in rows) / len(rows)
        n0, rungs = self.cfg.n0, self.cfg.rungs
        while True:
            sizes = [min(len(self.mixtures), n0 * self.cfg.eta**r) for r in range(rungs)]
            alive = len(self.candidates)
            projected = 0.0
            for i, n in enumerate(sizes):  # ASHA keeps ~1/eta per rung (at least one)
                survivors = alive if i == 0 else max(1, math.ceil(alive / self.cfg.eta**i))
                projected += n * audio_per_mix * max(cost[c] for c in self.candidates) * survivors
            if projected <= self.candidate_pool_left() or n0 <= 2:
                break
            n0 -= 1
        self.state["plan"] = {"rung_sizes": sizes, "cost_per_audio_s": cost, "audio_per_mixture_s": audio_per_mix,
                              "projected_candidate_s": projected, "fits_budget": projected <= self.candidate_pool_left()}
        self._save()
        return self.state["plan"]

    def _charge(self, cid: str, seconds: float) -> None:
        led = self.state["ledger"]
        if cid == self.baseline.id:
            led["baseline_consumed_s"] += seconds
        else:
            led["candidates"][cid] = led["candidates"].get(cid, 0.0) + seconds

    # --- rungs -------------------------------------------------------------------------------
    def run(self) -> dict:
        plan = self.pilot()
        sizes = plan["rung_sizes"]
        while self.state["rungs_done"] < len(sizes) and self.state["active"]:
            self.rung(self.state["rungs_done"], sizes)
        return self.state

    def rung(self, r: int, sizes: list[int]) -> list[dict]:
        final = r == len(sizes) - 1
        mixtures = self.mixtures[: sizes[r]]
        env = environment()
        ref = _reference(self.baseline, self.mixtures[: self.cfg.reference_mixtures], self.run_dir, r, self.scorers)
        ok, drift = check_reference(ref, self.pinned)
        self.state["baseline_reference"].append({"rung": r, "ok": ok, "max_drift": drift})
        base_rows = evaluate(self.baseline, mixtures, self.run_dir, self.scorers)  # baseline first: never starved
        self._charge(self.baseline.id, sum(x["compute_s"] for x in base_rows if not x["cached"]))
        records, rows_by = [], {}
        for cid in list(self.state["active"]):
            c = self.candidates[cid]
            need = sum(1 for m in mixtures if not (self.run_dir / "rows" / c.key() / f"{m.id}.json").exists())
            est = need * self.state["plan"]["audio_per_mixture_s"] * self.state["plan"]["cost_per_audio_s"][cid]
            if est > self.candidate_pool_left():
                action, reason, rows = "HUMAN_REVIEW", f"budget: needs ~{est:.0f} s, pool has {self.candidate_pool_left():.0f} s", []
            else:
                rows = evaluate(c, mixtures, self.run_dir, self.scorers)
                self._charge(cid, sum(x["compute_s"] for x in rows if not x["cached"]))
                rows_by[cid] = rows
                action = reason = None
            cmp = compare(rows, base_rows, self.cfg, self.cfg.seed + r) if rows else {}
            if action is None:
                action, reason = decide(cmp, final, rows, self.cfg)
            if not ok:
                action, reason = "HUMAN_REVIEW", f"baseline {self.baseline.id} drifted {drift:.4g} from its pinned reference: promotion halted"
            records.append((c, rows, cmp, action, reason))
        records = self._asha_quota(records, final)
        out = []
        for c, rows, cmp, action, reason in records:
            nxt = None
            if action == "CONTINUE" and not final:
                n_next = sizes[r + 1]
                new_audio = (n_next - sizes[r]) * self.state["plan"]["audio_per_mixture_s"]
                nxt = {"rung": r + 1, "mixtures": n_next, "audio_s": new_audio,
                       "compute_s": new_audio * self.state["plan"]["cost_per_audio_s"][c.id]}
                self.state["checkpoints"][c.id] = c.record()  # resume point carried to the next rung
            rec = self._record(c, r, mixtures, rows, base_rows, cmp, action, reason, nxt, env)
            result.append(self.results_path, rec)
            out.append(rec)
            self.state["decisions"].setdefault(c.id, []).append({"rung": r, "action": action, "reason": reason})
            if action != "CONTINUE":
                self.state["active"].remove(c.id)
        result.append(self.results_path, self._record(self.baseline, r, mixtures, base_rows, base_rows, {},
                                                      "CONTINUE", "protected baseline: always evaluated, never pruned",
                                                      None, env))
        self.state["rungs_done"] = r + 1
        self._save()
        return out

    def _asha_quota(self, records: list, final: bool) -> list:
        """Keep ceil(n/eta) of the CONTINUE set; cut only candidates a kept one confidently beats."""
        cont = [x for x in records if x[3] == "CONTINUE"]
        quota = max(1, math.ceil(len(cont) / self.cfg.eta))
        if final or len(cont) <= quota:
            return records
        rank = sorted(cont, key=lambda x: -sum(v["verdict"] == "better" for v in x[2].values()))
        kept, out = rank[:quota], []
        for x in records:
            if x in cont and x not in kept:
                for k in kept:
                    vs = compare(x[1], k[1], self.cfg, self.cfg.seed)
                    verdicts = [v["verdict"] for m, v in vs.items() if m in (*result.QUALITY, *result.PERCEPTUAL)]
                    if "worse" in verdicts and not {"better", "uncertain"} & set(verdicts):
                        x = (*x[:3], "STOP", f"ASHA quota: {k[0].id} is confidently better on every scored metric")
                        break
            out.append(x)
        return out

    def _record(self, c: Candidate, r: int, mixtures: list[Mixture], rows: list[dict], base_rows: list[dict],
                cmp: dict, action: str, reason: str, nxt: dict | None, env: dict) -> dict:
        means = _summary(rows)
        led = self.state["ledger"]
        consumed = led["baseline_consumed_s"] if c.id == self.baseline.id else led["candidates"].get(c.id, 0.0)
        per = [{k: v for k, v in row.items() if k != "cached"} for row in rows]
        return {
            "schema": result.SCHEMA, "run_id": self.state["run_id"], "experiment_id": self.experiment_id,
            "candidate": c.record(), "baseline_id": self.baseline.id, "rung": r,
            "inputs": {"set": str(mixtures[0].dir.parent) if mixtures else "", "mixture_ids": [m.id for m in mixtures] if rows else [],
                       "mixture_sha256": {m.id: m.sha256 for m in mixtures}, "order_seed": self.cfg.seed},
            "per_mixture": per,
            "dimensions": result.dimensions(means), "metrics": means,
            "paired_vs_baseline": cmp,
            "compute": {"unit": self.cfg.unit, "consumed_s": consumed,
                        "rung_s": sum(x["compute_s"] for x in rows), "audio_s": sum(x["audio_s"] for x in rows),
                        "candidate_pool_left_s": self.candidate_pool_left(), **env},
            "scorers": dict(self.scorers.notes),
            "decision": {"action": action, "reason": reason, "next_rung_estimate": nxt},
            "baseline_reference": self.state["baseline_reference"][-1],
            "human_verdict": None,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
