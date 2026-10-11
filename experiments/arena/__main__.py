"""CLI: ``python -m experiments.arena run|serve|tally``.

  run   --set tiny --baseline dfn_ll --candidates dfn,gtcrn,passthrough [--rungs 2 --eta 4 --n0 4]
  serve RUN_DIR --rung 1 [--port 0]
  tally RUN_DIR --rung 1
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from experiments.arena import arena
from experiments.arena.runner import Candidate, Scorers, load_mixtures
from experiments.arena.scheduler import Config, Study


def _cand(text: str) -> Candidate:
    """``id=spec`` or just ``spec`` (the id is then the spec)."""
    cid, _, spec = text.partition("=")
    return Candidate(id=cid.strip(), spec=(spec or cid).strip())


def _results_root() -> Path:
    return Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / "arena"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m experiments.arena")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", default="tiny")
    r.add_argument("--fixtures", type=Path, default=Path(os.environ.get("HEARING_FIXTURES_DIR", "eval/fixtures/generated")))
    r.add_argument("--baseline", default="dfn_ll", help="protected baseline, id=spec")
    r.add_argument("--candidates", required=True, help="comma list of id=spec")
    r.add_argument("--experiment", default="natural-denoise")
    r.add_argument("--run-dir", type=Path)
    r.add_argument("--rungs", type=int, default=2)
    r.add_argument("--eta", type=int, default=4)
    r.add_argument("--n0", type=int, default=4)
    r.add_argument("--budget-s", type=float, default=1800.0)
    r.add_argument("--baseline-share", type=float, default=0.25)
    r.add_argument("--dnsmos", default="auto", choices=("auto", "dnsmos", "mock", "off"))
    r.add_argument("--asr", default="off", help="off | mock | whisper (WER only on verified transcripts)")
    r.add_argument("--pinned-reference", type=Path, help="baseline reference scores (pinned on first use)")
    r.add_argument("--arena-trials", type=int, default=4)
    for name in ("serve", "tally"):
        s = sub.add_parser(name)
        s.add_argument("run_dir", type=Path)
        s.add_argument("--rung", type=int, default=1)
        s.add_argument("--port", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        from experiments.arena.serve import make_server

        srv = make_server(a.run_dir, a.rung, port=a.port)
        print(f"arena: http://127.0.0.1:{srv.server_address[1]}/", flush=True)
        srv.serve_forever()
        return 0
    if a.cmd == "tally":
        print(json.dumps(arena.tally(a.run_dir / "arena" / f"rung{a.rung}"), indent=2))
        return 0

    scorers = Scorers()
    if a.dnsmos != "off":
        from eval.dnsmos import get_dnsmos

        scorers.dnsmos = get_dnsmos(a.dnsmos)
        scorers.notes["dnsmos"] = scorers.dnsmos.name
    if a.asr != "off":
        from eval.transcribe import get_transcriber

        scorers.transcriber = get_transcriber(a.asr)
        scorers.notes["asr"] = getattr(scorers.transcriber, "name", a.asr)
    run_dir = a.run_dir or _results_root() / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(eta=a.eta, n0=a.n0, rungs=a.rungs, budget_s=a.budget_s, baseline_share=a.baseline_share)
    study = Study(a.experiment, _cand(a.baseline), [_cand(c) for c in a.candidates.split(",")],
                  load_mixtures(a.fixtures / a.set), run_dir, cfg, scorers, a.pinned_reference)
    state = study.run()
    print(json.dumps({"run_dir": str(run_dir), "plan": state["plan"], "decisions": state["decisions"],
                      "ledger": state["ledger"], "baseline_reference": state["baseline_reference"]}, indent=2))
    last = state["rungs_done"] - 1
    if last >= 0:
        print(f"arena batch: {arena.export_batch(run_dir, last, a.arena_trials)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
