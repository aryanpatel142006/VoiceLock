"""T-370: per-row numbers of the soft-gate live batch (``HEARING_GATED_SOFT``) + the T-326 target-attenuation check.

Reads ``$HEARING_RESULTS_DIR/<name>.log`` (``=== ROW setting=.. target=.. other=.. scene=.. load=.. start=..`` lines,
each followed by one ``scripts/demo_regress.py`` run), writes ``<name>.jsonl`` and prints one line per row:
wrong-speaker % (``other_dominant_s`` / scored s), leak, true dropouts, and target-active seconds pushed >= 10 dB / 6 dB
down (``eval.wrong_speaker_audit.session_atten``, pre-AGC output, after the 3.9 s cold start).
``.venv/bin/python -m eval.softgate_live [name]`` (default ``T-370_softgate_live``)."""
import json
import os
import re
import sys
from pathlib import Path

from eval.wrong_speaker_audit import session_atten


def main() -> int:
    R = Path(os.environ["HEARING_RESULTS_DIR"])
    F = Path(os.environ["HEARING_FIXTURES_DIR"])
    MAIN = R.parents[1]
    NAME = sys.argv[1] if len(sys.argv) > 1 else "T-370_softgate_live"
    log = (R / f"{NAME}.log").read_text()
    rows = []
    for seg in log.split("=== ROW ")[1:]:
        head = dict(kv.split("=", 1) for kv in seg.splitlines()[0].split() if "=" in kv and not kv.startswith("load"))
        load = re.search(r"load=(.*?) start=", seg.splitlines()[0]).group(1)
        m = re.findall(r"demo_regress_(\d{8}-\d{6})\.json", seg)
        if not m:
            rows.append({**head, "load": load, "error": "no json"})
            continue
        d = json.loads((R / f"demo_regress_{m[-1]}.json").read_text())
        s = d["scores"][0]
        sess = MAIN / "results/sessions" / s["session"]
        clip = F / "t141" / head["target"] / f"demo-team-{head['scene']}"
        at = session_atten(sess, clip)
        rows.append({**head, "load": load, "run": f"demo_regress_{m[-1]}", "session": str(sess),
                     "wrong_s": s["other_dominant_s"], "scored_s": s["seconds"], "wrong_pct": round(100 * s["other_dominant_s"] / s["seconds"], 1),
                     "leak_db": s["leak_db"], "si_sdri": s["si_sdri_db"], "si_sdri_pre": s.get("si_sdri_pre_agc_db"),
                     "dropout_s": s["true_dropout_s"], **at})
    out = R / f"{NAME}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    keys = ("setting", "target", "scene", "wrong_pct", "wrong_s", "leak_db", "dropout_s", "target_down_10dB_s", "target_down_6dB_s", "si_sdri", "si_sdri_pre", "load", "run")
    for r in rows:
        print(" | ".join(str(r.get(k)) for k in keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
