#!/usr/bin/env python3
"""No-reference score of a recorded replay session of a real-room clip (T-090).

    HEARING_SESSION_LOG=results/sessions .venv/bin/python -m hearing serve ...   # record
    .venv/bin/python scripts/session_score_nr.py <session dir> --clip eval/fixtures/demo/<name> [--json]

Prints DNSMOS SIG/BAK/OVRL in vs out, other-talker leak (dB), target preserved (dB), dropouts in
target-only speech, time to Isolated and the tier timeline. Definitions: eval/session_nr.py. The
clip's manifest.json (scripts/ingest_clip.py) supplies the speaking timeline instead of stems.
Without a session dir it scores the newest one under results/sessions.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import real_clip, session_nr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, nargs="?", help="tap directory (default: newest in results/sessions)")
    ap.add_argument("--clip", type=Path, required=True, help="clip dir, its manifest.json or clip.mp4")
    ap.add_argument("--dnsmos", default="auto", choices=["auto", "dnsmos", "mock"])
    ap.add_argument("--offset-s", type=float, default=0.0,
                    help="seconds of input before the clip started (mic audio before REPLAY was pressed)")
    ap.add_argument("--json", action="store_true", help="print the raw result as JSON")
    args = ap.parse_args(argv)
    session = args.session or max(Path("results/sessions").glob("*/"), key=lambda p: p.stat().st_mtime)
    result = session_nr.score(session, real_clip.load_manifest(args.clip), args.dnsmos, args.offset_s)
    print(json.dumps(result, indent=2) if args.json else session_nr.format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
