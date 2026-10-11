#!/usr/bin/env python3
"""Turn a phone video of two people into a replayable demo clip (T-090).

    .venv/bin/python scripts/ingest_clip.py kitchen.mp4 --target left [--name kitchen]

Writes eval/fixtures/demo/<name>/clip.mp4 (640x360 @ 25 fps H.264, mono 16 kHz AAC, speech level
matched to the GRID demo clip; gitignored) and manifest.json (target track/side, per-frame face
boxes, ASD speaking timeline per face; small, may be committed). The REPLAY button lists it, and
scripts/session_score_nr.py scores a recorded session of it. Recording recipe: docs/DEMO.md.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import real_clip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--target", default="largest", help="left | right | largest | track:<id> (default largest)")
    ap.add_argument("--name", help="clip directory name (default: the video's stem)")
    ap.add_argument("--out", type=Path, default=real_clip.DEMO_DIR, help="demo root (default eval/fixtures/demo)")
    ap.add_argument("--detector", default="auto", help="face detector: auto | mediapipe | mock")
    ap.add_argument("--asd", default="auto", help="active speaker detector: auto | light_asd | lipmotion")
    ap.add_argument("--max-s", type=float, help="keep only the first N seconds")
    ap.add_argument("--level-db", type=float, default=real_clip.GRID_LEVEL_DB,
                    help=f"speech level in dBFS (default {real_clip.GRID_LEVEL_DB}, the GRID demo clip's)")
    args = ap.parse_args(argv)
    path = real_clip.ingest(args.video, args.name, args.target, args.out, args.detector, args.asd,
                            args.max_s, args.level_db)
    m = real_clip.load_manifest(path)
    print(f"wrote {path.parent}/clip.mp4 + manifest.json  ({m['duration_s']} s, detector {m['detector']}, asd {m['asd']})")
    print(f"loudness: {json.dumps(m['loudness'])}")
    for f in m["faces"]:
        mark = "  <- target" if f["track_id"] == m["target"]["track_id"] else ""
        print(f"  track {f['track_id']:>3} {f['side']:<6} seen {f['frames_seen']} frames, "
              f"speaking {f['speaking_s']} s in {len(f['speaking'])} turns{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
