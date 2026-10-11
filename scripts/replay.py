"""Stream a local MP4 over /ws/session without camera/microphone access.

    .venv/bin/python scripts/replay.py                       # default clip: eval/fixtures/demo/demo-mf-easy
    .venv/bin/python scripts/replay.py eval/fixtures/demo/grid-two-face.mp4   # legacy loop
    .venv/bin/python scripts/replay.py eval/fixtures/demo/<name> --lock-target   # ingested clip (T-090)

An ingested clip (scripts/ingest_clip.py) is a directory with clip.mp4 + manifest.json. With
--lock-target the manifest's target is an oracle: the first telemetry face whose box centre is
within ORACLE_DIST of the manifest target's mean box is locked (and re-locked if that track is
lost). Without it the server's TargetSelector chooses, as in a live session.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hearing.replay import Clip, clip_manifest, feed

ORACLE_DIST = 0.15  # normalised distance between box centres that counts as the same face


class OracleLock:
    """Lock the face the ingest manifest says is the target, by box position (track ids differ per server)."""

    def __init__(self, manifest: dict | None):
        self.target = (manifest or {}).get("target")
        self.locked: int | None = None

    def on_message(self, msg: dict) -> dict | None:
        if self.target is None or msg.get("type") != "telemetry" or not msg.get("faces"):
            return None
        ids = {f["track_id"] for f in msg["faces"]}
        if self.locked in ids:
            return None
        x, y, w, h = self.target["box"]

        def dist(face):
            fx, fy, fw, fh = face["box"]
            return ((fx + fw / 2 - x - w / 2) ** 2 + (fy + fh / 2 - y - h / 2) ** 2) ** 0.5

        best = min(msg["faces"], key=dist)
        if dist(best) > ORACLE_DIST:
            return None
        self.locked = int(best["track_id"])
        return {"type": "control", "target": {"lock_track_id": self.locked}}


async def run(args):
    import websockets

    clip = await asyncio.to_thread(Clip.load, args.clip)
    async with websockets.connect(args.url, max_size=4 * 1024 * 1024) as ws:
        async def send(message):
            await ws.send(json.dumps(message) if isinstance(message, dict) else message)

        oracle = OracleLock(clip_manifest(args.clip)) if args.lock_target else None
        if args.lock_target and oracle.target is None:
            raise SystemExit(f"--lock-target needs an ingested clip with manifest.json, not {args.clip}")

        async def receive():
            async for message in ws:
                if isinstance(message, str):
                    if not args.quiet:
                        print(message, flush=True)
                    control = oracle.on_message(json.loads(message)) if oracle else None
                    if control:
                        print(json.dumps({"replay_oracle_lock": control}), flush=True)
                        await send(control)

        receiver = asyncio.create_task(receive())
        try:
            await feed(clip, send, loop=args.loop,
                       control={'mode': args.mode, 'suppression': args.suppression})
        finally:
            receiver.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip', type=Path, nargs='?', default=Path(__file__).resolve().parents[1] / 'eval/fixtures/demo/demo-mf-easy',
                        help='an MP4, or a clip directory / manifest.json (default eval/fixtures/demo/demo-mf-easy)')
    parser.add_argument('--url', default='ws://127.0.0.1:8000/ws/session')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--mode', default='natural', choices=['natural', 'focus_hq', 'synthetic', 'text', 'quiet', 'auto'])
    parser.add_argument('--suppression', type=float, default=80)
    parser.add_argument('--lock-target', action='store_true',
                        help="ingested clip: lock the manifest's target face (oracle) instead of letting TargetSelector choose")
    parser.add_argument('--quiet', action='store_true', help='do not print server messages')
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
