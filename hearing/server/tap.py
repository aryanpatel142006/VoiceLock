"""Session tap: record everything a live session sends and receives, for debugging.

Enabled by ``HEARING_SESSION_LOG=<dir>``; each WebSocket session gets its own
``<dir>/<utc-timestamp>-<n>/`` with:

* ``events.jsonl`` — every JSON event in either direction (``{"t", "dir", "event"}``)
  plus one summary line per binary frame (kind, seq, capture_ts, bytes);
* ``in.wav`` / ``out.wav`` — the client's microphone and the engine's output, PCM16 at the
  session's sample rate (``in`` is what the person heard without the app, ``out`` what
  the app played);
* ``out_pre_agc.wav`` — under the judge profile (16 kHz sessions), the same output before the output
  AGC (T-135), so separation (SI-SDRi) is scored before output leveling;
* ``frames/`` — every :data:`FRAME_EVERY`-th camera JPEG, so a target-lock problem can be
  seen afterwards.

Nothing here touches the engine: the tap only observes the wire, so it can't change
what the person hears. Off by default — recording audio is opt-in.
"""
from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path

from .protocol import HEADER, KIND_AUDIO_IN, KIND_AUDIO_OUT, KIND_VIDEO_IN

FRAME_EVERY = 15
_counter = 0


class SessionTap:
    def __init__(self, root: Path):
        global _counter
        _counter += 1
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        self.dir = Path(root) / f"{stamp}-{_counter}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "frames").mkdir(exist_ok=True)
        self.events = (self.dir / "events.jsonl").open("a")
        self.sample_rate = 16000
        self._wavs: dict[str, wave.Wave_write] = {}
        self._video = 0
        self.t0 = time.time()

    @classmethod
    def from_env(cls) -> SessionTap | None:
        root = os.environ.get("HEARING_SESSION_LOG")
        return cls(Path(root).expanduser()) if root else None

    # ------------------------------------------------------------------ wire
    def incoming(self, msg: bytes | str) -> None:
        if isinstance(msg, bytes):
            self._binary("in", msg)
            return
        try:
            event = json.loads(msg)
        except ValueError:
            event = {"raw": msg}
        if isinstance(event, dict) and event.get("type") == "hello":
            self.sample_rate = int(event.get("sample_rate", self.sample_rate))
        self._event("in", event)

    def outgoing(self, msg: bytes | dict) -> None:
        if isinstance(msg, bytes):
            self._binary("out", msg)
        else:
            self._event("out", msg)

    def pre_agc(self, pcm: bytes) -> None:
        """PCM16 output before the judge profile's AGC (``Session.pre_agc_sink``)."""
        self._wav("out_pre_agc").writeframes(pcm)

    def close(self) -> None:
        for w in self._wavs.values():
            w.close()
        self._event("tap", {"type": "tap_closed", "video_frames": self._video})
        self.events.close()

    # -------------------------------------------------------------- internals
    def _event(self, direction: str, event) -> None:
        self.events.write(json.dumps({"t": round(time.time() - self.t0, 3), "dir": direction,
                                      "event": event}) + "\n")
        self.events.flush()

    def _binary(self, direction: str, data: bytes) -> None:
        if len(data) < HEADER.size:
            self._event(direction, {"type": "short_binary", "bytes": len(data)})
            return
        kind, seq, capture_ts = HEADER.unpack_from(data, 0)
        payload = data[HEADER.size:]
        if kind == KIND_AUDIO_IN:
            self._wav("in").writeframes(payload)
        elif kind == KIND_AUDIO_OUT:
            self._wav("out").writeframes(payload)
        elif kind == KIND_VIDEO_IN:
            self._video += 1
            if self._video % FRAME_EVERY == 1:
                (self.dir / "frames" / f"{seq:08d}.jpg").write_bytes(payload)
        self._event(direction, {"type": "binary", "kind": kind, "seq": seq,
                                "capture_ts": round(capture_ts, 4), "bytes": len(payload)})

    def _wav(self, name: str) -> wave.Wave_write:
        w = self._wavs.get(name)
        if w is None:
            w = wave.open(str(self.dir / f"{name}.wav"), "wb")  # noqa: SIM115 - kept open for the session, closed in close()
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            self._wavs[name] = w
        return w
