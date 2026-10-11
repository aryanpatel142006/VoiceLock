"""Local MP4 -> browser-compatible paced PCM/JPEG messages. No device or network I/O."""
from __future__ import annotations

import asyncio
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from hearing.audio.io import float_to_pcm16
from hearing.server.protocol import pack_frame

DEFAULT_CLIP = 'demo-mf-easy'  # eval.make_fixtures --demo (T-097); 'grid-two-face.mp4' is the legacy loop


def clip_path(path: Path) -> Path:
    """``<name>/``, ``<name>/manifest.json`` or an MP4 → the MP4 to decode."""
    path = Path(path)
    if path.is_dir():
        return path / 'clip.mp4'
    if path.suffix.lower() == '.json':
        return path.with_name(json.loads(path.read_text()).get('clip', 'clip.mp4'))
    return path


def clip_manifest(path: Path) -> dict | None:
    """The ingest manifest next to an ingested clip (``scripts/ingest_clip.py``), else None."""
    mp4 = clip_path(path)
    manifest = mp4.with_name('manifest.json')
    if mp4.name != 'clip.mp4' or not manifest.is_file():
        return None
    return json.loads(manifest.read_text())


def list_clips(root: Path) -> list[dict]:
    """Every replayable clip under the demo directory, by the name POST /replay accepts."""
    root = Path(root)
    clips = [{'clip': p.name, 'label': p.stem + (' (legacy loop)' if p.name == 'grid-two-face.mp4' else '')}
             for p in sorted(root.glob('*.mp4'))]
    for mp4 in sorted(root.glob('*/clip.mp4')):
        entry = {'clip': f'{mp4.parent.name}/clip.mp4', 'label': mp4.parent.name}
        manifest = clip_manifest(mp4)
        if manifest:
            entry.update(duration_s=manifest.get('duration_s'), target=manifest.get('target', {}).get('side'))
        clips.append(entry)
    return clips


@dataclass(frozen=True)
class Clip:
    audio: np.ndarray
    images: tuple[bytes, ...]
    fps: int = 25

    @classmethod
    def load(cls, path: Path) -> Clip:
        """An MP4, or an ingested clip's directory / ``manifest.json`` (its ``clip.mp4``)."""
        from eval.fixtures import video

        path = clip_path(path)
        frames, _ = video.read_video(path, fps=25)
        audio = video.read_audio(path)
        images = []
        for frame in frames:
            out = io.BytesIO()
            Image.fromarray(frame).save(out, 'JPEG', quality=85)
            images.append(out.getvalue())
        if not len(audio) or not images:
            raise ValueError('clip needs both audio and video')
        return cls(audio, tuple(images))

    @property
    def duration(self) -> float:
        return min(len(self.audio) / 16000, len(self.images) / self.fps)


async def feed(clip: Clip, send, *, loop=False, clock=time.monotonic,
               sleep=asyncio.sleep, timestamp=None, control=None) -> None:
    """Schedule against absolute deadlines (processing never adds cumulative drift).

    ``send`` is an async transport callback accepting a JSON dict or wire bytes.
    Sequences count per media kind (audio 0, 1, 2, ...; video 0, 1, 2, ...) like the browser
    clients, so a seq gap in one stream is a lost frame of that stream (T-072). Sequences and
    capture timestamps stay monotonic across loops.
    """
    await send({'type': 'hello', 'sample_rate': 16000, 'frame_ms': 20,
                'video': True, 'client': 'replay'})
    await send({'type': 'control', **(control or {})})
    start = clock()
    epoch = time.time() if timestamp is None else timestamp
    seq = {1: 0, 2: 0}  # per kind: audio in, video in
    offset = 0.0
    while True:
        events = [(i / clip.fps, 2, image) for i, image in enumerate(clip.images)
                  if i / clip.fps < clip.duration]
        for i in range(0, int(clip.duration * 16000), 320):
            samples = clip.audio[i:min(i + 320, int(clip.duration * 16000))]
            samples = np.pad(samples, (0, 320 - len(samples)))
            events.append((i / 16000, 1, float_to_pcm16(samples)))
        # At a shared timestamp video goes first, so vision can update before the audio hop.
        events.sort(key=lambda item: (item[0], -item[1]))
        for at, kind, payload in events:
            deadline = offset + at
            await sleep(max(0.0, start + deadline - clock()))
            await send(pack_frame(kind, seq[kind], epoch + deadline, payload))
            seq[kind] += 1
        offset += clip.duration
        await sleep(max(0.0, start + offset - clock()))
        if not loop:
            return
