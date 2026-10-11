"""GRID audio-visual corpus (Cooke, Barker, Cunningham, Shao 2006; CC BY 4.0) for the AV fixture bank.

Source: Zenodo record 3625687 (``s<N>.zip`` per speaker, ~400 MB each, 1000 ``.mpg`` clips of
360x288 at 25 fps with 44.1 kHz audio; ``alignments.zip``, 20 MB, word timings for every clip).
Only the clips we use are fetched: a speaker zip is opened over HTTP ``Range`` requests and
single members are read out of it, so a tiny AV set costs ~2 MB of video plus the alignments.

Everything lands in ``$HEARING_FIXTURES_DIR/_sources/grid/`` (default ``eval/fixtures/generated``) (gitignored) and is reused on the
next run; with a warm cache nothing touches the network.

    for utt in iter_utterances("s1", limit=3):
        video_path, audio, transcript = utt      # audio: float32 mono 16 kHz from the clip
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.fixtures import video as fxvideo
from eval.metrics import normalize_text
from hearing.audio import io as aio

log = logging.getLogger(__name__)

SOURCES_DIR = (Path(os.environ["HEARING_FIXTURES_DIR"]) if os.environ.get("HEARING_FIXTURES_DIR") else Path(__file__).resolve().parent / "generated") / "_sources" / "grid"
ZENODO_RECORD = "3625687"
ZENODO_URL = "https://zenodo.org/api/records/" + ZENODO_RECORD + "/files/{name}/content"
DATASET = "GRID"
LICENSE = "CC BY 4.0 (GRID corpus, Cooke et al. 2006, doi:10.5281/zenodo.3625687)"
# One male and one female talker; sex from the corpus documentation, checked by eye and median f0.
SPEAKERS: dict[str, str] = {"s1": "M", "s4": "F"}
# Extra talkers for same-gender association pairs (T-069; median f0 s2 ~104 Hz, s7 ~168 Hz). Kept out of
# SPEAKERS so the av fixture sets (built from every SPEAKERS entry) do not change.
EXTRA_SPEAKERS: dict[str, str] = {"s2": "M", "s7": "F"}
ALIGN_UNITS_PER_S = 25000.0  # .align times are in 1/25000 s (1000 units per 25 fps frame)
SILENCE = frozenset({"sil", "sp"})
# Approximate face boxes (x, y, w, h normalised, hairline to chin) in a GRID 360x288 frame,
# measured by eye on frame 40 of one clip per talker; GRID framing is fixed per talker.
FACE_BOXES: dict[str, tuple[float, float, float, float]] = {
    "s1": (0.25, 0.24, 0.35, 0.63),
    "s4": (0.33, 0.38, 0.30, 0.47),
    # T-069: MediaPipe's median face box on the idle frames, raised 0.12 and grown 0.14 to the hairline
    "s2": (0.32, 0.32, 0.35, 0.57),
    "s7": (0.34, 0.39, 0.31, 0.53),
}
FACE_BOX_DEFAULT = (0.22, 0.15, 0.42, 0.72)  # head-and-shoulders centre for unmeasured talkers


def face_box(speaker: str) -> tuple[float, float, float, float]:
    return FACE_BOXES.get(speaker, FACE_BOX_DEFAULT)


# --- alignments -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Word:
    start_s: float
    end_s: float
    word: str


def parse_align(text: str) -> list[Word]:
    """Parse a GRID ``.align`` file (``start end word`` per line, 1/25000 s units)."""
    words = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        a, b, w = parts
        words.append(Word(int(a) / ALIGN_UNITS_PER_S, int(b) / ALIGN_UNITS_PER_S, w.lower()))
    return words


def transcript_of(words: list[Word]) -> str:
    return normalize_text(" ".join(w.word for w in words if w.word not in SILENCE))


def speech_intervals(words: list[Word], duration_s: float | None = None) -> list[list[float]]:
    """Merged ``[start_s, end_s]`` spans of non-silence words, clipped to ``duration_s``."""
    out: list[list[float]] = []
    for w in words:
        if w.word in SILENCE:
            continue
        s, e = w.start_s, w.end_s
        if duration_s is not None:
            s, e = min(s, duration_s), min(e, duration_s)
        if e <= s:
            continue
        if out and s <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [[round(s, 4), round(e, 4)] for s, e in out]


def loop_intervals(intervals: list[list[float]], period_s: float, duration_s: float) -> list[list[float]]:
    """Repeat intervals every ``period_s`` (a looped clip) and clip them to ``duration_s``."""
    if period_s <= 0:
        raise ValueError("period must be positive")
    out: list[list[float]] = []
    k = 0
    while k * period_s < duration_s:
        for s, e in intervals:
            s2, e2 = s + k * period_s, min(min(e, period_s) + k * period_s, duration_s)
            if s2 < e2:
                out.append([round(s2, 4), round(e2, 4)])
        k += 1
    return out


# --- remote zip -----------------------------------------------------------------------------


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP ``Range`` requests."""

    def __init__(self, url: str, client, counter: list[int] | None = None, retries: int = 5):
        super().__init__()
        self.url, self.client, self.retries = url, client, retries
        self.counter = counter if counter is not None else [0]
        self.pos = 0
        r = self._get(0, 0)
        self.size = int(r.headers["content-range"].rsplit("/", 1)[1])

    def _get(self, start: int, end: int):
        for attempt in range(self.retries):
            r = self.client.get(self.url, headers={"Range": f"bytes={start}-{end}"})
            if r.status_code == 206:
                self.counter[0] += len(r.content)
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("retry-after", 2 ** attempt))
                log.warning("GRID download: HTTP %s, retrying in %.0f s", r.status_code, wait)
                time.sleep(min(wait, 90.0))
                continue
            r.raise_for_status()
            raise RuntimeError(f"server ignored Range for {self.url} (HTTP {r.status_code})")
        raise RuntimeError(f"giving up on {self.url} after {self.retries} attempts")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:
        k = min(len(b), self.size - self.pos)
        if k <= 0:
            return 0
        data = self._get(self.pos, self.pos + k - 1).content
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


# --- corpus ---------------------------------------------------------------------------------


@dataclass
class GridUtterance:
    """One GRID clip. Iterating yields ``(video_path, audio, transcript)``."""

    speaker: str
    utt: str
    sex: str
    video_path: Path
    audio: np.ndarray  # float32 mono, 16 kHz, decoded from the clip's own audio track
    words: list[Word] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"grid-{self.speaker}-{self.utt}"

    @property
    def transcript(self) -> str:
        return transcript_of(self.words)

    def __iter__(self):
        return iter((self.video_path, self.audio, self.transcript))


class GridCorpus:
    """Cached access to GRID clips and alignments under ``root``; downloads only what is missing."""

    def __init__(self, root: str | os.PathLike = SOURCES_DIR, client=None):
        self.root = Path(root)
        self._client = client
        self._counter = [0]
        self._remote: dict[str, zipfile.ZipFile] = {}
        self._align_zip: zipfile.ZipFile | None = None

    @property
    def bytes_downloaded(self) -> int:
        return self._counter[0]

    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(follow_redirects=True, timeout=120.0)
        return self._client

    # alignments: the 20 MB zip is cached whole (tens of thousands of tiny members)
    def _alignments(self) -> zipfile.ZipFile:
        if self._align_zip is None:
            path = self.root / "alignments.zip"
            if not path.exists():
                self.root.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                log.info("downloading GRID alignments.zip (20 MB)")
                with self.client.stream("GET", ZENODO_URL.format(name="alignments.zip")) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_bytes(1 << 20):
                            f.write(chunk)
                            self._counter[0] += len(chunk)
                tmp.replace(path)
            self._align_zip = zipfile.ZipFile(path)
        return self._align_zip

    def utterance_ids(self, speaker: str) -> list[str]:
        prefix = f"alignments/{speaker}/"
        return sorted(
            n[len(prefix) : -len(".align")]
            for n in self._alignments().namelist()
            if n.startswith(prefix) and n.endswith(".align")
        )

    def words(self, speaker: str, utt: str) -> list[Word]:
        return parse_align(self._alignments().read(f"alignments/{speaker}/{utt}.align").decode())

    def video_path(self, speaker: str, utt: str) -> Path:
        """Local path of the clip, fetched from the speaker zip on Zenodo when not cached."""
        path = self.root / speaker / f"{utt}.mpg"
        if path.exists():
            return path
        if speaker not in self._remote:
            raw = RangeFile(ZENODO_URL.format(name=f"{speaker}.zip"), self.client, self._counter)
            self._remote[speaker] = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 16))
        member = f"{speaker}/{utt}.mpg"
        try:
            data = self._remote[speaker].read(member)
        except KeyError as e:
            raise FileNotFoundError(f"{member} is not in {speaker}.zip") from e
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path

    def load(self, speaker: str, utt: str) -> GridUtterance:
        path = self.video_path(speaker, utt)
        return GridUtterance(
            speaker=speaker,
            utt=utt,
            sex=SPEAKERS.get(speaker) or EXTRA_SPEAKERS.get(speaker, ""),
            video_path=path,
            audio=fxvideo.read_audio(path, aio.PIPELINE_SR),
            words=self.words(speaker, utt),
        )

    def pick(self, speaker: str, n: int, rng: np.random.Generator, exclude: set[str] = frozenset()) -> list[str]:
        """``n`` distinct utterance ids of ``speaker`` drawn with ``rng`` (deterministic per seed)."""
        ids = [u for u in self.utterance_ids(speaker) if u not in exclude]
        if len(ids) < n:
            raise ValueError(f"{speaker} has only {len(ids)} usable utterances, need {n}")
        return [ids[i] for i in sorted(rng.choice(len(ids), size=n, replace=False))]


def iter_utterances(
    speaker: str,
    ids: list[str] | None = None,
    limit: int | None = None,
    corpus: GridCorpus | None = None,
) -> Iterator[GridUtterance]:
    """Yield GRID clips of ``speaker`` (all, ``ids``, or the first ``limit``), downloading as needed."""
    corpus = corpus or GridCorpus()
    chosen = ids if ids is not None else corpus.utterance_ids(speaker)
    if limit is not None:
        chosen = chosen[:limit]
    for utt in chosen:
        try:
            yield corpus.load(speaker, utt)
        except FileNotFoundError as e:  # a handful of GRID clips are missing from the release
            log.warning("skipping %s: %s", utt, e)
