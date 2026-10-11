"""Long per-speaker recordings -> SIR-binned two-talker mixtures with ground truth (T-413).

Data-prep plan for Ahmad's ranked-approaches stack (approach 4, "tougher training", and the
confusion-metric eval every other approach is scored with). Input: a root holding one
directory per speaker with long takes (``<root>/<speaker>/profile-*.wav``, the 2-minute team
recordings). Output: fixtures in the ``eval/fixtures/mix.py`` layout plus ``enroll.wav``.

Rules (all deterministic from the seed):

* **Time split per take.** Each take is cut into contiguous regions with a 1 s guard between
  them: the speaker's first take opens with an *enrollment* region (``ENROLL_S``), and what is
  left of every take is split train / val / test by ``SPLIT_FRACTIONS``. Enrollment audio never
  appears in any mixture, and test sources never overlap train sources.
* **Speaker split.** ``holdout`` speakers (by name) give all their audio to ``test_unseen``;
  everyone else is "seen" (their test region is unseen *audio* of a seen voice).
* **Loudness levelling.** SIR is set on the *active speech level* (frame powers above a
  relative threshold, an ITU-T P.56-style approximation; no K-weighting, both stems are speech)
  so pauses do not bias it: ``sir_db = active_level(target) - active_level(interferer)``.
* **SIR bins.** Eval uses exact bins (default 0, -5, -15 dB, no jitter). Training draws a bin
  from ``TRAIN_SIR_BINS`` (asymmetric: most mass at or below 0 dB, the target quieter than
  the interferer) plus uniform jitter.
* **Noisy enrollment** (training only by default): the enrollment crop gets a different
  speaker (or a noise file) mixed in at ``ENROLL_SNR_DB``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.fixtures.mix import Manifest, Stems, write_fixture, write_index
from hearing.audio import io as aio

SR = aio.PIPELINE_SR
ENROLL_S = 30.0  # enrollment region at the start of each speaker's first take
GUARD_S = 1.0  # silence-free gap between regions so crops cannot straddle them
SPLIT_FRACTIONS = (("train", 0.8), ("val", 0.1), ("test", 0.1))
EVAL_SIR_BINS = (0.0, -5.0, -15.0)  # Ahmad: 0 dB required, -5 good, -15 aspirational
TRAIN_SIR_BINS = ((5.0, 0.15), (0.0, 0.3), (-5.0, 0.3), (-10.0, 0.15), (-15.0, 0.1))
TRAIN_SIR_JITTER_DB = 2.5
ENROLL_SNR_DB = (5.0, 20.0)  # noisy-enrollment range (training)
CROP_S = 4.0
ENROLL_CROP_S = 10.0
EVAL_LEVEL_DB = -26.0  # active level of the mix for eval items, dBFS
TRAIN_LEVEL_DB = (-35.0, -18.0)
PEAK = 0.99
MIN_ACTIVE = 0.5  # a crop needs at least this share of active frames
NO_NOISE_SNR_DB = 120.0  # Manifest.snr_db for items without a noise stem (JSON has no inf)
FRAME_S = 0.02
REL_THRESHOLD_DB = 25.0  # active frame: within this many dB of the 90th-percentile frame
ABS_FLOOR_DB = -70.0
DEFAULT_PATTERN = "profile-*.wav"


def _frame_power_db(x: np.ndarray, sr: int = SR) -> np.ndarray:
    n = max(1, int(FRAME_S * sr))
    frames = len(x) // n
    if frames == 0:
        return np.full(1, -200.0)
    f = np.asarray(x[: frames * n], np.float64).reshape(frames, n)
    return 10.0 * np.log10(np.mean(f * f, axis=1) + 1e-20)


def active_mask(x: np.ndarray, sr: int = SR) -> np.ndarray:
    p = _frame_power_db(x, sr)
    thr = max(ABS_FLOOR_DB, float(np.percentile(p, 90)) - REL_THRESHOLD_DB)
    return p > thr


def active_level_db(x: np.ndarray, sr: int = SR) -> float:
    """Mean power (dBFS) over active frames; ``-inf`` for a silent signal."""
    p = _frame_power_db(x, sr)
    if p.max() <= ABS_FLOOR_DB:
        return -math.inf
    active = p[active_mask(x, sr)]
    return float(10.0 * np.log10(np.mean(10.0 ** (active / 10.0))))


def active_share(x: np.ndarray, reference_db: float, sr: int = SR) -> float:
    """Share of frames within ``REL_THRESHOLD_DB`` of ``reference_db`` (the whole take's level)."""
    return float(np.mean(_frame_power_db(x, sr) > reference_db - REL_THRESHOLD_DB))


@dataclass(frozen=True)
class Region:
    speaker: str
    take: str  # path relative to the root
    split: str  # enroll | train | val | test | test_unseen
    start: int  # samples
    stop: int

    @property
    def seconds(self) -> float:
        return (self.stop - self.start) / SR


@dataclass
class Corpus:
    """Loaded takes and the region ledger; ``audio[take]`` is float32 at 16 kHz."""

    root: Path
    regions: list[Region]
    audio: dict[str, np.ndarray] = field(repr=False)
    levels: dict[str, float] = field(default_factory=dict)

    def level(self, take: str) -> float:
        if take not in self.levels:
            self.levels[take] = active_level_db(self.audio[take])
        return self.levels[take]

    def speakers(self, split: str) -> list[str]:
        return sorted({r.speaker for r in self.regions if r.split == split})

    def of(self, speaker: str, split: str) -> list[Region]:
        return [r for r in self.regions if r.speaker == speaker and r.split == split]

    def ledger(self) -> dict:
        out: dict = {}
        for r in self.regions:
            row = out.setdefault(r.speaker, {})
            row[r.split] = round(row.get(r.split, 0.0) + r.seconds, 2)
        return out


def discover(root: str | os.PathLike, pattern: str = DEFAULT_PATTERN, min_s: float = 60.0) -> dict[str, list[Path]]:
    """``{speaker: [takes]}`` for every ``<root>/<speaker>/<pattern>`` at least ``min_s`` long."""
    import soundfile as sf

    root = Path(root)
    found: dict[str, list[Path]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        takes = [p for p in sorted(d.glob(pattern)) if sf.info(str(p)).duration >= min_s]
        if takes:
            found[d.name] = takes
    return found


def plan_regions(
    takes: dict[str, list[Path]],
    lengths: dict[Path, int],
    root: Path,
    holdout: tuple[str, ...] = (),
    enroll_s: float = ENROLL_S,
) -> list[Region]:
    """The region ledger (pure function of take lengths; see the module docstring)."""
    guard = int(GUARD_S * SR)
    regions: list[Region] = []
    for speaker, paths in sorted(takes.items()):
        for k, path in enumerate(paths):
            rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            n = lengths[path]
            start = 0
            if k == 0:
                e = min(int(enroll_s * SR), n // 2)
                regions.append(Region(speaker, rel, "enroll", 0, e))
                start = e + guard
            if speaker in holdout:
                regions.append(Region(speaker, rel, "test_unseen", start, n))
                continue
            usable = n - start - guard * (len(SPLIT_FRACTIONS) - 1)
            for split, frac in SPLIT_FRACTIONS:
                stop = n if split == SPLIT_FRACTIONS[-1][0] else start + int(frac * usable)
                if stop - start > 0:
                    regions.append(Region(speaker, rel, split, start, stop))
                start = stop + guard
    return regions


def load_corpus(root, pattern: str = DEFAULT_PATTERN, holdout: tuple[str, ...] = (), min_s: float = 60.0,
                enroll_s: float = ENROLL_S) -> Corpus:
    root = Path(root)
    takes = discover(root, pattern, min_s)
    audio, lengths = {}, {}
    for paths in takes.values():
        for p in paths:
            x, _ = aio.load(p, SR)
            lengths[p] = len(x)
            audio[str(p.relative_to(root))] = x.astype(np.float32)
    return Corpus(root, plan_regions(takes, lengths, root, holdout, enroll_s), audio)


def _crop(corpus: Corpus, regions: list[Region], rng: np.random.Generator, n: int, tries: int = 20) -> tuple[np.ndarray, dict]:
    """A speech-active crop of ``n`` samples from one of ``regions`` (longest-first fallback)."""
    fits = [r for r in regions if r.stop - r.start >= n]
    if not fits:
        raise ValueError(f"no region of {regions[0].speaker if regions else '?'} holds {n / SR:.1f} s")
    weights = np.array([r.stop - r.start - n + 1 for r in fits], np.float64)
    best = None
    for _ in range(tries):
        r = fits[rng.choice(len(fits), p=weights / weights.sum())]
        s = int(rng.integers(r.start, r.stop - n + 1))
        x = corpus.audio[r.take][s : s + n]
        share = active_share(x, corpus.level(r.take))
        if best is None or share > best[2]:
            best = (x, {"take": r.take, "start_s": round(s / SR, 3)}, share)
        if share >= MIN_ACTIVE:
            break
    x, meta, share = best
    meta["active_share"] = round(share, 3)
    return x.astype(np.float64).copy(), meta


def _scale_to(x: np.ndarray, level_db: float) -> np.ndarray:
    cur = active_level_db(x)
    return x if not math.isfinite(cur) else x * 10.0 ** ((level_db - cur) / 20.0)


def draw_train_sir(rng: np.random.Generator) -> float:
    bins = np.array([b for b, _ in TRAIN_SIR_BINS])
    p = np.array([w for _, w in TRAIN_SIR_BINS])
    return float(rng.choice(bins, p=p / p.sum()) + rng.uniform(-TRAIN_SIR_JITTER_DB, TRAIN_SIR_JITTER_DB))


def render_item(
    corpus: Corpus,
    split: str,
    sir_db: float,
    rng: np.random.Generator,
    crop_s: float = CROP_S,
    enroll_crop_s: float = ENROLL_CROP_S,
    level_db: float = EVAL_LEVEL_DB,
    enroll_snr_db: float | None = None,
    noise: np.ndarray | None = None,
    snr_db: float | None = None,
    target: str | None = None,
) -> dict:
    """One mixture: ``mix == target + interferer + noise`` and an enrollment of the target.

    The torch-free core a training ``IterableDataset`` calls per item (approach 4 is a
    dataloader-only change); ``render_set`` calls it for eval fixtures.
    """
    pool = corpus.speakers(split)
    if len(pool) < 2 and split != "test_unseen":
        raise ValueError(f"split {split!r} needs at least two speakers, has {pool}")
    if split == "test_unseen":  # an unseen target may face any other speaker's test audio
        pool = sorted(set(pool) | set(corpus.speakers("test")))
    t = target or pool[int(rng.integers(len(pool)))]
    others = [s for s in pool if s != t]
    i = others[int(rng.integers(len(others)))]
    n, ne = int(crop_s * SR), int(enroll_crop_s * SR)
    t_split = split if corpus.of(t, split) else "test"
    i_split = split if corpus.of(i, split) else "test"
    tgt, t_meta = _crop(corpus, corpus.of(t, t_split), rng, n)
    itf, i_meta = _crop(corpus, corpus.of(i, i_split), rng, n)
    tgt = _scale_to(tgt, level_db)
    itf = _scale_to(itf, level_db - sir_db)
    nse = np.zeros(n)
    if noise is not None and snr_db is not None and len(noise) >= n:
        s = int(rng.integers(0, len(noise) - n + 1))
        nse = _scale_to(np.asarray(noise[s : s + n], np.float64), level_db - snr_db)
    enroll, e_meta = _crop(corpus, corpus.of(t, "enroll"), rng, ne)
    if enroll_snr_db is not None:
        others_e = [s for s in corpus.speakers("enroll") if s != t]
        if noise is not None and len(noise) >= ne:
            s = int(rng.integers(0, len(noise) - ne + 1))
            dist, e_meta["noise"] = np.asarray(noise[s : s + ne], np.float64), "noise"
        else:
            o = others_e[int(rng.integers(len(others_e)))]
            dist, _ = _crop(corpus, corpus.of(o, "enroll"), rng, ne)
            e_meta["noise"] = f"speaker:{o}"
        enroll = enroll + _scale_to(dist, active_level_db(enroll) - enroll_snr_db)
        e_meta["snr_db"] = round(enroll_snr_db, 2)
    mix = tgt + itf + nse
    peak = float(np.abs(mix).max())
    if peak > PEAK:
        g = PEAK / peak
        tgt, itf, nse, mix = tgt * g, itf * g, nse * g, mix * g
    peak_e = float(np.abs(enroll).max())
    if peak_e > PEAK:
        enroll = enroll * PEAK / peak_e
    f32 = np.float32
    return {
        "mix": mix.astype(f32), "target": tgt.astype(f32), "interferer": itf.astype(f32),
        "noise": nse.astype(f32), "enroll": enroll.astype(f32),
        "target_id": t, "interferer_id": i, "sir_db": float(sir_db),
        "measured_sir_db": round(active_level_db(tgt) - active_level_db(itf), 3),
        "snr_db": float(snr_db) if np.any(nse) else NO_NOISE_SNR_DB,
        "source": {"target": t_meta, "interferer": i_meta, "enroll": e_meta, "level": "active_p56"},
    }


def render_set(corpus: Corpus, out: str | os.PathLike, split: str = "test", bins=EVAL_SIR_BINS,
               per_bin: int = 20, seed: int = 413, enroll_snr_db: float | None = None) -> list[Manifest]:
    """Eval fixtures: ``per_bin`` items per exact SIR bin, targets cycled over the speakers."""
    out = Path(out)
    set_name = out.name
    manifests = []
    speakers = corpus.speakers(split) or corpus.speakers("test")
    for b in bins:
        for k in range(per_bin):
            sub = int(hashlib.sha256(f"{seed}:{split}:{b}:{k}".encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(sub)
            item = render_item(corpus, split, float(b), rng, target=speakers[k % len(speakers)],
                               enroll_snr_db=enroll_snr_db)
            fid = f"sir{int(b):+d}_{k:03d}"
            m = Manifest(
                id=fid, set=set_name, sample_rate=SR, duration_s=round(len(item["mix"]) / SR, 3),
                snr_db=item["snr_db"], sir_db=item["sir_db"], target_id=item["target_id"],
                interferer_id=item["interferer_id"], noise_type="none", transcript="",
                paths={name: f"{name}.wav" for name in ("mix", "target", "interferer", "noise")},
                seed=sub, source={**item["source"], "split": split,
                                  "measured_sir_db": item["measured_sir_db"], "enroll": {
                                      **item["source"]["enroll"], "path": "enroll.wav"}},
            )
            stems = Stems(item["mix"], item["target"], item["interferer"], item["noise"])
            d = write_fixture(out, m, stems)
            aio.save(d / "enroll.wav", item["enroll"], SR)
            manifests.append(m)
    write_index(out, set_name, seed, manifests)
    return manifests


def _default_root() -> Path:
    base = os.environ.get("HEARING_FIXTURES_DIR")
    if base:
        return Path(base) / "team"
    common = Path(os.popen("git rev-parse --path-format=absolute --git-common-dir").read().strip())
    return common.parent / "eval" / "fixtures" / "generated" / "team"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=("plan", "render"))
    ap.add_argument("--root", type=Path, default=None, help="<root>/<speaker>/<takes> (default: team fixtures)")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--holdout", nargs="*", default=(), help="speakers kept out of training entirely")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bins", type=float, nargs="+", default=list(EVAL_SIR_BINS))
    ap.add_argument("--per-bin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=413)
    ap.add_argument("--noisy-enroll-db", type=float, default=None, help="enrollment SNR (dB) for eval items")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    root = args.root or _default_root()
    corpus = load_corpus(root, args.pattern, tuple(args.holdout))
    if args.command == "plan":
        print(json.dumps({"root": str(root), "seconds": corpus.ledger()}, indent=2))
        return 0
    base = os.environ.get("HEARING_FIXTURES_DIR") or str(_default_root().parent)
    out = args.out or Path(base) / f"sir_mix_{args.split}"
    ms = render_set(corpus, out, args.split, args.bins, args.per_bin, args.seed, args.noisy_enroll_db)
    print(f"wrote {len(ms)} fixtures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
