"""Focus HQ target-speaker extraction (TSE): the candidate contract and its registry.

Every bake-off candidate (AV-MossFormer2, Dolphin, WeSep, ...) implements :class:`TSECandidate`
in its own module ``hearing/tse/<name>.py`` and registers itself with
:func:`register_candidate`. Heavy imports (torch, clearvoice, ...) belong inside ``load()`` so
that importing the module, and therefore :func:`candidates`, stays cheap and offline.

Inputs to ``extract`` (all optional except ``mix``; a candidate lists what it uses in ``needs``):

* ``mix``: mono float32 at ``sr`` (16 kHz in the pipeline).
* ``lips``: ``T x H x W`` uint8 grayscale mouth crops at ``lips_fps`` (25 fps, 88x88 default),
  frame 0 aligned with sample 0 of ``mix``.
* ``p_speaking``: per-video-frame probability that the target is speaking, same frame grid.
* ``enroll``: 16 kHz target-only audio, any length (3-5 s typical).
* ``other``: 16 kHz speech of the people *other* than the target (a negative enrollment), or None.
  Passed as an extra keyword only to candidates whose ``needs`` include ``"other"``; it is never
  required (the stage runs the window with ``other=None`` while it is being collected).
* ``asd``: the target track's ``p_speaking`` per 10 ms of ``mix`` taken from the session's tracks
  (NaN where the target has no visible track). Extra keyword, only for candidates whose ``needs``
  include ``"asd"``; unlike ``p_speaking`` it needs no lip feed and never skips a window.

The return value is the target estimate at 16 kHz with the same length as ``mix``.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

SR = 16000
LIPS_FPS = 25.0
LIPS_SIZE = 88
NEEDS = frozenset({"lips", "p_speaking", "enroll", "other", "asd"})
# Length mismatch tolerated (and fixed by pad/trim) in a candidate's output, e.g. resampling roundoff.
LENGTH_TOLERANCE = 160


@runtime_checkable
class TSECandidate(Protocol):
    """A target-speaker extractor under test.

    Optional attribute ``oracle: bool`` (default False): the harness then passes the clean
    target, sample-aligned with ``mix``, as ``enroll`` (upper-bound candidates only).
    Optional attribute ``device: str`` set by ``load()``; the harness reports it.
    """

    name: str
    licence: str
    needs: frozenset[str]

    def load(self) -> None:
        """Acquire weights through ``hearing.models`` on ``hearing.device.pick()``. Idempotent."""
        ...

    def extract(
        self,
        mix: np.ndarray,
        sr: int,
        *,
        lips: np.ndarray | None,
        lips_fps: float,
        p_speaking: np.ndarray | None,
        enroll: np.ndarray | None,
    ) -> np.ndarray:
        """Return the target estimate at 16 kHz, same length as ``mix``."""
        ...


Factory = Callable[[], TSECandidate]
F = TypeVar("F", bound=Factory)
_registry: dict[str, Factory] = {}
_discovered = False


def register_candidate(name: str) -> Callable[[F], F]:
    """Register a zero-argument factory (usually the class) under ``name``."""

    def decorate(factory: F) -> F:
        if name in _registry and _registry[name] is not factory:
            raise ValueError(f"TSE candidate already registered: {name!r}")
        _registry[name] = factory
        return factory

    return decorate


def _discover() -> None:
    """Import every ``hearing.tse`` submodule once so their candidates register themselves."""
    global _discovered
    if _discovered:
        return
    import hearing.tse as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):  # vendored code (_dolphin, ...), not a candidate module
            continue
        try:
            importlib.import_module(f"{pkg.__name__}.{mod.name}")
        except Exception as e:  # noqa: BLE001  (a broken candidate module must not hide the others
            log.warning(
                "hearing.tse.%s failed to import: %s: %s", mod.name, type(e).__name__, e
            )
    _discovered = True


def candidates() -> list[str]:
    """Names of all registered candidates (submodules of ``hearing.tse`` are imported first)."""
    _discover()
    return sorted(_registry)


def get_candidate(name: str, **kwargs) -> TSECandidate:
    """Build a fresh (unloaded) candidate by name; ``kwargs`` go to its factory."""
    if name not in _registry:
        _discover()
    try:
        factory = _registry[name]
    except KeyError:
        raise KeyError(
            f"Unknown TSE candidate {name!r}; known: {', '.join(candidates())}"
        ) from None
    return factory(**kwargs)


def conform_output(y: np.ndarray, n: int, name: str = "candidate") -> np.ndarray:
    """Coerce a candidate's output to mono float32 of length ``n``.

    A length off by at most :data:`LENGTH_TOLERANCE` samples is padded with zeros or trimmed;
    a larger mismatch or a non-1-D array is a contract violation and raises ``ValueError``.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2 and 1 in y.shape:
        y = y.reshape(-1)
    if y.ndim != 1:
        raise ValueError(f"{name}: extract must return 1-D audio, got shape {y.shape}")
    if abs(y.shape[0] - n) > LENGTH_TOLERANCE:
        raise ValueError(
            f"{name}: extract returned {y.shape[0]} samples for a {n}-sample mix"
        )
    if y.shape[0] < n:
        y = np.pad(y, (0, n - y.shape[0]))
    return np.ascontiguousarray(y[:n])
