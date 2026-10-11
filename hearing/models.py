"""Lazy model instances. Loaders own weight acquisition; this module never downloads."""

import os
from collections.abc import Callable
from pathlib import Path
from threading import RLock
from typing import TypeVar

Loader = TypeVar("Loader", bound=Callable[[], object])
_loaders: dict[str, Callable[[], object]] = {}
_cache: dict[str, object] = {}
_lock = RLock()


def register_loader(name: str) -> Callable[[Loader], Loader]:
    """Register a zero-argument loader without calling it."""

    def decorate(loader: Loader) -> Loader:
        with _lock:
            if name in _loaders:
                raise ValueError(f"Model loader already registered: {name!r}")
            _loaders[name] = loader
        return loader

    return decorate


def load(name: str) -> object:
    """Share one instance per name, including across concurrent callers."""
    with _lock:
        if name not in _cache:
            try:
                loader = _loaders[name]
            except KeyError:
                raise KeyError(
                    f"Unknown model {name!r}; known models: {', '.join(sorted(_loaders))}"
                ) from None
            _cache[name] = loader()
        return _cache[name]


def clear() -> None:
    """Drop cached references, retaining loaders; active callers keep their instances."""
    with _lock:
        _cache.clear()


def cache_dir() -> Path:
    """Resolve the configured path relative to cwd, without creating directories."""
    return Path(os.environ.get("HEARING_MODELS_DIR", "models")).expanduser().resolve()
