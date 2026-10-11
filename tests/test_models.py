from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest


def setup_models(monkeypatch):
    from hearing import models

    monkeypatch.setattr(models, "_loaders", {})
    monkeypatch.setattr(models, "_cache", {})
    return models


def test_load_caches_and_clear_keeps_loader(monkeypatch):
    models = setup_models(monkeypatch)
    calls = []

    @models.register_loader("test")
    def loader():
        calls.append(1)
        return object()

    first = models.load("test")
    assert models.load("test") is first
    assert len(calls) == 1
    models.clear()
    assert models.load("test") is not first
    assert len(calls) == 2


def test_none_result_is_cached(monkeypatch):
    models = setup_models(monkeypatch)
    calls = []

    @models.register_loader("none")
    def loader():
        calls.append(1)

    assert models.load("none") is None
    assert models.load("none") is None
    assert len(calls) == 1


def test_failed_load_can_retry(monkeypatch):
    models = setup_models(monkeypatch)
    calls = []

    @models.register_loader("retry")
    def loader():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("temporary failure")
        return "loaded"

    with pytest.raises(RuntimeError):
        models.load("retry")
    assert models.load("retry") == "loaded"


def test_concurrent_loads_share_instance(monkeypatch):
    models = setup_models(monkeypatch)
    started, release = Event(), Event()
    calls = []

    @models.register_loader("concurrent")
    def loader():
        calls.append(1)
        started.set()
        assert release.wait(timeout=1)
        return object()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(models.load, "concurrent")
        assert started.wait(timeout=1)
        second = pool.submit(models.load, "concurrent")
        release.set()
        assert first.result() is second.result()
    assert len(calls) == 1


def test_unknown_and_duplicate_loader_errors(monkeypatch):
    models = setup_models(monkeypatch)
    models.register_loader("known")(object)
    with pytest.raises(KeyError, match="missing.*known"):
        models.load("missing")
    with pytest.raises(ValueError, match="known"):
        models.register_loader("known")(object)


def test_cache_dir_resolves_without_creating_or_downloading(monkeypatch, tmp_path):
    models = setup_models(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEARING_MODELS_DIR", raising=False)
    assert models.cache_dir() == tmp_path / "models"
    monkeypatch.setenv("HEARING_MODELS_DIR", "custom/../weights")
    assert models.cache_dir() == tmp_path / "weights"
    assert list(tmp_path.iterdir()) == []
