"""hearing.server.warm: preload only what is on disk, report each model, never raise."""

import types

from hearing import models
from hearing.server import warm


def test_warm_loads_available_models_and_reports_the_rest(monkeypatch):
    loaded = []
    monkeypatch.setattr(models, "load", lambda name: loaded.append(name) or name)
    ok = types.ModuleType("t072_ok")
    ok.available = lambda: True
    off = types.ModuleType("t072_off")
    off.available = lambda: False
    monkeypatch.setitem(__import__("sys").modules, "t072_ok", ok)
    monkeypatch.setitem(__import__("sys").modules, "t072_off", off)
    plan = (("t072_ok", ("a", "b"), True), ("t072_off", ("c",), False), ("t072_missing_module", ("d",), False))
    done = warm.warm(plan)
    assert loaded == ["a", "b"]
    assert done["a"].startswith("ok") and done["b"].startswith("ok")
    assert done["c"] == "skipped: not available" and done["d"].startswith("skipped")


def test_a_failing_loader_is_reported_not_raised(monkeypatch):
    def boom(name):
        raise RuntimeError("no weights")

    monkeypatch.setattr(models, "load", boom)
    ok = types.ModuleType("t072_ok2")
    monkeypatch.setitem(__import__("sys").modules, "t072_ok2", ok)  # no available(): try it
    assert warm.warm((("t072_ok2", ("x",), False),)) == {"x": "RuntimeError: no weights"}


def test_warm_inference_skips_without_weights(monkeypatch):
    from hearing.tse import enrolled

    monkeypatch.setattr(enrolled, "available", lambda: False)
    assert warm.warm_inference() == "skipped: not available"


def test_health_reports_the_warm_state(monkeypatch):
    from fastapi.testclient import TestClient

    from hearing.server.app import create_app

    monkeypatch.setattr(warm, "_state", "warming")
    assert TestClient(create_app()).get("/health").json()["warm"] == "warming"
