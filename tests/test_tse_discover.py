import sys
import types

from hearing.tse import base


def test_discover_skips_underscore_modules(monkeypatch, tmp_path):
    (tmp_path / "_vendored").mkdir()
    (tmp_path / "_vendored" / "__init__.py").write_text("raise RuntimeError('imported')\n")
    (tmp_path / "_hidden.py").write_text("raise RuntimeError('imported')\n")
    (tmp_path / "shown.py").write_text("SHOWN = True\n")
    monkeypatch.setattr(sys.modules["hearing.tse"], "__path__", [str(tmp_path)])
    imported = []
    monkeypatch.setattr(base.importlib, "import_module", lambda n, *a: imported.append(n) or types.ModuleType(n))
    monkeypatch.setattr(base, "_discovered", False)
    base._discover()
    assert imported == ["hearing.tse.shown"]
