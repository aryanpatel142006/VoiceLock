import argparse
import os
from types import SimpleNamespace

from hearing import cli


def test_serve_hq_flags_reach_server_config(monkeypatch):
    import uvicorn

    from hearing.server import app as app_mod
    from hearing.server import warm

    def unexpected_warm():
        raise AssertionError("fast tests must not start model warm-up")

    monkeypatch.setattr(warm, "start", unexpected_warm)
    seen = {}
    def fake_app(config):
        seen["config"] = config
        return SimpleNamespace(state=SimpleNamespace())  # serve sets state.connect / state.cert_path

    monkeypatch.setattr(app_mod, "create_app", fake_app)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    parser = argparse.ArgumentParser()
    cli._add_serve(parser.add_subparsers())
    args = parser.parse_args(["serve", "--no-warm", "--hq-backend", "remote", "--remote-url", "http://gpu:8000"])
    assert cli._serve(args) == 0
    assert seen["config"].hq_backend == "remote"
    assert seen["config"].remote_url == "http://gpu:8000"


def test_serve_warms_by_default_but_can_disable_without_serving():
    parser = argparse.ArgumentParser()
    cli._add_serve(parser.add_subparsers())
    assert parser.parse_args(["serve"]).warm is True
    assert parser.parse_args(["serve", "--no-warm"]).warm is False


def test_load_dotenv_exports_keys_but_the_shell_wins(tmp_path, monkeypatch):
    from hearing.cli import load_dotenv

    (tmp_path / ".env").write_text("# key\nELEVENLABS_API_KEY='sk_test'\nexport T112_OTHER=2\nT112_KEEP=file\nbroken\n")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("T112_OTHER", raising=False)
    monkeypatch.setenv("T112_KEEP", "shell")
    assert load_dotenv(tmp_path) == [tmp_path / ".env"]
    assert os.environ["ELEVENLABS_API_KEY"] == "sk_test"
    assert os.environ["T112_OTHER"] == "2"
    assert os.environ["T112_KEEP"] == "shell"
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    monkeypatch.delenv("T112_OTHER")
    assert load_dotenv(tmp_path / "nowhere") == []
