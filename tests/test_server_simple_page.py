"""``/simple``: the stripped-down test page is served next to the console at ``/``."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hearing.server.app import create_app
from hearing.server.session import ServerConfig, default_console_dir


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")


def _client(console_dir: Path) -> TestClient:
    return TestClient(create_app(ServerConfig(console_dir=console_dir)))


def test_simple_serves_simple_html_and_root_still_serves_console(tmp_path):
    (tmp_path / "index.html").write_text("<h1>console</h1>")
    (tmp_path / "simple.html").write_text("<h1>simple</h1>")
    (tmp_path / "simple.js").write_text("console.log(2)")
    client = _client(tmp_path)
    r = client.get("/simple")
    assert r.status_code == 200
    assert "<h1>simple</h1>" in r.text
    assert r.headers["content-type"].startswith("text/html")
    assert "no-cache" in r.headers["cache-control"]
    assert "<h1>console</h1>" in client.get("/").text
    assert client.get("/simple.js").text == "console.log(2)"  # its relative imports resolve
    assert client.get("/health").json()["ok"] is True


def test_simple_is_404_without_the_page(tmp_path):
    (tmp_path / "index.html").write_text("<h1>console</h1>")
    assert _client(tmp_path).get("/simple").status_code == 404


def test_repo_simple_page_imports_the_console_protocol_module():
    console = default_console_dir()
    html = (console / "simple.html").read_text()
    js = (console / "simple.js").read_text()
    assert 'src="simple.js"' in html
    assert 'from "./protocol.js"' in js  # one wire implementation for both pages
    assert 'addModule("audio-worklet.js")' in js
    for el in ('data-mode="natural"', 'data-mode="focus_hq"', 'data-mode="synthetic"', 'data-mode="text"',
               'data-mode="quiet"', 'data-mode="auto"', 'id="suppression"', 'name="hq-backend"'):
        assert el in html


def test_judge_page_is_served(tmp_path):
    (tmp_path / "judge.html").write_text("<title>judge</title>")
    (tmp_path / "index.html").write_text("console")
    r = _client(tmp_path).get("/judge")
    assert r.status_code == 200 and "judge" in r.text and r.headers["cache-control"] == "no-cache"
