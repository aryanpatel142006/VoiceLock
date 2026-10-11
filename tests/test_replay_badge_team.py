"""T-173: under the team preset R / Shift+R replays the two-teammates mix (demo-team-easy) behind a REPLAY
badge that cannot be hidden while replay audio plays. Chromium with fake media; a tiny stand-in clip."""

from __future__ import annotations

import os

import pytest

from tests.test_device_picker import _serve
from tests.test_replay import clip


@pytest.mark.slow  # starts a server and Chromium; no weights, no network
def test_team_replay_clip_shows_an_unhideable_replay_badge(tmp_path, monkeypatch):
    sync_api = pytest.importorskip("playwright.sync_api")
    from eval.fixtures import video  # noqa: F401  (ffmpeg needed to build the clip)

    scene = tmp_path / "demo" / "demo-team-easy"
    scene.mkdir(parents=True)
    clip(scene, seconds=2.0).rename(scene / "clip.mp4")
    monkeypatch.setenv("HEARING_REPLAY_DIR", str(tmp_path / "demo"))
    monkeypatch.setenv("HEARING_DEMO_PRESET", "team")
    base, proc = _serve()
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                              "--autoplay-policy=no-user-gesture-required"])
            page = browser.new_context(permissions=["microphone", "camera"]).new_page()
            page.set_default_timeout(15000)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(base + "/judge")
            page.wait_for_function("window.hearingJudge.connected")
            page.wait_for_function("document.getElementById('replay-clip').value === 'demo-team-easy/clip.mp4'")
            badge = page.locator("#replay-badge")
            assert badge.is_hidden()

            page.keyboard.press("Shift+R")
            page.wait_for_function("window.hearingJudge.replaying === true || !document.getElementById('replay-badge').hidden")
            badge.wait_for(state="visible")
            assert badge.inner_text() == "REPLAY"

            # try every way of hiding it: the page puts it back and CSS keeps it displayed
            page.evaluate("""() => {
              const b = document.getElementById('replay-badge');
              b.hidden = true; b.style.display = 'none'; b.style.visibility = 'hidden'; b.style.opacity = '0';
            }""")
            page.wait_for_timeout(300)
            assert badge.is_visible()
            page.screenshot(path=str(os.environ.get("T173_SHOT") or tmp_path / "replay-badge.png"))

            page.keyboard.press("Shift+R")  # stop: the badge goes with the replay
            page.wait_for_function("document.getElementById('replay-badge').hidden")
            assert not errors, errors
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
