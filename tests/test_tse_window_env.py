"""T-419: HEARING_TSE_WINDOW_S / HEARING_TSE_HOP_S move the balanced enrolled window (1.0 / 0.5) and nothing else."""

from __future__ import annotations

import pytest

from hearing import registry
from hearing.server.rehearsal import ladder
from hearing.tse.stage import BALANCED_WINDOW, FocusHQ, balanced_window, tse_window

ENROLLED = 'tse(candidate="enrolled", window_s=1.0, hop_s=0.5, threaded=true, require_gpu=true)>mixer(max_duck_db=20)'


@pytest.fixture
def no_knobs(monkeypatch):
    monkeypatch.delenv("HEARING_TSE_WINDOW_S", raising=False)
    monkeypatch.delenv("HEARING_TSE_HOP_S", raising=False)
    return monkeypatch


def test_default_is_unchanged(no_knobs):
    assert balanced_window() == BALANCED_WINDOW == (1.0, 0.5)
    stage = FocusHQ(candidate="enrolled", window_s=1.0, hop_s=0.5)
    assert (stage.window_s, stage.hop_s, stage.latency_ms) == (1.0, 0.5, 1500.0)
    assert tse_window() == {"window_s": 1.0, "hop_s": 0.5, "delay_ms": 1500, "env": False}


def test_env_moves_the_balanced_enrolled_window(no_knobs):
    no_knobs.setenv("HEARING_TSE_WINDOW_S", "0.5")
    no_knobs.setenv("HEARING_TSE_HOP_S", "0.25")
    stage = FocusHQ(candidate="enrolled", window_s=1.0, hop_s=0.5)
    assert (stage.window_s, stage.hop_s, stage.latency_ms) == (0.5, 0.25, 750.0)
    assert stage.hops[0] == 4000  # the adaptive hop ladder starts at the new hop
    assert tse_window() == {"window_s": 0.5, "hop_s": 0.25, "delay_ms": 750, "env": True}
    tse = registry.build_pipeline(ENROLLED).stages[0]
    assert (tse.window_s, tse.hop_s) == (0.5, 0.25)  # the config/tiers.yaml spec as the ladder builds it


def test_env_leaves_other_windows_and_candidates(no_knobs):
    no_knobs.setenv("HEARING_TSE_WINDOW_S", "0.75")
    no_knobs.setenv("HEARING_TSE_HOP_S", "0.375")
    remote = FocusHQ(candidate="enrolled", window_s=0.5, hop_s=0.125)  # GPU box / speed preset keep theirs
    assert (remote.window_s, remote.hop_s) == (0.5, 0.125)
    mock = FocusHQ(candidate="identity", window_s=1.0, hop_s=0.5)
    assert (mock.window_s, mock.hop_s) == (1.0, 0.5)


def test_window_alone_halves_the_hop(no_knobs):
    no_knobs.setenv("HEARING_TSE_WINDOW_S", "0.75")
    assert balanced_window() == (0.75, 0.375)
    assert ladder()["balanced"]["window_s"] == 0.75 and ladder()["balanced"]["hop_s"] == 0.375


@pytest.mark.parametrize(("window", "hop"), [("abc", ""), ("0.5", "0.75"), ("0", "")])
def test_bad_knobs_raise(no_knobs, window, hop):
    no_knobs.setenv("HEARING_TSE_WINDOW_S", window)
    no_knobs.setenv("HEARING_TSE_HOP_S", hop)
    with pytest.raises(ValueError, match="HEARING_TSE"):
        balanced_window()


def test_health_reports_the_expected_delay(no_knobs, tmp_path):
    from fastapi.testclient import TestClient

    from hearing.server.app import create_app
    from hearing.server.session import ServerConfig

    no_knobs.setenv("HEARING_PROFILES_DIR", str(tmp_path))
    no_knobs.setenv("HEARING_TSE_WINDOW_S", "0.5")
    no_knobs.setenv("HEARING_TSE_HOP_S", "0.25")
    health = TestClient(create_app(ServerConfig())).get("/health").json()
    assert health["tse_window"] == {"window_s": 0.5, "hop_s": 0.25, "delay_ms": 750, "env": True}
