"""T-124: session_report per-client levels, demo_regress waits for warm, session_score first isolated."""
from __future__ import annotations

import numpy as np

from scripts import demo_regress, session_report, session_score


def _bin(t):
    return {"dir": "in", "t": t, "event": {"type": "binary", "kind": 1}}


def test_hello_segments_split_in_wav_by_client():
    rows = [{"dir": "in", "t": 0, "event": {"type": "hello", "client": "page", "frame_ms": 20}}]
    rows += [_bin(i) for i in range(3)]
    rows += [{"dir": "in", "t": 5, "event": {"type": "hello", "client": "replay", "frame_ms": 20}}]
    rows += [_bin(6 + i) for i in range(2)]
    assert session_report.hello_segments(rows, 16000) == [("page", 0, 960), ("replay", 960, 1600)]


def test_digital_zero_is_reported_apart_from_quiet_audio():
    sr = 16000
    f = sr // 50
    a = np.concatenate([np.zeros(f * 2), np.full(f, 1e-6), np.full(f, 0.1)]).astype(np.float32)
    st = session_report.level_stats(a, sr)
    assert st["frames"] == 4
    assert st["zero_pct"] == 50.0
    assert st["below_80_pct"] == 25.0
    assert "exact digital zero 50.0%" in session_report.format_level("x", st)


def test_wait_warm_polls_until_ready():
    states = iter([{"warm": "warming"}, OSError("down"), {"warm": "warming"}, {"warm": "ready"}])

    def fetch():
        v = next(states)
        if isinstance(v, Exception):
            raise v
        return v

    assert demo_regress.wait_warm(1, timeout=5, fetch=fetch, sleep=lambda s: None)


def test_wait_warm_gives_up():
    assert not demo_regress.wait_warm(1, timeout=0.05, fetch=lambda: {"warm": "warming"}, sleep=lambda s: None)


def test_first_isolated_camera_free_fallbacks():
    cam = [{"t": 1.0, "acquisition": {"state": "locked"}}]
    assert session_score.first_isolated(cam) == 1.0
    nc = [{"t": 1.0, "name_call": {"state": "locking"}}, {"t": 2.0, "name_call": {"state": "isolated"}}]
    assert session_score.first_isolated(nc) == 2.0
    enrolled = [{"t": 1.0, "tier": "dfn"}, {"t": 3.0, "tiers": {"active": "enrolled"}}]
    assert session_score.first_isolated(enrolled) == 3.0
    assert session_score.first_isolated([{"t": 1.0}]) is None
