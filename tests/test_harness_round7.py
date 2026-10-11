"""T-118: demo_regress prints the backend each run used, fails a local run under HEARING_REMOTE_URL,
warns on a remote tiers config that differs, and scores the cold start (DNSMOS first 7 s)."""
from __future__ import annotations

import numpy as np

from scripts import demo_regress as dr


def tick(hq_active: str | None) -> dict:
    return {"dir": "out", "event": {"type": "telemetry", "hq_active": hq_active}}


def test_backend_used_is_the_majority_hq_active():
    rows = [tick("local"), tick("remote"), tick("remote"), {"dir": "in", "event": {"type": "control"}}]
    assert dr.backend_used(rows) == "remote"
    assert dr.backend_used([tick(None)]) is None


def test_local_focus_run_with_a_remote_url_fails():
    url = "ws://127.0.0.1:18765/ws/session"
    assert dr.played_local_with_remote("focus_hq", "local", url)
    assert dr.played_local_with_remote("focus_hq@80", None, url)
    assert not dr.played_local_with_remote("focus_hq", "remote", url)
    assert not dr.played_local_with_remote("natural", "local", url)  # natural never uses the GPU
    assert not dr.played_local_with_remote("focus_hq", "local", None)


def test_health_url_from_the_session_url():
    assert dr.health_url("ws://127.0.0.1:18765/ws/session") == "http://127.0.0.1:18765/health"
    assert dr.health_url("wss://gpu.example:443/ws/session") == "https://gpu.example:443/health"


def test_remote_config_warning():
    local = {"tiers-remote.yaml": "abc123", "tiers.yaml": "fff"}
    assert dr.remote_config_warning({"tiers_config": {"tiers-remote.yaml": "abc123"}}, local) is None
    warn = dr.remote_config_warning({"tiers_config": {"tiers-remote.yaml": "999999"}, "git_rev": "x-dirty"}, local)
    assert "999999 != local abc123" in warn and "x-dirty" in warn
    assert "no tiers-remote.yaml hash" in dr.remote_config_warning({"ok": True}, local)


def test_dnsmos_first_scores_only_the_cold_start():
    rng = np.random.default_rng(0)
    y = np.concatenate([0.1 * rng.standard_normal(7 * dr.ss.SR), np.zeros(10 * dr.ss.SR)]).astype(np.float32)
    assert dr.dnsmos_first(y, "mock") is not None
    assert dr.dnsmos_first(np.zeros(20 * dr.ss.SR, np.float32), "mock") is None  # silence
    assert dr.dnsmos_first(y, "none") is None
    # only the first 7 s count: appending a different tail does not change the score
    assert dr.dnsmos_first(np.concatenate([y[: 7 * dr.ss.SR], np.ones(dr.ss.SR, np.float32)]), "mock") == dr.dnsmos_first(y, "mock")
