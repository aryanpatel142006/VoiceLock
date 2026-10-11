"""Playwright mobile emulation (iPhone 14, Pixel 7) of /simple over https against ``hearing serve --tls``."""

from __future__ import annotations

import os

import pytest

from scripts.phone_check import main

# Load per core above which the check is skipped, not failed: it once took 455 s and failed under
# load, then passed on rerun (Q-245). Override with HEARING_TEST_MAX_LOAD.
MAX_LOAD_PER_CORE = float(os.environ.get("HEARING_TEST_MAX_LOAD") or 1.5)


def load_per_core() -> float:
    return os.getloadavg()[0] / (os.cpu_count() or 1)


@pytest.mark.slow  # starts a server and Chromium (~7 s); no weights, no network
def test_simple_over_https_on_emulated_phones(monkeypatch):
    pytest.importorskip("playwright.sync_api")
    load = load_per_core()
    if load > MAX_LOAD_PER_CORE:
        pytest.skip(f"machine too loaded for a browser timing check: load {load:.2f}/core "
                    f"> {MAX_LOAD_PER_CORE} (HEARING_TEST_MAX_LOAD)")
    # Below the skip threshold, scale Playwright's waits with the load (1x idle, up to ~2.5x).
    if not os.environ.get("HEARING_TEST_SLOWDOWN"):
        monkeypatch.setattr("scripts.phone_check.SLOWDOWN", 1.0 + max(0.0, load))
    assert main([]) == 0
