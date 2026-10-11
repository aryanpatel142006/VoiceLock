"""Fast tests must never launch the server's real model preload worker."""
import os
import threading

import pytest


@pytest.fixture(autouse=True)
def no_model_warmup_in_fast_tests(request, monkeypatch):
    if request.node.get_closest_marker('slow') or request.node.get_closest_marker('network'):
        return
    from hearing.server import warm

    def forbidden():
        raise AssertionError('fast tests must not launch model warm-up; use --no-warm')

    monkeypatch.setattr(warm, 'start', forbidden)
    assert not any(thread.name == 'warm-models' for thread in threading.enumerate())


def scaled(seconds: float) -> float:
    """Timing tests scale timeouts/budgets by HEARING_TEST_SLOWDOWN (default 1; the harness sets it on busy machines)."""
    try:
        return seconds * max(1.0, float(os.environ.get("HEARING_TEST_SLOWDOWN", "1")))
    except ValueError:
        return seconds
