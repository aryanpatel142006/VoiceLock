"""T-328: under the team preset's audio_only the judge flow's lock is the saved voice, not a face."""
from __future__ import annotations

import asyncio
import json

import websockets

import hearing.replay
from scripts import judge_check as jc

TEAM = {"name": "team", "pins": {"audio_only": True}}


def tick(tier: str, acq: str, detail: str = "", source: str | None = None, preset: dict | None = TEAM) -> dict:
    t = {"type": "telemetry", "profile": "judge", "tier": tier, "target": {"track_id": None},
         "acquisition": {"state": acq, "detail": detail},
         "enrollment": {"state": "ready" if source else "collecting", "source": source,
                        "profile": "aryan" if source else None}}
    if preset is not None:
        t["preset"] = preset
    return t


class FakeWS:
    def __init__(self, ticks: list[dict]):
        self.ticks = ticks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, message):
        pass

    async def __aiter__(self):
        for t in self.ticks:
            yield json.dumps(t)
            await asyncio.sleep(0.02)


def run_flow(monkeypatch, ticks: list[dict]) -> dict:
    async def feed(clip, send, loop=True, control=None):
        await asyncio.sleep(0.05 + 0.02 * len(ticks))

    monkeypatch.setattr(websockets, "connect", lambda *a, **k: FakeWS(ticks))
    monkeypatch.setattr(hearing.replay, "feed", feed)
    return asyncio.run(jc.judge_flow("ws://fake", object(), 5.0))


def test_is_profile_lock():
    assert jc.is_profile_lock(tick("audio-only", "searching", source="profile"))
    assert jc.is_profile_lock(tick("enrolled", "locked", detail="known voice: aryan"))
    assert not jc.is_profile_lock(tick("enrolled", "locked", detail="calibrated", source="calibrated"))
    assert not jc.is_profile_lock(tick("audio-only", "searching"))


def test_audio_only_locks_on_the_saved_voice(monkeypatch):
    ticks = [tick("audio-only", "searching")] * 3 + [tick("audio-only", "acquiring", source="profile")] * 5 \
        + [tick("enrolled", "locked", detail="known voice: aryan", source="profile")] * 3
    r = run_flow(monkeypatch, ticks)
    assert r["lock_kind"] == "profile" and r["profile"] == "aryan"
    assert r["marks"]["locked"] < r["marks"]["isolated"]
    assert r["lock_to_isolated_s"] is not None and 0 < r["lock_to_isolated_s"] < 1


def test_audio_only_isolated_is_the_enrolled_tier_while_acquisition_searches(monkeypatch):
    # what the session sends under audio_only today: no face, so acquisition never leaves "searching"
    look = "looking for a speaking face"
    ticks = [tick("audio-only", "searching", look, source="profile")] * 4 \
        + [tick("enrolled", "searching", look, source="profile")] * 3
    r = run_flow(monkeypatch, ticks)
    assert r["marks"]["locked"] < r["marks"]["isolated"] and 0 < r["lock_to_isolated_s"] < 1


def test_audio_only_without_profile_never_locks(monkeypatch):
    r = run_flow(monkeypatch, [tick("audio-only", "searching")] * 2 + [tick("enrolled", "searching")] * 2)
    assert r["lock_kind"] == "profile" and "locked" not in r["marks"] and r["lock_to_isolated_s"] is None


def test_without_audio_only_a_face_is_still_the_lock(monkeypatch):
    plain = [tick("audio-only", "searching", source="profile", preset=None)] * 2
    face = tick("enrolled", "locked", source="profile", preset=None)
    face["target"] = {"track_id": 3, "locked": True}
    r = run_flow(monkeypatch, plain + [face] * 2)
    assert r["lock_kind"] == "face" and r["lock_to_isolated_s"] == 0.0
