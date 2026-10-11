"""T-233: demo_regress flags rows whose enrolled tier (profile path) never played."""
from scripts import demo_regress as dr


def _log(tiers):
    return [{"dir": "out", "event": {"type": "telemetry", "tier": t}} for t in tiers]


def test_enrolled_never_played_is_not_measured():
    share = dr.enrolled_share(_log(["audio-only"] * 10))
    assert share == 0.0 and dr.not_measured(True, share)
    assert dr.not_measured(True, None)


def test_enrolled_playing_is_measured():
    share = dr.enrolled_share(_log(["enrolled"] * 8 + ["gate"] * 2))
    assert share == 0.8 and not dr.not_measured(True, share)


def test_no_enrolled_request_never_flagged():
    assert not dr.not_measured(False, 0.0)


def test_exit_2_and_loud_line_when_tier_never_played(capsys):
    share = dr.enrolled_share(_log(["audio-only"] * 10))
    assert dr.not_measured(True, share)
    assert dr.unmeasured_exit(["profile row"], allow=False)
    assert "NOT MEASURED" in capsys.readouterr().err


def test_exit_0_when_tier_played_or_allowed(capsys):
    assert not dr.unmeasured_exit([], allow=False)
    assert not dr.unmeasured_exit(["profile row"], allow=True)
    assert "NOT MEASURED" not in capsys.readouterr().err
