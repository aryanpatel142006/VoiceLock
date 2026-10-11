"""T-129: demo_regress --voice-profile sends the T-125 profile enroll control with the mode."""

from scripts.demo_regress import voice_profile_enroll


def test_voice_profile_enroll_name_only():
    assert voice_profile_enroll("aryan") == {"action": "profile", "name": "aryan"}


def test_voice_profile_enroll_with_negative():
    assert voice_profile_enroll("aryan:joseph") == {"action": "profile", "name": "aryan", "other": "joseph"}


def test_isolation_flag_parses():
    import pytest

    from scripts import demo_regress

    with pytest.raises(SystemExit):
        demo_regress.run(["--isolation", "loose"])
