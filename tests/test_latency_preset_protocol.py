"""Latency preset in the wire protocol."""

import pytest

from hearing.server.protocol import Control, parse_message


def test_control_accepts_latency_preset():
    """Control message can include latency_preset field."""
    msg = Control(type="control", latency_preset="speed")
    assert msg.latency_preset == "speed"


def test_latency_preset_values():
    """Only valid preset values are accepted."""
    # Valid values
    for preset in ["speed", "balanced", "accuracy"]:
        msg = Control(type="control", latency_preset=preset)
        assert msg.latency_preset == preset

    # Invalid value
    with pytest.raises(ValueError):
        Control(type="control", latency_preset="invalid")


def test_parse_control_with_latency_preset():
    """Parse a control message with latency_preset from JSON."""
    data = {
        "type": "control",
        "mode": "focus_hq",
        "latency_preset": "balanced"
    }
    msg = parse_message(data)
    assert isinstance(msg, Control)
    assert msg.latency_preset == "balanced"


def test_latency_preset_is_optional():
    """Latency preset is optional (for backward compatibility)."""
    data = {"type": "control", "mode": "natural"}
    msg = parse_message(data)
    assert msg.latency_preset is None


def test_control_partial_update_with_preset():
    """Only latency_preset can be sent in a control message."""
    data = {"type": "control", "latency_preset": "accuracy"}
    msg = parse_message(data)
    assert msg.latency_preset == "accuracy"
    assert msg.mode is None
