from tools.timeline import gate_state, timeline


def test_gate_state_rules():
    assert gate_state(None) is None
    assert gate_state({"confidence": 0.9}) is None
    assert gate_state({"gated": False, "confidence": 0.9}) == "SOLO"
    assert gate_state({"gated": False, "confidence": 0.1}) == "UNSURE"
    assert gate_state({"gated": True, "soft_gated": False, "confidence": 0}) == "UNSURE"
    assert gate_state({"gated": True, "soft_gated": True, "confidence": 0}) == "OVERLAP-DUCKED"


def test_timeline_runs():
    def row(t, tse):
        return {"t": t, "dir": "out", "event": {"type": "telemetry", "tse": tse}}

    solo, duck = {"gated": False, "confidence": 0.9}, {"gated": True, "soft_gated": True, "confidence": 0}
    runs = timeline([row(0, solo), row(1, solo), row(2, duck), row(3, None), row(4, solo)])
    assert runs == [(0.0, 1.0, "SOLO"), (2.0, 2.0, "OVERLAP-DUCKED"), (4.0, 4.0, "SOLO")]
