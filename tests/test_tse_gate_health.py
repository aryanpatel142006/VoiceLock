"""T-382: the enrolled pick's gate state reaches health() (and so telemetry.tse) for the UI's gate pill."""
from hearing.tse.stage import FocusHQ


def test_health_reports_gated_and_soft_gated():
    st = FocusHQ(candidate="enrolled")
    h = st.health()
    assert "gated" in h and "soft_gated" in h
    st.cand.gated, st.cand.soft_gated = True, True
    st._health_key = None  # health() is cached until a window resolves
    h = st.health()
    assert h["gated"] is True and h["soft_gated"] is True
