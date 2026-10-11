"""T-149: eval/quiet_toggle.py caption attribution and scoring (no server)."""

import numpy as np

from eval.quiet_toggle import attribute, score

TARGET = [(0.0, 3.0), (10.0, 14.0)]
OTHER = [(4.0, 8.0), (12.0, 16.0)]


def test_attribute_by_speaking_spans_and_loop():
    assert attribute(0.5, 2.5, TARGET, OTHER, 20.0) == "target"
    assert attribute(4.5, 7.5, TARGET, OTHER, 20.0) == "other"
    assert attribute(12.0, 14.0, TARGET, OTHER, 20.0) == "target"  # overlap: the target spoke throughout
    assert attribute(14.5, 15.5, TARGET, OTHER, 20.0) == "other"  # the other talker alone after the overlap
    assert attribute(24.5, 27.5, TARGET, OTHER, 20.0) == "other"  # second loop of the clip
    assert attribute(8.5, 9.5, TARGET, OTHER, 20.0) == "none"


def test_score_counts_one_toggle():
    manifest = {"duration_s": 20.0, "faces": [{"role": "target", "speaking": TARGET},
                                               {"role": "other", "speaking": OTHER}]}
    tele = [(t, {"effective_mode": "quiet" if 1.0 <= t < 4.0 else "focus_hq", "target": {"track_id": 3},
                 "acquisition": {"state": "searching"}, "quiet": {"feed": "focus"}, "tiers": {"delay_ms": 0, "active": "enrolled"},
                 "enrollment": {"profile": "aryan", "state": "calibrated"}})
            for t in np.arange(0.0, 6.0, 0.1)]
    log = {"presses": [("quiet", 1.0), ("focus_hq", 4.0)], "telemetry": tele, "epoch": 100.0,
           "audio": [(t, np.zeros(320, np.float32)) for t in np.arange(1.0, 4.0, 0.02)],
           "captions": [(2.5, {"text": "one two three", "final": True, "t0": 100.5, "t1": 102.5})], "errors": []}
    r = score(log, manifest)
    assert (r["target_unchanged"], r["quiet_silent"], r["back_isolated"]) == (1, 1, 1)
    assert r["caption_words"] == 3 and r["other_words"] == 0 and r["first_final_p50_s"] == 1.5
