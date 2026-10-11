"""T-419: eval.t419_rows reads the delay from the run (session_score's measured lag), never from the windows."""

from __future__ import annotations

import json

from eval.t419_rows import delay_columns, summary


def test_delay_columns_weight_the_enrolled_runs(tmp_path):
    run = tmp_path / "demo_regress_x.json"
    run.write_text(json.dumps({"scores": [{"alignment": "reported", "delays": [
        {"start_s": 0.0, "end_s": 4.0, "tier": "audio-only", "measured_ms": 13.0, "corr": 0.7},
        {"start_s": 4.0, "end_s": 20.0, "tier": "enrolled", "measured_ms": 750.0, "corr": 0.8},
        {"start_s": 20.0, "end_s": 28.0, "tier": "enrolled", "measured_ms": 870.0, "corr": 0.6}]}]}))
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(json.dumps({"dir": "out", "event": {"type": "telemetry", "tier": t, "latency_ms": {"total": v}}})
                                for t, v in (("audio-only", 30.0), ("enrolled", 770.0), ("enrolled", 780.0), ("enrolled", 900.0))))
    cols = delay_columns(run, events)
    assert cols["delay_ms"] == 790.0 and cols["delay_max_ms"] == 870.0 and cols["corr"] == 0.6
    assert cols["total_ms"] == 780.0


def test_summary_means_per_arm():
    rows = [{"scene": "t141-easy", "label": "ft+0.5/0.25", "od_pct": p, "leak": -10.0, "target_down_10dB_s": 1.0,
             "p95": 150.0, "misses": "0 / 116", "sdri_pre": 5.0, "delay_ms": 750.0, "delay_max_ms": 750.0, "total_ms": 770.0}
            for p in (2.0, 4.0)]
    line = summary(rows).splitlines()[2]
    assert "| t141 | ft+0.5/0.25 | 2 | 750.0, 750.0 | 770.0 | 3.0 % | -10.0 | 2.0 | 150 | 0 / 232 | +5.0 |" == line
