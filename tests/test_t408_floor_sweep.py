from eval.t408_floor_sweep import pick, summarize


def _row(arm, target, scene, od, muted, leak=-15.0):
    return {"label": arm, "target": target, "other": "x", "scene": scene, "od_pct": od,
            "target_down_10dB_s": muted, "leak": leak, "sdri_pre": 5.0}


def test_summarize_groups_and_joseph_muting():
    rows = [_row("t387+floor20", "joseph", "t141-hard", 2.0, 8.0), _row("t387+floor20", "aryan", "holdout-sir0-cafe", 1.0, 1.0)]
    s = summarize(rows)["t387+floor20"]
    assert s["all"] == {"n": 2, "wrong_pct": 1.5, "leak_db": -15.0, "muted_s": 9.0, "muted_joseph_s": 8.0,
                        "muted15_s": None, "muted20_s": None, "sdri_pre": 5.0}
    assert s["t141"]["n"] == 1 and s["holdout"]["muted_joseph_s"] is None


def test_pick_least_muting_within_wrong_speaker_tolerance():
    rows = []
    for arm, od, muted in (("t387+floor20", 2.0, 10.0), ("t387+floor15", 2.4, 6.0), ("t387+floor12", 3.0, 3.0),
                           ("t387+floor25", 1.5, 12.0), ("t374+floor20", 1.0, 1.0)):
        rows.append(_row(arm, "joseph", "t141-easy", od, muted))
    s = summarize(rows)
    assert pick(s) == "t387+floor15"  # floor 12 is +1.0 pt; t374 is never a candidate
    assert pick(s, tol=1.0) == "t387+floor12"
    assert pick({}) is None


def test_pick_tie_on_10db_goes_to_shallower_muting():
    a, b = _row("t387+floor20", "joseph", "t141-hard", 2.0, 5.0), _row("t387+floor25", "joseph", "t141-hard", 1.9, 5.0)
    a |= {"target_down_15dB_s": 5.0, "target_down_20dB_s": 2.0}
    b |= {"target_down_15dB_s": 5.0, "target_down_20dB_s": 4.5}
    assert pick(summarize([a, b])) == "t387+floor20"
