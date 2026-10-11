"""T-070 ladder harness: config rewrite and policy replay (no weights, no scenes)."""

import numpy as np

from eval import association_ladder as al
from hearing.fusion.tiers import LadderConfig


def test_ladder_config_runs_tse_inline_and_strict_adds_the_pause_preset():
    cfg = LadderConfig.from_dict(al.ladder_config())
    enrolled = next(t for t in cfg.tiers if t.name == "enrolled")
    assert enrolled.pipelines[0].startswith('tse(threaded=false, candidate="enrolled"')
    assert enrolled.avoid == {"same_bucket"}
    strict = next(t for t in LadderConfig.from_dict(al.ladder_config(strict=True)).tiers if t.name == "enrolled")
    assert 'pause_action="duck"' in strict.pipelines[0] and "pause_p=0.5" in strict.pipelines[0]
    av = next(t for t in LadderConfig.from_dict(al.ladder_config(strict=True)).tiers if t.name == "focus_hq")
    assert "pause" not in av.pipelines[0] and "threaded=false" in av.pipelines[0]


def test_replay_applies_the_upgrade_hold_and_steady_skips_fades():
    order = ("enrolled", "gate")
    ok = np.array([[False, True]] * 50 + [[True, True]] * 150)  # enrolled eligible from 0.5 s
    tiers = al.replay(ok, order, upgrade_after_s=1.0)
    assert tiers[:150] == ["gate"] * 150 and tiers[150:] == ["enrolled"] * 50  # up after 1.0 s
    assert al._steady(tiers, 0, 4, settle=15) == "gate"  # the start is not a switch
    assert al._steady(tiers, 152, 156, settle=15) is None  # inside the crossfade after the switch
    assert al._steady(tiers, 170, 174, settle=15) == "enrolled"


def test_all_devices_drops_the_cuda_only_restriction():
    av = next(t for t in LadderConfig.from_dict(al.ladder_config()).tiers if t.name == "focus_hq")
    assert av.devices == {"cuda"}  # T-072: the host's own ladder
    av = next(t for t in LadderConfig.from_dict(al.ladder_config(all_devices=True)).tiers if t.name == "focus_hq")
    assert not av.devices
