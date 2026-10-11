"""Latency preset -> enrolled tse spec mapping."""

import pytest

from hearing.fusion.tiers import TierLadder
from hearing.latency_preset import PRESETS, apply_preset, enrolled_window
from hearing.server.session import load_stage_modules

SPEC = 'tse(candidate="enrolled", window_s=1.0, hop_s=0.5, require_gpu=true)>mixer'


def test_windows_grow_with_preset():
    delays = [sum(enrolled_window(p)) for p in PRESETS]
    assert delays == sorted(delays) and len(set(delays)) == 3
    assert enrolled_window("balanced") == (1.0, 0.5)


@pytest.mark.parametrize("preset,expect", [("speed", "window_s=0.5, hop_s=0.25"),
                                           ("balanced", "window_s=1.0, hop_s=0.5"),
                                           ("accuracy", "window_s=1.5, hop_s=0.75")])
def test_apply_preset(preset, expect):
    out = apply_preset(SPEC, preset)
    assert expect in out and out.endswith("require_gpu=true)>mixer")
    if preset == "balanced":
        assert out == SPEC  # never rewritten


def test_non_enrolled_untouched():
    av = 'tse(candidate="av_mossformer2", window_s=2.0, hop_s=1.0)>mixer'
    assert apply_preset(av, "speed") == av
    assert apply_preset("dfn_ll>gate>mixer", "speed") == "dfn_ll>gate>mixer"


@pytest.mark.parametrize("spec", ['tse(candidate=enrolled)>mixer',
                                 'tse(candidate="enrolled", window_s="2", hop_s="1")>mixer'])
def test_preset_handles_shorthand_and_string_window_arguments(spec):
    assert apply_preset(spec, "accuracy") == (
        'tse(candidate="enrolled", window_s=1.5, hop_s=0.75)>mixer')


def test_preset_does_not_rewrite_another_tse_in_the_same_pipeline():
    spec = SPEC + '>tse(candidate="oracle", window_s=2.0, hop_s=1.0)'
    assert apply_preset(spec, "speed").endswith(
        '>tse(candidate="oracle", window_s=2.0, hop_s=1.0)')


def test_preset_replaces_existing_ladder_argument():
    assert apply_preset('tiers(preset=speed, only="gate,audio-only")>mixer', "accuracy") == (
        'tiers(preset="accuracy", only="gate,audio-only")>mixer')
    assert apply_preset('tiers(preset=speed)', "balanced") == 'tiers(preset="balanced")'
    assert apply_preset("tiers", "balanced") == "tiers"


def test_unknown_preset():
    with pytest.raises(ValueError):
        enrolled_window("fast")


def test_ladder_applies_preset():
    load_stage_modules()
    cfg = {"tiers": [{"name": "enrolled", "pipeline": SPEC}, {"name": "audio-only", "pipeline": "passthrough"}]}
    ladder = TierLadder(cfg, preset="speed")
    specs = [s.pipelines[0] for s in ladder.config.tiers]
    assert "window_s=0.5, hop_s=0.25" in specs[0] and specs[1] == "passthrough"
    with pytest.raises(ValueError):
        TierLadder(cfg, preset="nope")


def test_balanced_keeps_a_short_enrolled_config():
    """T-078: the configured spec IS balanced; only speed/accuracy rewrite the windows."""
    load_stage_modules()
    short = 'tse(candidate="enrolled", window_s=0.5, hop_s=0.25)>mixer'
    cfg = {"tiers": [{"name": "enrolled", "pipeline": short},
                     {"name": "audio-only", "pipeline": "passthrough"}]}
    ladder = TierLadder(cfg, preset="balanced")
    assert ladder.config.tiers[0].pipelines[0] == short
    assert ladder.built["enrolled"].delay_ms == pytest.approx(750.0)
    assert "window_s=1.5, hop_s=0.75" in TierLadder(cfg, preset="accuracy").config.tiers[0].pipelines[0]


def test_balanced_honours_the_gpu_box_enrolled_spec():
    """config/tiers.yaml remote.enrolled (served through REMOTE_PIPELINES) runs as written."""
    import yaml

    from hearing.fusion.tiers import DEFAULT_CONFIG

    gpu = yaml.safe_load(DEFAULT_CONFIG.read_text())["remote"]["enrolled"]
    assert "window_s=0.5, hop_s=0.125" in gpu  # T-086 A100 pick
    assert apply_preset(gpu, "balanced") == gpu
    assert "window_s=1.5, hop_s=0.75" in apply_preset(gpu, "accuracy")


@pytest.mark.slow
def test_bakeoff_writes_partial_json_per_window(tmp_path):
    import json

    from eval import bakeoff
    from eval.bakeoff_fixtures import synthetic_av_set

    fixtures = synthetic_av_set(n=1, out=tmp_path / "fixtures")
    bakeoff.run(str(fixtures), ["identity"], [0.5, 1.0], limit=1, partial_dir=tmp_path)
    for w in ("0.5", "1"):
        data = json.loads((tmp_path / f"bakeoff_{w}.json").read_text())
        assert data["rows"][0]["candidate"] == "identity"
