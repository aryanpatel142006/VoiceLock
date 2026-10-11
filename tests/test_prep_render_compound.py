import json

import numpy as np

from eval.fixtures.mix import Manifest, Stems, write_fixture
from eval.prep_render_compound import build_plan, render_compound, transform_audio


def _record(tmp_path):
    source = tmp_path / "src" / "tiny" / "source"
    source.mkdir(parents=True)
    x = np.sin(np.linspace(0, 30, 1600)).astype(np.float32) * 0.2
    stems = Stems(mix=x, target=x, interferer=x * 0.7, noise=x * 0.1)
    manifest = Manifest(
        id="source",
        set="tiny",
        sample_rate=16000,
        duration_s=0.1,
        snr_db=0.0,
        sir_db=0.0,
        target_id="a",
        interferer_id="b",
        noise_type="pink",
        transcript="target words",
        interferer_transcript="other words",
        paths={"mix": "mix.wav", "target": "target.wav", "interferer": "interferer.wav", "noise": "noise.wav"},
        seed=1,
        source={"dataset": "test", "split": "development", "target_speaker": "a", "interferer_speaker": "b"},
    )
    write_fixture(tmp_path / "src" / "tiny", manifest, stems)
    return tmp_path / "src", manifest.to_dict(), x


def test_transform_does_not_mutate_and_changes_compound():
    x = np.ones(1600, dtype=np.float32)
    before = x.copy()
    y = transform_audio(x, "noise_reverb_pitch_bandlimit", "target")
    assert np.array_equal(x, before)
    assert y.shape == x.shape
    assert not np.array_equal(y, x)


def test_plan_cap_is_balanced_across_roles_and_hard_levels(tmp_path):
    _source_root, record, _ = _record(tmp_path)
    plan = build_plan({"fixtures": [record]}, cap=8)
    assert len(plan) == 8
    assert {p["role_swap"] for p in plan} == {False, True}
    assert {p["hard_target_db"] for p in plan} == {-10.0, -15.0}


def test_render_preserves_additivity_and_provenance(tmp_path):
    source_root, record, _original = _record(tmp_path)
    result = render_compound(
        record,
        source_root / "tiny",
        tmp_path / "out" / "compound_heldout",
        "pause_onset",
        -15.0,
        True,
        7,
    )
    case = tmp_path / "out" / "compound_heldout" / result["id"]
    loaded = json.loads((case / "manifest.json").read_text())
    assert loaded["source"]["role_swap"] is True
    assert loaded["source"]["hard_target_db"] == -15.0
    assert loaded["source"]["transcript_available"] is True
    for name in ("mix", "target", "interferer", "noise"):
        assert (case / f"{name}.wav").exists()
