import json

import numpy as np

from eval import prep_headroom as prep


def test_smoothing_is_bounded_and_silence_has_no_gain():
    desired = np.full(8, 20.0, dtype=np.float32)
    active = np.array([True, True, False, False, True, True, True, True])
    gain = prep.smooth_gain_db(desired, active, attack_frames=2, release_frames=2)
    assert np.max(gain) <= prep.MAX_GAIN_DB
    assert gain[2] == 0 and gain[3] == 0


def test_forecast_distinguishes_float_overrange_and_pcm_clipping():
    x = np.array([0.5, 1.2, -1.3], dtype=np.float32)
    result = prep.gain_forecast(x)
    assert result["clipping_forecast"]["float_overrange_samples"] == result["clipping_forecast"]["pcm_clip_samples"]


def test_cli_analyses_clean_stems_and_resumes(tmp_path, capsys):
    fixture = tmp_path / "tiny" / "case"
    fixture.mkdir(parents=True)
    x = np.zeros(320, dtype=np.float32)
    x[32:160] = 0.1
    prep.aio.save(fixture / "target.wav", x, prep.SR)
    prep.aio.save(fixture / "interferer.wav", x * 0.5, prep.SR)
    prep.aio.save(fixture / "noise.wav", x * 0.1, prep.SR)
    (fixture / "manifest.json").write_text(json.dumps({
        "id": "case", "set": "tiny", "target_id": "t", "interferer_id": "i",
        "transcript": "consented utterance", "paths": {"target": "target.wav", "interferer": "interferer.wav", "noise": "noise.wav"},
    }))
    out = tmp_path / "out"
    assert prep.main(["--fixtures", str(fixture.parent), "--out", str(out), "--render"]) == 0
    first = capsys.readouterr().out
    assert '"completed_cases": 1' in first
    assert (out / "waveforms/case_baseline_sum.wav").exists()
    assert prep.main(["--fixtures", str(fixture.parent), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["available_cases"] == 1
