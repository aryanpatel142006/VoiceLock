import json

import numpy as np
import pytest
import soundfile as sf

from eval.prep_team_audio_audit import audit_paths, write_markdown


def test_audit_paths_reports_signal_health_and_writes_document(tmp_path):
    root = tmp_path / "profiles" / "aryan"
    root.mkdir(parents=True)
    signal = np.concatenate(
        [np.zeros(800, dtype=np.float32), np.full(800, 0.5, dtype=np.float32)]
    )
    sf.write(root / "clip.wav", signal, 16_000)

    records = audit_paths([tmp_path / "profiles"])

    assert len(records) == 1
    record = records[0]
    assert record["speaker"] == "aryan"
    assert record["duration_s"] == 0.1
    assert record["sample_rate"] == 16_000
    assert record["peak_amplitude"] == 0.5
    assert record["rms_energy"] == pytest.approx(np.sqrt(0.125))
    assert record["speech_fraction"] == 0.5
    assert record["snr_db"] > 50

    output = tmp_path / "audit.json"
    output.write_text(json.dumps({"recordings": records}), encoding="utf-8")
    markdown = tmp_path / "audit.md"
    write_markdown(records, markdown)
    assert "clip.wav" in markdown.read_text(encoding="utf-8")
