from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from eval.prep_output_audit import AuditConfig, audit_case, run_manifest

SR = 16_000


def write_wav(path: Path, samples: np.ndarray) -> None:
    data = np.asarray(np.clip(samples, -1, 1), dtype=np.float32)
    pcm = (data * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm)


def tone(n: int = SR) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    return 0.2 * np.sin(2 * np.pi * 220 * t)


def test_delayed_output_reports_lag_and_matched_level(tmp_path: Path) -> None:
    reference = tone()
    delayed = np.concatenate([np.zeros(80, dtype=np.float32), reference])
    ref_path, out_path = tmp_path / "ref.wav", tmp_path / "out.wav"
    write_wav(ref_path, reference)
    write_wav(out_path, delayed)

    result = audit_case(
        {"id": "delay", "reference": str(ref_path), "output": str(out_path)},
        AuditConfig(max_lag_ms=30),
    )

    assert result["alignment"]["lag_ms"] == 5.0
    assert result["alignment"]["overlap_s"] > 0.99
    assert result["matched_level"]["scale_db"] == 0.0


def test_silence_is_not_a_dropout_and_clipping_is_reported(tmp_path: Path) -> None:
    reference = np.concatenate([tone(8000), np.zeros(8000, dtype=np.float32)])
    output = reference.copy()
    output[100] = 1.0
    mask = np.concatenate([np.ones(8000, dtype=np.uint8), np.zeros(8000, dtype=np.uint8)])
    paths = [tmp_path / name for name in ("ref.wav", "out.wav", "mask.wav")]
    write_wav(paths[0], reference)
    write_wav(paths[1], output)
    write_wav(paths[2], mask)

    result = audit_case(
        {"id": "edges", "reference": str(paths[0]), "output": str(paths[1]), "target_active": str(paths[2])},
        AuditConfig(),
    )

    assert result["dropouts"]["active_frames"] > 0
    assert result["dropouts"]["dropout_frames"] == 0
    assert result["clipping"]["clipped_samples"] > 0
    assert result["alignment"]["lag_ms"] == 0.0


def test_transition_discontinuity_is_distinguished_from_waveform_changes(tmp_path: Path) -> None:
    reference = tone()
    output = reference.copy()
    output[SR // 2 :] += 0.7
    write_wav(tmp_path / "ref.wav", reference)
    write_wav(tmp_path / "out.wav", output)

    result = audit_case(
        {"id": "click", "reference": str(tmp_path / "ref.wav"), "output": str(tmp_path / "out.wav")},
        AuditConfig(click_z=8.0),
    )

    assert result["clicks"]["candidate_boundaries"] >= 1
    assert result["clicks"]["ordinary_change_p95"] > 0
    assert result["clicks"]["suspect_boundaries"] >= 1


def test_manifest_resume_and_hashes(tmp_path: Path) -> None:
    write_wav(tmp_path / "ref.wav", tone(400))
    write_wav(tmp_path / "out.wav", tone(400))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source": {"speaker": "synthetic", "utterance": "offline tone", "interval": [0, 0.025]},
                "cases": [{"id": "one", "reference": "ref.wav", "output": "out.wav"}],
            }
        )
    )
    first = run_manifest(manifest, tmp_path / "results")
    second = run_manifest(manifest, tmp_path / "results", resume=True)
    assert first["available_cases"] == 1
    assert second["resumed_cases"] == 1
    assert second["config_sha256"]
    assert second["cases"][0]["provenance"]["speaker"] == "synthetic"
