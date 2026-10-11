"""Unit tests for overnight voice-vector robustness evaluation (T-212).

All tests run fully offline without network access or model weight downloads.
Verifies ECAPA unit embeddings, negative profile rejection, LDA route activation,
crop windowing, stream selection, and dev/heldout threshold calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.overnight_voice_vectors import (
    CROP_DURATIONS,
    SR,
    MockEmbedder,
    ProbeCase,
    _speech_rms,
    _unit_vector,
    apply_transformation,
    build_probe_suite,
    calibrate_dev_thresholds,
    evaluate_heldout_with_thresholds,
    evaluate_probe_case,
    extract_voiced_crops,
    generate_synthetic_fixtures,
    is_voiced,
    mix_contamination,
    run_overnight_voice_vectors,
    verify_saved_profile_lda,
)
from hearing.tse.blind import BlindSeparator
from hearing.tse.blind import MockBackend as MockSepBackend
from hearing.tse.enroll import profile_clip, split_pieces
from hearing.tse.enrolled import fit_lda


def test_unit_vector_normalization() -> None:
    v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    u = _unit_vector(v)
    assert u.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-5)

    zero = np.zeros(5, dtype=np.float32)
    u_zero = _unit_vector(zero)
    assert np.all(u_zero == 0.0)

    nan_v = np.array([np.nan, 1.0], dtype=np.float32)
    u_nan = _unit_vector(nan_v)
    assert np.all(u_nan == 0.0)


def test_is_voiced_and_crop_extraction() -> None:
    sr = SR
    t = np.arange(sr * 2, dtype=np.float32) / sr
    loud_tone = 0.2 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)

    assert is_voiced(loud_tone, min_rms=0.01)
    assert not is_voiced(silence, min_rms=0.01)

    combined = np.concatenate([loud_tone, silence])
    crops = extract_voiced_crops(combined, sr=sr, crop_s=0.5, step_s=0.5, min_rms=0.01)
    # The loud_tone is 2.0s -> four 0.5s crops; silence is rejected
    assert len(crops) == 4
    for c in crops:
        assert len(c) == int(0.5 * sr)
        assert _speech_rms(c) >= 0.01


def test_apply_transformation_types() -> None:
    sr = SR
    t = np.arange(sr, dtype=np.float32) / sr
    x = 0.3 * np.sin(2 * np.pi * 400 * t).astype(np.float32)

    # Quiet transformation
    quiet = apply_transformation(x, "quiet")
    rms_orig = _speech_rms(x)
    rms_quiet = _speech_rms(quiet)
    ratio_db = 20.0 * np.log10(rms_quiet / rms_orig)
    np.testing.assert_allclose(ratio_db, -12.0, atol=0.5)

    # Noise transformation
    noisy = apply_transformation(x, "noise_15db", seed=42)
    assert len(noisy) == len(x)
    assert not np.array_equal(noisy, x)

    # Telephony transformation
    tel = apply_transformation(x, "telephony")
    assert len(tel) == len(x)
    assert np.max(np.abs(tel)) <= 1.0


def test_mix_contamination_sir() -> None:
    sr = SR
    t = np.arange(sr, dtype=np.float32) / sr
    target = 0.2 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    interferer = 0.2 * np.sin(2 * np.pi * 500 * t).astype(np.float32)

    mix_0db = mix_contamination(target, interferer, sir_db=0.0)
    assert len(mix_0db) == len(target)
    assert not np.array_equal(mix_0db, target)

    mix_neg10db = mix_contamination(target, interferer, sir_db=-10.0)
    # At -10 dB SIR, interferer is significantly amplified
    assert _speech_rms(mix_neg10db) > _speech_rms(mix_0db)


def test_mock_embedder_and_separator() -> None:
    embedder = MockEmbedder(192)
    audio = np.random.randn(8000).astype(np.float32) * 0.1
    emb = embedder.embed(audio)
    assert emb.shape == (192,)
    np.testing.assert_allclose(np.linalg.norm(emb), 1.0, atol=1e-5)

    separator = BlindSeparator(backend=MockSepBackend())
    streams = separator.separate(audio, SR)
    assert len(streams) == 2
    assert len(streams[0]) == len(audio)


def test_fit_lda_separation() -> None:
    rng = np.random.default_rng(42)
    # 2 classes with clear separation along first dimension
    class_a = rng.standard_normal((10, 192)).astype(np.float32)
    class_a[:, 0] += 5.0
    class_b = rng.standard_normal((10, 192)).astype(np.float32)
    class_b[:, 0] -= 5.0

    res = fit_lda(class_a, class_b)
    assert res is not None
    w, b = res
    assert w.shape == (192,)
    # Class A should score near +1 and Class B should score near -1
    score_a = float(np.mean([w @ x - b for x in class_a]))
    score_b = float(np.mean([w @ x - b for x in class_b]))
    assert score_a > score_b
    np.testing.assert_allclose(score_a, 1.0, atol=0.2)
    np.testing.assert_allclose(score_b, -1.0, atol=0.2)


def test_saved_profile_lda_verification_conditions() -> None:
    sr = SR
    fixtures = generate_synthetic_fixtures(seed=101)
    a = fixtures["aryan_profile"]
    j = fixtures["joseph_profile"]

    pieces_a = split_pieces(a, sr, 2.5)
    pieces_j = split_pieces(j, sr, 2.5)
    clip_a = profile_clip(pieces_a)
    clip_j = profile_clip(pieces_j)

    class OrthogonalMockEmb:
        device = "cpu"

        def embed(self, audio: np.ndarray) -> np.ndarray:
            x = np.asarray(audio, dtype=np.float32)
            fft = np.abs(np.fft.rfft(x, n=512))
            emb = np.zeros(192, dtype=np.float32)
            emb[: min(len(fft), 192)] = fft[: min(len(fft), 192)]
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 1e-12 else emb

    embedder = OrthogonalMockEmb()
    separator = BlindSeparator(backend=MockSepBackend())

    verif = verify_saved_profile_lda(clip_a, clip_j, embedder, separator, sr)
    assert "saved_profile_scoring" in verif
    assert "activation_rules_verified" in verif
    assert verif["num_target_pieces"] >= 2
    assert verif["num_other_pieces"] >= 2
    assert verif["activation_rules_verified"]["requires_closed_set_true"]
    assert verif["activation_rules_verified"]["requires_two_or_more_pieces"]


def test_evaluate_probe_case_agreement_and_wrong_duration() -> None:
    fixtures = generate_synthetic_fixtures(seed=202)
    a = fixtures["aryan_profile"]
    j = fixtures["joseph_profile"]

    pieces_a = split_pieces(a, SR, 2.5)
    pieces_j = split_pieces(j, SR, 2.5)
    clip_a = profile_clip(pieces_a)
    clip_j = profile_clip(pieces_j)

    embedder = MockEmbedder(192)
    separator = BlindSeparator(backend=MockSepBackend())

    t_emb = _unit_vector(embedder.embed(clip_a))
    o_emb = _unit_vector(embedder.embed(clip_j))

    crop = fixtures["aryan_fixture"][: SR * 2]
    other_crop = fixtures["joseph_fixture"][: SR * 2]

    probe = ProbeCase(
        probe_id="test_probe_001",
        pair="team",
        target_speaker="aryan",
        other_speaker="joseph",
        condition="separated",
        crop_duration_s=2.0,
        source_group="dev",
        audio=mix_contamination(crop, other_crop, sir_db=0.0),
        target_clean=crop,
        other_clean=other_crop,
    )

    result = evaluate_probe_case(probe, clip_a, clip_j, t_emb, o_emb, embedder, separator)
    assert result.probe_id == "test_probe_001"
    assert isinstance(result.same_person_cos, float)
    assert isinstance(result.cosine_margin, float)
    assert result.selected_stream in (0, 1)
    assert result.oracle_stream in (0, 1)
    assert result.oracle_selected_agreement in (True, False)
    assert result.wrong_speaker_duration_s in (0.0, 2.0)


def test_dev_threshold_calibration_and_heldout_eval() -> None:
    fixtures = generate_synthetic_fixtures(seed=303)
    a = fixtures["aryan_profile"]
    j = fixtures["joseph_profile"]
    pieces_a = split_pieces(a, SR, 2.5)
    pieces_j = split_pieces(j, SR, 2.5)
    clip_a = profile_clip(pieces_a)
    clip_j = profile_clip(pieces_j)

    embedder = MockEmbedder(192)
    separator = BlindSeparator(backend=MockSepBackend())
    t_emb = _unit_vector(embedder.embed(clip_a))
    o_emb = _unit_vector(embedder.embed(clip_j))

    dev_results = []
    heldout_results = []

    for idx, split_name in enumerate(["dev", "heldout"]):
        for c_idx in range(6):
            crop = fixtures["aryan_fixture"][:SR]
            probe = ProbeCase(
                probe_id=f"probe_{split_name}_{c_idx}",
                pair="team",
                target_speaker="aryan",
                other_speaker="joseph",
                condition="clean",
                crop_duration_s=1.0,
                source_group=split_name,
                audio=crop,
                target_clean=crop,
                other_clean=fixtures["joseph_fixture"][:SR],
            )
            res = evaluate_probe_case(probe, clip_a, clip_j, t_emb, o_emb, embedder, separator)
            if split_name == "dev":
                dev_results.append(res)
            else:
                heldout_results.append(res)

    thresholds = calibrate_dev_thresholds(dev_results, target_far=0.05)
    assert "calibrated_margin_threshold" in thresholds
    assert thresholds["n_dev_probes"] == 6

    eval_heldout = evaluate_heldout_with_thresholds(heldout_results, thresholds)
    assert eval_heldout["n_heldout_probes"] == 6
    assert 0.0 <= eval_heldout["tpr_clean"] <= 1.0
    assert 0.0 <= eval_heldout["far_wrong_speaker"] <= 1.0


def test_probe_cap_enforcement() -> None:
    audio_data = generate_synthetic_fixtures(seed=404)
    probes_20 = build_probe_suite(audio_data, cap=20, seed=404)
    assert len(probes_20) == 20

    probes_default = build_probe_suite(audio_data, cap=150, seed=404)
    assert len(probes_default) <= 150


def test_run_overnight_voice_vectors_dry_run_contract(tmp_path: Path) -> None:
    output_path = tmp_path / "results" / "test_ovv.json"
    fixtures_dir = tmp_path / "fixtures"

    report = run_overnight_voice_vectors(
        fixtures_dir=fixtures_dir,
        profiles_dir=None,
        output_path=output_path,
        cap=12,
        dry_run=True,
        device="cpu",
        seed=505,
    )

    assert output_path.is_file()
    assert report["task_id"] == "T-212"
    assert report["status"] == "complete"
    assert report["counts"]["total_probes"] == 12
    assert report["counts"]["cap"] == 12
    assert "lda_route_verification" in report
    assert "crop_duration_breakdown" in report
    for c in CROP_DURATIONS:
        assert f"{c:g}s" in report["crop_duration_breakdown"]
    assert "condition_breakdown" in report
    assert "heldout_evaluation" in report

    # Verify JSON file round-trip
    saved_data = json.loads(output_path.read_text())
    assert saved_data["config_hash"] == report["config_hash"]
    assert len(saved_data["probes"]) == 12
