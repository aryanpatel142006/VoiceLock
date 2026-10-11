"""Overnight voice-vector robustness evaluation (T-212).

Evaluates 192-D ECAPA unit speaker embeddings, negative profile rejection,
and separated-speech crops across varying crop durations (0.5s, 1.0s, 2.0s, 4.0s)
and acoustic conditions (clean, transformed, contaminated, separated).

Measures:
* Same-person vs different-person cosine similarity and margin.
* Negative rejection under SAME_VOICE (0.70) cutoff.
* Verification of saved-profile / LDA route activation.
* Oracle-vs-selected stream agreement and wrong-speaker duration on separated crops.
* Strict dev-only threshold calibration and evaluation on untouched heldout source groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hearing.audio import io as aio
from hearing.tse.blind import BlindSeparator
from hearing.tse.blind import MockBackend as MockSepBackend
from hearing.tse.enroll import profile_clip, profile_pieces, split_pieces
from hearing.tse.enrolled import (
    CONF_MARGIN,
    CONF_MARGIN_LDA,
    SAME_VOICE,
    EnrolledCandidate,
    fit_lda,
)

SR = 16000
CROP_DURATIONS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
DEFAULT_CAP = 144
VOICED_RMS_FLOOR = 0.01
VOICED_MIN_DB = -40.0
TARGET_FAR = 0.05


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return "missing"
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_str(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(v))
    if not v.size or not np.isfinite(v).all() or norm <= 1e-12:
        return np.zeros_like(v)
    return v / norm


def _speech_rms(audio: np.ndarray) -> float:
    if not len(audio):
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))


def _speech_dbfs(audio: np.ndarray) -> float:
    rms = _speech_rms(audio)
    if rms <= 1e-9:
        return -120.0
    return float(20.0 * np.log10(rms))


def is_voiced(audio: np.ndarray, min_rms: float = VOICED_RMS_FLOOR, min_db: float = VOICED_MIN_DB) -> bool:
    if len(audio) < 160:
        return False
    rms = _speech_rms(audio)
    return rms >= min_rms and _speech_dbfs(audio) >= min_db


def extract_voiced_crops(
    audio: np.ndarray,
    sr: int = SR,
    crop_s: float = 1.0,
    step_s: float | None = None,
    min_rms: float = VOICED_RMS_FLOOR,
    max_crops: int = 16,
) -> list[np.ndarray]:
    """Slice audio into crops of length ``crop_s``, keeping only voiced crops."""
    win = max(1, round(crop_s * sr))
    hop = round((step_s if step_s is not None else crop_s) * sr)
    crops: list[np.ndarray] = []
    for start in range(0, len(audio) - win + 1, hop):
        segment = audio[start : start + win]
        if is_voiced(segment, min_rms=min_rms):
            crops.append(segment)
            if len(crops) >= max_crops:
                break
    return crops


class MockEmbedder:
    """Deterministic 192-D mock embedder with distinctive speaker spectral centroids."""

    device = "cpu"

    def __init__(self, in_features: int = 192) -> None:
        self.in_features = in_features

    def embed(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32)
        if len(x) == 0:
            return np.zeros(self.in_features, dtype=np.float32)
        n_fft = min(512, len(x))
        spec = np.abs(np.fft.rfft(x, n=n_fft))
        emb = np.zeros(self.in_features, dtype=np.float32)
        emb[: min(len(spec), self.in_features)] = spec[: min(len(spec), self.in_features)]
        # Add deterministic projection
        rms = _speech_rms(x)
        emb[0] += rms * 10.0
        norm = np.linalg.norm(emb)
        return (emb / norm).astype(np.float32) if norm > 1e-12 else emb


def apply_transformation(
    audio: np.ndarray,
    transform_type: str,
    seed: int = 0,
) -> np.ndarray:
    """Apply acoustic transformations (quiet, noise babble/pink, bandlimit)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(audio, dtype=np.float32).copy()
    if transform_type == "quiet":
        # -12 dB quiet proxy
        return x * float(10 ** (-12.0 / 20.0))
    elif transform_type in ("noise_pink", "noise_babble", "noise_5db", "noise_15db"):
        snr_db = 5.0 if "5db" in transform_type else 15.0
        rms_sig = _speech_rms(x)
        rms_noise = max(1e-6, rms_sig / (10 ** (snr_db / 20.0)))
        noise = rng.standard_normal(len(x)).astype(np.float32)
        noise = noise * float(rms_noise / (np.std(noise) + 1e-9))
        return np.clip(x + noise, -1.0, 1.0)
    elif transform_type == "telephony":
        # Simple simulated bandpass (300 Hz to 3400 Hz) via spectral masking
        spec = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / SR)
        mask = (freqs >= 300.0) & (freqs <= 3400.0)
        spec[~mask] *= 0.05
        filtered = np.fft.irfft(spec, n=len(x))
        return np.clip(filtered.astype(np.float32), -1.0, 1.0)
    return x


def mix_contamination(
    target: np.ndarray,
    interferer: np.ndarray,
    sir_db: float = 0.0,
) -> np.ndarray:
    """Mix target and interferer at a specified Signal-to-Interference Ratio (SIR)."""
    n = min(len(target), len(interferer))
    t_stem = target[:n].astype(np.float32)
    i_stem = interferer[:n].astype(np.float32)
    rms_t = _speech_rms(t_stem)
    rms_i = _speech_rms(i_stem)
    if rms_i > 1e-9:
        scale_i = (rms_t / (10 ** (sir_db / 20.0))) / rms_i
        i_stem = i_stem * float(scale_i)
    return np.clip(t_stem + i_stem, -1.0, 1.0)


@dataclass
class ProbeCase:
    probe_id: str
    pair: str  # "team" or "public"
    target_speaker: str
    other_speaker: str
    condition: str  # "clean", "transformed", "contaminated", "separated"
    crop_duration_s: float
    source_group: str  # "dev" or "heldout"
    audio: np.ndarray = field(repr=False)
    target_clean: np.ndarray | None = field(default=None, repr=False)
    other_clean: np.ndarray | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResult:
    probe_id: str
    pair: str
    target_speaker: str
    other_speaker: str
    condition: str
    crop_duration_s: float
    source_group: str
    same_person_cos: float
    different_person_cos: float
    cosine_margin: float
    lda_margin: float | None
    lda_active: bool
    pair_cos: float
    negative_active: bool
    negative_rejected: bool
    selected_stream: int | None
    oracle_stream: int | None
    oracle_selected_agreement: bool | None
    wrong_speaker_duration_s: float
    confidence: float | None


def verify_saved_profile_lda(
    target_clip: np.ndarray,
    other_clip: np.ndarray,
    embedder: Any,
    separator: Any,
    sr: int = SR,
) -> dict[str, Any]:
    """Verify under what conditions the saved-profile / LDA route is active in EnrolledCandidate."""
    # Test 1: Full saved profiles registered via profile_clip
    cand_saved = EnrolledCandidate(
        separator=separator,
        embedder=embedder,
        closed_set=True,
    )
    cand_saved.prepare(target_clip, other_clip, sr)
    cand_saved.extract(target_clip[: min(sr * 2, len(target_clip))], sr, enroll=target_clip, other=other_clip)

    # Test 2: Unregistered raw audio (not registered in _PROFILE_CLIPS)
    rng = np.random.default_rng(123)
    raw_target = (rng.standard_normal(min(sr * 4, len(target_clip))) * 0.05).astype(np.float32)
    raw_other = (rng.standard_normal(min(sr * 4, len(other_clip))) * 0.05).astype(np.float32)
    cand_raw = EnrolledCandidate(
        separator=separator,
        embedder=embedder,
        closed_set=True,
    )
    cand_raw.prepare(raw_target, raw_other, sr)
    cand_raw.extract(raw_target[: min(sr * 2, len(raw_target))], sr, enroll=raw_target, other=raw_other)

    # Test 3: closed_set explicitly turned off
    cand_no_closed = EnrolledCandidate(
        separator=separator,
        embedder=embedder,
        closed_set=False,
    )
    cand_no_closed.prepare(target_clip, other_clip, sr)
    cand_no_closed.extract(target_clip[: min(sr * 2, len(target_clip))], sr, enroll=target_clip, other=other_clip)

    # Test 4: Single target only (other=None)
    cand_single = EnrolledCandidate(
        separator=separator,
        embedder=embedder,
        closed_set=True,
    )
    cand_single.prepare(target_clip, None, sr)
    cand_single.extract(target_clip[: min(sr * 2, len(target_clip))], sr, enroll=target_clip, other=None)

    target_pieces = profile_pieces(target_clip)
    other_pieces = profile_pieces(other_clip)
    num_target_pieces = len(target_pieces) if target_pieces is not None else 0
    num_other_pieces = len(other_pieces) if other_pieces is not None else 0

    return {
        "saved_profile_scoring": cand_saved.scoring,
        "saved_profile_lda_active": cand_saved.scoring == "lda",
        "saved_profile_pair_cos": cand_saved.pair_cos,
        "saved_profile_negative_active": cand_saved.negative_active,
        "num_target_pieces": num_target_pieces,
        "num_other_pieces": num_other_pieces,
        "raw_audio_scoring": cand_raw.scoring,
        "raw_audio_lda_active": cand_raw.scoring == "lda",
        "closed_set_false_scoring": cand_no_closed.scoring,
        "closed_set_false_lda_active": cand_no_closed.scoring == "lda",
        "single_target_scoring": cand_single.scoring,
        "single_target_lda_active": cand_single.scoring == "lda",
        "activation_rules_verified": {
            "requires_both_profiles_registered": cand_saved.scoring == "lda" and cand_raw.scoring == "cosine",
            "requires_closed_set_true": cand_no_closed.scoring == "cosine",
            "requires_distinct_negative": cand_saved.negative_active == (cand_saved.pair_cos <= SAME_VOICE),
            "requires_two_or_more_pieces": (num_target_pieces >= 2 and num_other_pieces >= 2),
        },
    }


def evaluate_probe_case(
    probe: ProbeCase,
    target_clip: np.ndarray,
    other_clip: np.ndarray,
    target_emb: np.ndarray,
    other_emb: np.ndarray,
    embedder: Any,
    separator: Any,
) -> ProbeResult:
    """Evaluate a single probe against target and negative enrollment profiles."""
    sr = SR
    audio = probe.audio
    crop_emb = _unit_vector(embedder.embed(audio))

    same_person_cos = float(np.dot(target_emb, crop_emb))
    different_person_cos = float(np.dot(other_emb, crop_emb))
    cosine_margin = same_person_cos - different_person_cos

    pair_cos = float(np.dot(target_emb, other_emb))
    negative_rejected = pair_cos > SAME_VOICE
    negative_active = not negative_rejected

    # Check LDA route
    target_pieces = profile_pieces(target_clip)
    other_pieces = profile_pieces(other_clip)
    lda_margin = None
    lda_active = False

    if (
        negative_active
        and target_pieces is not None
        and other_pieces is not None
        and len(target_pieces) >= 2
        and len(other_pieces) >= 2
    ):
        t_units = [_unit_vector(embedder.embed(p)) for p in target_pieces]
        o_units = [_unit_vector(embedder.embed(p)) for p in other_pieces]
        lda_fit = fit_lda(np.stack(t_units), np.stack(o_units))
        if lda_fit is not None:
            w, b = lda_fit
            score_probe = float(w @ crop_emb) - b
            # In LDA units, target centroid is +1 and other centroid is -1
            lda_margin = score_probe
            lda_active = True

    # For separated or contaminated speech, measure stream selection and wrong-speaker duration
    selected_stream = None
    oracle_stream = None
    oracle_selected_agreement = None
    wrong_speaker_duration_s = 0.0
    confidence = None

    if probe.condition == "separated":
        # Separate mixture audio into streams
        streams = separator.separate(audio, sr)
        if len(streams) >= 2:
            st_embs = [_unit_vector(embedder.embed(s)) for s in streams[:2]]
            if lda_active and lda_fit is not None:
                w, b = lda_fit
                scores = [float(w @ e) - b for e in st_embs]
                full_conf = CONF_MARGIN_LDA
            else:
                sims = [float(np.dot(target_emb, e)) for e in st_embs]
                scores = (
                    [s - float(np.dot(other_emb, e)) for s, e in zip(sims, st_embs, strict=True)]
                    if negative_active
                    else sims
                )
                full_conf = CONF_MARGIN

            selected_stream = int(np.argmax(scores))
            margin_score = float(scores[selected_stream] - scores[1 - selected_stream])
            confidence = min(1.0, max(0.0, margin_score / full_conf)) if full_conf > 0 else 0.5

            if probe.target_clean is not None and probe.other_clean is not None:
                n_eval = min(len(probe.target_clean), len(streams[0]))
                corr0 = float(np.dot(streams[0][:n_eval], probe.target_clean[:n_eval]))
                corr1 = float(np.dot(streams[1][:n_eval], probe.target_clean[:n_eval]))
                oracle_stream = 0 if corr0 >= corr1 else 1
                oracle_selected_agreement = selected_stream == oracle_stream
                if not oracle_selected_agreement:
                    wrong_speaker_duration_s = probe.crop_duration_s

    elif probe.condition == "contaminated":
        # Target + interferer mixed: selection between target and negative
        # Decision is whether margin favors target (> 0)
        oracle_stream = 0  # target was active
        selected_stream = 0 if cosine_margin >= 0.0 else 1
        oracle_selected_agreement = selected_stream == oracle_stream
        if not oracle_selected_agreement:
            wrong_speaker_duration_s = probe.crop_duration_s

    return ProbeResult(
        probe_id=probe.probe_id,
        pair=probe.pair,
        target_speaker=probe.target_speaker,
        other_speaker=probe.other_speaker,
        condition=probe.condition,
        crop_duration_s=probe.crop_duration_s,
        source_group=probe.source_group,
        same_person_cos=round(same_person_cos, 4),
        different_person_cos=round(different_person_cos, 4),
        cosine_margin=round(cosine_margin, 4),
        lda_margin=round(lda_margin, 4) if lda_margin is not None else None,
        lda_active=lda_active,
        pair_cos=round(pair_cos, 4),
        negative_active=negative_active,
        negative_rejected=negative_rejected,
        selected_stream=selected_stream,
        oracle_stream=oracle_stream,
        oracle_selected_agreement=oracle_selected_agreement,
        wrong_speaker_duration_s=round(wrong_speaker_duration_s, 3),
        confidence=round(confidence, 4) if confidence is not None else None,
    )


def calibrate_dev_thresholds(
    dev_results: list[ProbeResult],
    target_far: float = TARGET_FAR,
) -> dict[str, float]:
    """Fit decision and rejection thresholds strictly on dev source group."""
    # Negative probes: where target was NOT present or different-person cosine
    diff_cos = [r.different_person_cos for r in dev_results]
    if diff_cos:
        # Threshold where only target_far fraction of false positives occur
        margin_thr = float(np.percentile(diff_cos, 100.0 * (1.0 - target_far)))
    else:
        margin_thr = 0.0

    return {
        "calibrated_margin_threshold": round(margin_thr, 4),
        "target_far": target_far,
        "n_dev_probes": len(dev_results),
    }


def evaluate_heldout_with_thresholds(
    heldout_results: list[ProbeResult],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Evaluate frozen thresholds on untouched heldout source group."""
    thr = thresholds["calibrated_margin_threshold"]
    same_cos = [r.same_person_cos for r in heldout_results]
    diff_cos = [r.different_person_cos for r in heldout_results]

    tpr_clean = [
        r.same_person_cos >= thr for r in heldout_results if r.condition == "clean"
    ]
    tpr_transformed = [
        r.same_person_cos >= thr for r in heldout_results if r.condition == "transformed"
    ]
    far_wrong = [r.different_person_cos >= thr for r in heldout_results]

    agreement_cases = [
        r.oracle_selected_agreement
        for r in heldout_results
        if r.oracle_selected_agreement is not None
    ]
    wrong_dur = sum(r.wrong_speaker_duration_s for r in heldout_results)

    return {
        "n_heldout_probes": len(heldout_results),
        "mean_same_person_cos": round(float(np.mean(same_cos)), 4) if same_cos else 0.0,
        "mean_different_person_cos": round(float(np.mean(diff_cos)), 4) if diff_cos else 0.0,
        "mean_margin": round(float(np.mean([r.cosine_margin for r in heldout_results])), 4)
        if heldout_results
        else 0.0,
        "tpr_clean": round(float(np.mean(tpr_clean)), 4) if tpr_clean else 1.0,
        "tpr_transformed": round(float(np.mean(tpr_transformed)), 4) if tpr_transformed else 1.0,
        "far_wrong_speaker": round(float(np.mean(far_wrong)), 4) if far_wrong else 0.0,
        "oracle_agreement_rate": round(float(np.mean(agreement_cases)), 4) if agreement_cases else 1.0,
        "total_wrong_speaker_duration_s": round(wrong_dur, 3),
    }


def generate_synthetic_fixtures(seed: int = 212) -> dict[str, Any]:
    """Deterministic synthetic fixture generator for tests and dry runs."""
    rng = np.random.default_rng(seed)
    sr = SR

    def synth_voice(freq: float, duration_s: float) -> np.ndarray:
        t = np.arange(round(duration_s * sr), dtype=np.float32) / sr
        harmonics = [
            (0.35, freq),
            (0.20, freq * 2),
            (0.12, freq * 3),
            (0.08, freq * 4),
        ]
        sig = np.zeros_like(t)
        for amp, f in harmonics:
            sig += amp * np.sin(2 * np.pi * f * t + rng.uniform(0, np.pi))
        # Add slight natural amplitude modulation
        envelope = 0.8 + 0.2 * np.sin(2 * np.pi * 3.5 * t)
        sig = sig * envelope
        return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.4).astype(np.float32)

    return {
        "aryan_profile": synth_voice(140.0, 10.0),
        "joseph_profile": synth_voice(125.0, 10.0),
        "aryan_fixture": synth_voice(140.0, 15.0),
        "joseph_fixture": synth_voice(125.0, 15.0),
        "public_1320_profile": synth_voice(160.0, 10.0),
        "public_6930_profile": synth_voice(110.0, 10.0),
        "public_1320_fixture": synth_voice(160.0, 15.0),
        "public_6930_fixture": synth_voice(110.0, 15.0),
    }


def load_source_audio(fixtures_dir: Path, profiles_dir: Path | None) -> dict[str, Any]:
    """Load source recordings or fallback gracefully."""
    p_dir = profiles_dir if profiles_dir and profiles_dir.is_dir() else fixtures_dir / "profiles"
    team_dir = fixtures_dir / "team"
    pub_dir = fixtures_dir / "public_overnight"

    audio_data: dict[str, Any] = {}

    # Team profiles
    aryan_prof_path = p_dir / "aryan" / "profile-01.wav"
    joseph_prof_path = p_dir / "joseph" / "profile-01.wav"
    if aryan_prof_path.is_file() and joseph_prof_path.is_file():
        audio_data["aryan_profile"], _ = aio.load(aryan_prof_path, SR)
        audio_data["joseph_profile"], _ = aio.load(joseph_prof_path, SR)
        audio_data["aryan_profile_path"] = str(aryan_prof_path)
        audio_data["joseph_profile_path"] = str(joseph_prof_path)
        audio_data["aryan_profile_sha256"] = _sha256_file(aryan_prof_path)
        audio_data["joseph_profile_sha256"] = _sha256_file(joseph_prof_path)

    # Team fixtures
    aryan_fix_path = team_dir / "aryan" / "fixture-01.wav"
    joseph_fix_path = team_dir / "joseph" / "fixture-01-20261009-1353.wav"
    if aryan_fix_path.is_file() and joseph_fix_path.is_file():
        audio_data["aryan_fixture"], _ = aio.load(aryan_fix_path, SR)
        audio_data["joseph_fixture"], _ = aio.load(joseph_fix_path, SR)
        audio_data["aryan_fixture_path"] = str(aryan_fix_path)
        audio_data["joseph_fixture_path"] = str(joseph_fix_path)
        audio_data["aryan_fixture_sha256"] = _sha256_file(aryan_fix_path)
        audio_data["joseph_fixture_sha256"] = _sha256_file(joseph_fix_path)

    # Public profiles
    p1320_path = pub_dir / "public_profiles" / "1320.wav"
    p6930_path = pub_dir / "public_profiles" / "6930.wav"
    if p1320_path.is_file() and p6930_path.is_file():
        audio_data["public_1320_profile"], _ = aio.load(p1320_path, SR)
        audio_data["public_6930_profile"], _ = aio.load(p6930_path, SR)
        audio_data["public_1320_profile_path"] = str(p1320_path)
        audio_data["public_6930_profile_path"] = str(p6930_path)
        audio_data["public_1320_profile_sha256"] = _sha256_file(p1320_path)
        audio_data["public_6930_profile_sha256"] = _sha256_file(p6930_path)

    # Public fixtures
    pub_fix_1320_path = pub_dir / "take0_1320_sir0_snr15" / "target.wav"
    pub_fix_6930_path = pub_dir / "take0_6930_sir0_snr5" / "target.wav"
    if pub_fix_1320_path.is_file() and pub_fix_6930_path.is_file():
        audio_data["public_1320_fixture"], _ = aio.load(pub_fix_1320_path, SR)
        audio_data["public_6930_fixture"], _ = aio.load(pub_fix_6930_path, SR)
        audio_data["public_1320_fixture_path"] = str(pub_fix_1320_path)
        audio_data["public_6930_fixture_path"] = str(pub_fix_6930_path)
        audio_data["public_1320_fixture_sha256"] = _sha256_file(pub_fix_1320_path)
        audio_data["public_6930_fixture_sha256"] = _sha256_file(pub_fix_6930_path)

    return audio_data


def build_probe_suite(
    audio_data: dict[str, Any],
    cap: int = DEFAULT_CAP,
    seed: int = 212,
) -> list[ProbeCase]:
    """Build bounded probe suite respecting <= 150 probes cap across conditions and crops."""
    probes: list[ProbeCase] = []
    pid = 0

    pairs_to_run = [
        ("team", "aryan", "joseph", "aryan_profile", "joseph_profile", "aryan_fixture", "joseph_fixture"),
        ("team", "joseph", "aryan", "joseph_profile", "aryan_profile", "joseph_fixture", "aryan_fixture"),
        (
            "public",
            "1320",
            "6930",
            "public_1320_profile",
            "public_6930_profile",
            "public_1320_fixture",
            "public_6930_fixture",
        ),
        (
            "public",
            "6930",
            "1320",
            "public_6930_profile",
            "public_1320_profile",
            "public_6930_fixture",
            "public_1320_fixture",
        ),
    ]

    for pair_name, target_spk, other_spk, _, _, fix_t_key, fix_o_key in pairs_to_run:
        target_audio = audio_data.get(fix_t_key)
        other_audio = audio_data.get(fix_o_key)
        if target_audio is None or other_audio is None:
            continue

        # Split source into dev (first 50%) and heldout (second 50%)
        mid_t = len(target_audio) // 2
        mid_o = len(other_audio) // 2
        splits = [
            ("dev", target_audio[:mid_t], other_audio[:mid_o]),
            ("heldout", target_audio[mid_t:], other_audio[mid_o:]),
        ]

        for split_name, t_split, o_split in splits:
            for crop_s in CROP_DURATIONS:
                t_crops = extract_voiced_crops(t_split, SR, crop_s=crop_s, max_crops=2)
                o_crops = extract_voiced_crops(o_split, SR, crop_s=crop_s, max_crops=2)
                if not t_crops or not o_crops:
                    continue

                # 1. Clean condition
                for ci, c in enumerate(t_crops[:1]):
                    probes.append(
                        ProbeCase(
                            probe_id=f"probe_{pid:03d}_{pair_name}_{target_spk}_clean_{crop_s}s_{split_name}_{ci}",
                            pair=pair_name,
                            target_speaker=target_spk,
                            other_speaker=other_spk,
                            condition="clean",
                            crop_duration_s=crop_s,
                            source_group=split_name,
                            audio=c,
                            target_clean=c,
                            other_clean=o_crops[0],
                            metadata={"transform": "none"},
                        )
                    )
                    pid += 1

                # 2. Transformed condition (quiet -12 dB, noise babble 15 dB)
                for ci, c in enumerate(t_crops[:1]):
                    trans_type = "quiet" if ci % 2 == 0 else "noise_15db"
                    trans_audio = apply_transformation(c, trans_type, seed=seed + pid)
                    probes.append(
                        ProbeCase(
                            probe_id=f"probe_{pid:03d}_{pair_name}_{target_spk}_transformed_{crop_s}s_{split_name}_{ci}",
                            pair=pair_name,
                            target_speaker=target_spk,
                            other_speaker=other_spk,
                            condition="transformed",
                            crop_duration_s=crop_s,
                            source_group=split_name,
                            audio=trans_audio,
                            target_clean=c,
                            other_clean=o_crops[0],
                            metadata={"transform": trans_type},
                        )
                    )
                    pid += 1

                # 3. Contaminated condition (SIR -5 dB overlap)
                for ci, c in enumerate(t_crops[:1]):
                    o_c = o_crops[min(ci, len(o_crops) - 1)]
                    mixed = mix_contamination(c, o_c, sir_db=-5.0)
                    probes.append(
                        ProbeCase(
                            probe_id=f"probe_{pid:03d}_{pair_name}_{target_spk}_contaminated_{crop_s}s_{split_name}_{ci}",
                            pair=pair_name,
                            target_speaker=target_spk,
                            other_speaker=other_spk,
                            condition="contaminated",
                            crop_duration_s=crop_s,
                            source_group=split_name,
                            audio=mixed,
                            target_clean=c,
                            other_clean=o_c,
                            metadata={"sir_db": -5.0},
                        )
                    )
                    pid += 1

                # 4. Separated condition (mixture to be separated)
                for ci, c in enumerate(t_crops[:1]):
                    o_c = o_crops[min(ci, len(o_crops) - 1)]
                    mixed = mix_contamination(c, o_c, sir_db=0.0)
                    probes.append(
                        ProbeCase(
                            probe_id=f"probe_{pid:03d}_{pair_name}_{target_spk}_separated_{crop_s}s_{split_name}_{ci}",
                            pair=pair_name,
                            target_speaker=target_spk,
                            other_speaker=other_spk,
                            condition="separated",
                            crop_duration_s=crop_s,
                            source_group=split_name,
                            audio=mixed,
                            target_clean=c,
                            other_clean=o_c,
                            metadata={"sir_db": 0.0, "mixture": True},
                        )
                    )
                    pid += 1

                if len(probes) >= cap:
                    break
            if len(probes) >= cap:
                break
        if len(probes) >= cap:
            break

    return probes[:cap]


def run_overnight_voice_vectors(
    fixtures_dir: Path,
    profiles_dir: Path | None,
    output_path: Path,
    cap: int = DEFAULT_CAP,
    dry_run: bool = False,
    device: str = "cpu",
    seed: int = 212,
    stop_file: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute bounded overnight evaluation and save reproducible results."""
    t0 = time.time()
    cap = min(cap, 150)  # strict hard cap <= 150
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")

    # Load audio
    audio_data = load_source_audio(fixtures_dir, profiles_dir)
    synthetic_used = False
    if "aryan_profile" not in audio_data or "aryan_fixture" not in audio_data:
        # Fall back to synthetic fixtures for offline testing
        synthetic_used = True
        audio_data.update(generate_synthetic_fixtures(seed))

    # Load models
    if dry_run or synthetic_used:
        embedder = MockEmbedder(192)
        separator = BlindSeparator(backend=MockSepBackend())
        model_status = "mock_deterministic"
    else:
        from hearing import models
        from hearing.tse.enrolled import MODEL as ECAPA_MODEL

        try:
            embedder = models.load(ECAPA_MODEL)
            separator = BlindSeparator()
            separator.load()
            model_status = "real_weights_loaded"
        except (RuntimeError, ImportError) as exc:
            embedder = MockEmbedder(192)
            separator = BlindSeparator(backend=MockSepBackend())
            model_status = f"mock_fallback_due_to_{exc.__class__.__name__}"

    # Prepare registered profile clips
    def prepare_profile(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pieces = split_pieces(x, SR, piece_s=2.5)
        clip = profile_clip(pieces)
        emb = _unit_vector(embedder.embed(clip))
        return clip, emb

    team_t_clip, team_t_emb = prepare_profile(audio_data["aryan_profile"])
    team_o_clip, team_o_emb = prepare_profile(audio_data["joseph_profile"])

    pub_t_clip, pub_t_emb = prepare_profile(audio_data["public_1320_profile"])
    pub_o_clip, pub_o_emb = prepare_profile(audio_data["public_6930_profile"])

    # Verify LDA route
    team_lda_verification = verify_saved_profile_lda(
        team_t_clip, team_o_clip, embedder, separator, SR
    )
    pub_lda_verification = verify_saved_profile_lda(
        pub_t_clip, pub_o_clip, embedder, separator, SR
    )

    # Build probe suite
    probes = build_probe_suite(audio_data, cap=cap, seed=seed)

    # Check resumption
    completed_results: list[ProbeResult] = []
    resumed = False
    if resume and checkpoint_path.is_file():
        try:
            chk_data = json.loads(checkpoint_path.read_text())
            if chk_data.get("seed") == seed and chk_data.get("cap") == cap:
                completed_results = [ProbeResult(**r) for r in chk_data.get("probes", [])]
                resumed = True
        except (OSError, json.JSONDecodeError):
            completed_results = []

    completed_ids = {r.probe_id for r in completed_results}
    probes_to_run = [p for p in probes if p.probe_id not in completed_ids]

    # Run evaluation loop
    for probe in probes_to_run:
        if stop_file and stop_file.is_file():
            break

        if probe.pair == "team":
            if probe.target_speaker == "aryan":
                t_clip, o_clip = team_t_clip, team_o_clip
                t_emb, o_emb = team_t_emb, team_o_emb
            else:
                t_clip, o_clip = team_o_clip, team_t_clip
                t_emb, o_emb = team_o_emb, team_t_emb
        else:
            if probe.target_speaker == "1320":
                t_clip, o_clip = pub_t_clip, pub_o_clip
                t_emb, o_emb = pub_t_emb, pub_o_emb
            else:
                t_clip, o_clip = pub_o_clip, pub_t_clip
                t_emb, o_emb = pub_o_emb, pub_t_emb

        res = evaluate_probe_case(
            probe, t_clip, o_clip, t_emb, o_emb, embedder, separator
        )
        completed_results.append(res)

        # Periodic checkpoint
        if len(completed_results) % 8 == 0:
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "cap": cap,
                        "probes": [asdict(r) for r in completed_results],
                    },
                    indent=2,
                )
            )

    # Threshold calibration on dev
    dev_results = [r for r in completed_results if r.source_group == "dev"]
    heldout_results = [r for r in completed_results if r.source_group == "heldout"]

    thresholds = calibrate_dev_thresholds(dev_results, target_far=TARGET_FAR)
    heldout_eval = evaluate_heldout_with_thresholds(heldout_results, thresholds)

    # Aggregated breakdown by crop duration
    crop_stats: dict[str, Any] = {}
    for crop_s in CROP_DURATIONS:
        c_probes = [r for r in completed_results if r.crop_duration_s == crop_s]
        c_heldout = [r for r in c_probes if r.source_group == "heldout"]
        same = [r.same_person_cos for r in c_probes]
        diff = [r.different_person_cos for r in c_probes]
        margin = [r.cosine_margin for r in c_probes]
        agreements = [
            r.oracle_selected_agreement
            for r in c_heldout
            if r.oracle_selected_agreement is not None
        ]
        wrong_s = sum(r.wrong_speaker_duration_s for r in c_heldout)

        crop_stats[f"{crop_s:g}s"] = {
            "n_probes": len(c_probes),
            "same_person_cos_p50": round(float(np.median(same)), 4) if same else 0.0,
            "different_person_cos_p50": round(float(np.median(diff)), 4) if diff else 0.0,
            "cosine_margin_p50": round(float(np.median(margin)), 4) if margin else 0.0,
            "cosine_margin_p05": round(float(np.percentile(margin, 5)), 4) if margin else 0.0,
            "cosine_margin_p95": round(float(np.percentile(margin, 95)), 4) if margin else 0.0,
            "oracle_agreement_rate": round(float(np.mean(agreements)), 4) if agreements else 1.0,
            "wrong_speaker_duration_s": round(wrong_s, 3),
        }

    # Condition stats
    condition_stats: dict[str, Any] = {}
    for cond in ("clean", "transformed", "contaminated", "separated"):
        cond_probes = [r for r in completed_results if r.condition == cond]
        same = [r.same_person_cos for r in cond_probes]
        diff = [r.different_person_cos for r in cond_probes]
        margin = [r.cosine_margin for r in cond_probes]
        condition_stats[cond] = {
            "n_probes": len(cond_probes),
            "mean_same_cos": round(float(np.mean(same)), 4) if same else 0.0,
            "mean_diff_cos": round(float(np.mean(diff)), 4) if diff else 0.0,
            "mean_margin": round(float(np.mean(margin)), 4) if margin else 0.0,
        }

    duration_total_s = time.time() - t0

    # Build final report document
    config_dict = {
        "cap": cap,
        "seed": seed,
        "crop_durations": list(CROP_DURATIONS),
        "device": device,
        "same_voice_threshold": SAME_VOICE,
        "target_far": TARGET_FAR,
    }
    config_hash = _sha256_str(json.dumps(config_dict, sort_keys=True))

    results_document: dict[str, Any] = {
        "task_id": "T-212",
        "title": "Overnight voice-vector robustness: ECAPA identity, negative profiles and separated-speech crops",
        "status": "complete",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_total_s": round(duration_total_s, 2),
        "device": device,
        "model_status": model_status,
        "synthetic_used": synthetic_used,
        "resumed": resumed,
        "config": config_dict,
        "config_hash": config_hash,
        "source_hashes": {k: v for k, v in audio_data.items() if k.endswith("_sha256")},
        "counts": {
            "total_probes": len(completed_results),
            "dev_probes": len(dev_results),
            "heldout_probes": len(heldout_results),
            "cap": cap,
        },
        "lda_route_verification": {
            "team_pair": team_lda_verification,
            "public_pair": pub_lda_verification,
        },
        "negative_rejection": {
            "team_pair_cos": round(float(np.dot(team_t_emb, team_o_emb)), 4),
            "team_negative_rejected": float(np.dot(team_t_emb, team_o_emb)) > SAME_VOICE,
            "public_pair_cos": round(float(np.dot(pub_t_emb, pub_o_emb)), 4),
            "public_negative_rejected": float(np.dot(pub_t_emb, pub_o_emb)) > SAME_VOICE,
            "rule": f"cos(target, other) > {SAME_VOICE} => negative rejected as same voice",
        },
        "dev_threshold_calibration": thresholds,
        "heldout_evaluation": heldout_eval,
        "crop_duration_breakdown": crop_stats,
        "condition_breakdown": condition_stats,
        "probes": [asdict(r) for r in completed_results],
    }

    output_path.write_text(json.dumps(results_document, indent=2))
    if checkpoint_path.is_file():
        checkpoint_path.unlink()

    return results_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overnight voice-vector robustness evaluation (T-212)"
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path(os.environ.get("HEARING_FIXTURES_DIR", "eval/fixtures/generated")),
        help="Root path for fixtures",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=Path(os.environ.get("HEARING_PROFILES_DIR", "eval/fixtures/generated/profiles")),
        help="Root path for voice profiles",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results"))
        / "overnight_voice_vectors.json",
        help="Output JSON results path",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_CAP,
        help="Maximum probes to evaluate (capped <= 150)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast offline mock pipeline without real model weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (cpu/mps/cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=212,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=None,
        help="Path to stop file for clean interruption",
    )

    args = parser.parse_args()
    report = run_overnight_voice_vectors(
        fixtures_dir=args.fixtures_dir,
        profiles_dir=args.profiles_dir,
        output_path=args.output,
        cap=args.cap,
        dry_run=args.dry_run,
        device=args.device,
        seed=args.seed,
        stop_file=args.stop_file,
        resume=args.resume,
    )
    print(f"T-212 Completed: {report['counts']['total_probes']} probes evaluated.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
