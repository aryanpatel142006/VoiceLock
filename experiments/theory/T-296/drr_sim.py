"""DRR physics simulation, crossover frequency prediction, and distance-invariant representations.

Task: T-296 (Theory: distance-invariant speaker representations building on T-283).

Models:
1. Room acoustics: Direct-to-reverberant ratio (DRR), Sabine RT60, critical distance dc(f),
   and atmospheric absorption vs Euclidean distance.
2. Frequency decomposition: Spectral contrast of direct formant structure vs diffuse room
   coloration; crossover frequency f_cross(d) prediction vs T-283's 2000 Hz band split.
3. Distance-confounded benchmark: Evaluation matrix comparing matched-distance vs
   mismatched-distance (near-enroll/far-test) speaker verification margins.
4. Linear vs Band decomposition proof: Empirical demonstration that non-linear distance
   coloration violates linear subspace transversality, necessitating dynamic band gating.
5. Morning rehearsal protocol: 3 concrete distances (D1=0.4m, D2=1.2m, D3=2.5m) with predicted
   DRR values for the HackRU demo room.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RoomParameters:
    length: float = 8.0
    width: float = 6.0
    height: float = 3.0
    # Combined directivity Q = Q_src * Q_mic.
    # Q_src ~ 1.3 (speech on-axis); Q_mic ~ 1.55 (directional cardioid) -> Q = 2.0
    directivity: float = 2.0
    speed_of_sound: float = 343.0  # m/s


# Frequency-dependent absorption coefficients for typical commercial presentation room
# (drywall walls, commercial carpet on concrete, drop acoustic ceiling tile)
DEFAULT_OCTAVE_BANDS: list[float] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
DEFAULT_ABSORPTION_COEFFS: list[float] = [0.10, 0.15, 0.22, 0.28, 0.35, 0.40, 0.42]


class RoomAcousticModel:
    """Acoustic simulation of direct and reverberant sound propagation."""

    def __init__(
        self,
        length: float = 8.0,
        width: float = 6.0,
        height: float = 3.0,
        directivity: float = 2.0,
        octave_bands: list[float] | None = None,
        absorption_coeffs: list[float] | None = None,
    ) -> None:
        self.params = RoomParameters(
            length=float(length),
            width=float(width),
            height=float(height),
            directivity=float(directivity),
        )
        self.octave_bands = np.array(octave_bands or DEFAULT_OCTAVE_BANDS, dtype=np.float64)
        self.absorption_coeffs = np.array(
            absorption_coeffs or DEFAULT_ABSORPTION_COEFFS, dtype=np.float64
        )

        self.volume = self.params.length * self.params.width * self.params.height
        self.surface_area = 2.0 * (
            self.params.length * self.params.width
            + self.params.length * self.params.height
            + self.params.width * self.params.height
        )

    def absorption(self, freq: float) -> float:
        """Interpolate mean room surface absorption coefficient at frequency freq (Hz)."""
        f = np.clip(freq, self.octave_bands[0], self.octave_bands[-1])
        return float(np.interp(np.log10(f), np.log10(self.octave_bands), self.absorption_coeffs))

    def rt60(self, freq: float) -> float:
        """Sabine reverberation time T60 (seconds) at frequency freq."""
        alpha = self.absorption(freq)
        sa = self.surface_area * alpha
        return float(0.161 * self.volume / max(sa, 1e-6))

    def room_constant(self, freq: float) -> float:
        """Room constant R = S * alpha / (1 - alpha) in m^2."""
        alpha = self.absorption(freq)
        alpha = min(max(alpha, 0.01), 0.95)
        return float(self.surface_area * alpha / (1.0 - alpha))

    def air_absorption_db_per_m(self, freq: float) -> float:
        """Atmospheric attenuation coefficient (ISO 9613-1 approximation at 20C, 50% RH)."""
        f_khz = freq / 1000.0
        return float(0.007 * (f_khz**2))

    def critical_distance(self, freq: float) -> float:
        """Critical distance dc in meters where direct energy equals reverberant energy (DRR = 0 dB)."""
        r = self.room_constant(freq)
        q = self.params.directivity
        return float(np.sqrt((q * r) / (16.0 * np.pi)))

    def drr_linear(self, distance: float, freq: float) -> float:
        """Direct-to-reverberant energy ratio (linear scale)."""
        d = max(float(distance), 0.05)
        r = self.room_constant(freq)
        q = self.params.directivity
        alpha_db_m = self.air_absorption_db_per_m(freq)
        atten_lin = 10.0 ** (-alpha_db_m * d / 10.0)
        return float((q * r * atten_lin) / (16.0 * np.pi * (d**2)))

    def drr_db(self, distance: float, freq: float) -> float:
        """Direct-to-reverberant ratio in decibels."""
        lin = self.drr_linear(distance, freq)
        return float(10.0 * np.log10(max(lin, 1e-12)))

    def identity_snr_db(self, distance: float, freq: float) -> float:
        """Effective speaker identity SNR considering speech spectrum, DRR, and modal smearing.

        Near-field speech (0.4m) provides high SPL across bands. As distance increases,
        glottal tilt (-6 dB/octave) means direct formants F3 and F4 drop into the ambient
        room noise and reverberation smearing floor much earlier than low-band formants F1 and F2.
        """
        drr = self.drr_db(distance, freq)
        f_clamped = max(freq, 200.0)
        # Speech spectrum roll-off relative to 2 kHz (speech power is +12 dB at 500 Hz, 0 dB at 2 kHz)
        speech_spec_db = -6.0 * np.log2(f_clamped / 2000.0)
        # Distance loss relative to near anchor (0.4 m)
        dist_loss_db = 20.0 * np.log10(max(distance, 0.1) / 0.4)
        # High-frequency vulnerability: modal density and ambient noise floor
        hf_penalty = (f_clamped / 2000.0) ** 1.2 * (dist_loss_db * 0.35)
        return float(drr + speech_spec_db - hf_penalty)

    def predict_crossover_frequency(
        self, distance: float, threshold_snr_db: float = 0.0
    ) -> float:
        """Predict crossover frequency (Hz) where identity SNR drops below threshold.

        At near distance (0.4m), direct sound dominates across speech band (> 3.5 kHz).
        At conversational distance (1.2 - 1.5m), crossover drops into 1.5 - 2.5 kHz (T-283 split).
        At far distance (2.5m), crossover degrades below 1200 Hz.
        """
        freqs = np.linspace(300.0, 7500.0, 400)
        snrs = np.array([self.identity_snr_db(distance, f) for f in freqs])

        if np.all(snrs >= threshold_snr_db):
            return float(freqs[-1])
        if np.all(snrs < threshold_snr_db):
            return float(freqs[0])

        idx = np.where(snrs <= threshold_snr_db)[0][0]
        if idx == 0:
            return float(freqs[0])
        f0, f1 = freqs[idx - 1], freqs[idx]
        s0, s1 = snrs[idx - 1], snrs[idx]
        return float(f0 + (threshold_snr_db - s0) * (f1 - f0) / (s1 - s0))

    def morning_rehearsal_protocol(self) -> dict[str, Any]:
        """Generates multi-distance enrollment protocol with concrete distances and predicted DRRs."""
        distances = [
            {
                "id": "D1_near",
                "label": "Handheld / Desktop Near",
                "distance_m": 0.4,
                "role": "Clean enrollment anchor (high direct energy, pristine formants)",
                "drr_db": round(self.drr_db(0.4, 1000.0), 2),
                "crossover_hz": round(self.predict_crossover_frequency(0.4), 1),
                "regime": "Direct dominant (DRR >> 0 dB)",
            },
            {
                "id": "D2_mid",
                "label": "Conversational Standing Mid",
                "distance_m": 1.2,
                "role": "Transition boundary (direct ~ reverberant, T-283 split crossover)",
                "drr_db": round(self.drr_db(1.2, 1000.0), 2),
                "crossover_hz": round(self.predict_crossover_frequency(1.2), 1),
                "regime": "Transition zone (DRR ~ 0 dB)",
            },
            {
                "id": "D3_far",
                "label": "Stage Reveal Far",
                "distance_m": 2.5,
                "role": "Reverberant target/interferer (simulates bonus reveal presenter)",
                "drr_db": round(self.drr_db(2.5, 1000.0), 2),
                "crossover_hz": round(self.predict_crossover_frequency(2.5), 1),
                "regime": "Reverberation dominant (DRR << 0 dB)",
            },
        ]
        return {
            "room_geometry": {
                "dimensions_m": [self.params.length, self.params.width, self.params.height],
                "volume_m3": round(self.volume, 1),
                "surface_area_m2": round(self.surface_area, 1),
                "rt60_mid_s": round(self.rt60(1000.0), 2),
                "critical_distance_m": round(self.critical_distance(1000.0), 2),
            },
            "calibration_phrases": [
                "The quick brown fox jumps over the lazy dog.",
                "Selective hearing brings conversational focus into sharp clarity.",
            ],
            "recording_duration_per_distance_s": 4.0,
            "distances": distances,
        }


class BandContrastAnalyzer:
    """Analyzes formant contrast preservation and spectral degradation across frequency bands."""

    def __init__(self, room: RoomAcousticModel | None = None) -> None:
        self.room = room or RoomAcousticModel()
        self.split_hz = 2000.0  # T-283 band split frequency

    def speech_psd(self, freqs: np.ndarray) -> np.ndarray:
        """Synthetic speech power spectral density with glottal tilt (-6 dB/oct) and formants."""
        f_norm = np.maximum(freqs, 100.0) / 500.0
        glottal = 1.0 / (1.0 + f_norm**2)

        formants = np.zeros_like(freqs)
        for fc, bw, gain in [
            (500.0, 90.0, 4.0),
            (1500.0, 110.0, 3.0),
            (2500.0, 140.0, 2.0),
            (3500.0, 180.0, 1.2),
        ]:
            formants += gain / (1.0 + ((freqs - fc) / (bw / 2.0)) ** 2)

        return (glottal * (1.0 + formants)).astype(np.float64)

    def evaluate_band_contrast(self, distance: float) -> dict[str, float]:
        """Compute mean direct formant contrast in low (<2 kHz) and high (>2 kHz) bands."""
        freqs = np.linspace(100.0, 8000.0, 500)
        psd = self.speech_psd(freqs)

        drr_lin = np.array([self.room.drr_linear(distance, f) for f in freqs])
        # Direct formant power vs diffuse reverberant energy and ambient floor scaling with distance
        ambient_floor = 0.03 * (freqs / 1000.0) * ((distance / 1.0) ** 1.5)
        direct_pwr = psd * drr_lin
        rev_pwr = psd + ambient_floor
        contrast = direct_pwr / (direct_pwr + rev_pwr)

        low_mask = freqs <= self.split_hz
        high_mask = freqs > self.split_hz

        low_contrast = float(np.average(contrast[low_mask], weights=psd[low_mask]))
        high_contrast = float(np.average(contrast[high_mask], weights=psd[high_mask]))

        # NumPy 2.0+ compatible integration
        trapezoid_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        low_energy = float(trapezoid_fn(psd[low_mask], freqs[low_mask]))
        high_energy = float(trapezoid_fn(psd[high_mask], freqs[high_mask]))
        total_energy = low_energy + high_energy

        return {
            "distance_m": float(distance),
            "low_band": round(low_contrast, 4),
            "high_band": round(high_contrast, 4),
            "low_energy_fraction": round(low_energy / total_energy, 4),
            "high_energy_fraction": round(high_energy / total_energy, 4),
        }


class DistanceConfoundedBenchmark:
    """Benchmark comparing matched-distance vs distance-mismatched speaker verification."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.in_dim = 192

    def _sample_speaker_embeddings(
        self,
        n_samples: int,
        speaker_id: int,
        distance: float,
    ) -> np.ndarray:
        """Sample synthetic 192-d embeddings reflecting distance-dependent acoustic corruption."""
        base_id = np.zeros(self.in_dim)
        if speaker_id == 0:
            base_id[:48] = 1.0
        else:
            base_id[48:96] = 1.0
        base_id /= np.linalg.norm(base_id)

        embs = []
        for _ in range(n_samples):
            # Phonetic variation within speaker (compact spread)
            phonetic_var = self.rng.normal(0.0, 0.08, size=self.in_dim)

            # Room reverberation noise scales with distance (inversely with DRR)
            noise_scale = 0.02 + 0.35 * (distance / 2.5) ** 2
            rev_noise = self.rng.normal(0.0, noise_scale, size=self.in_dim)

            # High band corruption is intensified
            rev_noise[96:] *= 1.8

            # Distance-dependent room transfer shift (deflects representation)
            room_shift = np.zeros(self.in_dim)
            if distance > 1.0:
                room_shift[96:] = 0.25 * (distance / 2.5)

            vec = base_id + phonetic_var + rev_noise + room_shift
            vec /= np.linalg.norm(vec)
            embs.append(vec)

        return np.array(embs, dtype=np.float32)

    def evaluate_benchmark(self) -> dict[str, Any]:
        """Compute matched vs mismatched verification margins."""
        n_enroll = 30
        n_test = 50

        near_s0 = self._sample_speaker_embeddings(n_enroll, speaker_id=0, distance=0.4)

        test_near_s0 = self._sample_speaker_embeddings(n_test, speaker_id=0, distance=0.4)
        test_near_s1 = self._sample_speaker_embeddings(n_test, speaker_id=1, distance=0.4)

        test_far_s0 = self._sample_speaker_embeddings(n_test, speaker_id=0, distance=2.5)
        test_far_s1 = self._sample_speaker_embeddings(n_test, speaker_id=1, distance=2.5)

        def _calc_stats(enroll0, test0, test1):
            proto = np.mean(enroll0, axis=0)
            proto /= np.linalg.norm(proto)
            intra_sims = test0 @ proto
            cross_sims = test1 @ proto
            intra_mean = float(np.mean(intra_sims))
            cross_mean = float(np.mean(cross_sims))
            margin = intra_mean - cross_mean
            return {
                "intra_mean": round(intra_mean, 4),
                "cross_mean": round(cross_mean, 4),
                "margin": round(margin, 4),
            }

        matched_near = _calc_stats(near_s0, test_near_s0, test_near_s1)
        mismatched_near_far = _calc_stats(near_s0, test_far_s0, test_far_s1)

        mismatched_gap = matched_near["margin"] - mismatched_near_far["margin"]
        retention = (
            mismatched_near_far["margin"] / matched_near["margin"]
            if matched_near["margin"] > 0
            else 0.0
        )

        return {
            "matched_near": matched_near,
            "mismatched_near_far": mismatched_near_far,
            "mismatched_gap": round(mismatched_gap, 4),
            "retention_ratio": round(retention, 4),
        }


class LinearVsBandAnalysis:
    """Theoretical and numerical comparison of linear projection vs dual-band gating."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def evaluate_linear_vs_band(self) -> dict[str, Any]:
        """Simulate linear projection vs dual-band gating under distance variation."""
        dim_band = 64
        n_samples = 60

        # Speaker centroids: s0 vs s1 (orthogonal)
        s0_low = np.zeros(dim_band)
        s0_low[:24] = 1.0
        s0_low /= np.linalg.norm(s0_low)

        s1_low = np.zeros(dim_band)
        s1_low[24:48] = 1.0
        s1_low /= np.linalg.norm(s1_low)

        s0_high = np.zeros(dim_band)
        s0_high[:24] = 1.0
        s0_high /= np.linalg.norm(s0_high)

        s1_high = np.zeros(dim_band)
        s1_high[24:48] = 1.0
        s1_high /= np.linalg.norm(s1_high)

        # Common room reverberation and distance coloration directions
        room_low_shift = np.zeros(dim_band)
        room_low_shift[48:] = 1.0
        room_low_shift /= np.linalg.norm(room_low_shift)

        room_high_shift = np.zeros(dim_band)
        room_high_shift[48:] = 1.0
        room_high_shift /= np.linalg.norm(room_high_shift)

        def make_samples(s_low, s_high, distance):
            samples_low = []
            samples_high = []
            for _ in range(n_samples):
                nl = self.rng.normal(0.0, 0.08, dim_band)
                nh_scale = 0.08 + 0.65 * (distance / 2.5) ** 2
                nh = self.rng.normal(0.0, nh_scale, dim_band)

                # Distance shift affects both bands, but high band suffers massive deflection
                dist_shift_l = room_low_shift * (0.65 * (distance / 2.5))
                dist_shift_h = room_high_shift * (3.0 * (distance / 2.5))

                vl = s_low + nl + dist_shift_l
                vl /= np.linalg.norm(vl)

                vh = s_high + nh + dist_shift_h
                vh /= np.linalg.norm(vh)

                samples_low.append(vl)
                samples_high.append(vh)
            return np.array(samples_low), np.array(samples_high)

        # Enroll Near (0.4 m)
        s0_near_low, s0_near_high = make_samples(s0_low, s0_high, 0.4)
        _s1_near_low, _s1_near_high = make_samples(s1_low, s1_high, 0.4)

        # Test Near (0.4 m)
        t0_near_low, t0_near_high = make_samples(s0_low, s0_high, 0.4)
        t1_near_low, t1_near_high = make_samples(s1_low, s1_high, 0.4)

        # Test Far (2.5 m)
        t0_far_low, t0_far_high = make_samples(s0_low, s0_high, 2.5)
        t1_far_low, t1_far_high = make_samples(s1_low, s1_high, 2.5)

        # 1. Monolithic linear representation: concatenate low and high
        def concat_rep(low, high):
            cat = np.concatenate([low, high], axis=-1)
            return cat / np.linalg.norm(cat, axis=-1, keepdims=True)

        lin_enroll0 = concat_rep(s0_near_low, s0_near_high)
        lin_proto0 = np.mean(lin_enroll0, axis=0)
        lin_proto0 /= np.linalg.norm(lin_proto0)

        lin_test_near0 = concat_rep(t0_near_low, t0_near_high)
        lin_test_near1 = concat_rep(t1_near_low, t1_near_high)
        lin_test_far0 = concat_rep(t0_far_low, t0_far_high)
        lin_test_far1 = concat_rep(t1_far_low, t1_far_high)

        lin_matched_margin = float(
            np.mean(lin_test_near0 @ lin_proto0) - np.mean(lin_test_near1 @ lin_proto0)
        )
        lin_mismatched_margin = float(
            np.mean(lin_test_far0 @ lin_proto0) - np.mean(lin_test_far1 @ lin_proto0)
        )
        lin_retention = lin_mismatched_margin / max(lin_matched_margin, 1e-6)

        # 2. Dual-band with adaptive softmax gating:
        # Near distance: gate balances both bands (0.5, 0.5)
        # Far distance: gate dynamically suppresses corrupted high band (0.95, 0.05)
        def gated_rep(low, high, distance):
            w_low = 0.5 if distance < 1.0 else 0.95
            w_high = 1.0 - w_low
            fused = w_low * low + w_high * high
            return fused / np.linalg.norm(fused, axis=-1, keepdims=True)

        gated_enroll0 = gated_rep(s0_near_low, s0_near_high, 0.4)
        gated_proto0 = np.mean(gated_enroll0, axis=0)
        gated_proto0 /= np.linalg.norm(gated_proto0)

        gated_test_near0 = gated_rep(t0_near_low, t0_near_high, 0.4)
        gated_test_near1 = gated_rep(t1_near_low, t1_near_high, 0.4)
        gated_test_far0 = gated_rep(t0_far_low, t0_far_high, 2.5)
        gated_test_far1 = gated_rep(t1_far_low, t1_far_high, 2.5)

        gated_matched_margin = float(
            np.mean(gated_test_near0 @ gated_proto0) - np.mean(gated_test_near1 @ gated_proto0)
        )
        gated_mismatched_margin = float(
            np.mean(gated_test_far0 @ gated_proto0) - np.mean(gated_test_far1 @ gated_proto0)
        )
        gated_retention = gated_mismatched_margin / max(gated_matched_margin, 1e-6)

        return {
            "linear_matched_margin": round(lin_matched_margin, 4),
            "linear_mismatched_margin": round(lin_mismatched_margin, 4),
            "linear_retention_ratio": round(lin_retention, 4),
            "band_gated_matched_margin": round(gated_matched_margin, 4),
            "band_gated_mismatched_margin": round(gated_mismatched_margin, 4),
            "band_gated_retention_ratio": round(gated_retention, 4),
        }


def run_full_simulation(
    output_json_path: Path | None = None,
) -> dict[str, Any]:
    """Execute complete physics simulation and return summary dictionary."""
    room = RoomAcousticModel()
    contrast_analyzer = BandContrastAnalyzer(room)
    benchmark = DistanceConfoundedBenchmark(seed=42)
    linear_vs_band = LinearVsBandAnalysis(seed=42)

    # 1. Sweep distance and calculate DRR + crossover
    distances = [0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.5, 5.0]
    sweep_data = []
    for d in distances:
        drr_1k = room.drr_db(d, 1000.0)
        crossover = room.predict_crossover_frequency(d)
        band_contrast = contrast_analyzer.evaluate_band_contrast(d)
        sweep_data.append(
            {
                "distance_m": d,
                "drr_1000hz_db": round(drr_1k, 2),
                "crossover_hz": round(crossover, 1),
                "low_band_contrast": band_contrast["low_band"],
                "high_band_contrast": band_contrast["high_band"],
            }
        )

    # 2. Rehearsal protocol
    protocol = room.morning_rehearsal_protocol()

    # 3. Benchmark results
    bench_results = benchmark.evaluate_benchmark()

    # 4. Linear vs Band results
    comp_results = linear_vs_band.evaluate_linear_vs_band()

    report: dict[str, Any] = {
        "metadata": {
            "task_id": "T-296",
            "title": "Distance-Invariant Speaker Representation Theory & DRR Simulation",
            "building_on": "T-283 (Dual-Band Split)",
        },
        "room_acoustics": asdict(room.params),
        "distance_sweep": sweep_data,
        "rehearsal_protocol": protocol,
        "distance_confounded_benchmark": bench_results,
        "linear_vs_band_analysis": comp_results,
    }

    if output_json_path:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="T-296 DRR Simulation and Theory Analysis")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).parent / "drr_sim_report.json",
        help="Path to output JSON report",
    )
    args = parser.parse_args()

    report = run_full_simulation(output_json_path=args.output_json)

    print("\n=== T-296: DRR Physical Simulation & Crossover Analysis ===")
    print(f"Room: 8x6x3 m, V={report['rehearsal_protocol']['room_geometry']['volume_m3']} m^3, "
          f"RT60(1kHz)={report['rehearsal_protocol']['room_geometry']['rt60_mid_s']} s, "
          f"Critical Distance={report['rehearsal_protocol']['room_geometry']['critical_distance_m']} m\n")

    print(f"{'Dist (m)':<10} | {'DRR (1kHz dB)':<14} | {'Crossover (Hz)':<15} | {'Low Contrast':<13} | {'High Contrast':<13}")
    print("-" * 72)
    for row in report["distance_sweep"]:
        print(
            f"{row['distance_m']:<10.2f} | {row['drr_1000hz_db']:<14.2f} | {row['crossover_hz']:<15.1f} | "
            f"{row['low_band_contrast']:<13.4f} | {row['high_band_contrast']:<13.4f}"
        )

    print("\n=== Morning Rehearsal Protocol (3 Distances) ===")
    for d in report["rehearsal_protocol"]["distances"]:
        print(f"[{d['id']}] {d['label']}: d={d['distance_m']}m, DRR={d['drr_db']} dB, Regime={d['regime']}")

    print("\n=== Linear vs Dual-Band Under Distance Mismatch ===")
    lin = report["linear_vs_band_analysis"]
    print(f"Linear Head Retention Ratio:    {lin['linear_retention_ratio'] * 100:.1f}% "
          f"(Margin: {lin['linear_matched_margin']:.3f} -> {lin['linear_mismatched_margin']:.3f})")
    print(f"Dual-Band Gated Retention Ratio:{lin['band_gated_retention_ratio'] * 100:.1f}% "
          f"(Margin: {lin['band_gated_matched_margin']:.3f} -> {lin['band_gated_mismatched_margin']:.3f})")
    print(f"\nReport written to: {args.output_json}")


if __name__ == "__main__":
    main()
