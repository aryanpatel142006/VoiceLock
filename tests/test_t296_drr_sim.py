"""Unit tests for T-296 distance-invariance theory and DRR simulation (drr_sim.py).

Verifies:
1. Room acoustics: Sabine RT60, room constant, critical distance, and DRR vs distance.
2. Frequency-dependent absorption and atmospheric attenuation.
3. Crossover frequency prediction vs T-283's 2 kHz band split.
4. Distance-mismatched evaluation protocol (matched vs mismatched margin degradation).
5. Mathematical linear projection condition vs dual-band gating failure modes.
6. Morning rehearsal 3-distance predicted DRR values and protocol output schema.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments" / "theory" / "T-296"
_MODULE_PATH = _EXP_DIR / "drr_sim.py"


def _load_drr_sim_module():
    spec = importlib.util.spec_from_file_location("drr_sim", _MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {_MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["drr_sim"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sim():
    return _load_drr_sim_module()


def test_room_acoustics_geometry_and_rt60(sim):
    """Room parameters compute volume, area, and Sabine RT60 correctly."""
    room = sim.RoomAcousticModel(length=8.0, width=6.0, height=3.0)
    assert np.isclose(room.volume, 144.0)
    assert np.isclose(room.surface_area, 180.0)

    # Mid-frequency RT60 for typical demo room (1000 Hz) should be ~0.4 - 0.6 s
    rt60_1k = room.rt60(1000.0)
    assert 0.35 <= rt60_1k <= 0.65


def test_drr_monotonically_decreases_with_distance(sim):
    """Direct-to-reverberant ratio decreases monotonically with distance according to 1/d^2."""
    room = sim.RoomAcousticModel(length=8.0, width=6.0, height=3.0)
    distances = np.array([0.3, 0.6, 1.2, 2.4, 4.8])
    drr_vals = [room.drr_db(d, freq=1000.0) for d in distances]

    # Check strictly monotonically decreasing
    for i in range(len(drr_vals) - 1):
        assert drr_vals[i] > drr_vals[i + 1]

    # Doubling distance should drop direct power by ~6 dB (ignoring tiny air absorption diff)
    for i in range(len(drr_vals) - 1):
        drop = drr_vals[i] - drr_vals[i + 1]
        assert np.isclose(drop, 6.02, atol=0.5)


def test_critical_distance_corresponds_to_zero_drr(sim):
    """At critical distance d_c, DRR should be 0 dB (+-0.1 dB)."""
    room = sim.RoomAcousticModel(length=8.0, width=6.0, height=3.0)
    dc = room.critical_distance(freq=1000.0)
    assert 1.0 <= dc <= 3.5

    drr_at_dc = room.drr_db(dc, freq=1000.0)
    assert np.isclose(drr_at_dc, 0.0, atol=0.1)


def test_crossover_frequency_prediction_vs_t283_split(sim):
    """At conversational distance (~1.2 - 1.5 m), predicted crossover is near T-283's 2 kHz split."""
    room = sim.RoomAcousticModel(length=8.0, width=6.0, height=3.0)
    # At near distance (0.4 m), direct sound dominates across speech band (> 4 kHz)
    cross_near = room.predict_crossover_frequency(distance=0.4)
    assert cross_near >= 3500.0

    # At conversational distance (1.2 - 1.5 m), crossover drops into the 1.5 - 2.5 kHz window
    cross_mid = room.predict_crossover_frequency(distance=1.3)
    assert 1500.0 <= cross_mid <= 2500.0

    # At far distance (2.5 m), crossover degrades below 1200 Hz
    cross_far = room.predict_crossover_frequency(distance=2.5)
    assert cross_far < 1200.0


def test_band_energy_and_formant_contrast_decay(sim):
    """Formant contrast in high band (>2 kHz) decays faster than low band (<2 kHz)."""
    analyzer = sim.BandContrastAnalyzer()
    near_contrast = analyzer.evaluate_band_contrast(distance=0.4)
    far_contrast = analyzer.evaluate_band_contrast(distance=2.5)

    # In near condition, both bands maintain high contrast (> 0.75)
    assert near_contrast["low_band"] > 0.80
    assert near_contrast["high_band"] > 0.75

    # In far condition, high-band contrast degrades significantly more than low band
    assert far_contrast["low_band"] > far_contrast["high_band"]
    high_decay = near_contrast["high_band"] - far_contrast["high_band"]
    low_decay = near_contrast["low_band"] - far_contrast["low_band"]
    assert high_decay > low_decay


def test_distance_confounded_benchmark_degradation(sim):
    """Cross-distance evaluation exhibits severe margin collapse compared to matched eval."""
    bench = sim.DistanceConfoundedBenchmark(seed=42)
    results = bench.evaluate_benchmark()

    # Matched near margin should be high
    matched_near_margin = results["matched_near"]["margin"]
    assert matched_near_margin > 0.5

    # Mismatched near-enroll / far-test margin should collapse significantly
    mismatched_margin = results["mismatched_near_far"]["margin"]
    assert mismatched_margin < matched_near_margin
    assert results["mismatched_gap"] > 0.2
    assert results["retention_ratio"] < 0.75


def test_linear_projection_condition_vs_band_gating(sim):
    """Linear projection fails when distance variation intersects speaker subspace."""
    sim_proof = sim.LinearVsBandAnalysis(seed=42)
    eval_res = sim_proof.evaluate_linear_vs_band()

    # Linear projection without band decomposition suffers severe test degradation under distance shift
    assert eval_res["linear_matched_margin"] > 0.5
    assert eval_res["linear_mismatched_margin"] < eval_res["linear_matched_margin"] * 0.6

    # Gated dual-band maintains significantly higher retention ratio
    assert eval_res["band_gated_retention_ratio"] > eval_res["linear_retention_ratio"]


def test_morning_rehearsal_protocol_specification(sim):
    """Morning rehearsal protocol defines 3 concrete distances with valid DRR values."""
    room = sim.RoomAcousticModel(length=8.0, width=6.0, height=3.0)
    protocol = room.morning_rehearsal_protocol()

    assert "distances" in protocol
    dists = protocol["distances"]
    assert len(dists) == 3

    # D1 (Near): 0.4 m, positive high DRR (> +7 dB)
    assert dists[0]["id"] == "D1_near"
    assert np.isclose(dists[0]["distance_m"], 0.4)
    assert dists[0]["drr_db"] >= 7.0

    # D2 (Mid): 1.2 m, transitional DRR (-1 to +4 dB)
    assert dists[1]["id"] == "D2_mid"
    assert np.isclose(dists[1]["distance_m"], 1.2)
    assert -2.0 <= dists[1]["drr_db"] <= 5.0

    # D3 (Far): 2.5 m, reverberant dominated (negative DRR < -3 dB)
    assert dists[2]["id"] == "D3_far"
    assert np.isclose(dists[2]["distance_m"], 2.5)
    assert dists[2]["drr_db"] <= -3.0
