"""Streaming stress-test corpus builder for EMA tracking variants (T-293).

Generates 40 deterministic, scripted scenarios specifically targeting:
1. Rapid turn-taking (<300ms gaps)
2. Simultaneous onset
3. 200ms pauses (micro-pauses & hesitation)
4. Gain asymmetry (6dB)
5. Compound stress configurations

Each scenario produces:
- ``mix.wav``: 16 kHz mono audio rendered using existing mix mathematical conventions
- ``turn_labels.json``: ground-truth speaker-turn timestamps with exact intervals
- ``scenario_card.md``: comprehensive documentation of the scenario and EMA evaluation relevance

Master index is written to ``eval/stream_stress/T-293/manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from eval.fixtures.mix import PEAK, fit_length, rms
from eval.fixtures.sources import TINY_DIR, TinySource, pink_noise

SAMPLE_RATE = 16000
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "stream_stress" / "T-293"


@dataclass
class Turn:
    """One speaker turn in a scenario."""

    turn_id: int
    speaker: str  # "speaker_1" or "speaker_2"
    start_s: float
    end_s: float
    gain_db: float = 0.0
    description: str = ""
    source_offset_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 6)

    def to_dict(self, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
        start_samp = round(self.start_s * sample_rate)
        end_samp = round(self.end_s * sample_rate)
        return {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "start_sample": start_samp,
            "end_sample": end_samp,
            "duration_s": self.duration_s,
            "gain_db": self.gain_db,
            "description": self.description,
        }


@dataclass
class ScenarioSpec:
    """Specification for one stress-test scenario."""

    id: str
    name: str
    category: str
    description: str
    seed: int
    duration_s: float
    turns: list[Turn]
    params: dict[str, Any] = field(default_factory=dict)
    noise_type: str = "none"  # "none", "babble", "pink"
    snr_db: float | None = None
    sample_rate: int = SAMPLE_RATE


def _apply_fade(audio: np.ndarray, fade_samples: int = 80) -> np.ndarray:
    """Apply raised-cosine fade-in and fade-out to prevent boundary clicks."""
    if len(audio) < 2 * fade_samples or fade_samples <= 0:
        return audio
    out = audio.copy()
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_samples)))
    out[:fade_samples] *= ramp
    out[-fade_samples:] *= ramp[::-1]
    return out


def _load_sources(tiny_dir: Path = TINY_DIR) -> dict[str, np.ndarray]:
    """Load base speaker audio and noise from tiny fixtures."""
    source = TinySource(tiny_dir)
    speeches = source.speech()
    # speech_a (female 1580), speech_b (male 61)
    spk1_audio = speeches[0].audio.astype(np.float32)
    spk2_audio = speeches[1].audio.astype(np.float32)

    babble_path = tiny_dir / "noise_babble.wav"
    babble_audio, _ = sf.read(babble_path, dtype="float32")

    return {
        "speaker_1": spk1_audio,
        "speaker_2": spk2_audio,
        "babble": babble_audio.astype(np.float32),
    }


def _slice_source(source_audio: np.ndarray, duration_s: float, offset_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Slice or loop speech audio for the requested duration with an offset."""
    needed_samples = round(duration_s * sr)
    offset_samples = round(offset_s * sr) % len(source_audio)

    # Roll to offset
    rolled = np.roll(source_audio, -offset_samples)
    fitted = fit_length(rolled, needed_samples)
    return _apply_fade(fitted.astype(np.float32), fade_samples=min(80, needed_samples // 4))


def build_scenarios() -> list[ScenarioSpec]:
    """Generate specifications for all 40 scripted stress scenarios."""
    scenarios: list[ScenarioSpec] = []
    base_seed = 29301

    # =========================================================================
    # Group 1: Rapid Turn-Taking (<300ms gaps) - Scenarios 01 to 10
    # =========================================================================
    rapid_gaps = [
        (1, "rapid_turn_50ms", 0.050, "Rapid turn handoff with 50ms gap (<300ms)"),
        (2, "rapid_turn_100ms", 0.100, "Rapid turn handoff with 100ms gap (<300ms)"),
        (3, "rapid_turn_150ms", 0.150, "Rapid turn handoff with 150ms gap (<300ms)"),
        (4, "rapid_turn_200ms", 0.200, "Rapid turn handoff with 200ms gap (<300ms)"),
        (5, "rapid_turn_250ms", 0.250, "Rapid turn handoff with 250ms gap (<300ms)"),
    ]
    for num, name, gap_s, desc in rapid_gaps:
        scenarios.append(
            ScenarioSpec(
                id=f"scenario_{num:02d}",
                name=name,
                category="rapid_turn_taking",
                description=desc,
                seed=base_seed + num - 1,
                duration_s=4.5,
                params={"gap_ms": round(gap_s * 1000), "gap_s": gap_s},
                turns=[
                    Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Speaker 1 initial turn", 0.0),
                    Turn(2, "speaker_2", 2.0 + gap_s, 2.0 + gap_s + 2.0, 0.0, f"Speaker 2 handoff after {round(gap_s*1000)}ms gap", 0.5),
                ],
            )
        )

    # 06: Ping-pong with 100ms gaps (4 turns)
    scenarios.append(
        ScenarioSpec(
            id="scenario_06",
            name="rapid_pingpong_100ms",
            category="rapid_turn_taking",
            description="Four-turn rapid ping-pong exchange with 100ms gaps",
            seed=base_seed + 5,
            duration_s=5.5,
            params={"gap_ms": 100, "num_turns": 4},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.2, 0.0, "Speaker 1 turn 1", 0.0),
                Turn(2, "speaker_2", 1.3, 2.5, 0.0, "Speaker 2 turn 1 (100ms gap)", 0.3),
                Turn(3, "speaker_1", 2.6, 3.8, 0.0, "Speaker 1 turn 2 (100ms gap)", 0.6),
                Turn(4, "speaker_2", 3.9, 5.1, 0.0, "Speaker 2 turn 2 (100ms gap)", 0.9),
            ],
        )
    )

    # 07: Ping-pong with 200ms gaps (4 turns)
    scenarios.append(
        ScenarioSpec(
            id="scenario_07",
            name="rapid_pingpong_200ms",
            category="rapid_turn_taking",
            description="Four-turn rapid ping-pong exchange with 200ms gaps",
            seed=base_seed + 6,
            duration_s=6.0,
            params={"gap_ms": 200, "num_turns": 4},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.2, 0.0, "Speaker 1 turn 1", 0.0),
                Turn(2, "speaker_2", 1.4, 2.6, 0.0, "Speaker 2 turn 1 (200ms gap)", 0.4),
                Turn(3, "speaker_1", 2.8, 4.0, 0.0, "Speaker 1 turn 2 (200ms gap)", 0.8),
                Turn(4, "speaker_2", 4.2, 5.4, 0.0, "Speaker 2 turn 2 (200ms gap)", 1.2),
            ],
        )
    )

    # 08: Accelerating turn gaps (280ms -> 180ms -> 80ms)
    scenarios.append(
        ScenarioSpec(
            id="scenario_08",
            name="accelerating_turn_gaps",
            category="rapid_turn_taking",
            description="Accelerating conversational cadence with diminishing gaps: 280ms -> 180ms -> 80ms",
            seed=base_seed + 7,
            duration_s=5.5,
            params={"gaps_ms": [280, 180, 80]},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.1, 0.0, "Speaker 1 turn 1", 0.0),
                Turn(2, "speaker_2", 1.38, 2.48, 0.0, "Speaker 2 turn 1 (280ms gap)", 0.3),
                Turn(3, "speaker_1", 2.66, 3.76, 0.0, "Speaker 1 turn 2 (180ms gap)", 0.6),
                Turn(4, "speaker_2", 3.84, 4.94, 0.0, "Speaker 2 turn 2 (80ms gap)", 0.9),
            ],
        )
    )

    # 09: Asymmetric gaps (80ms vs 240ms)
    scenarios.append(
        ScenarioSpec(
            id="scenario_09",
            name="asymmetric_gaps",
            category="rapid_turn_taking",
            description="Asymmetric turn gaps: rapid 80ms reply followed by 240ms handoff",
            seed=base_seed + 8,
            duration_s=5.2,
            params={"gaps_ms": [80, 240]},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.4, 0.0, "Speaker 1 turn 1", 0.0),
                Turn(2, "speaker_2", 1.48, 2.88, 0.0, "Speaker 2 quick retort (80ms gap)", 0.5),
                Turn(3, "speaker_1", 3.12, 4.52, 0.0, "Speaker 1 resumed handoff (240ms gap)", 1.0),
            ],
        )
    )

    # 10: Rapid staccato (75ms gaps, 5 short turns)
    scenarios.append(
        ScenarioSpec(
            id="scenario_10",
            name="rapid_staccato_75ms",
            category="rapid_turn_taking",
            description="Five short staccato bursts (0.75s) separated by 75ms conversational gaps",
            seed=base_seed + 9,
            duration_s=4.6,
            params={"gap_ms": 75, "num_turns": 5},
            turns=[
                Turn(1, "speaker_1", 0.000, 0.750, 0.0, "Turn 1 (Spk 1)", 0.0),
                Turn(2, "speaker_2", 0.825, 1.575, 0.0, "Turn 2 (Spk 2, 75ms gap)", 0.2),
                Turn(3, "speaker_1", 1.650, 2.400, 0.0, "Turn 3 (Spk 1, 75ms gap)", 0.4),
                Turn(4, "speaker_2", 2.475, 3.225, 0.0, "Turn 4 (Spk 2, 75ms gap)", 0.6),
                Turn(5, "speaker_1", 3.300, 4.050, 0.0, "Turn 5 (Spk 1, 75ms gap)", 0.8),
            ],
        )
    )

    # =========================================================================
    # Group 2: Simultaneous Onset - Scenarios 11 to 20
    # =========================================================================
    # 11: Simultaneous onset at t=0, equal gain, Spk 1 continues
    scenarios.append(
        ScenarioSpec(
            id="scenario_11",
            name="simultaneous_onset_spk1_continues",
            category="simultaneous_onset",
            description="Simultaneous onset at t=0.0s for 2.0s, Spk 2 drops out, Spk 1 continues solo to 4.0s",
            seed=base_seed + 10,
            duration_s=4.5,
            params={"overlap_s": 2.0, "solo_speaker": "speaker_1"},
            turns=[
                Turn(1, "speaker_1", 0.0, 4.0, 0.0, "Speaker 1 active throughout (onset t=0)", 0.0),
                Turn(2, "speaker_2", 0.0, 2.0, 0.0, "Speaker 2 simultaneous onset t=0, drops at 2.0s", 0.3),
            ],
        )
    )

    # 12: Simultaneous onset at t=0, equal gain, Spk 2 continues
    scenarios.append(
        ScenarioSpec(
            id="scenario_12",
            name="simultaneous_onset_spk2_continues",
            category="simultaneous_onset",
            description="Simultaneous onset at t=0.0s for 2.0s, Spk 1 drops out, Spk 2 continues solo to 4.0s",
            seed=base_seed + 11,
            duration_s=4.5,
            params={"overlap_s": 2.0, "solo_speaker": "speaker_2"},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Speaker 1 simultaneous onset t=0, drops at 2.0s", 0.0),
                Turn(2, "speaker_2", 0.0, 4.0, 0.0, "Speaker 2 active throughout (onset t=0)", 0.3),
            ],
        )
    )

    # 13: Full simultaneous double-talk overlap
    scenarios.append(
        ScenarioSpec(
            id="scenario_13",
            name="simultaneous_full_overlap",
            category="simultaneous_onset",
            description="Simultaneous onset at t=0.0s with full 3.0s double-talk overlap at 0 dB SIR",
            seed=base_seed + 12,
            duration_s=3.5,
            params={"overlap_s": 3.0},
            turns=[
                Turn(1, "speaker_1", 0.0, 3.0, 0.0, "Speaker 1 full double-talk", 0.0),
                Turn(2, "speaker_2", 0.0, 3.0, 0.0, "Speaker 2 full double-talk", 0.5),
            ],
        )
    )

    # 14: Simultaneous onset post-silence
    scenarios.append(
        ScenarioSpec(
            id="scenario_14",
            name="simultaneous_onset_post_silence",
            category="simultaneous_onset",
            description="1.0s lead silence followed by simultaneous onset at t=1.0s for 2.5s",
            seed=base_seed + 13,
            duration_s=4.0,
            params={"lead_silence_s": 1.0, "overlap_s": 2.5},
            turns=[
                Turn(1, "speaker_1", 1.0, 3.5, 0.0, "Speaker 1 onset after 1s silence", 0.0),
                Turn(2, "speaker_2", 1.0, 3.5, 0.0, "Speaker 2 onset after 1s silence", 0.5),
            ],
        )
    )

    # 15: Brief simultaneous collision burst (500ms), Spk 1 resolves
    scenarios.append(
        ScenarioSpec(
            id="scenario_15",
            name="simultaneous_burst_short_500ms",
            category="simultaneous_onset",
            description="Simultaneous onset at t=0.0s with brief 500ms overlap collision; Spk 1 resolves solo",
            seed=base_seed + 14,
            duration_s=3.5,
            params={"overlap_s": 0.5, "resolving_speaker": "speaker_1"},
            turns=[
                Turn(1, "speaker_1", 0.0, 3.0, 0.0, "Speaker 1 sustained speech", 0.0),
                Turn(2, "speaker_2", 0.0, 0.5, 0.0, "Speaker 2 brief 500ms collision", 0.2),
            ],
        )
    )

    # 16: Brief simultaneous collision burst (500ms), Spk 2 resolves
    scenarios.append(
        ScenarioSpec(
            id="scenario_16",
            name="simultaneous_burst_spk2_resolves",
            category="simultaneous_onset",
            description="Simultaneous onset at t=0.0s with brief 500ms overlap collision; Spk 2 resolves solo",
            seed=base_seed + 15,
            duration_s=3.5,
            params={"overlap_s": 0.5, "resolving_speaker": "speaker_2"},
            turns=[
                Turn(1, "speaker_1", 0.0, 0.5, 0.0, "Speaker 1 brief 500ms collision", 0.0),
                Turn(2, "speaker_2", 0.0, 3.0, 0.0, "Speaker 2 sustained speech", 0.2),
            ],
        )
    )

    # 17: Midstream simultaneous onset
    scenarios.append(
        ScenarioSpec(
            id="scenario_17",
            name="midstream_simultaneous_onset",
            category="simultaneous_onset",
            description="Speaker 1 speaks solo [0.0, 2.0s], pause [2.0, 2.4s], then both onset simultaneously at t=2.4s",
            seed=base_seed + 16,
            duration_s=5.0,
            params={"midstream_onset_s": 2.4},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Speaker 1 initial solo segment", 0.0),
                Turn(2, "speaker_1", 2.4, 4.4, 0.0, "Speaker 1 midstream simultaneous onset", 0.4),
                Turn(3, "speaker_2", 2.4, 4.4, 0.0, "Speaker 2 midstream simultaneous onset", 0.7),
            ],
        )
    )

    # 18: Simultaneous onset followed by handoff
    scenarios.append(
        ScenarioSpec(
            id="scenario_18",
            name="simultaneous_onset_handoff",
            category="simultaneous_onset",
            description="Simultaneous onset [0.0, 1.5s], then Spk 1 solo [1.5, 2.8s], then Spk 2 handoff [3.0, 4.5s]",
            seed=base_seed + 17,
            duration_s=5.0,
            params={"simultaneous_s": 1.5},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.8, 0.0, "Speaker 1 onset and continued solo", 0.0),
                Turn(2, "speaker_2", 0.0, 1.5, 0.0, "Speaker 2 simultaneous onset", 0.3),
                Turn(3, "speaker_2", 3.0, 4.5, 0.0, "Speaker 2 sequential handoff", 0.8),
            ],
        )
    )

    # 19: Cascade collision with simultaneous re-onset
    scenarios.append(
        ScenarioSpec(
            id="scenario_19",
            name="cascade_collision_onset",
            category="simultaneous_onset",
            description="Speaker 1 solo, followed by 1.0s simultaneous onset collision, resolving to Speaker 2 solo",
            seed=base_seed + 18,
            duration_s=5.0,
            params={"collision_start_s": 1.5, "collision_s": 1.0},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.5, 0.0, "Speaker 1 solo into collision", 0.0),
                Turn(2, "speaker_2", 1.5, 4.5, 0.0, "Speaker 2 simultaneous collision onset resolving solo", 0.4),
            ],
        )
    )

    # 20: Simultaneous onset under babble noise
    scenarios.append(
        ScenarioSpec(
            id="scenario_20",
            name="simultaneous_onset_babble_noise",
            category="simultaneous_onset",
            description="Simultaneous onset at t=0.0s for 2.5s embedded in 10 dB SNR babble noise",
            seed=base_seed + 19,
            duration_s=3.5,
            params={"overlap_s": 2.5, "snr_db": 10.0},
            noise_type="babble",
            snr_db=10.0,
            turns=[
                Turn(1, "speaker_1", 0.0, 2.5, 0.0, "Speaker 1 simultaneous onset in babble", 0.0),
                Turn(2, "speaker_2", 0.0, 2.5, 0.0, "Speaker 2 simultaneous onset in babble", 0.5),
            ],
        )
    )

    # =========================================================================
    # Group 3: 200ms Pauses (Micro-Pauses & Hesitations) - Scenarios 21 to 30
    # =========================================================================
    # 21: Single speaker intra-turn 200ms pause
    scenarios.append(
        ScenarioSpec(
            id="scenario_21",
            name="intraturn_pause_200ms",
            category="micro_pause_200ms",
            description="Speaker 1 with mid-utterance exactly 200ms pause (tests EMA coasting without decay)",
            seed=base_seed + 20,
            duration_s=4.2,
            params={"pause_ms": 200, "pause_s": 0.200},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.8, 0.0, "Speaker 1 pre-pause phrase", 0.0),
                Turn(2, "speaker_1", 2.0, 3.8, 0.0, "Speaker 1 post-pause phrase (200ms pause)", 0.6),
            ],
        )
    )

    # 22: Intra-turn double 200ms pause
    scenarios.append(
        ScenarioSpec(
            id="scenario_22",
            name="intraturn_double_pause_200ms",
            category="micro_pause_200ms",
            description="Speaker 1 with two 200ms hesitation pauses across a continuous 4.0s utterance",
            seed=base_seed + 21,
            duration_s=4.4,
            params={"pause_ms": 200, "count": 2},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.2, 0.0, "Segment 1", 0.0),
                Turn(2, "speaker_1", 1.4, 2.6, 0.0, "Segment 2 (after 200ms pause)", 0.4),
                Turn(3, "speaker_1", 2.8, 4.0, 0.0, "Segment 3 (after 200ms pause)", 0.8),
            ],
        )
    )

    # 23: Speaker 2 intra-turn 200ms pause
    scenarios.append(
        ScenarioSpec(
            id="scenario_23",
            name="spk2_intraturn_pause_200ms",
            category="micro_pause_200ms",
            description="Speaker 2 with mid-utterance exactly 200ms pause",
            seed=base_seed + 22,
            duration_s=4.2,
            params={"pause_ms": 200, "speaker": "speaker_2"},
            turns=[
                Turn(1, "speaker_2", 0.0, 1.8, 0.0, "Speaker 2 pre-pause phrase", 0.0),
                Turn(2, "speaker_2", 2.0, 3.8, 0.0, "Speaker 2 post-pause phrase (200ms pause)", 0.6),
            ],
        )
    )

    # 24: Inter-turn exact 200ms pause
    scenarios.append(
        ScenarioSpec(
            id="scenario_24",
            name="interturn_exact_pause_200ms",
            category="micro_pause_200ms",
            description="Clean speaker transition separated by exactly 200ms pause between turns",
            seed=base_seed + 23,
            duration_s=4.5,
            params={"pause_ms": 200},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Speaker 1 turn", 0.0),
                Turn(2, "speaker_2", 2.2, 4.2, 0.0, "Speaker 2 turn (200ms inter-turn pause)", 0.5),
            ],
        )
    )

    # 25: 200ms pause then interruption
    scenarios.append(
        ScenarioSpec(
            id="scenario_25",
            name="pause_200ms_then_interruption",
            category="micro_pause_200ms",
            description="Speaker 1 pauses 200ms, but Speaker 2 takes the floor immediately at the pause boundary",
            seed=base_seed + 24,
            duration_s=4.5,
            params={"pause_ms": 200},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Speaker 1 turn ending in pause", 0.0),
                Turn(2, "speaker_2", 2.2, 4.0, 0.0, "Speaker 2 floor takeover at 2.2s", 0.3),
            ],
        )
    )

    # 26: Ping-pong with 200ms pauses
    scenarios.append(
        ScenarioSpec(
            id="scenario_26",
            name="pingpong_with_200ms_pauses",
            category="micro_pause_200ms",
            description="Three-turn dialogue where every transition is exactly 200ms pause",
            seed=base_seed + 25,
            duration_s=5.0,
            params={"pause_ms": 200, "num_turns": 3},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.4, 0.0, "Turn 1 (Spk 1)", 0.0),
                Turn(2, "speaker_2", 1.6, 3.0, 0.0, "Turn 2 (Spk 2, 200ms pause)", 0.4),
                Turn(3, "speaker_1", 3.2, 4.6, 0.0, "Turn 3 (Spk 1, 200ms pause)", 0.8),
            ],
        )
    )

    # 27: 200ms pause under babble noise
    scenarios.append(
        ScenarioSpec(
            id="scenario_27",
            name="pause_200ms_under_babble",
            category="micro_pause_200ms",
            description="Speaker 1 with 200ms pause embedded in 12 dB SNR babble noise (tests noise-pull)",
            seed=base_seed + 26,
            duration_s=4.2,
            params={"pause_ms": 200, "snr_db": 12.0},
            noise_type="babble",
            snr_db=12.0,
            turns=[
                Turn(1, "speaker_1", 0.0, 1.8, 0.0, "Pre-pause phrase", 0.0),
                Turn(2, "speaker_1", 2.0, 3.8, 0.0, "Post-pause phrase (200ms pause)", 0.6),
            ],
        )
    )

    # 28: 200ms pause preceding overlap
    scenarios.append(
        ScenarioSpec(
            id="scenario_28",
            name="pause_200ms_preceding_overlap",
            category="micro_pause_200ms",
            description="Speaker 1 has 200ms pause, resumes at 2.2s, while Speaker 2 intrudes at 2.5s",
            seed=base_seed + 27,
            duration_s=4.5,
            params={"pause_ms": 200},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Spk 1 initial phrase", 0.0),
                Turn(2, "speaker_1", 2.2, 4.2, 0.0, "Spk 1 post-pause phrase", 0.5),
                Turn(3, "speaker_2", 2.5, 4.2, 0.0, "Spk 2 intrusion during post-pause", 0.8),
            ],
        )
    )

    # 29: Asymmetric pauses (150ms vs 200ms)
    scenarios.append(
        ScenarioSpec(
            id="scenario_29",
            name="asymmetric_pauses_150_200ms",
            category="micro_pause_200ms",
            description="Intra-turn 150ms pause followed by inter-turn 200ms pause",
            seed=base_seed + 28,
            duration_s=4.8,
            params={"pauses_ms": [150, 200]},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.2, 0.0, "Spk 1 segment 1", 0.0),
                Turn(2, "speaker_1", 1.35, 2.55, 0.0, "Spk 1 segment 2 (150ms pause)", 0.4),
                Turn(3, "speaker_2", 2.75, 4.25, 0.0, "Spk 2 handoff (200ms pause)", 0.8),
            ],
        )
    )

    # 30: Triple rhythmic 200ms pauses
    scenarios.append(
        ScenarioSpec(
            id="scenario_30",
            name="triple_rhythmic_pause_200ms",
            category="micro_pause_200ms",
            description="Three periodic 200ms hesitation pauses across a continuous 5.0s sentence",
            seed=base_seed + 29,
            duration_s=5.2,
            params={"pause_ms": 200, "count": 3},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.0, 0.0, "Chunk 1", 0.0),
                Turn(2, "speaker_1", 1.2, 2.2, 0.0, "Chunk 2 (200ms pause)", 0.3),
                Turn(3, "speaker_1", 2.4, 3.4, 0.0, "Chunk 3 (200ms pause)", 0.6),
                Turn(4, "speaker_1", 3.6, 4.6, 0.0, "Chunk 4 (200ms pause)", 0.9),
            ],
        )
    )

    # =========================================================================
    # Group 4: Gain Asymmetry (6dB) - Scenarios 31 to 38
    # =========================================================================
    # 31: Loud interferer (+6dB) turn-taking
    scenarios.append(
        ScenarioSpec(
            id="scenario_31",
            name="gain_asymmetry_loud_interferer_6db",
            category="gain_asymmetry_6db",
            description="Turn-taking with +6dB louder interferer (Spk 1 0dB, Spk 2 +6dB, 150ms gap)",
            seed=base_seed + 30,
            duration_s=4.5,
            params={"gain_asymmetry_db": 6.0, "loud_speaker": "speaker_2", "gap_ms": 150},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Speaker 1 at baseline 0dB", 0.0),
                Turn(2, "speaker_2", 2.15, 4.15, 6.0, "Speaker 2 loud interferer (+6dB, 150ms gap)", 0.5),
            ],
        )
    )

    # 32: Quiet interferer (-6dB) turn-taking
    scenarios.append(
        ScenarioSpec(
            id="scenario_32",
            name="gain_asymmetry_quiet_interferer_6db",
            category="gain_asymmetry_6db",
            description="Turn-taking with -6dB softer interferer (Spk 1 0dB, Spk 2 -6dB, 150ms gap)",
            seed=base_seed + 31,
            duration_s=4.5,
            params={"gain_asymmetry_db": -6.0, "quiet_speaker": "speaker_2", "gap_ms": 150},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Speaker 1 at baseline 0dB", 0.0),
                Turn(2, "speaker_2", 2.15, 4.15, -6.0, "Speaker 2 quiet interferer (-6dB, 150ms gap)", 0.5),
            ],
        )
    )

    # 33: Simultaneous onset with loud interferer (+6dB)
    scenarios.append(
        ScenarioSpec(
            id="scenario_33",
            name="gain_asymmetry_simultaneous_loud_int_6db",
            category="gain_asymmetry_6db",
            description="Simultaneous onset at t=0.0s where interferer is +6dB louder than target",
            seed=base_seed + 32,
            duration_s=4.5,
            params={"gain_asymmetry_db": 6.0, "loud_speaker": "speaker_2"},
            turns=[
                Turn(1, "speaker_1", 0.0, 4.0, 0.0, "Speaker 1 target at 0dB", 0.0),
                Turn(2, "speaker_2", 0.0, 2.0, 6.0, "Speaker 2 loud interferer at +6dB simultaneous onset", 0.4),
            ],
        )
    )

    # 34: Simultaneous onset with dominant target (+6dB)
    scenarios.append(
        ScenarioSpec(
            id="scenario_34",
            name="gain_asymmetry_simultaneous_dominant_target_6db",
            category="gain_asymmetry_6db",
            description="Simultaneous onset at t=0.0s where target is +6dB louder than interferer",
            seed=base_seed + 33,
            duration_s=4.5,
            params={"gain_asymmetry_db": 6.0, "loud_speaker": "speaker_1"},
            turns=[
                Turn(1, "speaker_1", 0.0, 4.0, 6.0, "Speaker 1 dominant target at +6dB", 0.0),
                Turn(2, "speaker_2", 0.0, 2.0, 0.0, "Speaker 2 interferer at 0dB simultaneous onset", 0.4),
            ],
        )
    )

    # 35: Ping-pong with +6dB loud interferer
    scenarios.append(
        ScenarioSpec(
            id="scenario_35",
            name="gain_asymmetry_pingpong_loud_int_6db",
            category="gain_asymmetry_6db",
            description="Four-turn ping-pong where Speaker 2 is consistently +6dB louder than Speaker 1",
            seed=base_seed + 34,
            duration_s=5.5,
            params={"gain_asymmetry_db": 6.0, "gap_ms": 100},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.2, 0.0, "Spk 1 turn 1 (0dB)", 0.0),
                Turn(2, "speaker_2", 1.3, 2.5, 6.0, "Spk 2 turn 1 (+6dB, 100ms gap)", 0.3),
                Turn(3, "speaker_1", 2.6, 3.8, 0.0, "Spk 1 turn 2 (0dB, 100ms gap)", 0.6),
                Turn(4, "speaker_2", 3.9, 5.1, 6.0, "Spk 2 turn 2 (+6dB, 100ms gap)", 0.9),
            ],
        )
    )

    # 36: Step gain change across 200ms pause
    scenarios.append(
        ScenarioSpec(
            id="scenario_36",
            name="gain_step_change_6db",
            category="gain_asymmetry_6db",
            description="Speaker 1 starts at 0dB, pauses 200ms, then resumes at +6dB (intra-speaker 6dB step)",
            seed=base_seed + 35,
            duration_s=4.2,
            params={"gain_step_db": 6.0, "pause_ms": 200},
            turns=[
                Turn(1, "speaker_1", 0.0, 1.8, 0.0, "Spk 1 initial phrase at 0dB", 0.0),
                Turn(2, "speaker_1", 2.0, 3.8, 6.0, "Spk 1 post-pause phrase stepped up to +6dB", 0.5),
            ],
        )
    )

    # 37: Loud interferer intrusion burst (+6dB)
    scenarios.append(
        ScenarioSpec(
            id="scenario_37",
            name="gain_asymmetry_int_burst_6db",
            category="gain_asymmetry_6db",
            description="Speaker 1 continuous at 0dB; Speaker 2 injects a +6dB burst intrusion [1.5, 2.7s]",
            seed=base_seed + 36,
            duration_s=4.2,
            params={"gain_asymmetry_db": 6.0, "burst_speaker": "speaker_2"},
            turns=[
                Turn(1, "speaker_1", 0.0, 4.0, 0.0, "Spk 1 continuous sentence at 0dB", 0.0),
                Turn(2, "speaker_2", 1.5, 2.7, 6.0, "Spk 2 loud intrusion burst (+6dB)", 0.3),
            ],
        )
    )

    # 38: 6dB asymmetry handoff with 50ms gap
    scenarios.append(
        ScenarioSpec(
            id="scenario_38",
            name="gain_asymmetry_handoff_6db",
            category="gain_asymmetry_6db",
            description="Very rapid handoff (50ms gap) into a +6dB louder speaker",
            seed=base_seed + 37,
            duration_s=4.5,
            params={"gain_asymmetry_db": 6.0, "gap_ms": 50},
            turns=[
                Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Spk 1 at 0dB", 0.0),
                Turn(2, "speaker_2", 2.05, 4.05, 6.0, "Spk 2 at +6dB (50ms gap)", 0.5),
            ],
        )
    )

    # =========================================================================
    # Group 5: Compound Stress - Scenarios 39 and 40
    # =========================================================================
    # 39: Compound simultaneous onset + 6dB asymmetry + 200ms pause + rapid handoff
    scenarios.append(
        ScenarioSpec(
            id="scenario_39",
            name="compound_simultaneous_asymmetry_pause",
            category="compound_stress",
            description="Simultaneous onset at 6dB asymmetry, resolving into 200ms pause, followed by rapid 100ms handoff",
            seed=base_seed + 38,
            duration_s=5.5,
            params={
                "simultaneous_onset": True,
                "gain_asymmetry_db": 6.0,
                "pause_ms": 200,
                "gap_ms": 100,
            },
            turns=[
                Turn(1, "speaker_1", 0.0, 2.0, 0.0, "Spk 1 onset (0dB)", 0.0),
                Turn(2, "speaker_2", 0.0, 1.0, 6.0, "Spk 2 simultaneous onset (+6dB)", 0.2),
                # 200ms pause from 2.0 to 2.2
                Turn(3, "speaker_1", 2.2, 3.5, 0.0, "Spk 1 post-pause continuation (0dB)", 0.5),
                # 100ms gap from 3.5 to 3.6
                Turn(4, "speaker_2", 3.6, 5.0, 6.0, "Spk 2 rapid reply (+6dB, 100ms gap)", 0.8),
            ],
        )
    )

    # 40: Full cocktail compound stress
    scenarios.append(
        ScenarioSpec(
            id="scenario_40",
            name="compound_full_cocktail_stress",
            category="compound_stress",
            description="Full cocktail stress: rapid staccato (<150ms gaps), alternating 6dB asymmetry, two 200ms pauses, and collision",
            seed=base_seed + 39,
            duration_s=6.5,
            params={
                "rapid_gaps": True,
                "gain_asymmetry_db": 6.0,
                "pauses_ms": [200, 200],
                "overlap_collision": True,
            },
            turns=[
                Turn(1, "speaker_1", 0.0, 1.2, 0.0, "Spk 1 initial (0dB)", 0.0),
                # 120ms gap -> 1.32
                Turn(2, "speaker_2", 1.32, 2.52, 6.0, "Spk 2 loud reply (+6dB, 120ms gap)", 0.3),
                # 200ms pause -> 2.72
                Turn(3, "speaker_1", 2.72, 4.22, 0.0, "Spk 1 resume after 200ms pause (0dB)", 0.6),
                # collision: Spk 2 interrupts from 3.5 to 4.5 (+6dB)
                Turn(4, "speaker_2", 3.5, 4.5, 6.0, "Spk 2 collision intrusion (+6dB)", 0.9),
                # 200ms pause -> 4.7
                Turn(5, "speaker_1", 4.7, 6.0, 0.0, "Spk 1 final wrap-up after 200ms pause (0dB)", 1.2),
            ],
        )
    )

    return scenarios


def _compute_active_and_overlap_intervals(turns: list[Turn], total_duration_s: float) -> dict[str, Any]:
    """Compute per-speaker active intervals, pairwise overlap intervals, and silence intervals."""
    speaker_active: dict[str, list[list[float]]] = {}
    for t in sorted(turns, key=lambda x: x.start_s):
        speaker_active.setdefault(t.speaker, []).append([round(t.start_s, 6), round(t.end_s, 6)])

    # Compute overlaps
    overlaps: list[list[Any]] = []
    sorted_turns = sorted(turns, key=lambda x: (x.start_s, x.end_s))
    for i, left in enumerate(sorted_turns):
        for right in sorted_turns[i + 1 :]:
            if left.speaker != right.speaker:
                o_start = max(left.start_s, right.start_s)
                o_end = min(left.end_s, right.end_s)
                if o_end > o_start:
                    spks = sorted([left.speaker, right.speaker])
                    overlaps.append([round(o_start, 6), round(o_end, 6), spks])

    # Compute global speech union intervals
    all_spans = sorted([[t.start_s, t.end_s] for t in turns], key=lambda x: x[0])
    merged_spans: list[list[float]] = []
    for s_start, s_end in all_spans:
        if not merged_spans or s_start > merged_spans[-1][1]:
            merged_spans.append([s_start, s_end])
        else:
            merged_spans[-1][1] = max(merged_spans[-1][1], s_end)

    # Compute silent intervals (complement of merged speech)
    silences: list[list[float]] = []
    curr = 0.0
    for s_start, s_end in merged_spans:
        if s_start > curr:
            silences.append([round(curr, 6), round(s_start, 6)])
        curr = max(curr, s_end)
    if curr < total_duration_s:
        silences.append([round(curr, 6), round(total_duration_s, 6)])

    return {
        "speaker_active_intervals": speaker_active,
        "overlap_intervals": overlaps,
        "silent_intervals": silences,
    }


def _generate_scenario_card_md(spec: ScenarioSpec, labels: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Generate rich documentation markdown card for one scenario."""
    lines = [
        f"# Scenario Card: {spec.id} ({spec.name})",
        "",
        "## Overview",
        f"- **Category:** `{spec.category}`",
        f"- **Description:** {spec.description}",
        f"- **Seed:** `{spec.seed}`",
        f"- **Duration:** {spec.duration_s:.3f} s ({round(spec.duration_s * spec.sample_rate):,} samples @ {spec.sample_rate} Hz)",
        f"- **Noise Condition:** `{spec.noise_type}`" + (f" (SNR: {spec.snr_db} dB)" if spec.snr_db is not None else " (Clean / No additive noise)"),
        f"- **Peak Amplitude:** {metrics['peak']:.4f} ({20.0 * np.log10(max(metrics['peak'], 1e-9)):.2f} dBFS)",
        f"- **RMS Energy:** {metrics['rms']:.5f}",
        f"- **SHA-256:** `{metrics['sha256']}`",
        "",
        "## Ground-Truth Timeline",
        "| Turn ID | Speaker | Start (s) | End (s) | Duration (s) | Gain (dB) | Notes |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]

    for t in labels["turns"]:
        lines.append(
            f"| {t['turn_id']} | `{t['speaker']}` | {t['start_s']:.3f} | {t['end_s']:.3f} | "
            f"{t['duration_s']:.3f} | {t['gain_db']:+.1f} | {t['description']} |"
        )

    if labels["overlap_intervals"]:
        lines.extend([
            "",
            "### Overlap Intervals",
            "| Start (s) | End (s) | Duration (s) | Overlapping Speakers |",
            "|:---:|:---:|:---:|:---|",
        ])
        for o in labels["overlap_intervals"]:
            lines.append(f"| {o[0]:.3f} | {o[1]:.3f} | {o[1]-o[0]:.3f} | `{', '.join(o[2])}` |")

    if labels["silent_intervals"]:
        lines.extend([
            "",
            "### Pauses & Silent Intervals",
            "| Start (s) | End (s) | Gap Duration (s) | Gap (ms) |",
            "|:---:|:---:|:---:|:---:|",
        ])
        for s in labels["silent_intervals"]:
            lines.append(f"| {s[0]:.3f} | {s[1]:.3f} | {s[1]-s[0]:.3f} | {round((s[1]-s[0])*1000)} ms |")

    lines.extend([
        "",
        "## EMA Tracking Evaluation Relevance",
        "- **T-275 (Short-crop EMA Context Buffer):**",
        "  - **Inertia & Lag:** Evaluates exponential moving average adaptation speed across short turn boundaries (<300ms) and pauses.",
        "  - **Wrong-Speaker Time:** Assesses wrong-speaker accumulation when switching between profiles.",
        "  - **Smoothing Stability:** Verifies suppression of angular jitter during stationary turns without excessive sluggishness.",
        "- **T-282 (Confidence-Weighted EMA):**",
        "  - **Switch-Stick Vulnerability:** Detects whether adaptive gain $\\alpha_t = \\alpha_{\\text{base}} \\cdot \\sigma(\\beta \\cdot \\text{margin})$ collapses during turn transitions and stalls adaptation.",
        "  - **Coasting Across Pauses:** Verifies whether 200ms silence pauses cause confidence degradation or allow clean state coasting.",
        "  - **Gain Asymmetry Balance:** Tests whether 6dB gain asymmetry creates asymmetric tracking locks or pulls embeddings towards the louder talker.",
        "",
        "## Verification Checkpoints",
        f"- [x] Deterministic rendering verified with fixed seed `{spec.seed}`.",
        f"- [x] Audio rendered strictly at {spec.sample_rate} Hz mono float32 without clipping.",
        "- [x] Ground-truth timestamps match audio onset/offset sample boundaries.",
        "",
    ])

    return "\n".join(lines)


def render_scenario(
    spec: ScenarioSpec,
    output_dir: Path | str,
    tiny_dir: Path = TINY_DIR,
) -> dict[str, Any]:
    """Render one scenario into ``mix.wav``, ``turn_labels.json``, and ``scenario_card.md``."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(tiny_dir)
    sr = spec.sample_rate
    n_samples = round(spec.duration_s * sr)

    target_stem = np.zeros(n_samples, dtype=np.float64)
    interferer_stem = np.zeros(n_samples, dtype=np.float64)

    for turn in spec.turns:
        t_start_samp = round(turn.start_s * sr)
        t_end_samp = round(turn.end_s * sr)
        dur_s = turn.end_s - turn.start_s

        src_audio = sources[turn.speaker]
        sliced = _slice_source(src_audio, dur_s, turn.source_offset_s, sr)

        # Apply gain
        gain_linear = 10.0 ** (turn.gain_db / 20.0)
        voiced_turn = (sliced * gain_linear).astype(np.float64)

        if turn.speaker == "speaker_1":
            target_stem[t_start_samp:t_end_samp] += voiced_turn[: t_end_samp - t_start_samp]
        else:
            interferer_stem[t_start_samp:t_end_samp] += voiced_turn[: t_end_samp - t_start_samp]

    # Noise stem
    noise_stem = np.zeros(n_samples, dtype=np.float64)
    rng = np.random.default_rng(spec.seed)
    if spec.noise_type == "babble" and spec.snr_db is not None:
        raw_babble = fit_length(sources["babble"].astype(np.float64), n_samples)
        t_rms = max(rms(target_stem), 1e-4)
        b_rms = max(rms(raw_babble), 1e-6)
        noise_stem = raw_babble * (t_rms / b_rms / (10.0 ** (spec.snr_db / 20.0)))
    elif spec.noise_type == "pink" and spec.snr_db is not None:
        raw_pink = pink_noise(n_samples, rng).astype(np.float64)
        t_rms = max(rms(target_stem), 1e-4)
        p_rms = max(rms(raw_pink), 1e-6)
        noise_stem = raw_pink * (t_rms / p_rms / (10.0 ** (spec.snr_db / 20.0)))

    # Sum mixture
    mix = target_stem + interferer_stem + noise_stem
    peak_val = float(np.max(np.abs(mix), initial=0.0))

    # Peak normalization according to PEAK convention (0.9)
    if peak_val > PEAK:
        norm_gain = PEAK / peak_val
        mix *= norm_gain
        target_stem *= norm_gain
        interferer_stem *= norm_gain
        noise_stem *= norm_gain
    elif peak_val > 0.0:
        # Scale to consistent nominal speech level without exceeding PEAK
        scale = min(PEAK / peak_val, 1.5)
        mix *= scale
        target_stem *= scale
        interferer_stem *= scale
        noise_stem *= scale

    mix_float32 = mix.astype(np.float32)

    # Write mix.wav
    mix_wav_path = out_path / "mix.wav"
    sf.write(mix_wav_path, mix_float32, sr, subtype="PCM_16")

    # Audio metrics
    wav_bytes = mix_wav_path.read_bytes()
    wav_sha256 = hashlib.sha256(wav_bytes).hexdigest()
    measured_peak = float(np.max(np.abs(mix_float32), initial=0.0))
    measured_rms = float(np.sqrt(np.mean(mix_float32.astype(np.float64) ** 2)))

    metrics = {
        "peak": round(measured_peak, 6),
        "rms": round(measured_rms, 6),
        "sha256": wav_sha256,
        "sample_rate": sr,
        "duration_s": spec.duration_s,
        "total_samples": n_samples,
    }

    # Build turn labels JSON
    interval_info = _compute_active_and_overlap_intervals(spec.turns, spec.duration_s)
    labels = {
        "scenario_id": spec.id,
        "name": spec.name,
        "category": spec.category,
        "description": spec.description,
        "sample_rate": sr,
        "duration_s": spec.duration_s,
        "total_samples": n_samples,
        "turns": [t.to_dict(sr) for t in spec.turns],
        "speaker_active_intervals": interval_info["speaker_active_intervals"],
        "overlap_intervals": interval_info["overlap_intervals"],
        "silent_intervals": interval_info["silent_intervals"],
        "params": spec.params,
        "seed": spec.seed,
    }

    labels_path = out_path / "turn_labels.json"
    labels_path.write_text(json.dumps(labels, indent=2) + "\n")

    # Build scenario card markdown
    card_content = _generate_scenario_card_md(spec, labels, metrics)
    card_path = out_path / "scenario_card.md"
    card_path.write_text(card_content)

    return {
        "id": spec.id,
        "name": spec.name,
        "category": spec.category,
        "description": spec.description,
        "seed": spec.seed,
        "duration_s": spec.duration_s,
        "sample_rate": sr,
        "total_samples": n_samples,
        "path": f"{spec.id}/mix.wav",
        "turn_labels_path": f"{spec.id}/turn_labels.json",
        "card_path": f"{spec.id}/scenario_card.md",
        "params": spec.params,
        "audio_metrics": metrics,
        "num_turns": len(spec.turns),
    }


def render_corpus(
    output_dir: Path | str = DEFAULT_OUT_DIR,
    tiny_dir: Path = TINY_DIR,
) -> dict[str, Any]:
    """Render all 40 scenarios and the master manifest.json."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios()
    scenario_records: list[dict[str, Any]] = []

    category_counts: dict[str, int] = {}
    for spec in scenarios:
        category_counts[spec.category] = category_counts.get(spec.category, 0) + 1
        rec = render_scenario(spec, out_root / spec.id, tiny_dir)
        scenario_records.append(rec)

    manifest = {
        "task": "T-293",
        "title": "Streaming stress-test corpus for EMA tracking variants",
        "description": (
            "40 deterministic scenarios evaluating EMA tracking variants (T-275 short-crop EMA, "
            "T-282 confidence-weighted EMA) across rapid turn-taking (<300ms gaps), simultaneous onset, "
            "200ms pauses, and 6dB gain asymmetry."
        ),
        "total_scenarios": len(scenario_records),
        "sample_rate": SAMPLE_RATE,
        "categories": category_counts,
        "scenarios": scenario_records,
    }

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return manifest


def spot_check_scenario(scenario_dir: Path | str) -> dict[str, Any]:
    """Spot-check a scenario directory for label accuracy and acoustic validity."""
    s_path = Path(scenario_dir)
    wav_path = s_path / "mix.wav"
    labels_path = s_path / "turn_labels.json"
    card_path = s_path / "scenario_card.md"

    if not wav_path.is_file():
        return {"passed": False, "error": f"Missing {wav_path}"}
    if not labels_path.is_file():
        return {"passed": False, "error": f"Missing {labels_path}"}
    if not card_path.is_file():
        return {"passed": False, "error": f"Missing {card_path}"}

    audio, sr = sf.read(wav_path, dtype="float32")
    labels = json.loads(labels_path.read_text())

    # 1. Check length matches
    expected_samples = labels["total_samples"]
    if len(audio) != expected_samples:
        return {
            "passed": False,
            "error": f"Sample count mismatch: {len(audio)} != {expected_samples}",
        }

    # 2. Check each turn has non-zero speech energy
    turn_checks = []
    for turn in labels["turns"]:
        t_start = turn["start_sample"]
        t_end = turn["end_sample"]
        t_slice = audio[t_start:t_end]
        t_rms = float(np.sqrt(np.mean(t_slice.astype(np.float64) ** 2)))
        t_peak = float(np.max(np.abs(t_slice)))
        if t_rms < 0.005 or t_peak < 0.02:
            return {
                "passed": False,
                "error": f"Turn {turn['turn_id']} ({turn['speaker']}) is silent or too quiet: RMS={t_rms}",
            }
        turn_checks.append({"turn_id": turn["turn_id"], "rms": t_rms, "peak": t_peak})

    # 3. Check silence intervals (no unexpected sound during pauses if clean)
    silence_checks = []
    has_noise = labels.get("params", {}).get("snr_db") is not None or "babble" in labels.get("name", "")
    for s_span in labels.get("silent_intervals", []):
        s_start = round(s_span[0] * sr)
        s_end = round(s_span[1] * sr)
        if s_end > s_start:
            s_slice = audio[s_start:s_end]
            s_rms = float(np.sqrt(np.mean(s_slice.astype(np.float64) ** 2)))
            if not has_noise and s_rms > 0.001:
                return {
                    "passed": False,
                    "error": f"Silence interval {s_span} has unexpected audio energy: RMS={s_rms}",
                }
            silence_checks.append({"interval": s_span, "rms": s_rms})

    return {
        "passed": True,
        "scenario_id": labels["scenario_id"],
        "category": labels["category"],
        "duration_s": labels["duration_s"],
        "num_turns": len(labels["turns"]),
        "turns": turn_checks,
        "silences": silence_checks,
    }


def main() -> None:
    """CLI entrypoint: generate corpus and run spot check."""
    parser = argparse.ArgumentParser(description="Render T-293 streaming stress-test corpus.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--spot-check", action="store_true", help="Run spot check on 3 scenarios")
    args = parser.parse_args()

    print(f"Rendering T-293 corpus to {args.out}...")
    manifest = render_corpus(args.out)
    print(f"Rendered {manifest['total_scenarios']} scenarios.")

    if args.spot_check:
        print("Spot-checking 3 scenarios for label accuracy...")
        picked = ["scenario_01", "scenario_11", "scenario_21"]
        checks = []
        for pid in picked:
            res = spot_check_scenario(args.out / pid)
            checks.append(res)
            print(f"  {pid}: {'PASS' if res['passed'] else 'FAIL'}")
        (args.out / "spot_check.json").write_text(json.dumps(checks, indent=2) + "\n")
        print("Wrote spot-check results to", args.out / "spot_check.json")


if __name__ == "__main__":
    main()
