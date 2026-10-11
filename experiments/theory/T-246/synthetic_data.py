"""Synthetic speaker embedding generator with acoustic and separator factors (T-246).

Simulates:
- 192-D ECAPA-TDNN speaker embedding latent geometry.
- Disjoint speaker and room/channel acoustic subspaces.
- Crop duration variance (1.0s window vs 2.5s enrollment).
- Blind separation artifacts (MossFormer2 inter-stream leakage / bleed).
- Quiet target SNR degradation (-6 dB, -12 dB).
- Target-absent room noise / interference windows.
- Strict disjoint splits: train / dev / heldout (speakers and rooms disjoint).
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np


@dataclasses.dataclass(frozen=True)
class EmbeddingSample:
    embedding: np.ndarray  # [192] unit normalized float32
    speaker_id: int  # -1 for target-absent noise
    room_id: int
    condition: str  # "enrollment", "target_stream", "interferer_stream", "quiet_target", "target_absent"
    crop_duration_s: float
    snr_db: float
    is_hard_pair: bool = False


@dataclasses.dataclass(frozen=True)
class StreamPairSample:
    """A separated two-speaker audio window with two competing streams."""

    target_profile: np.ndarray  # [192] unit normalized enrollment embedding
    target_speaker_id: int
    other_profile: np.ndarray | None  # [192] negative enrollment profile if enrolled
    other_speaker_id: int | None
    stream_0: np.ndarray  # [192] stream 0 candidate
    stream_1: np.ndarray  # [192] stream 1 candidate
    correct_stream: int  # 0 or 1 for target-present, -1 for target-absent
    condition: str  # "normal", "hard_pair", "quiet_target", "target_absent"
    snr_db: float


class SyntheticSpeakerGenerator:
    """Generates synthetic 192-D ECAPA speaker embeddings with realistic acoustic properties."""

    def __init__(
        self,
        dim: int = 192,
        spk_dim: int = 96,
        room_dim: int = 48,
        noise_dim: int = 48,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.spk_dim = spk_dim
        self.room_dim = room_dim
        self.noise_dim = noise_dim
        self.rng = np.random.default_rng(seed)

        # Orthogonal projection basis mapping latent subspaces into 192-D observation space
        q, _ = np.linalg.qr(self.rng.standard_normal((dim, dim)))
        self.w_spk = q[:, :spk_dim]  # speaker identity basis
        self.w_room = q[:, spk_dim : spk_dim + room_dim]  # room/mic acoustic basis
        self.w_noise = q[:, spk_dim + room_dim :]  # channel/separator noise basis

    def generate_speaker_vectors(self, num_speakers: int) -> dict[int, np.ndarray]:
        """Generate latent speaker vocal-tract vectors."""
        spk_dict = {}
        for spk_id in range(num_speakers):
            # Formant / gender cluster (first 8 dims) + fine timbre (remaining dims)
            cluster = self.rng.choice([-1.0, 1.0])
            coarse = np.full(8, cluster) + 0.2 * self.rng.standard_normal(8)
            fine = self.rng.standard_normal(self.spk_dim - 8)
            vec = np.concatenate([coarse, fine])
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            spk_dict[spk_id] = vec
        return spk_dict

    def generate_room_vectors(self, num_rooms: int) -> dict[int, np.ndarray]:
        """Generate latent room acoustics / microphone transfer vectors."""
        room_dict = {}
        for room_id in range(num_rooms):
            vec = self.rng.standard_normal(self.room_dim)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            room_dict[room_id] = vec
        return room_dict

    def synthesize_crop(
        self,
        spk_latent: np.ndarray | None,
        room_latent: np.ndarray,
        duration_s: float = 1.0,
        leak_latent: np.ndarray | None = None,
        leak_ratio: float = 0.0,
        snr_db: float = 20.0,
        is_noise: bool = False,
    ) -> np.ndarray:
        """Synthesize an observed 192-D unit embedding under specific acoustic conditions."""
        # Crop duration effect: variance scales as 1 / duration
        var_scale = 1.0 / np.sqrt(max(duration_s, 0.2))

        # Acoustic components
        if is_noise or spk_latent is None:
            spk_comp = np.zeros(self.dim)
        else:
            if leak_latent is not None and leak_ratio > 0.0:
                mixed_spk = (1.0 - leak_ratio) * spk_latent + leak_ratio * leak_latent
                mixed_spk = mixed_spk / (np.linalg.norm(mixed_spk) + 1e-12)
            else:
                mixed_spk = spk_latent
            spk_comp = self.w_spk @ mixed_spk

        room_comp = 0.25 * (self.w_room @ room_latent)

        # Noise / separator distortion component based on SNR
        noise_level = 10.0 ** (-snr_db / 20.0)
        noise_latent = self.rng.standard_normal(self.noise_dim)
        noise_latent = noise_latent / (np.linalg.norm(noise_latent) + 1e-12)
        noise_comp = noise_level * (self.w_noise @ noise_latent)

        # Intrinsic estimation jitter due to finite crop length
        jitter = 0.15 * var_scale * self.rng.standard_normal(self.dim)

        combined = spk_comp + room_comp + noise_comp + jitter
        norm = np.linalg.norm(combined)
        return (combined / norm if norm > 1e-12 else combined).astype(np.float32)


class DatasetPartition:
    """Container for a structured partition of synthetic training or evaluation data."""

    def __init__(
        self,
        name: Literal["train", "dev", "heldout"],
        speakers: dict[int, np.ndarray],
        rooms: dict[int, np.ndarray],
        generator: SyntheticSpeakerGenerator,
    ) -> None:
        self.name = name
        self.speakers = speakers
        self.rooms = rooms
        self.generator = generator

    def sample_contrastive_batch(
        self,
        batch_size: int = 64,
        crops_per_speaker: int = 4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a multi-crop supervised contrastive batch with labels.

        Returns (features: [N, 192], labels: [N]).
        """
        spk_ids = list(self.speakers.keys())
        num_spks = max(1, batch_size // crops_per_speaker)
        chosen_spks = self.generator.rng.choice(
            spk_ids,
            size=min(num_spks, len(spk_ids)),
            replace=False,
        )

        features = []
        labels = []
        room_ids = list(self.rooms.keys())

        for spk_id in chosen_spks:
            spk_latent = self.speakers[spk_id]
            for _ in range(crops_per_speaker):
                r_id = int(self.generator.rng.choice(room_ids))
                room_latent = self.rooms[r_id]

                # Sample diverse acoustic crop conditions: clean, separated, or noisy
                crop_cond = self.generator.rng.choice(["clean", "separated", "noisy"])
                if crop_cond == "clean":
                    dur = 2.5
                    leak_ratio = 0.0
                    snr = 25.0
                elif crop_cond == "separated":
                    dur = 1.0
                    other_spk = int(self.generator.rng.choice([s for s in spk_ids if s != spk_id]))
                    leak_latent = self.speakers[other_spk]
                    leak_ratio = float(self.generator.rng.uniform(0.05, 0.20))
                    snr = float(self.generator.rng.uniform(10.0, 20.0))
                else:  # noisy
                    dur = 1.0
                    leak_latent = None
                    leak_ratio = 0.0
                    snr = float(self.generator.rng.uniform(0.0, 10.0))

                emb = self.generator.synthesize_crop(
                    spk_latent=spk_latent,
                    room_latent=room_latent,
                    duration_s=dur,
                    leak_latent=leak_latent if crop_cond == "separated" else None,
                    leak_ratio=leak_ratio,
                    snr_db=snr,
                )
                features.append(emb)
                labels.append(spk_id)

        return np.stack(features).astype(np.float32), np.array(labels, dtype=np.int64)

    def generate_evaluation_stream_pairs(
        self,
        count: int = 100,
    ) -> list[StreamPairSample]:
        """Generate evaluation test cases simulating 2-stream separated windows."""
        spk_ids = list(self.speakers.keys())
        room_ids = list(self.rooms.keys())
        samples: list[StreamPairSample] = []

        for i in range(count):
            cond_roll = i % 4
            target_spk = int(self.generator.rng.choice(spk_ids))
            other_candidates = [s for s in spk_ids if s != target_spk]
            other_spk = int(self.generator.rng.choice(other_candidates))
            r_id = int(self.generator.rng.choice(room_ids))
            room_latent = self.rooms[r_id]

            # Clean enrollment profiles (2.5 s)
            t_profile = self.generator.synthesize_crop(
                self.speakers[target_spk],
                room_latent,
                duration_s=2.5,
                snr_db=25.0,
            )
            o_profile = self.generator.synthesize_crop(
                self.speakers[other_spk],
                room_latent,
                duration_s=2.5,
                snr_db=25.0,
            )

            if cond_roll == 0:  # Normal equal-level mixture
                cond = "normal"
                snr = 0.0
                stream_t = self.generator.synthesize_crop(
                    self.speakers[target_spk],
                    room_latent,
                    duration_s=1.0,
                    leak_latent=self.speakers[other_spk],
                    leak_ratio=0.10,
                    snr_db=15.0,
                )
                stream_o = self.generator.synthesize_crop(
                    self.speakers[other_spk],
                    room_latent,
                    duration_s=1.0,
                    leak_latent=self.speakers[target_spk],
                    leak_ratio=0.10,
                    snr_db=15.0,
                )
                # Randomize stream positions (0 vs 1)
                t_idx = int(self.generator.rng.choice([0, 1]))
                s0 = stream_t if t_idx == 0 else stream_o
                s1 = stream_o if t_idx == 0 else stream_t
                correct = t_idx

            elif cond_roll == 1:  # Hard pair (male-male / similar formant)
                cond = "hard_pair"
                snr = 0.0
                stream_t = self.generator.synthesize_crop(
                    self.speakers[target_spk],
                    room_latent,
                    duration_s=1.0,
                    leak_latent=self.speakers[other_spk],
                    leak_ratio=0.18,  # elevated separator bleed
                    snr_db=12.0,
                )
                stream_o = self.generator.synthesize_crop(
                    self.speakers[other_spk],
                    room_latent,
                    duration_s=1.0,
                    leak_latent=self.speakers[target_spk],
                    leak_ratio=0.18,
                    snr_db=12.0,
                )
                t_idx = int(self.generator.rng.choice([0, 1]))
                s0 = stream_t if t_idx == 0 else stream_o
                s1 = stream_o if t_idx == 0 else stream_t
                correct = t_idx

            elif cond_roll == 2:  # Quiet target (-10 dB SNR)
                cond = "quiet_target"
                snr = -10.0
                stream_t = self.generator.synthesize_crop(
                    self.speakers[target_spk],
                    room_latent,
                    duration_s=1.0,
                    leak_latent=self.speakers[other_spk],
                    leak_ratio=0.25,  # higher distortion when quiet
                    snr_db=5.0,
                )
                stream_o = self.generator.synthesize_crop(
                    self.speakers[other_spk],
                    room_latent,
                    duration_s=1.0,
                    leak_latent=None,
                    leak_ratio=0.0,
                    snr_db=20.0,
                )
                t_idx = int(self.generator.rng.choice([0, 1]))
                s0 = stream_t if t_idx == 0 else stream_o
                s1 = stream_o if t_idx == 0 else stream_t
                correct = t_idx

            else:  # Target absent (target silent in window, interferer + noise only)
                cond = "target_absent"
                snr = -30.0
                stream_o = self.generator.synthesize_crop(
                    self.speakers[other_spk],
                    room_latent,
                    duration_s=1.0,
                    snr_db=15.0,
                )
                stream_noise = self.generator.synthesize_crop(
                    None,
                    room_latent,
                    duration_s=1.0,
                    is_noise=True,
                    snr_db=-5.0,
                )
                o_idx = int(self.generator.rng.choice([0, 1]))
                s0 = stream_o if o_idx == 0 else stream_noise
                s1 = stream_noise if o_idx == 0 else stream_o
                correct = -1  # Neither stream is the target!

            samples.append(
                StreamPairSample(
                    target_profile=t_profile,
                    target_speaker_id=target_spk,
                    other_profile=o_profile,
                    other_speaker_id=other_spk,
                    stream_0=s0,
                    stream_1=s1,
                    correct_stream=correct,
                    condition=cond,
                    snr_db=snr,
                )
            )

        return samples


def create_synthetic_splits(
    seed: int = 42,
    train_speakers: int = 40,
    dev_speakers: int = 10,
    heldout_speakers: int = 10,
    train_rooms: int = 5,
    dev_rooms: int = 2,
    heldout_rooms: int = 2,
) -> tuple[DatasetPartition, DatasetPartition, DatasetPartition]:
    """Factory creating strictly disjoint Train, Dev, and Heldout dataset partitions."""
    gen = SyntheticSpeakerGenerator(seed=seed)
    total_spks = train_speakers + dev_speakers + heldout_speakers
    total_rooms = train_rooms + dev_rooms + heldout_rooms

    all_spks = gen.generate_speaker_vectors(total_spks)
    all_rooms = gen.generate_room_vectors(total_rooms)

    # Disjoint speaker slices
    train_spk_dict = {i: all_spks[i] for i in range(train_speakers)}
    dev_spk_dict = {i: all_spks[i] for i in range(train_speakers, train_speakers + dev_speakers)}
    heldout_spk_dict = {i: all_spks[i] for i in range(train_speakers + dev_speakers, total_spks)}

    # Disjoint room slices
    train_room_dict = {i: all_rooms[i] for i in range(train_rooms)}
    dev_room_dict = {i: all_rooms[i] for i in range(train_rooms, train_rooms + dev_rooms)}
    heldout_room_dict = {i: all_rooms[i] for i in range(train_rooms + dev_rooms, total_rooms)}

    train_part = DatasetPartition("train", train_spk_dict, train_room_dict, gen)
    dev_part = DatasetPartition("dev", dev_spk_dict, dev_room_dict, gen)
    heldout_part = DatasetPartition("heldout", heldout_spk_dict, heldout_room_dict, gen)

    return train_part, dev_part, heldout_part
