"""T-402: gate for the fine-tuned fixed-order slot (``HEARING_FIXED_GATE=1``, default off).

Under ``HEARING_PICK=fixed`` + ``HEARING_SEP_CKPT`` (T-386) the slot already is the right person; the
ECAPA/LDA contrast gate still muted Joseph's slot (4.6-8.2 s per 30 s scene with floor 20). This gate decides
"play the target slot" from the slots themselves: a 3-class (target / partner / none) multinomial logistic
on the target slot's energy against the partner slot and the mix (last half of the window, the part that
plays) plus ECAPA cosine of the target slot to both profiles, with short hysteresis. ECAPA alone is kept as
a stranger check: an active target slot that matches neither profile stays ducked. In ``EnrolledCandidate``
the gate only rescues: it un-mutes a window the ECAPA contrast gate muted, never mutes one it played (gating
every window by the slot raised wrong-speaker 2-3 points on the Aryan rows offline, T-402), and not while the partner
slot outscores the target slot by more than :data:`RESCUE_MAX_PARTNER_LEAD`; a window still muted may swap to the
partner slot (:meth:`FixedGate.swap`) when the net routed the target there. Live A/B: not adopted (docs/BENCHMARKS.md);
the orchestrator closed this line (Q-1445): T-386's arm 4 stays the demo setting, the knob stays off.

Weights: :data:`WEIGHTS` were fitted by ``eval/fixed_gate_train.py`` on the team train takes
(profile-02..05, the T-374 train split) separated by the T-374 fixed checkpoint; ``$HEARING_FIXED_GATE_HEAD``
may name a JSON file ``{"weights": [[...] x 3]}`` with another fit.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

FIXED_GATE_ENV = "HEARING_FIXED_GATE"  # "1" = on; unset / "0" = off (the ECAPA contrast gate as in T-386)
FIXED_GATE_HEAD_ENV = "HEARING_FIXED_GATE_HEAD"  # optional JSON weights replacing WEIGHTS
CLASSES = ("target", "partner", "none")
DB_CLIP = 40.0  # energy ratios are clipped to +-40 dB
FEATURES = ("t_over_p_db", "t_over_mix_db", "p_over_mix_db", "cos_t_target", "cos_t_partner", "bias")
OPEN_P = 0.5  # p(target) that opens a closed gate
CLOSE_P = 0.3  # an open gate stays open while p(target) >= this ...
HOLD_WINDOWS = 1  # ... or for this many windows below it (short hysteresis)
STRANGER_COS = 0.1  # target slot active (>= STRANGER_ACTIVE_DB re the mix) and cos < this to both profiles
STRANGER_ACTIVE_DB = -10.0
# swap rescue (T-402 v3): a window the gates still mute whose target slot is silent while the partner slot carries the
# window and scores on the target's side (LDA >= SWAP_SCORE; Joseph routed into Aryan's slot scored -0.16..+0.17
# offline, Aryan alone -0.36..-0.68) plays the partner slot instead
SWAP_SCORE = -0.2
SWAP_T_OVER_P_DB = -15.0
SWAP_P_OVER_MIX_DB = -6.0
# rescue guard (T-402 v4): no rescue while the partner slot outscores the target slot by more than this (LDA units);
# offline every rescued window that let Aryan back in on the Aryan rows had the partner slot ahead by 0.03-0.41,
# 4 of 6 by > 0.15
RESCUE_MAX_PARTNER_LEAD = 0.15
MIN_CONF = 0.6  # confidence of an open window: max(p(target), MIN_CONF) (the mixer ducks the residual by it)
# fitted by eval/fixed_gate_train.py (rows = CLASSES, columns = FEATURES); see docs/BENCHMARKS.md T-402
WEIGHTS = np.array([  # 1600 rows (995 target / 365 partner / 240 none), train acc 0.796, target recall 0.891
    [-0.0519, 0.1461, -0.0056, 0.6354, -0.7763, 3.4105],
    [0.0563, -0.0605, 0.1109, -0.1422, 0.7311, 0.8732],
    [-0.0044, -0.0856, -0.1053, -0.4932, 0.0452, -4.2837],
])


def enabled() -> bool:
    return os.environ.get(FIXED_GATE_ENV, "").strip().lower() in ("1", "on", "true")


def _db(a: float, b: float) -> float:
    v = 10.0 * math.log10((a + 1e-10) / (b + 1e-10))
    return min(max(v, -DB_CLIP), DB_CLIP)


def features(target_slot, partner_slot, mix, cos_target: float, cos_partner: float) -> np.ndarray:
    """One window's feature row (:data:`FEATURES`); energies over the window's last half."""
    n = len(mix) // 2
    e_t, e_p, e_m = (float(np.mean(np.square(np.asarray(x[n:], dtype=np.float64)))) for x in
                     (target_slot, partner_slot, mix))
    return np.array([_db(e_t, e_p), _db(e_t, e_m), _db(e_p, e_m), float(cos_target), float(cos_partner), 1.0])


def softmax_probs(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = np.asarray(weights, dtype=np.float64) @ np.asarray(x, dtype=np.float64)
    z = np.exp(z - z.max())
    return z / z.sum()


def load_weights(path: str | os.PathLike | None = None) -> np.ndarray:
    path = path or os.environ.get(FIXED_GATE_HEAD_ENV, "").strip()
    if not path:
        return WEIGHTS.copy()
    w = np.asarray(json.loads(Path(path).expanduser().read_text())["weights"], dtype=np.float64)
    if w.shape != (len(CLASSES), len(FEATURES)):
        raise ValueError(f"fixed gate weights must be {len(CLASSES)}x{len(FEATURES)}, got {w.shape}")
    return w


class FixedGate:
    """Streaming gate: :meth:`step` per window -> (play the target slot?, confidence)."""

    def __init__(self, weights=None):
        self.weights = load_weights() if weights is None else np.asarray(weights, dtype=np.float64)
        self.reset()

    def reset(self):
        self.open = False
        self._below = 0
        self.probs = None  # last window's p(target, partner, none)
        self.stranger = False
        self.x = None  # last window's feature row
        self.swapped = False

    def swap(self, partner_score: float) -> bool:
        """After :meth:`step`: play the partner slot this window? (target slot silent, partner slot on the target's
        side of the LDA). Only consulted for windows that stay muted."""
        x = self.x
        self.swapped = bool(x is not None and not self.stranger and x[0] <= SWAP_T_OVER_P_DB
                            and x[2] >= SWAP_P_OVER_MIX_DB and partner_score >= SWAP_SCORE)
        return self.swapped

    def step(self, target_slot, partner_slot, mix, cos_target: float, cos_partner: float) -> tuple[bool, float]:
        x = features(target_slot, partner_slot, mix, cos_target, cos_partner)
        self.x, self.swapped = x, False
        p = softmax_probs(self.weights, x)
        self.probs = [float(v) for v in p]
        pt = float(p[0])
        self.stranger = bool(x[1] >= STRANGER_ACTIVE_DB and max(cos_target, cos_partner) < STRANGER_COS)
        if self.stranger:
            self.open, self._below = False, 0
        elif self.open:
            self._below = 0 if pt >= CLOSE_P else self._below + 1
            self.open = self._below <= HOLD_WINDOWS
        else:
            self.open, self._below = pt >= OPEN_P, 0
        return self.open, (max(pt, MIN_CONF) if self.open else 0.0)
