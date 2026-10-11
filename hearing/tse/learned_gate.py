"""T-407: learned continuous-gain gate for the fine-tuned fixed-order slot (``HEARING_FIXED_GATE=learned``, default off).

T-402's 3-class rescue gate cut Joseph's muting but let Aryan back in (+1.4-2.1 pt wrong-speaker, ~6 dB leak). This
gate does not take a hard target / partner decision. Two logistic heads read one window of the fixed separator:

- ``p_target``: the target is talking in this window (its reference >= :data:`ACTIVE_DB` re the mix);
- ``p_leak``:   only someone else is (target reference < :data:`SILENT_DB`, partner or stranger >= :data:`ACTIVE_DB`),
  so playing the target slot would let them in;

from the slot energies over the mix, the ECAPA cosines of both slots and of the mix to both profiles, a slot VAD
(slot >= :data:`VAD_DB` re the mix) and the previous window's target-slot level, cosine contrast and VAD. The
played gain is continuous: ``s = p_target - LEAK_WEIGHT * p_leak`` maps linearly from ``LO`` (floor, -20 dB) to
``HI`` (0 dB); an open window (gain > -6 dB) lowers ``LO`` / ``HI`` by ``HYST`` for the next one (hysteresis), and
``p_leak >= LEAK_CLOSE`` closes it whatever ``p_target`` says (cost-weighted toward never letting the partner in).

Weights: fitted by ``eval/learned_gate_fit.py`` on T-391's per-window features of the v3 separator (train renders
only; policy constants tuned on the val renders; scored on the holdout scenes). ``$HEARING_FIXED_GATE_HEAD`` may name
the JSON it writes (``{"kind": "learned", "mean", "std", "w_target", "w_leak", "policy"}``).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

ACTIVE_DB = -15.0  # a voice's reference energy over the mix at or above this = talking in the window
SILENT_DB = -25.0  # below this = not talking
VAD_DB = -20.0  # slot energy over the mix at or above this = slot active
DB_LO = -60.0  # slot / mix dB clipped to [DB_LO, +10]
FLOOR_DB = -20.0  # gain at s <= LO (the live HEARING_GATED_FLOOR_DB=20 room level)
OPEN_DB = -6.0  # a window played above this is "open" for the hysteresis
FEATURES = (
    "mix_dbfs", "t_db", "p_db", "t_over_p", "ct_t", "ct_p", "cp_t", "cp_p", "cm_t", "cm_p", "ct_contrast",
    "cp_contrast", "vad_t", "vad_p", "prev_t_db", "prev_ct_contrast", "prev_vad_t", "has_prev",
)
POLICY = {"leak_weight": 1.0, "lo": 0.2, "hi": 0.6, "hyst": 0.1, "leak_close": 0.6}


def _clip_db(v: float) -> float:
    return min(max(float(v), DB_LO), 10.0)


def window_row(mix_dbfs, t_db, p_db, ct_t, ct_p, cp_t, cp_p, cm_t, cm_p, prev=None) -> np.ndarray:
    """One window's raw feature row (:data:`FEATURES`). ``*_db`` are slot energy over the mix (dB); ``ct_t`` = cos of
    the target slot to the target profile, ``ct_p`` to the partner's, ``cp_*`` the partner slot, ``cm_*`` the mix.
    ``prev`` is the previous window's row (None at a start)."""
    t, p = _clip_db(t_db), _clip_db(p_db)
    row = [max(float(mix_dbfs), -80.0) / 20.0, t / 20.0, p / 20.0, (t - p) / 20.0, ct_t, ct_p, cp_t, cp_p, cm_t, cm_p,
           ct_t - ct_p, cp_t - cp_p, float(t >= VAD_DB), float(p >= VAD_DB)]
    if prev is None:
        row += [0.0, 0.0, 0.0, 0.0]
    else:
        row += [float(prev[1]), float(prev[10]), float(prev[12]), 1.0]
    return np.asarray(row, dtype=np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def gain_db(p_target: float, p_leak: float, was_open: bool, policy: dict | None = None) -> float:
    """The policy: gain in dB in [FLOOR_DB, 0] from both probabilities and the previous window's state."""
    q = POLICY if policy is None else policy
    if p_leak >= q["leak_close"]:
        return FLOOR_DB
    s = p_target - q["leak_weight"] * p_leak
    lo, hi = q["lo"], q["hi"]
    if was_open:
        lo, hi = lo - q["hyst"], hi - q["hyst"]
    a = min(max((s - lo) / max(hi - lo, 1e-6), 0.0), 1.0)
    return FLOOR_DB * (1.0 - a)


class LearnedHead:
    """Standardised features -> ``(p_target, p_leak)``."""

    def __init__(self, mean, std, w_target, w_leak, policy=None):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.w_target = np.asarray(w_target, dtype=np.float64)
        self.w_leak = np.asarray(w_leak, dtype=np.float64)
        self.policy = dict(POLICY if policy is None else policy)
        n = len(FEATURES)
        if self.mean.shape != (n,) or self.std.shape != (n,) or self.w_target.shape != (n + 1,) or \
                self.w_leak.shape != (n + 1,):
            raise ValueError(f"learned gate head must have {n} features (+ bias)")

    @classmethod
    def from_dict(cls, d: dict) -> LearnedHead:
        return cls(d["mean"], d["std"], d["w_target"], d["w_leak"], d.get("policy"))

    def to_dict(self) -> dict:
        return {"kind": "learned", "features": list(FEATURES), "mean": self.mean.tolist(), "std": self.std.tolist(),
                "w_target": self.w_target.tolist(), "w_leak": self.w_leak.tolist(), "policy": self.policy}

    def probs(self, x: np.ndarray) -> tuple:
        z = np.concatenate([(np.atleast_2d(x) - self.mean) / self.std, np.ones((np.atleast_2d(x).shape[0], 1))], 1)
        pt, pl = sigmoid(z @ self.w_target), sigmoid(z @ self.w_leak)
        return (float(pt[0]), float(pl[0])) if np.ndim(x) == 1 else (pt, pl)


def load_head(path: str | os.PathLike | None = None) -> LearnedHead | None:
    """The head named by ``path`` / ``$HEARING_FIXED_GATE_HEAD`` if it is a learned one, else the shipped
    :data:`HEAD` (None until a fit has been pasted in)."""
    path = path or os.environ.get("HEARING_FIXED_GATE_HEAD", "").strip()
    if path:
        d = json.loads(Path(path).expanduser().read_text())
        if d.get("kind") == "learned":
            return LearnedHead.from_dict(d)
    return None if HEAD is None else LearnedHead.from_dict(HEAD)


class LearnedGate:
    """Streaming: :meth:`step` per window -> gain in dB (``FLOOR_DB`` .. 0)."""

    def __init__(self, head: LearnedHead):
        self.head = head
        self.reset()

    def reset(self):
        self.prev = None
        self.open = False
        self.p_target = self.p_leak = None
        self.gain_db = FLOOR_DB

    def step(self, mix_dbfs, t_db, p_db, ct_t, ct_p, cp_t, cp_p, cm_t, cm_p) -> float:
        x = window_row(mix_dbfs, t_db, p_db, ct_t, ct_p, cp_t, cp_p, cm_t, cm_p, self.prev)
        self.prev = x
        self.p_target, self.p_leak = self.head.probs(x)
        self.gain_db = gain_db(self.p_target, self.p_leak, self.open, self.head.policy)
        self.open = self.gain_db > OPEN_DB
        return self.gain_db


def db_over(x, ref) -> float:
    """Energy of ``x`` over ``ref`` in dB (-100 when ``x`` is silent)."""
    e = float(np.sum(np.square(np.asarray(x, dtype=np.float64))))
    if e < 1e-10:
        return -100.0
    return 10.0 * math.log10(e / (float(np.sum(np.square(np.asarray(ref, dtype=np.float64)))) + 1e-12))


HEAD = None  # filled from eval/learned_gate_fit.py's JSON only if the gate is adopted (docs/BENCHMARKS.md T-407)
