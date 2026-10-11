"""Preference model: log what the user picks in which room, learn it, recommend, and drive Auto.

* :class:`PrefLog` appends one JSON line per labelled choice (context features + the
  mode the user chose). Never apparent age/gender: only the 1 Hz room features.
* :class:`PrefModel` is a tiny softmax regression on the five context features plus
  their pairwise products (21 inputs, 5 classes, numpy + scipy L-BFGS, ~ms to refit).
* :func:`simulate` is a scripted user in five kinds of rooms. It seeds the model so
  recommendations work before any real choices exist and gives the held-out accuracy
  number (``python -m hearing.context.prefs``). Real choices outweigh it as they accrue.
* :class:`Advisor` turns 1 Hz context vectors into ``recommendation`` events (toasts,
  throttled) and, in Auto mode, mode switches with hysteresis and a cooldown; while the model
  is unsure Auto follows the 2-talker rule (focus_hq at 2+ speakers, natural 5 s after).

Time is the session's audio clock in seconds (one context vector per second), so the
behaviour is deterministic and testable.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from hearing.context.features import FEATURES, ContextVector

LABELS = ("natural", "focus_hq", "synthetic", "text", "quiet")
AUTO_MODES = ("natural", "focus_hq", "synthetic")  # Auto never goes silent on its own
SOURCE_WEIGHT = {"explicit": 1.0, "accepted": 1.0, "dwell": 0.3}


# ---------------------------------------------------------------- features -> inputs

_CENTER = np.array([-40.0, 10.0, 1.5, 0.25, 0.5])
_SCALE = np.array([15.0, 10.0, 1.5, 0.25, 0.5])
_PAIRS = [(i, j) for i in range(len(FEATURES)) for j in range(i, len(FEATURES))]


def design(x: np.ndarray) -> np.ndarray:
    """Rows of raw features -> fixed-scale features, their pairwise products, and a bias."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64)).copy()
    x[:, 2] = np.minimum(x[:, 2], 4)  # 4+ speakers is "a crowd"
    z = (x - _CENTER) / _SCALE
    quad = np.stack([z[:, i] * z[:, j] for i, j in _PAIRS], axis=1)
    return np.hstack([z, quad, np.ones((len(z), 1))])


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class PrefModel:
    """Multinomial logistic regression over :func:`design` features."""

    def __init__(self, l2: float = 1e-3):
        self.l2 = l2
        self.w = np.zeros((len(FEATURES) + len(_PAIRS) + 1, len(LABELS)))

    def fit(self, x, y, weights=None, max_iter: int = 300) -> PrefModel:
        phi = design(x)
        y_idx = np.array([LABELS.index(v) for v in y])
        onehot = np.eye(len(LABELS))[y_idx]
        sw = np.ones(len(phi)) if weights is None else np.asarray(weights, dtype=np.float64)
        total = sw.sum()
        shape = self.w.shape

        def loss(flat):
            w = flat.reshape(shape)
            p = _softmax(phi @ w)
            nll = -np.sum(sw * np.log(np.sum(p * onehot, axis=1) + 1e-12)) / total
            grad = phi.T @ (sw[:, None] * (p - onehot)) / total
            reg = self.l2 * w[:-1]  # no penalty on the bias row
            return nll + 0.5 * np.sum(reg * w[:-1]), (grad + np.vstack([reg, np.zeros((1, shape[1]))])).ravel()

        res = minimize(loss, self.w.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
        self.w = res.x.reshape(shape)
        return self

    def proba(self, x) -> np.ndarray:
        return _softmax(design(x) @ self.w)

    def predict(self, x) -> list[str]:
        return [LABELS[i] for i in self.proba(x).argmax(axis=1)]

    def accuracy(self, x, y) -> float:
        return float(np.mean(np.array(self.predict(x)) == np.array(y)))

    def copy(self) -> PrefModel:
        other = PrefModel(self.l2)
        other.w = self.w.copy()
        return other


# ---------------------------------------------------------------- simulated user

# room -> (noise_db, snr_db, speakers, overlap) ranges
ROOMS = {
    "quiet_room": ((-70, -50), (15, 40), (0, 1), (0.0, 0.0)),
    "meeting": ((-55, -40), (8, 30), (2, 4), (0.0, 0.3)),
    "street": ((-38, -20), (0, 12), (0, 1), (0.0, 0.0)),
    "cafe": ((-42, -25), (2, 15), (1, 4), (0.0, 0.5)),
    "party": ((-30, -12), (0, 8), (2, 6), (0.2, 0.8)),
}


def simulated_choice(f: np.ndarray, user: str = "default") -> str:
    """The scripted user's preferred mode for one feature row.

    ``default`` keeps the real voice when it can and re-voices only when the room is
    hopeless; ``revoicer`` (a second persona, to measure personalisation) switches to
    Synthetic as soon as two people talk in a noisy room.
    """
    noise, snr, speakers, overlap, conf = f
    if speakers == 0 and noise > -35:
        return "quiet"
    if noise > -22 and snr < 3:
        return "text"
    if user == "revoicer" and noise > -40 and speakers >= 2:
        return "synthetic"
    if noise < -45 and overlap < 0.2:
        return "natural"
    if snr < 5 and (speakers >= 3 or overlap > 0.4):
        return "synthetic"
    if conf >= 0.5 and speakers >= 1:
        return "focus_hq"
    return "natural" if noise < -35 else "synthetic"


def simulate(n: int, seed: int = 0, user: str = "default", label_noise: float = 0.1):
    """``n`` rows of (features, chosen mode) from the scripted user; ``label_noise`` of the
    labels are replaced by a random mode (people are not consistent)."""
    rng = np.random.default_rng(seed)
    rooms = list(ROOMS.values())
    x = np.zeros((n, len(FEATURES)))
    y = []
    for k in range(n):
        noise, snr, spk, ov = rooms[rng.integers(len(rooms))]
        speakers = int(rng.integers(spk[0], spk[1] + 1))
        x[k] = [
            rng.uniform(*noise),
            rng.uniform(*snr),
            speakers,
            rng.uniform(*ov) if speakers >= 2 else 0.0,
            rng.uniform(0, 1) if speakers else 0.0,
        ]
        label = simulated_choice(x[k], user)
        if rng.random() < label_noise:
            label = LABELS[rng.integers(len(LABELS))]
        y.append(label)
    return x, y


@lru_cache(maxsize=1)
def _prior() -> tuple[np.ndarray, tuple[str, ...], PrefModel]:
    x, y = simulate(2000, seed=0)
    return x, tuple(y), PrefModel().fit(x, y)


def prior_model() -> PrefModel:
    """The seed model trained on the simulated user (cached per process)."""
    return _prior()[2].copy()


# ---------------------------------------------------------------- log


@dataclass
class PrefRow:
    features: list[float]
    chosen: str
    previous: str
    source: str  # explicit | accepted | dwell
    ts: float = field(default_factory=time.time)


class PrefLog:
    """Labelled choices in memory, appended to a JSONL file when ``path`` is set."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).expanduser() if path else None
        self.rows: list[PrefRow] = self.load(self.path) if self.path else []

    @staticmethod
    def load(path: Path) -> list[PrefRow]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
                row = PrefRow([float(d[k]) for k in FEATURES], d["chosen"], d.get("previous", ""),
                              d.get("source", "explicit"), float(d.get("ts", 0.0)))
            except (ValueError, KeyError, TypeError):
                continue  # a torn or foreign line must not break the session
            if row.chosen in LABELS:
                rows.append(row)
        return rows

    def append(self, row: PrefRow) -> None:
        self.rows.append(row)
        if self.path is None:
            return
        record = {**dict(zip(FEATURES, row.features, strict=True)), "chosen": row.chosen,
                  "previous": row.previous, "source": row.source, "ts": round(row.ts, 3)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def append_attention(self, settings: dict[str, float], preset: str | None = None,
                         ts: float | None = None) -> None:
        """Record an attention choice (dwell_s, hold_s, switch_margin) so a per-person default
        can be learned later. Not a mode label: :meth:`load` skips these lines."""
        if self.path is None:
            return
        record = {"kind": "attention", **settings, "preset": preset,
                  "ts": round(time.time() if ts is None else ts, 3)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------- recommend


@dataclass(frozen=True)
class Recommendation:
    mode: str
    confidence: float
    reason: str

    def message(self, **extra) -> dict:
        return {"type": "recommendation", "mode": self.mode, "confidence": round(self.confidence, 2),
                "reason": self.reason, **extra}


def describe(v: ContextVector) -> str:
    """Short human reason built from the features that matter most."""
    parts = []
    if v.noise_db > -35:
        parts.append(f"noisy ({v.noise_db:.0f} dB)")
    elif v.noise_db < -50:
        parts.append("quiet room")
    if v.snr_db < 6 and v.speakers:
        parts.append(f"low SNR ({v.snr_db:.0f} dB)")
    if v.speakers >= 3:
        parts.append(f"{v.speakers} speakers")
    elif v.speakers == 2:
        parts.append("2 speakers")
    elif v.speakers == 0:
        parts.append("nobody talking")
    if v.overlap > 0.3:
        parts.append("overlapping talk")
    if v.target_conf >= 0.6:
        parts.append("clear target")
    elif v.speakers and v.target_conf < 0.3:
        parts.append("no clear target")
    return ", ".join(parts) or "room unchanged"


class Recommender:
    """Seed model + this user's logged choices; refits after every new choice."""

    def __init__(self, log: PrefLog | None = None, prior_weight: float = 50.0):
        self.log = log or PrefLog()
        self.prior_weight = prior_weight
        self.model = prior_model()
        self.fit_ms = 0.0
        if self.log.rows:
            self.refit()

    def refit(self) -> None:
        px, py, _ = _prior()
        rows = self.log.rows
        x = np.vstack([px, [r.features for r in rows]]) if rows else px
        y = list(py) + [r.chosen for r in rows]
        w = np.concatenate([np.full(len(px), self.prior_weight / len(px)),
                            [SOURCE_WEIGHT.get(r.source, 1.0) for r in rows]])
        t0 = time.perf_counter()
        self.model.fit(x, y, w, max_iter=60)  # warm start from the current weights
        self.fit_ms = 1000.0 * (time.perf_counter() - t0)

    def observe(self, vector: ContextVector, chosen: str, previous: str, source: str) -> None:
        if chosen not in LABELS:
            return
        self.log.append(PrefRow(vector.values(), chosen, previous, source))
        self.refit()

    def recommend(self, vector: ContextVector, allowed=LABELS) -> Recommendation:
        p = self.model.proba(vector.values())[0]
        idx = [LABELS.index(m) for m in allowed]
        best = idx[int(np.argmax(p[idx]))]
        conf = float(p[best] / p[idx].sum())
        return Recommendation(LABELS[best], conf, describe(vector))


# ---------------------------------------------------------------- toasts and Auto


@dataclass
class AdvisorConfig:
    min_conf: float = 0.6  # below this, say nothing and do not switch
    stable_s: int = 3  # the same recommendation this many seconds in a row
    toast_cooldown_s: float = 20.0  # between any two toasts
    repeat_s: float = 60.0  # before toasting the same mode again
    user_quiet_s: float = 10.0  # after the user changes mode by hand
    auto_cooldown_s: float = 10.0  # between two Auto switches
    dwell_s: float = 30.0  # log "kept this mode" every this many seconds
    accept_window_s: float = 30.0  # a change to the toasted mode within this counts as accepted
    # Auto's fallback while the model is unsure (< min_conf): the synth feed's rule (T-076)
    rule_speakers: int = 2  # this many talkers -> focus_hq at once
    rule_release_s: int = 5  # ... back to natural after this many seconds below that


class Advisor:
    """One per session. Call :meth:`on_context` once per context vector and
    :meth:`on_user_mode` on every explicit mode change."""

    def __init__(self, recommender: Recommender | None = None, config: AdvisorConfig | None = None,
                 initial_auto: str = "natural"):
        self.rec = recommender or Recommender()
        self.cfg = config or AdvisorConfig()
        self.auto_mode = initial_auto
        self.last: ContextVector | None = None
        self.now = 0.0
        self._streak_mode: str | None = None
        self._streak = 0
        self._auto_streak_mode: str | None = None
        self._auto_streak = 0
        self._last_toast = -1e9
        self._toasted: dict[str, float] = {}
        self._last_user = -1e9
        self._last_switch = -1e9
        self._dwell_since = 0.0
        self._setting: str | None = None  # the setting at the previous tick
        self._rule_low_s = 0  # seconds in a row below rule_speakers

    def on_user_mode(self, mode: str, previous: str) -> None:
        """Log an explicit choice against the latest room features."""
        self._last_user = self._dwell_since = self.now
        if self.last is None or mode == "auto" or mode == previous:
            return
        toasted = self._toasted.get(mode)
        accepted = toasted is not None and self.now - toasted <= self.cfg.accept_window_s
        self.rec.observe(self.last, mode, previous, "accepted" if accepted else "explicit")

    def on_context(self, vector: ContextVector, setting: str, now: float) -> list[dict]:
        """``setting`` is the user's mode (maybe ``auto``); returns events to send."""
        self.now, self.last = now, vector
        cfg, out = self.cfg, []
        if setting == "auto":
            first, self._setting = self._setting != "auto", setting
            return self._auto(vector, now, first)
        self._setting = setting
        if now - self._dwell_since >= cfg.dwell_s:
            self._dwell_since = now
            self.rec.observe(vector, setting, setting, "dwell")
        pick = self.rec.recommend(vector)
        self._streak = self._streak + 1 if pick.mode == self._streak_mode else 1
        self._streak_mode = pick.mode
        if (pick.mode != setting and pick.confidence >= cfg.min_conf and self._streak >= cfg.stable_s
                and now - self._last_toast >= cfg.toast_cooldown_s
                and now - self._toasted.get(pick.mode, -1e9) >= cfg.repeat_s
                and now - self._last_user >= cfg.user_quiet_s):
            self._last_toast = self._toasted[pick.mode] = now
            out.append(pick.message(auto=False))
        return out

    def _rule(self, vector: ContextVector) -> tuple[str, str]:
        """The deterministic Auto rule (the synth feed's, ``Session._update_auto_feed``):
        focus_hq at once with 2+ talkers; below that, hold the current mode for
        ``rule_release_s`` seconds, then natural. Returns (mode, reason)."""
        n = self.cfg.rule_speakers
        if vector.speakers >= n:
            self._rule_low_s = 0
            return "focus_hq", f"rule: {vector.speakers} talkers"
        self._rule_low_s += 1
        if self.auto_mode != "natural" and self._rule_low_s < self.cfg.rule_release_s:
            return self.auto_mode, f"rule: {self._rule_low_s} s below {n} talkers, holding"
        return "natural", f"rule: fewer than {n} talkers"

    def _auto(self, vector: ContextVector, now: float, first: bool) -> list[dict]:
        """One Auto tick: the learned model when it is confident (stable for ``stable_s``,
        ``auto_cooldown_s`` between its switches), else :meth:`_rule`, so Auto is never
        dumber than the rule. Every switch, and the first tick in Auto, is a recommendation."""
        cfg = self.cfg
        if first:
            self._rule_low_s = 0
        pick = self.rec.recommend(vector, AUTO_MODES)
        self._auto_streak = self._auto_streak + 1 if pick.mode == self._auto_streak_mode else 1
        self._auto_streak_mode = pick.mode
        rule, why = self._rule(vector)  # always ticked so its release clock stays current
        if pick.confidence >= cfg.min_conf:
            if (pick.mode != self.auto_mode and self._auto_streak >= cfg.stable_s
                    and now - self._last_switch >= cfg.auto_cooldown_s):
                self.auto_mode = pick.mode
                self._last_switch = now
                return [pick.message(auto=True, source="model")]
            if first:  # entering Auto: say what it plays and who decides
                why = f"model ({pick.mode} {pick.confidence:.2f}): {pick.reason}"
                return [Recommendation(self.auto_mode, pick.confidence, why).message(auto=True, source="model")]
            return []
        if rule != self.auto_mode or first:
            if rule != self.auto_mode:
                self.auto_mode, self._last_switch = rule, now
            return [Recommendation(rule, 1.0, why).message(auto=True, source="rule")]
        return []


# ---------------------------------------------------------------- evaluation


def evaluate(seed: int = 1, log: Path | None = None) -> dict:
    """Held-out accuracy of the seed model, and of personalisation for a second persona."""
    model = prior_model()
    x_test, y_test = simulate(1000, seed=seed)
    clean_test = [simulated_choice(f) for f in x_test]
    majority = max(set(y_test), key=list(y_test).count)
    result = {
        "prior_heldout_acc": model.accuracy(x_test, y_test),
        "prior_heldout_acc_vs_policy": model.accuracy(x_test, clean_test),
        "majority_baseline_acc": float(np.mean(np.array(y_test) == majority)),
        "label_noise": 0.1,
    }
    # personalisation: a revoicer persona contributes 60 choices; test on 500 more of theirs
    xr, yr = simulate(560, seed=seed + 1, user="revoicer")
    plog = PrefLog()
    for f, label in zip(xr[:60], yr[:60], strict=True):
        plog.rows.append(PrefRow(list(f), label, "", "explicit"))
    personal = Recommender(plog)
    result["revoicer_acc_prior_only"] = model.accuracy(xr[60:], yr[60:])
    result["revoicer_acc_after_60_choices"] = personal.model.accuracy(xr[60:], yr[60:])
    result["refit_ms"] = round(personal.fit_ms, 1)
    vec = ContextVector(-30, 5, 3, 0.4, 0.8, "natural")
    t0 = time.perf_counter()
    for _ in range(100):
        personal.recommend(vec)
    result["recommend_ms"] = round(10.0 * (time.perf_counter() - t0), 3)
    if log is not None:
        rows = PrefLog.load(Path(log))
        if len(rows) >= 10:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(rows))
            cut = max(1, len(rows) // 5)
            test, train = [rows[i] for i in order[:cut]], [rows[i] for i in order[cut:]]
            r = Recommender(PrefLog())
            r.log.rows = train
            r.refit()
            result["log_rows"] = len(rows)
            result["log_heldout_acc"] = r.model.accuracy([t.features for t in test], [t.chosen for t in test])
        else:
            result["log_rows"] = len(rows)
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Held-out accuracy of the context preference model.")
    ap.add_argument("--log", type=Path, help="a prefs JSONL log to evaluate as well (80/20 split)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    for k, v in evaluate(args.seed, args.log).items():
        print(f"{k:32s} {v}")


if __name__ == "__main__":
    main()
