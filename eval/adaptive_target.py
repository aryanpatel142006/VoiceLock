"""Adaptive target prototype from confident solo crops (T-380, offline only, knob default off).

``AdaptiveTargetCandidate`` is an :class:`EnrolledCandidate` whose closed-set LDA target class also learns
from the stream it just played, but only on a window that is (a) confident: the pick's ``margin_score`` is at
or above the hold threshold, the F0 cue did not overturn it, nothing held it and it was not gated, and
(b) solo: exactly one active stream (``StreamVad``: some voiced time, none shared). The played stream's
unit embedding is appended to the target piece embeddings (most recent ``max_crops`` kept) and the LDA is
refit; the saved profile pieces and the negative are never changed, and ``reset`` drops the learned crops.
"""
from __future__ import annotations

import numpy as np

from hearing.tse.enrolled import (
    LDA_HOLD_SCALE,
    EnrolledCandidate,
    _unit_embedding,
    fit_lda,
)


def solo_activity(act: dict) -> bool:
    """Exactly one active stream in the window: voiced time exists and no stream pair shares it."""
    return act["any_s"] > 0.0 and act["both_s"] == 0.0 and not act["overlap"]


class AdaptiveTargetCandidate(EnrolledCandidate):
    def __init__(self, *args, max_crops: int = 12, **kw):
        self.max_crops = int(max_crops)
        self.crops: list[np.ndarray] = []
        self.updates = 0
        super().__init__(*args, **kw)

    def reset(self):
        super().reset()
        self.crops = []
        self.updates = 0

    def _closed_set(self):
        t, o = self._pieces.get("target"), self._pieces.get("other")
        if not self.crops or not self.closed_set or t is None or o is None or len(t[1]) < 2 or len(o[1]) < 2:
            return super()._closed_set()
        key = (t[0], o[0], self.updates)
        if self._lda is None or self._lda[0] != key:
            self._lda = (key, fit_lda(np.vstack([t[1], np.stack(self.crops)]), o[1]))
        return self._lda[1]

    def confident(self) -> bool:
        """The last window's pick may teach the target: margin >= hold, not vetoed/held/gated."""
        hold = self.margin * LDA_HOLD_SCALE if self.scoring == "lda" else self.margin
        return (self.margin_score is not None and self.margin_score >= hold and not self.vetoed
                and not self.held and not self.gated and self.selected_stream is not None)

    def learn(self, stream: np.ndarray, act: dict) -> bool:
        """Offer the played stream after ``extract``; True when it was added to the target class."""
        if self.scoring != "lda" or not self.confident() or not solo_activity(act):
            return False
        self.crops.append(_unit_embedding(self.embedder, stream))
        del self.crops[: -self.max_crops]
        self.updates += 1
        return True
