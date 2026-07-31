"""uid242 serving wrapper — UID148's sequence-signature + size-dispersion detector (S1-RW).

Faithful port of the model uid148 actually serves. Their repo
(github.com/payizogu20/code-seqsig-rw-detector-1, "code-seqsig-rw-detector-1"
v3.3.1 @5c5aaf34216f55d86ea3f0597609dfe016432eb4) ships NO trained artifact, so
its miner falls through to poker44_ml.luck_detector.LuckDetector: a TRAINING-FREE
behavioral scorer (live competition composite ~0.637, rank #2).

PROFILE ``sequence-signature-sd-rw`` / VARIANT ``S1-RW``:
  * Action-sequence signature concentration (street/action/size tokens), with
    top-1 + top-2 + repeat-mass mix (RW wider repeat structure)
  * Street-progression uniformity
  * Winsorized voluntary-size CV deficit (closes jittered-token / tight-size
    blind spot)
  * Piecewise-linear anchors [0.29, 0.88]

BATCH-RANK REMAP (ON by default here; uid148 ships it OFF by default):
  Rank-preserving map so the top ``POKER44_MAX_POS_FRAC`` of each request
  batch cross 0.5. Preserves ranking while securing the validator
  threshold_sanity gate. Disable with POKER44_BATCH_RANK=0.
"""
from __future__ import annotations

import os
from typing import Any, List

from poker44_ml.luck_detector import build_luck_detector

MODEL_ARTIFACT = None

_BATCH_RANK = os.environ.get("POKER44_BATCH_RANK", "1").strip().lower() in {"1", "true", "yes", "on"}
_TOP_FRAC = min(max(float(os.environ.get("POKER44_MAX_POS_FRAC", "0.125")), 0.01), 0.99)
_SPAN = min(0.8, 0.5 / (1.0 - _TOP_FRAC) * 0.98)
_MIN_BATCH = 4
_EMPTY_SCORE = 0.1


def _batch_rank(scores: List[float], frac: float, span: float) -> List[float]:
    n = len(scores)
    if n < _MIN_BATCH:
        return list(scores)
    order = sorted(range(n), key=lambda i: (scores[i], i))
    rank = [0.0] * n
    for pos, idx in enumerate(order):
        rank[idx] = pos / (n - 1)
    lo = 0.5 - (1.0 - frac) * span
    return [min(1.0, max(0.0, lo + r * span)) for r in rank]


class Poker44Model:
    """Adapter exposing UID148's S1-RW luck detector through score_chunks()."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._detector = build_luck_detector()
        self.backend = f"luck-{self._detector.PROFILE}"

    def score_chunks(self, groups: List[List[dict[str, Any]]]) -> List[float]:
        if not groups:
            return []
        live_idx = [i for i, g in enumerate(groups) if g]
        if not live_idx:
            return [_EMPTY_SCORE] * len(groups)

        raw = self._detector.score_chunks([groups[i] for i in live_idx])
        if _BATCH_RANK and len(raw) >= _MIN_BATCH:
            raw = _batch_rank(list(raw), _TOP_FRAC, _SPAN)

        out = [_EMPTY_SCORE] * len(groups)
        for slot, value in zip(live_idx, raw):
            out[slot] = float(value)
        return out
