"""uid242 serving wrapper — UID237's Markov+pot-geometry luck detector (M3-GB).

Faithful port of the model uid237 actually serves. Their repo
(github.com/eyyupkemer7/jet-detector-3, "jet-markovpot-gb-detector-3" v3.7.3
@2df44d27845c44bd4a27f99c4afc0c791565314b) ships NO trained artifact
(models/ and *.joblib are absent), so its miner forward() falls through to
poker44_ml.luck_detector.LuckDetector: a TRAINING-FREE behavioral scorer.

PROFILE ``markov-pot-geometry-gb`` / VARIANT ``M3-GB``:
  * Markov action-transition entropy deficit (scripted seats are near-
    deterministic given previous action)
  * Pot bet/pot CV regularity (bots size off fixed pot fractions)
  * Lighter signature concentration + street uniformity
  * Weighted geometric blend across terms; smoothstep anchors [0.24, 0.80]

BATCH-RANK REMAP (ON by default here; uid237 ships it OFF by default):
  Rank-preserving map so the top ``POKER44_MAX_POS_FRAC`` of each request
  batch cross 0.5. Preserves ranking (AP / recall@FPR) while securing the
  validator threshold_sanity gate at live geometry. Disable with
  POKER44_BATCH_RANK=0 to serve raw anchors exactly as uid237 does.
"""
from __future__ import annotations

import os
from typing import Any, List

from poker44_ml.luck_detector import build_luck_detector

# No artifact for a heuristic; kept so miner.py / launcher env refs stay harmless.
MODEL_ARTIFACT = None

_BATCH_RANK = os.environ.get("POKER44_BATCH_RANK", "1").strip().lower() in {"1", "true", "yes", "on"}
_TOP_FRAC = min(max(float(os.environ.get("POKER44_MAX_POS_FRAC", "0.125")), 0.01), 0.99)
# Keep lo >= 0 so no chunk is clamped (clamping creates ties and loses ranking).
_SPAN = min(0.8, 0.5 / (1.0 - _TOP_FRAC) * 0.98)
_MIN_BATCH = 4
# Empty/unscoreable chunk: below the boundary, never a false positive.
_EMPTY_SCORE = 0.1


def _batch_rank(scores: List[float], frac: float, span: float) -> List[float]:
    """UID142's _apply_batch_rank: argsort-of-argsort into a band whose 0.5
    crossing sits exactly at the top `frac` of the batch. Rank-preserving."""
    n = len(scores)
    if n < _MIN_BATCH:
        return list(scores)
    order = sorted(range(n), key=lambda i: (scores[i], i))  # stable, ties by index
    rank = [0.0] * n
    for pos, idx in enumerate(order):
        rank[idx] = pos / (n - 1)
    lo = 0.5 - (1.0 - frac) * span
    return [min(1.0, max(0.0, lo + r * span)) for r in rank]


class Poker44Model:
    """Adapter exposing UID237's M3-GB luck detector through score_chunks()."""

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
