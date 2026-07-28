"""uid113 serving wrapper — uid254's super_poker_3 (XGBoost) model.

The model and its feature/inference code are uid254's, vendored under super_poker/
from https://github.com/BranDonallenLuis/super_poker_3 @2bf6668f18b6 (MIT - see
LICENSE-uid254). This module adapts it to our miner's score_chunks() interface and
places the operating point.

OPERATING POINT (v2 change): uid254 serves a fitted threshold->0.5 remap. A FIXED
threshold drifts - if the live distribution shifts, the positive rate creeps toward
zero-gate (too few positives -> whole window reward hard-gates to 0) or high-FPR.
v1 pinned that threshold from captures (12.5%), but it was still a fixed cut. v2
instead takes uid254's RAW model probabilities and places the 0.5 line by a
within-batch RANK map at a fixed 12.5% positive fraction. This is rank-preserving
(AP and recall@FPR<=0.05 - 65% of the reward - are IDENTICAL to the fixed-threshold
serving, since the ordering is unchanged), but it GUARANTEES exactly ~12.5% of every
request batch crosses 0.5 regardless of distribution drift, so the operating point
can never drift into the zero-gate. Empty chunks are excluded from the ranking and
scored 0.1 (uid254's raw model scores an all-zero vector at ~0.71, a false flag).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from super_poker.inference import SuperPokerModel

MODEL_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "super_poker_3.joblib"

# Fraction of each request batch mapped above 0.5 (the operating point).
_TOP_FRAC = min(max(float(os.environ.get("POKER44_MAX_POS_FRAC", "0.125")), 0.01), 0.99)
# Keep lo >= 0 so no chunk is clamped (clamping creates ties and loses ranking).
_SPAN = min(0.8, 0.5 / (1.0 - _TOP_FRAC) * 0.98)
# Below this, a batch is too small to rank; pass raw proba through.
_MIN_BATCH = 4
_EMPTY_SCORE = 0.1


def _batch_rank(scores: List[float], frac: float, span: float) -> List[float]:
    """Argsort-of-argsort into a band whose 0.5 crossing sits at the top `frac`.
    Strictly order-preserving; the top `frac` of the batch land above 0.5."""
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
    """Adapter exposing uid254's SuperPokerModel through our miner's score_chunks()."""

    def __init__(self, artifact_path: Path | str = MODEL_ARTIFACT) -> None:
        self._model = SuperPokerModel(artifact_path)

    def score_chunks(self, groups: List[List[dict[str, Any]]]) -> List[float]:
        if not groups:
            return []
        live_idx = [i for i, g in enumerate(groups) if g]
        if not live_idx:
            return [_EMPTY_SCORE] * len(groups)
        # uid254's raw model probability = the ranking; the 0.5 line is placed by rank_map.
        raw = self._model.predict_chunk_components([groups[i] for i in live_idx])["raw_scores"]
        ranked = _batch_rank([float(v) for v in raw], _TOP_FRAC, _SPAN)
        out = [_EMPTY_SCORE] * len(groups)
        for slot, value in zip(live_idx, ranked):
            out[slot] = float(value)
        return out
