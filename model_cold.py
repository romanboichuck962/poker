"""uid242 serving wrapper — UID142's "cold" stacked model.

The model itself (539 robust features, 5-tree stack, logistic meta with
hard-bot reweighting, blended isotonic calibrator, threshold remap) is
UID142's, vendored under poker44_ml/ from
https://github.com/david10301-code/Poker44-cold-poker1 (MIT, see
LICENSE-uid142). This module only adapts it to our miner's score_chunks()
interface and fixes the serving operating point.

WHY THE BATCH-RANK REMAP IS ON:
Scored against 600 captured live validator chunks, the artifact's fixed
threshold remap (0.70/0.25) puts EVERY live chunk above 0.5 -- the live
calibrated-score distribution is compressed into [0.78, 0.94] (q50 0.882,
q90 0.895), so a fixed cut is a knife-edge: 0.88 -> 55% positive, 0.90 -> 7%.
Serving that as-is means FPR@0.5 ~ 1.0, threshold_sanity collapses, and the
validator reward hard-gates to 0. UID142's own diagnose_live_scores.py flags
exactly this case (bot_rate > 0.85 => HIGH RISK).

So we enable UID142's own batch-rank remap (their neurons/miner.py
::_apply_batch_rank, env-gated there) instead of relying on the fixed cut. It
is strictly rank-preserving, so AP and recall@FPR<=0.05 -- 0.65 of the reward
-- are untouched; it only decides where 0.5 sits, which makes the operating
point immune to the benchmark->live distribution shift.

span is chosen so lo = 0.5 - (1-frac)*span >= 0, keeping the map strictly
increasing. Their default span of 0.8 at frac 0.125 would clamp the bottom
~34% of every batch to 0.0, creating ties and discarding that ranking.

frac 0.125 was selected on the labeled holdout (last 2 releases) at live
geometry -- 100-chunk windows, 20% bots, scored exactly as served:
mean reward 0.9571, p10 0.9278, 0/40 zero-gates, FPR@0.5 0.0003.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from poker44_ml.inference import Poker44Model as _ColdModel

MODEL_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "poker44_cold.joblib"

# Fraction of each request batch mapped above 0.5.
_TOP_FRAC = min(max(float(os.environ.get("POKER44_MAX_POS_FRAC", "0.125")), 0.01), 0.99)
# Keep lo >= 0 so no chunk is clamped (clamping creates ties and loses ranking).
_SPAN = min(0.8, 0.5 / (1.0 - _TOP_FRAC) * 0.98)
# UID142 applies the rank map only once a batch is large enough to rank.
_MIN_BATCH = 4
# Score for an empty/unscoreable chunk: below the boundary, never a false positive.
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
    """Adapter exposing UID142's model through our miner's score_chunks()."""

    def __init__(self, artifact_path: Path | str = MODEL_ARTIFACT) -> None:
        self._model = _ColdModel(artifact_path)

    def score_chunks(self, groups: List[List[dict[str, Any]]]) -> List[float]:
        if not groups:
            return []
        # Empty chunks are handled separately: featurizing them is meaningless
        # and they must not consume a positive slot in the batch budget.
        live_idx = [i for i, group in enumerate(groups) if group]
        if not live_idx:
            return [_EMPTY_SCORE] * len(groups)

        raw = self._model.predict_chunk_scores([groups[i] for i in live_idx])
        ranked = _batch_rank(raw, _TOP_FRAC, _SPAN)

        out = [_EMPTY_SCORE] * len(groups)
        for slot, value in zip(live_idx, ranked):
            out[slot] = float(value)
        return out
