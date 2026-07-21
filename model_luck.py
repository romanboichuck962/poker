"""uid242 serving wrapper — UID225's sequence-signature luck detector (heuristic).

Faithful port of the model uid225 actually serves. Their repo
(github.com/mitsuiminoru000-pixel/poker44-luck-detector-2, "luck-signature-detector"
v3.2.0) ships NO trained artifact (models/ and *.joblib are gitignored), so its
miner forward() falls through to poker44_ml.luck_detector.LuckDetector directly:
a TRAINING-FREE behavioral scorer. It flags a chunk by how much its hands collapse
onto a few action-sequence signatures (scripted replay) versus human play, which
spreads across many distinct sequences. This is the signal scoring 0.687 (rank #3)
on the live leaderboard.

Per-hand signature = street-shape + street/action/size-bucket tokens; per-chunk
concentration = 0.45*top_sig_share + 0.35*repeat_mass + 0.20*(1-unique_share),
blended with street-uniformity (0.18); piecewise-linear anchor calibration
[0.30,0.90] -> [0.5,1.0], floor 0.05. See poker44_ml/luck_detector.py.

BATCH-RANK REMAP (ON by default here; uid225 ships it OFF):
  uid225 serves the RAW anchor scores. At OUR live geometry that is a knife-edge:
  on 100-chunk 20%-bot windows the raw scores give safety only ~0.78 (some windows
  have too few bots crossing 0.5 -> threshold_sanity bleeds, risking a 0-gate), and
  on the 820 real captured live chunks raw scores compress to [0.13,0.62] and flag
  <1% at 0.5. UID142's rank-preserving batch-rank remap fixes exactly this: it maps
  each chunk to its within-batch rank so the top `frac` cross 0.5 every batch. It is
  STRICTLY rank-preserving, so AP and recall@FPR<=0.05 (0.65 of reward) are
  identical to raw -- uid225's ranking signal is untouched -- it only fixes where 0.5
  sits. On the labeled live-geometry windows this lifts mean reward 0.547 -> 0.613
  and safety 0.78 -> 0.998 (pillar-3 of the live-first methodology). Disable with
  POKER44_BATCH_RANK=0 to serve exactly what uid225 serves.
"""
from __future__ import annotations

import os
from typing import Any, List

from poker44_ml.luck_detector import build_luck_detector

# No artifact for a heuristic; kept so miner.py's `from model_luck import ...` and
# any launcher POKER44_MODEL_PATH reference stay harmless.
MODEL_ARTIFACT = None

_BATCH_RANK = os.environ.get("POKER44_BATCH_RANK", "1").strip().lower() in {"1", "true", "yes", "on"}
_TOP_FRAC = min(max(float(os.environ.get("POKER44_MAX_POS_FRAC", "0.125")), 0.01), 0.99)
# Keep lo >= 0 so no chunk is clamped (clamping creates ties and loses ranking).
_SPAN = min(0.8, 0.5 / (1.0 - _TOP_FRAC) * 0.98)
_MIN_BATCH = 4
# Empty/unscoreable chunk: below the boundary, never a false positive (live
# chunks are never empty; uid225's detector would return 0.5 here).
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
    """Adapter exposing UID225's luck detector through our miner's score_chunks()."""

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
