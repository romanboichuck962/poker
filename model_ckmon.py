"""uid167 serving wrapper — UID157's ckmon stacked coherence model.

Vendored under poker44_ml/ from https://github.com/TakashiAkio/poker44-first
@d226c492 (MIT - see LICENSE-uid157). Architecture:

* Features: hand-order-stat aggregates + fixed n-gram vocab + pd-coast V4
  cross-hand coherence block (~700+ dims).
* Stack: LightGBM + XGBoost + CatBoost + ExtraTrees + RandomForest +
  HistGradientBoosting -> LogisticRegression meta, blended isotonic.
* Operating point: within-batch RANK map at POKER44_MAX_POS_FRAC (default 0.16).
  UID157's fixed score_calibrator compresses live scores; the rank map keeps
  the 0.5 crossing gate-safe without touching AP/recall ranking.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from poker44_ml.inference import Poker44Model as _CkmonModel

MODEL_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "poker44_ckmon_1.joblib"

_TOP_FRAC = min(max(float(os.environ.get("POKER44_MAX_POS_FRAC", "0.16")), 0.01), 0.99)
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
    """Adapter exposing UID157's Poker44Model through score_chunks()."""

    def __init__(self, artifact_path: Path | str = MODEL_ARTIFACT) -> None:
        self._model = _CkmonModel(artifact_path)

    def score_chunks(self, groups: List[List[dict[str, Any]]]) -> List[float]:
        if not groups:
            return []
        live_idx = [i for i, group in enumerate(groups) if group]
        if not live_idx:
            return [_EMPTY_SCORE] * len(groups)
        raw = self._model.predict_chunk_scores([groups[i] for i in live_idx])
        ranked = _batch_rank([float(v) for v in raw], _TOP_FRAC, _SPAN)
        out = [_EMPTY_SCORE] * len(groups)
        for slot, value in zip(live_idx, ranked):
            out[slot] = float(value)
        return out
