"""uid113 serving wrapper — uid254's super_poker_3 (XGBoost) model.

The model and its feature/inference code are uid254's, vendored under super_poker/
from https://github.com/BranDonallenLuis/super_poker_3 @2bf6668f18b6 (MIT - see
LICENSE-uid254). This module only adapts it to our miner's score_chunks() interface
and guards empty chunks.

Operating point: uid254 serves a fitted threshold->0.5 piecewise remap and no batch
budget. Their benchmark-fitted threshold on our 63-release retrain was 0.5759, which
put 96.5% of captured live chunks above 0.5 (raw live median 0.657) -> reward hard-gate
risk. So the artifact's threshold is capture-calibrated instead: quantile(live raw
proba, 0.875) over our captured validator chunks = 0.738, i.e. a 12.5% live positive
rate. This keeps uid254's exact serving mechanism (their SuperPokerModel._remap), just
with the threshold set from our live distribution rather than the benchmark.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from super_poker.inference import SuperPokerModel

MODEL_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "super_poker_3.joblib"

# Empty/unscoreable chunk -> below the 0.5 boundary, never a false positive.
# (uid254's raw model scores an all-zero feature vector at ~0.71, a false bot flag.)
_EMPTY_SCORE = 0.1


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
        scores = self._model.predict_chunk_scores([groups[i] for i in live_idx])
        out = [_EMPTY_SCORE] * len(groups)
        for slot, value in zip(live_idx, scores):
            out[slot] = float(value)
        return out
