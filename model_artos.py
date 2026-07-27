"""uid167 serving wrapper — uid225's artos model.

The model and its feature/ensemble/policy code are uid225's, vendored under
poker44_ml/ from https://github.com/judev113/p44-artos-v1 @0976ec00f74c (MIT -
see LICENSE-artos). This module only adapts artos's Detector to our miner's
score_chunks() interface.

Operating point is artos's OWN within-batch rank map (poker44_ml.policy.rank_map)
at the positive_fraction stored in the artifact (0.07, chosen by their gate-safe
fraction sweep on our benchmark). It flags exactly ~7% of each request batch and
can never trip the zero-true-positive gate, so no external batch-rank fix is
needed. Empty/degenerate chunks are left to artos: rank_map ranks them at the
bottom band (~0.01), i.e. correctly negative.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

from poker44_ml.inference import Detector

MODEL_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "artos_v1.joblib"


class Poker44Model:
    """Adapter exposing uid225's artos Detector through our miner's score_chunks()."""

    def __init__(self, artifact_path: Path | str = MODEL_ARTIFACT) -> None:
        self._detector = Detector.load(artifact_path)

    def score_chunks(self, groups: List[List[dict[str, Any]]]) -> List[float]:
        if not groups:
            return []
        # artos applies its within-batch rank_map internally (>=8 chunks); a live
        # request carries ~100, so the operating point is placed by artos itself.
        return [float(s) for s in self._detector.predict_chunk_scores(groups)]
