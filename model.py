"""Poker44 serving model for uid 242 — V4 coherent rank-robust detector.

Faithful adaptation of UID176's public pd-coast model_v4 ("p44-v4-coherent
-rank-robust", v4.1): 821 order-invariant features (353-col distribution view
+ 468 chunk-coherence columns), nine branches (six on raw features, three on
request-relative feature percentiles), walk-forward-selected branch weights /
blend mode / positive fraction, and an exact rank-preserving boundary map that
places the top fraction of each request batch just above 0.5.

The request-relative percentile branches are the core idea: live validator
payloads are far outside the labeled benchmark envelope (we measured ~37% of
feature cells outside q01-q99 on captured chunks, matching UID176's own
report), which collapses raw probabilities. Within-request feature ranks keep
the behavioral ordering, which is all AP/recall need.

Interface kept identical to the uid 77 model: MODEL_ARTIFACT + Poker44Model
with score_chunks(chunks) -> list[float].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np

from v4.calibration import apply_mapper
from v4.features import FEATURE_NAMES, FEATURE_SCHEMA_SHA256, matrix_for_chunks
from v4.mapping import chunk_tie_key, exact_rank_map
from v4.model import BRANCH_NAMES, blend_branches
from v4.schema import clean_hand

MODEL_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "poker44_model.joblib"

# Percentile fusion and the exact rank map are meaningful only for a
# validator-sized request; smaller calls use calibrated probability behavior.
_MIN_BATCH_FOR_RANK = 8
_EMPTY_SCORE = 0.1


class Poker44Model:
    """Load and serve a V4 coherent artifact."""

    def __init__(self, artifact_path: Path | str = MODEL_ARTIFACT) -> None:
        artifact: Dict[str, Any] = joblib.load(artifact_path)
        if artifact.get("artifact_version") != 4:
            raise ValueError("not a Poker44 V4 artifact")
        if list(artifact.get("feature_names") or []) != FEATURE_NAMES:
            raise ValueError("V4 feature schema mismatch; retrain the artifact")
        if artifact.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
            raise ValueError("V4 feature schema fingerprint mismatch; retrain the artifact")
        self.artifact = artifact
        self.model = artifact["model"]
        if tuple(getattr(self.model, "branch_names", ())) != BRANCH_NAMES:
            raise ValueError("V4 branch schema mismatch; retrain the artifact")
        self.mapper = dict(artifact["mapper"])
        self.blend_mode = str(artifact.get("blend_mode", "probability"))
        if self.blend_mode not in {"probability", "rank"}:
            raise ValueError("V4 blend mode must be probability or rank")
        fraction = os.getenv("P44_TOP_FRAC") or artifact.get("batch_top_fraction", 0.10)
        self.batch_top_fraction = float(fraction)
        if not 0.0 < self.batch_top_fraction < 1.0:
            raise ValueError("V4 batch_top_fraction must be between zero and one")
        if not {"cut", "scale"}.issubset(self.mapper) or float(self.mapper["scale"]) <= 0.0:
            raise ValueError("V4 mapper is invalid")

    @staticmethod
    def _clean(chunks: Sequence[Sequence[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        return [
            [clean_hand(hand) for hand in (chunk or []) if isinstance(hand, dict)]
            for chunk in chunks
        ]

    def score_chunks(self, chunks: Sequence[Sequence[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        clean = self._clean(chunks)
        empty = np.asarray([len(chunk) == 0 for chunk in clean], dtype=bool)
        matrix = matrix_for_chunks(clean)
        branches = self.model.branch_scores(matrix)
        mode = self.blend_mode if len(clean) >= _MIN_BATCH_FOR_RANK else "probability"
        keys = [chunk_tie_key(chunk) for chunk in clean]
        raw = blend_branches(branches, self.model.branch_weights_, mode, tie_keys=keys)
        # Empty chunks carry no behavior: rank them at the bottom so the exact
        # rank map can never spend a positive slot on one.
        raw = np.where(empty, 1e-6, raw)
        if len(clean) >= _MIN_BATCH_FOR_RANK:
            scores = exact_rank_map(raw, self.batch_top_fraction, tie_keys=keys)
        else:
            scores = apply_mapper(raw, self.mapper)
        scores = np.nan_to_num(scores, nan=0.5, posinf=0.99, neginf=0.01)
        scores = np.where(empty, _EMPTY_SCORE, scores)
        return [round(float(np.clip(value, 0.01, 0.99)), 8) for value in scores]
