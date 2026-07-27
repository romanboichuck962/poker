"""Artifact loading and independent chunk inference."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from super_poker.features import chunk_features


class SuperPokerModel:
    def __init__(self, artifact_path: str | Path):
        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        self.feature_names = list(artifact["feature_names"])
        self.threshold = float(artifact["threshold"])
        self.metadata = dict(artifact.get("metadata") or {})

    @staticmethod
    def _remap(score: float, threshold: float) -> float:
        threshold = min(max(threshold, 1e-6), 1 - 1e-6)
        if score >= threshold:
            return 0.5 + 0.5 * (score - threshold) / (1 - threshold)
        return 0.5 * score / threshold

    def predict_chunk_components(self, chunks: list[list[dict]]) -> dict[str, list[float]]:
        if not chunks:
            return {"raw_scores": [], "final_scores": []}
        frame = pd.DataFrame([chunk_features(chunk) for chunk in chunks])
        frame = frame.reindex(columns=self.feature_names, fill_value=0.0).fillna(0.0)
        raw = self.model.predict_proba(frame.astype(float))[:, 1]
        scores = []
        for value in raw:
            score = self._remap(float(value), self.threshold)
            scores.append(round(min(1.0, max(0.0, score)) if math.isfinite(score) else 0.5, 6))
        return {
            "raw_scores": [round(float(value), 6) for value in raw],
            "final_scores": scores,
        }

    def predict_chunk_scores(self, chunks: list[list[dict]]) -> list[float]:
        return self.predict_chunk_components(chunks)["final_scores"]
