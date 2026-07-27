"""Small, picklable heterogeneous probability ensemble."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ProbabilityEnsemble:
    def __init__(self, models: Sequence[object], weights: Sequence[float]):
        if not models or len(models) != len(weights):
            raise ValueError("models and weights must be non-empty and aligned")
        clean = np.asarray(weights, dtype=float)
        if np.any(clean < 0) or not np.isfinite(clean).all() or clean.sum() <= 0:
            raise ValueError("ensemble weights must be finite, non-negative, and non-zero")
        self.models = list(models)
        self.weights = (clean / clean.sum()).tolist()

    def predict_proba(self, frame):
        columns = [np.asarray(model.predict_proba(frame), dtype=float)[:, 1] for model in self.models]
        bot = np.average(np.stack(columns, axis=1), axis=1, weights=self.weights)
        bot = np.clip(bot, 0.0, 1.0)
        return np.stack([1.0 - bot, bot], axis=1)
