"""Fixed sigmoid score mapper (small-batch fallback), from pd-coast model_v3."""

from __future__ import annotations

from typing import Dict

import numpy as np


def fit_fixed_mapper(
    scores: np.ndarray,
    labels: np.ndarray,
    target_human_fpr: float = 0.05,
) -> Dict[str, float]:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    human = s[y == 0]
    if human.size == 0:
        cut = float(np.median(s))
    else:
        cut = float(np.quantile(human, 1.0 - target_human_fpr, method="higher"))
    spread = float(np.quantile(s, 0.75) - np.quantile(s, 0.25))
    return {"cut": cut, "scale": max(spread, 0.03), "target_human_fpr": target_human_fpr}


def apply_mapper(scores: np.ndarray, mapper: Dict[str, float]) -> np.ndarray:
    z = (np.asarray(scores, float) - float(mapper["cut"])) / float(mapper["scale"])
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -20, 20))), 0.01, 0.99)
