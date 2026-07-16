"""RocketEnsemble: four decorrelated components fused by weighted log-odds.

Mirrors UID163's rocket_ensemble.py + d0_features split:
  * stack -> StackingClassifier(LGBM+XGBoost+CatBoost+ExtraTrees+RF -> LR meta)
             trained on the PH ("phasberg") view: hero + behavioral features.
  * mono  -> soft-voting committee of sign-constrained/monotone XGBoost models,
             also on the PH view.
  * mlp   -> soft-voting committee of PCA->MLP models on the UN (union of PH+V2)
             view.
  * drse  -> drift-robust subspace ensemble (bagged ExtraTrees/HistGBM on random
             feature subspaces) on the V2 ("hero-free") view.

One shared `blend()` is used by both the trainer's walk-forward weight search
and the serving path, so the weights are always selected against the exact
function that ships.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

PARTS = ("stack", "mono", "mlp", "drse")
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def blend(parts: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Weighted log-odds fusion. `parts[name]` are per-component P(bot)."""
    wsum = sum(weights[name] for name in PARTS)
    z = np.zeros_like(np.asarray(next(iter(parts.values())), dtype=float))
    for name in PARTS:
        z = z + weights[name] * _logit(parts[name])
    return _sigmoid(z / wsum)


class RocketEnsemble:
    """Picklable container for the four fitted components + their weights."""

    fusion = "weighted-logit"

    def __init__(self, stack, mono, mlp, drse, cols_ph, cols_v2, weights):
        self.stack = stack
        self.mono = mono
        self.mlp = mlp
        self.drse = drse
        self.cols_ph = list(cols_ph)
        self.cols_v2 = list(cols_v2)
        self.weights = dict(weights)

    def components(self, ph: np.ndarray, v2: np.ndarray) -> Dict[str, np.ndarray]:
        un = np.hstack([v2, ph])
        return {
            "stack": self.stack.predict_proba(ph)[:, 1],
            "mono": self.mono.predict_proba(ph)[:, 1],
            "mlp": self.mlp.predict_proba(un)[:, 1],
            "drse": self.drse.predict_proba(v2)[:, 1],
        }

    def score(self, ph: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return blend(self.components(ph, v2), self.weights)
