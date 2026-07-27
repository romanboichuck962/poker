"""Current Poker44 reward wrapper and standard diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from poker44.score.scoring import reward


def metrics(labels, scores) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    network_reward, details = reward(p, y)
    return {
        "reward": float(network_reward),
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        **{key: float(value) for key, value in details.items()},
    }
