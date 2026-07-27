"""The validator reward — imported from upstream, never reimplemented.

Every competing miner repo we studied made the same mistake at least once: they
copied the reward formula into their own metrics module, the subnet changed the
formula, and their training silently optimized a stale objective for weeks.

Observed formula history on netuid 126:

    early       (0.65*AP + 0.35*recall@0.5) * (1 - fpr)**2, hard zero at fpr>=0.10
    2026-06-26   0.75*AP + 0.25*recall@fpr<=0.05
    current      0.35*AP + 0.30*recall@fpr<=0.05 + 0.20*tsq + 0.10*tsq + 0.05

There is exactly one reward implementation in this repo and it lives upstream.
If you find yourself writing `AP_WEIGHT = ...` anywhere, stop.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

# The single source of truth. Do not shadow, wrap-and-modify, or re-derive.
from poker44.score.scoring import reward as validator_reward

__all__ = ["validator_reward", "reward_metrics", "format_metrics"]


def reward_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
) -> Dict[str, float]:
    """Run the authoritative reward and flatten its detail dict.

    Returned keys mirror upstream's names so a formula change surfaces as a
    KeyError in your reporting rather than as a silently wrong number.
    """
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if s.size != y.size:
        raise ValueError(f"score/label length mismatch: {s.size} vs {y.size}")

    value, details = validator_reward(s, y)
    out: Dict[str, Any] = {"reward": float(value)}
    out.update({key: float(item) for key, item in details.items()})
    out["positive_count"] = int(np.sum(s >= 0.5))
    out["n"] = int(s.size)
    return out


def format_metrics(metrics: Dict[str, float]) -> str:
    """One-line summary for training / walk-forward logs."""
    return (
        f"reward={metrics.get('reward', 0.0):.4f} "
        f"ap={metrics.get('ap_score', 0.0):.4f} "
        f"recall@fpr05={metrics.get('bot_recall', 0.0):.4f} "
        f"tsq={metrics.get('threshold_sanity_quality', 0.0):.3f} "
        f"fpr@0.5={metrics.get('hard_fpr', 0.0):.4f} "
        f"flagged={metrics.get('positive_count', 0)}/{metrics.get('n', 0)}"
    )
