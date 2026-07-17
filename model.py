"""Poker44 serving model for uid 242 — D0Draco (UID172's method).

Faithful adaptation of UID172's public poker44-benchmark-huge-2 (commit
67ec343, model "poker44-draco" v3.6, serving logic from model/infer.py):

  * two feature views, identical code paths to training (train == serve):
    phasberg (~293 dims: 40 per-hand scalars x 7 order-stats + 12 replay
    signature features + hand_count) and features_v2 (~250 hero-free,
    sanitization-invariant dims);
  * D0Draco: weighted 4-member RANK-blend {stack .28, mono .24, mlp .28,
    drse .20} — only each member's chunk ordering matters, which is what the
    AP-dominated reward scores;
  * rank-preserving post-processing: monotone remap of the deploy threshold
    to 0.5, then a batch safety budget capping >=0.5 calls per request —
    ranking (AP / recall@FPR) is never altered, only the decision boundary.

Interface kept identical for miner.py: MODEL_ARTIFACT + Poker44Model with
score_chunks(chunks) -> list[float].
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np

from d0.ensemble import D0Draco  # noqa: F401  (needed to unpickle the artifact)
from d0.drse import DRSE  # noqa: F401  (DRSE instances live inside the pickle)
from d0.views import phasberg_dict, v2_dict

MODEL_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "poker44_model.joblib"

_EMPTY_SCORE = 0.1


def _remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    """Monotone piecewise-linear remap moving decision threshold t to 0.5."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


def _apply_batch_safety_budget(scores: np.ndarray, max_frac: float) -> np.ndarray:
    """Cap the fraction of >=0.5 calls per batch WITHOUT changing the ranking."""
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0 or max_frac >= 1.0:
        return s
    k = max(1, int(np.floor(max_frac * n)))
    positive = np.flatnonzero(s >= 0.5)
    if positive.size <= k:
        return s
    order = positive[np.argsort(-s[positive], kind="stable")]
    squeeze = order[k:]
    below = s[s < 0.5]
    lo = min(float(below.max()) if below.size else 0.45, 0.499)
    span = 0.5 - lo
    out = s.copy()
    m = squeeze.size
    for rank, idx in enumerate(squeeze):
        out[idx] = lo + span * (m - rank) / (m + 1.0)
    return np.clip(out, 0.0, 1.0)


class Poker44Model:
    """Load and serve a D0Draco artifact."""

    def __init__(self, artifact_path: Path | str = MODEL_ARTIFACT) -> None:
        artifact: Dict[str, Any] = joblib.load(artifact_path)
        if artifact.get("kind") != "d0_draco":
            raise ValueError("not a D0Draco artifact")
        self.ens: D0Draco = artifact["ens"]
        self.threshold = float(artifact["threshold"])
        self.max_pos_frac = float(
            os.getenv("POKER44_MAX_POS_FRAC") or artifact.get("max_pos_frac", 0.16)
        )
        self.cols_ph = list(artifact["cols_ph"])
        self.cols_v2 = list(artifact["cols_v2"])
        self.artifact = artifact

    def _matrices(self, chunks: Sequence[List[Dict[str, Any]]]):
        ph = np.array([[float(d.get(c, 0.0)) for c in self.cols_ph]
                       for d in (phasberg_dict(c) for c in chunks)], dtype=float)
        v2 = np.array([[float(d.get(c, 0.0)) for c in self.cols_v2]
                       for d in (v2_dict(c) for c in chunks)], dtype=float)
        ph = np.nan_to_num(ph, nan=0.0, posinf=0.0, neginf=0.0)
        v2 = np.nan_to_num(v2, nan=0.0, posinf=0.0, neginf=0.0)
        return ph, v2

    def score_chunks(self, chunks: Sequence[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        ph, v2 = self._matrices(chunks)
        p = self.ens.score(ph, v2)
        scores = _remap_to_threshold(np.asarray(p, dtype=float), self.threshold)
        scores = _apply_batch_safety_budget(scores, self.max_pos_frac)
        return [_EMPTY_SCORE if not chunk else round(float(s), 6)
                for chunk, s in zip(chunks, scores)]
