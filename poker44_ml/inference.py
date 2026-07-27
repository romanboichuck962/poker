"""The serving path — the only place a score is ever produced.

Training, walk-forward and the miner all score through ``Detector`` so an
offline number and a live number can never diverge. If you find yourself
calling ``ensemble.score`` directly outside this module, you have created a
train/serve skew bug.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from poker44_ml.features import feature_matrix
from poker44_ml.policy import chunk_tie_key, rank_map

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

# Below this many chunks the within-batch policy has too little to rank, so the
# raw score passes through. Live requests carry ~100 chunks; this only guards
# single-chunk debugging calls.
MIN_BATCH_FOR_POLICY = 8


class Detector:
    """Load an artifact and score chunks exactly as the miner will."""

    def __init__(self, artifact: Dict[str, Any]):
        self.ensemble = artifact["ensemble"]
        self.feature_names: List[str] = list(artifact["feature_names"])
        self.metadata: Dict[str, Any] = dict(artifact.get("metadata") or {})
        self.positive_fraction = float(self.metadata.get("positive_fraction", 0.10))
        if not 0.0 < self.positive_fraction < 1.0:
            raise ValueError(
                f"positive_fraction must be in (0,1), got {self.positive_fraction}"
            )
        if not self.feature_names:
            raise ValueError("artifact carries no feature_names")

    @classmethod
    def load(cls, path: str | Path) -> "Detector":
        if joblib is None:
            raise RuntimeError("joblib is required to load an artifact")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"artifact not found: {path}. Train first: python -m poker44_ml.train"
            )
        return cls(joblib.load(path))

    # ------------------------------------------------------------------ score

    def raw_scores(self, chunks: Sequence[Sequence[Dict[str, Any]]]) -> np.ndarray:
        """Blended within-batch rank score, before the 0.5 line is placed."""
        if not chunks:
            return np.zeros(0, dtype=float)
        rows = np.asarray(
            feature_matrix(chunks, self.feature_names), dtype=np.float64
        )
        rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(self.ensemble.score(rows), dtype=float)

    def predict_chunk_scores(
        self,
        chunks: Sequence[Sequence[Dict[str, Any]]],
        *,
        apply_policy: bool = True,
    ) -> List[float]:
        """One risk score per chunk. Length always equals ``len(chunks)``."""
        if not chunks:
            return []

        raw = self.raw_scores(chunks)
        if not apply_policy or len(raw) < MIN_BATCH_FOR_POLICY:
            return [round(float(np.clip(v, 0.0, 1.0)), 8) for v in raw]

        keys = [chunk_tie_key(chunk) for chunk in chunks]
        mapped = rank_map(raw, self.positive_fraction, tie_keys=keys)
        return [round(float(np.clip(v, 0.0, 1.0)), 8) for v in mapped]

    def predict_chunk_score(self, chunk: Sequence[Dict[str, Any]]) -> float:
        scores = self.predict_chunk_scores([chunk], apply_policy=False)
        return scores[0] if scores else 0.5

    # ------------------------------------------------------------ diagnostics

    def drift_report(
        self,
        chunks: Sequence[Sequence[Dict[str, Any]]],
    ) -> Dict[str, float]:
        """How far this batch sits outside the training feature distribution.

        The benchmark and live traffic differ in chunk size and pot scale, so a
        live batch that reads far outside training quantiles is the early warning
        that an artifact has gone stale — visible before the leaderboard shows it.
        Requires an artifact trained with reference quantiles recorded.
        """
        reference = self.metadata.get("feature_reference")
        if not reference or not chunks:
            return {}

        rows = np.asarray(feature_matrix(chunks, self.feature_names), dtype=np.float64)
        rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
        q01 = np.asarray(reference["q01"], dtype=float)
        q99 = np.asarray(reference["q99"], dtype=float)
        median = np.asarray(reference["median"], dtype=float)
        iqr = np.maximum(
            np.asarray(reference["q75"], dtype=float)
            - np.asarray(reference["q25"], dtype=float),
            1e-9,
        )
        outside = (rows < q01) | (rows > q99)
        shift = np.abs(np.median(rows, axis=0) - median) / iqr
        return {
            "outside_q01_q99_rate": float(outside.mean()),
            "features_mostly_outside": int(np.sum(outside.mean(axis=0) > 0.5)),
            "median_shift_iqr": float(np.median(shift)),
            "p90_shift_iqr": float(np.quantile(shift, 0.90)),
        }

    def benchmark_latency(
        self,
        chunks: Sequence[Sequence[Dict[str, Any]]],
        repeats: int = 3,
    ) -> Dict[str, float]:
        """Wall-clock per request. The validator times out at 180s."""
        if not chunks:
            return {"total_ms": 0.0, "per_chunk_ms": 0.0}
        repeats = max(1, int(repeats))
        started = time.perf_counter()
        for _ in range(repeats):
            self.predict_chunk_scores(chunks)
        total_ms = (time.perf_counter() - started) * 1000.0 / repeats
        return {"total_ms": total_ms, "per_chunk_ms": total_ms / len(chunks)}
