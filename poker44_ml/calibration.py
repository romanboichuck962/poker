"""Stack-calibration helpers.

Active:
  * :class:`BlendedIsotonicCalibrator` — the isotonic stack calibrator, dulled by
    blending its (step-shaped) output back toward the raw score so plateaus
    smooth out and ranking resolution is preserved.

Retained for backward-compatibility only (do NOT wire into new training code):
  * :class:`BlendedQuantileCalibrator` — the removed quantile calibrator, kept so
    previously-saved ``.joblib`` artifacts that pickled it inside their
    ``StackedEnsemble`` can still be unpickled and scored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import QuantileTransformer

from poker44.score.scoring import reward


class BlendedIsotonicCalibrator:
    """Isotonic stack calibration, dulled by blending toward identity.

    Pure isotonic regression is piecewise-constant: many distinct raw scores
    collapse onto the same plateau (sharp steps + abrupt jumps), erasing
    within-plateau ranking resolution. Blending the isotonic output with the raw
    score pulls it back toward identity, smoothing the steps while staying
    monotone (a convex combination of two monotone functions is monotone, and the
    identity term makes it strictly increasing so ties are broken)::

        out = blend * isotonic(raw) + (1 - blend) * raw

    ``blend=1.0`` -> pure isotonic (sharpest); ``blend=0.0`` -> passthrough
    (no calibration). Lower ``blend`` = duller / smoother.
    """

    def __init__(self, blend: float = 0.5) -> None:
        self.blend = float(max(0.0, min(1.0, blend)))
        self._iso: Optional[IsotonicRegression] = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "BlendedIsotonicCalibrator":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(
            np.asarray(scores, dtype=float).ravel(),
            np.asarray(labels, dtype=float).ravel(),
        )
        self._iso = iso
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(scores, dtype=float).ravel(), 0.0, 1.0)
        if self._iso is None:
            return values
        isotonic = np.asarray(self._iso.transform(values), dtype=float)
        mixed = self.blend * isotonic + (1.0 - self.blend) * values
        return np.clip(mixed, 0.0, 1.0)


class BlendedQuantileCalibrator:
    """Monotone score spreader for collapsed stacked probabilities.

    Deprecated/unused: kept only for unpickling legacy saved models. See the
    module docstring.
    """

    def __init__(self, blend: float = 0.9, max_quantiles: int = 256) -> None:
        self.blend = float(max(0.0, min(1.0, blend)))
        self.max_quantiles = int(max(8, max_quantiles))
        self._qt: Optional[QuantileTransformer] = None

    def fit(self, scores: np.ndarray) -> "BlendedQuantileCalibrator":
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        n_quantiles = int(max(8, min(self.max_quantiles, len(values))))
        qt = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="uniform",
            subsample=max(len(values), 1000),
            random_state=42,
        )
        qt.fit(values)
        self._qt = qt
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self._qt is None:
            return np.clip(values.ravel(), 0.0, 1.0)
        uniformized = self._qt.transform(values).ravel()
        base = np.clip(values.ravel(), 0.0, 1.0)
        mixed = self.blend * uniformized + (1.0 - self.blend) * base
        return np.clip(mixed, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Reward-aware, FPR-capped score calibrator (ported from pd-coast).
#
# The stacked head ranks well but its absolute score scale is arbitrary: the
# validator's 0.5 decision boundary can land in the wrong place, so a model that
# separates bots from humans in *rank* can still flag almost every live chunk as
# a bot (high FPR@0.5) if the whole score band sits above 0.5. This is exactly
# the "bot=98 human=2 on a human-heavy batch" symptom.
#
# :class:`ScoreCalibrator` fixes the score *geometry* without retraining, via up
# to four monotone stages applied in a fixed order::
#
#     raw -> quantile spread -> isotonic -> threshold_logit remap -> logit shift
#
# Every stage is monotone, so average precision (rank) is invariant; calibration
# only moves recall and FPR. :meth:`fit` grid-searches the stages to maximize the
# validator reward subject to a HARD FPR ceiling (kept under 0.10 so live drift
# has margin). Serialize with :meth:`to_dict` and restore with :meth:`from_dict`.
# ---------------------------------------------------------------------------


def validator_reward(
    scores: Sequence[float], labels: Sequence[int]
) -> Tuple[float, Dict[str, float]]:
    """Thin wrapper around the AUTHORITATIVE validator ``reward``.

    This does NOT re-implement or modify the scoring system: it calls the same
    ``poker44.score.scoring.reward`` the validator uses (identical to the
    Poker44-subnet reference) at the fixed 0.5 boundary. It exists only so the
    training-time calibrator can score candidates against the real objective.
    """
    rew, details = reward(np.asarray(scores, dtype=float), np.asarray(labels, dtype=int))
    return float(rew), {key: float(value) for key, value in details.items()}


def reward_metrics(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    """Authoritative reward components plus score-separation diagnostics.

    Diagnostics (``human_prob_max`` etc.) are derived from the scores directly
    and do not alter the reward; the reward fields come straight from the
    validator ``reward`` output.
    """
    truth = np.asarray([int(value) for value in labels], dtype=int)
    values = np.clip(np.asarray([float(value) for value in scores]), 0.0, 1.0)
    reward_value, details = validator_reward(values, truth)
    metrics: Dict[str, float] = {
        "validator_reward": reward_value,
        "validator_fpr": details["fpr"],
        "validator_bot_recall": details["bot_recall"],
        "validator_ap_score": details["ap_score"],
        "validator_base_score": details["base_score"],
        "human_safety_penalty": details["human_safety_penalty"],
        "hard_fpr": details["hard_fpr"],
        "hard_bot_recall": details["hard_bot_recall"],
        "positive_prediction_rate": details["positive_prediction_rate"],
    }
    humans = values[truth == 0]
    bots = values[truth == 1]
    metrics["human_prob_max"] = float(humans.max()) if humans.size else 0.0
    metrics["bot_prob_min"] = float(bots.min()) if bots.size else 1.0
    metrics["score_gap_at_0_5"] = metrics["bot_prob_min"] - metrics["human_prob_max"]
    metrics["prob_min"] = float(values.min()) if values.size else 0.0
    metrics["prob_max"] = float(values.max()) if values.size else 0.0
    metrics["prob_mean"] = float(values.mean()) if values.size else 0.0
    return metrics


def _cal_clamp01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _cal_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _cal_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def fit_quantile_spread(
    scores: np.ndarray,
    *,
    blend: float = 0.9,
    n_knots: int = 256,
) -> Optional[Tuple[List[float], List[float]]]:
    """Fit a monotone empirical-CDF spreader on a fixed [0, 1] grid.

    Returns ``(grid, y)`` knot lists for ``np.interp``, or ``None`` when the
    input has no spread to learn from. The map is ``y = blend * empiricalCDF(grid)
    + (1 - blend) * grid``: at ``blend=1`` it is the pure rank transform, at
    ``blend=0`` the identity. Strictly monotone, so it never reorders scores; its
    sole job is anti-collapse when a live score band squashes into a narrow range.
    """
    raw = np.asarray(scores, dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size < 8 or float(np.ptp(raw)) < 1e-9:
        return None
    blend = float(np.clip(blend, 0.0, 1.0))
    n_knots = int(max(8, n_knots))
    grid = np.linspace(0.0, 1.0, n_knots)
    sorted_raw = np.sort(np.clip(raw, 0.0, 1.0))
    cdf = np.searchsorted(sorted_raw, grid, side="right") / float(sorted_raw.size)
    y = blend * cdf + (1.0 - blend) * grid
    y = np.maximum.accumulate(y)
    y = y + np.linspace(0.0, 1e-6, n_knots)
    y = _cal_clamp01(y)
    return grid.tolist(), y.tolist()


def _threshold_logit(scores: np.ndarray, threshold: float, temperature: float) -> np.ndarray:
    """Map ``threshold`` to 0.5 with a sigmoid of slope ``1 / temperature``."""
    temperature = max(float(temperature), 1e-6)
    return _cal_sigmoid((np.clip(scores, 1e-6, 1.0 - 1e-6) - float(threshold)) / temperature)


def _logit_shift(scores: np.ndarray, bias: float, temperature: float) -> np.ndarray:
    if abs(float(bias)) < 1e-12 and abs(float(temperature) - 1.0) < 1e-12:
        return _cal_clamp01(scores)
    temperature = max(float(temperature), 1e-6)
    return _cal_sigmoid((_cal_logit(scores) + float(bias)) / temperature)


def _objective_key(metrics: Dict[str, float], objective: str) -> Tuple[float, float, float]:
    """Lexicographic sort key for the calibration grid search."""
    ap = float(metrics.get("validator_ap_score", 0.0))
    recall = float(metrics.get("validator_bot_recall", 0.0))
    reward_value = float(metrics.get("validator_reward", 0.0))
    if objective == "reward":
        return (reward_value, ap, recall)
    if objective == "recall":
        return (recall, reward_value, ap)
    return (ap, recall, reward_value)  # ap_first


def _passes_fpr_constraint(metrics: Dict[str, float], max_fpr: float) -> bool:
    """Reject any candidate whose held-out chunk-level FPR is unsafe."""
    return metrics.get("validator_fpr", 1.0) < max_fpr - 1e-9


def conformal_bias_for_target_fpr(
    human_scores: np.ndarray,
    target_fpr: float,
    *,
    max_abs_bias: float = 5.0,
) -> float:
    """Logit bias that drops the human-score ``1 - target_fpr`` quantile to ~0.5."""
    if human_scores.size == 0:
        return 0.0
    target_fpr = float(min(max(target_fpr, 1e-4), 0.5))
    quantile = float(np.quantile(human_scores, 1.0 - target_fpr))
    quantile = min(max(quantile, 1e-6), 1.0 - 1e-6)
    bias = -float(np.log(quantile / (1.0 - quantile)))
    return float(max(-abs(max_abs_bias), min(abs(max_abs_bias), bias)))


class ScoreCalibrator:
    """Monotone, reward-aware score post-processor (see module comment)."""

    def __init__(
        self,
        *,
        spread_x: Optional[List[float]] = None,
        spread_y: Optional[List[float]] = None,
        isotonic_x: Optional[List[float]] = None,
        isotonic_y: Optional[List[float]] = None,
        remap: Optional[Dict[str, float]] = None,
        logit_bias: float = 0.0,
        logit_temperature: float = 1.0,
        objective: str = "reward",
        target_fpr: float = 0.04,
        max_fpr: float = 0.05,
    ) -> None:
        self.spread_x = spread_x
        self.spread_y = spread_y
        self.isotonic_x = isotonic_x
        self.isotonic_y = isotonic_y
        self.remap = remap or {}
        self.logit_bias = float(logit_bias)
        self.logit_temperature = float(logit_temperature)
        self.objective = str(objective)
        self.target_fpr = float(target_fpr)
        self.max_fpr = float(max_fpr)

    # ------------------------------------------------------------------ apply

    def _apply_spread(self, scores: np.ndarray) -> np.ndarray:
        if not self.spread_x or not self.spread_y:
            return _cal_clamp01(scores)
        xp = np.asarray(self.spread_x, dtype=float)
        fp = np.asarray(self.spread_y, dtype=float)
        return _cal_clamp01(np.interp(np.clip(scores, 0.0, 1.0), xp, fp))

    def _apply_isotonic(self, scores: np.ndarray) -> np.ndarray:
        if not self.isotonic_x or not self.isotonic_y:
            return _cal_clamp01(scores)
        xp = np.asarray(self.isotonic_x, dtype=float)
        fp = np.asarray(self.isotonic_y, dtype=float)
        return _cal_clamp01(np.interp(np.clip(scores, 0.0, 1.0), xp, fp))

    def _apply_remap(self, scores: np.ndarray) -> np.ndarray:
        if not self.remap:
            return _cal_clamp01(scores)
        return _cal_clamp01(
            _threshold_logit(
                scores,
                threshold=float(self.remap.get("threshold", 0.5)),
                temperature=float(self.remap.get("temperature", 0.25)),
            )
        )

    def transform(self, scores: Sequence[float]) -> np.ndarray:
        """Apply quantile spread -> isotonic -> remap -> logit shift, in order."""
        out = np.asarray(scores, dtype=float)
        out = self._apply_spread(out)
        out = self._apply_isotonic(out)
        out = self._apply_remap(out)
        out = _logit_shift(out, self.logit_bias, self.logit_temperature)
        return _cal_clamp01(out)

    @property
    def is_identity(self) -> bool:
        return (
            not self.spread_x
            and not self.isotonic_x
            and not self.remap
            and abs(self.logit_bias) < 1e-12
            and abs(self.logit_temperature - 1.0) < 1e-12
        )

    # ------------------------------------------------------------------- fit

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        *,
        use_spread: bool = True,
        spread_blend: float = 0.9,
        spread_knots: int = 256,
        use_isotonic: bool = True,
        isotonic_identity_blend: float = 0.05,
        remap_temperature_grid: Sequence[float] = (0.12, 0.18, 0.25, 0.35, 0.5, 0.65, 0.85, 1.0, 1.25),
        logit_bias_grid: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
        logit_temperature_grid: Sequence[float] = (0.6, 0.8, 1.0, 1.2),
    ) -> "ScoreCalibrator":
        """Tune all stages on a held-out calibration split under the FPR ceiling."""
        raw = np.asarray(scores, dtype=float)
        lab = np.asarray(labels, dtype=int)
        if raw.size == 0 or lab.sum() == 0 or lab.sum() == lab.size:
            return self  # need both classes to calibrate; leave as identity

        # Stage 0: quantile spread (monotone anti-collapse).
        self.spread_x = self.spread_y = None
        if use_spread:
            knots = fit_quantile_spread(raw, blend=spread_blend, n_knots=spread_knots)
            if knots is not None:
                self.spread_x, self.spread_y = knots
        base = self._apply_spread(raw)

        # Stage 1: isotonic recalibration (monotone, preserves ranking).
        self.isotonic_x = self.isotonic_y = None
        if use_isotonic and len(set(lab.tolist())) >= 2:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(base, lab.astype(float))
            grid = np.linspace(0.0, 1.0, 256)
            iso_y = np.clip(iso.predict(grid), 0.0, 1.0)
            # Blend a slice of the identity so the curve stays STRICTLY increasing:
            # a step-shaped isotonic fit collapses whole score intervals to one
            # value, which would destroy live ranking (AP) even when val looks perfect.
            blend = float(np.clip(isotonic_identity_blend, 0.0, 1.0))
            iso_y = (1.0 - blend) * iso_y + blend * grid
            self.isotonic_x = grid.tolist()
            self.isotonic_y = iso_y.tolist()
        post_iso = self._apply_isotonic(base)

        # Stage 2: threshold-logit remap (recenters the 0.5 boundary).
        humans = post_iso[lab == 0]
        bots = post_iso[lab == 1]
        thresholds: set[float] = set()
        for q in np.linspace(0.40, 0.995, 24):
            thresholds.add(float(np.quantile(humans, q)))
        for q in np.linspace(0.005, 0.60, 20):
            thresholds.add(float(np.quantile(bots, q)))
        thresholds.update({0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30})

        self.remap = {}
        baseline = reward_metrics(lab.tolist(), post_iso.tolist())
        best_key = (
            _objective_key(baseline, self.objective)
            if _passes_fpr_constraint(baseline, self.max_fpr)
            else None
        )
        for threshold in sorted(thresholds):
            for temperature in remap_temperature_grid:
                remapped = _threshold_logit(post_iso, threshold, temperature)
                metrics = reward_metrics(lab.tolist(), remapped.tolist())
                if not _passes_fpr_constraint(metrics, self.max_fpr):
                    continue
                key = _objective_key(metrics, self.objective)
                if best_key is None or key > best_key:
                    best_key = key
                    self.remap = {"threshold": float(threshold), "temperature": float(temperature)}
        post_remap = self._apply_remap(post_iso)

        # Stage 3: logit shift. Seed with the conformal bias for the target FPR.
        conformal = conformal_bias_for_target_fpr(post_remap[lab == 0], self.target_fpr)
        bias_candidates = sorted(
            {float(b) for b in logit_bias_grid}
            | {conformal}
            | {conformal + d for d in (0.0, 0.25, 0.5, 1.0, 1.5)}
        )
        self.logit_bias, self.logit_temperature = 0.0, 1.0
        baseline = reward_metrics(lab.tolist(), post_remap.tolist())
        best_key = (
            _objective_key(baseline, self.objective)
            if _passes_fpr_constraint(baseline, self.max_fpr)
            else None
        )
        for bias in bias_candidates:
            for temperature in logit_temperature_grid:
                if abs(bias) < 1e-12 and abs(temperature - 1.0) < 1e-12:
                    continue
                shifted = _logit_shift(post_remap, bias, temperature)
                metrics = reward_metrics(lab.tolist(), shifted.tolist())
                if not _passes_fpr_constraint(metrics, self.max_fpr):
                    continue
                key = _objective_key(metrics, self.objective)
                if best_key is None or key > best_key:
                    best_key = key
                    self.logit_bias, self.logit_temperature = float(bias), float(temperature)
        return self

    # --------------------------------------------------------- (de)serialize

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "score_calibrator_v1",
            "spread_x": self.spread_x,
            "spread_y": self.spread_y,
            "isotonic_x": self.isotonic_x,
            "isotonic_y": self.isotonic_y,
            "remap": dict(self.remap),
            "logit_bias": self.logit_bias,
            "logit_temperature": self.logit_temperature,
            "objective": self.objective,
            "target_fpr": self.target_fpr,
            "max_fpr": self.max_fpr,
        }

    @classmethod
    def from_dict(cls, state: Optional[Dict[str, Any]]) -> Optional["ScoreCalibrator"]:
        if not state or state.get("kind") != "score_calibrator_v1":
            return None
        return cls(
            spread_x=state.get("spread_x"),
            spread_y=state.get("spread_y"),
            isotonic_x=state.get("isotonic_x"),
            isotonic_y=state.get("isotonic_y"),
            remap=state.get("remap"),
            logit_bias=float(state.get("logit_bias", 0.0)),
            logit_temperature=float(state.get("logit_temperature", 1.0)),
            objective=str(state.get("objective", "reward")),
            target_fpr=float(state.get("target_fpr", 0.04)),
            max_fpr=float(state.get("max_fpr", 0.05)),
        )

    def summary(self, labels: Sequence[int], raw_scores: Sequence[float]) -> str:
        """Before/after reward summary, for logging at the end of training."""
        before = reward_metrics(labels, raw_scores)
        after = reward_metrics(labels, self.transform(raw_scores))
        reward_before, _ = validator_reward(
            np.asarray(raw_scores, float), np.asarray(labels, int)
        )
        reward_after, _ = validator_reward(
            self.transform(raw_scores), np.asarray(labels, int)
        )
        return (
            f"spread={'on' if self.spread_x else 'off'} "
            f"remap={self.remap or None} logit_bias={self.logit_bias:.4f} "
            f"logit_temp={self.logit_temperature:.4f} isotonic={'on' if self.isotonic_x else 'off'} | "
            f"reward {reward_before:.4f}->{reward_after:.4f} "
            f"fpr {before['validator_fpr']:.4f}->{after['validator_fpr']:.4f} "
            f"recall {before['validator_bot_recall']:.4f}->{after['validator_bot_recall']:.4f} "
            f"human_prob_max {before['human_prob_max']:.4f}->{after['human_prob_max']:.4f}"
        )
