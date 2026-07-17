"""Robust real-data ensemble for coherent Poker44 chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .features import BASE_FEATURE_COUNT


BRANCH_NAMES = (
    "coherent_extra_trees",
    "coherent_random_forest",
    "coherent_hist_gradient",
    "combined_regularized_extra",
    "combined_logistic",
    "base_human_prototype",
    "rank_coherent_hist_gradient",
    "rank_combined_regularized_extra",
    "rank_combined_logistic",
)
RAW_BRANCH_COUNT = 6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(x, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class HumanTailPrototype:
    """Small covariance-distance branch without the V3 neural dependency."""

    max_features: int = 64

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HumanTailPrototype":
        values = np.asarray(x, dtype=float)
        labels = np.asarray(y, dtype=int)
        human, bot = values[labels == 0], values[labels == 1]
        scale = np.std(values, axis=0) + 1e-6
        effect = np.abs((bot.mean(axis=0) - human.mean(axis=0)) / scale)
        self.indices_ = np.argsort(-effect, kind="mergesort")[: min(self.max_features, values.shape[1])]
        self.scaler_ = StandardScaler().fit(values[:, self.indices_])
        normalized = self.scaler_.transform(values[:, self.indices_])
        normalized_human = normalized[labels == 0]
        normalized_bot = normalized[labels == 1]
        self.human_ = LedoitWolf().fit(normalized_human)
        self.bot_ = LedoitWolf().fit(normalized_bot)
        human_distance = self.human_.mahalanobis(normalized_human)
        bot_distance = self.bot_.mahalanobis(normalized_human)
        margin = human_distance - bot_distance
        self.center_ = float(np.median(margin))
        self.scale_ = float(np.std(margin) + 1e-6)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        normalized = self.scaler_.transform(values[:, self.indices_])
        margin = self.human_.mahalanobis(normalized) - self.bot_.mahalanobis(normalized)
        probability = _sigmoid((margin - self.center_) / self.scale_)
        return np.column_stack((1.0 - probability, probability))


@dataclass(frozen=True)
class ModelConfig:
    trees: int = 700
    hist_iterations: int = 700
    max_depth: int = 9
    learning_rate: float = 0.03


def percentile_feature_matrix(x: np.ndarray) -> np.ndarray:
    """Convert columns to tie-averaged request-relative percentiles.

    Equal values receive equal ranks, making the transform invariant to request
    order without allowing an identifier-derived tie break into model inputs.
    """
    values = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    if values.ndim != 2:
        raise ValueError("feature matrix must be two-dimensional")
    row_count, column_count = values.shape
    if row_count <= 1:
        return np.full_like(values, 0.5)
    ranked = np.empty_like(values)
    denominator = float(row_count - 1)
    for column in range(column_count):
        series = values[:, column]
        order = np.argsort(series, kind="mergesort")
        ordered = series[order]
        starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        ends = np.r_[starts[1:], row_count]
        for start, end in zip(starts, ends):
            average_rank = (float(start) + float(end - 1)) / (2.0 * denominator)
            ranked[order[start:end], column] = average_rank
    return ranked


def grouped_percentile_feature_matrix(
    x: np.ndarray,
    groups: Sequence[str] | None,
) -> np.ndarray:
    """Apply request-relative percentiles independently inside training dates."""
    values = np.asarray(x, dtype=float)
    if groups is None:
        return percentile_feature_matrix(values)
    keys = np.asarray([str(value) for value in groups], dtype=str)
    if keys.shape != (values.shape[0],):
        raise ValueError("groups must match feature rows")
    ranked = np.empty_like(values)
    for key in sorted(set(keys.tolist())):
        indices = np.flatnonzero(keys == key)
        ranked[indices] = percentile_feature_matrix(values[indices])
    return ranked


class CoherentEnsemble:
    """Diverse tree ensemble with explicit coherent and stability views."""

    branch_names = BRANCH_NAMES

    def __init__(self, seed: int = 44, config: ModelConfig | None = None) -> None:
        self.seed = int(seed)
        self.config = config or ModelConfig()
        cfg = self.config
        self.coherent_extra = ExtraTreesClassifier(
            n_estimators=cfg.trees,
            max_depth=cfg.max_depth,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        self.coherent_forest = RandomForestClassifier(
            n_estimators=cfg.trees,
            max_depth=cfg.max_depth,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed + 7,
        )
        self.coherent_hist = make_pipeline(
            VarianceThreshold(1e-12),
            HistGradientBoostingClassifier(
                max_iter=cfg.hist_iterations,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                min_samples_leaf=2,
                l2_regularization=1.0,
                random_state=seed + 13,
            ),
        )
        self.combined_extra = ExtraTreesClassifier(
            n_estimators=cfg.trees,
            max_depth=7,
            min_samples_leaf=5,
            max_features=0.45,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed + 19,
        )
        self.combined_logistic = make_pipeline(
            VarianceThreshold(1e-12),
            RobustScaler(),
            LogisticRegression(
                C=0.08,
                class_weight="balanced",
                max_iter=4000,
                random_state=seed + 23,
            ),
        )
        self.base_prototype = HumanTailPrototype(max_features=64)
        self.rank_coherent_hist = make_pipeline(
            VarianceThreshold(1e-12),
            HistGradientBoostingClassifier(
                max_iter=cfg.hist_iterations,
                learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth,
                min_samples_leaf=2,
                l2_regularization=1.0,
                random_state=seed + 29,
            ),
        )
        self.rank_combined_extra = ExtraTreesClassifier(
            n_estimators=cfg.trees,
            max_depth=7,
            min_samples_leaf=5,
            max_features=0.45,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed + 31,
        )
        self.rank_combined_logistic = make_pipeline(
            VarianceThreshold(1e-12),
            RobustScaler(),
            LogisticRegression(
                C=0.08,
                class_weight="balanced",
                max_iter=4000,
                random_state=seed + 37,
            ),
        )
        # Public winner-inspired tree blend is the safe pre-selection default.
        self.branch_weights_ = np.asarray(
            [0.45, 0.25, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=float,
        )

    @staticmethod
    def _views(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        combined = np.asarray(x, dtype=float)
        base = combined[:, :BASE_FEATURE_COUNT]
        coherent = combined[:, BASE_FEATURE_COUNT:]
        if coherent.shape[1] == 0:
            raise ValueError("V4 coherent feature view is empty")
        return combined, base, coherent

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Sequence[float] | None = None,
        groups: Sequence[str] | None = None,
    ) -> "CoherentEnsemble":
        combined, base, coherent = self._views(x)
        rank_combined = grouped_percentile_feature_matrix(combined, groups)
        _, _, rank_coherent = self._views(rank_combined)
        labels = np.asarray(y, dtype=int)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.coherent_extra.fit(coherent, labels, sample_weight=weights)
        self.coherent_forest.fit(coherent, labels, sample_weight=weights)
        self.coherent_hist.fit(coherent, labels, histgradientboostingclassifier__sample_weight=weights)
        self.combined_extra.fit(combined, labels, sample_weight=weights)
        self.combined_logistic.fit(combined, labels, logisticregression__sample_weight=weights)
        # The density branch has no sample_weight API; date balancing is handled
        # by keeping it a selectable specialist rather than a mandatory weight.
        self.base_prototype.fit(base, labels)
        self.rank_coherent_hist.fit(
            rank_coherent,
            labels,
            histgradientboostingclassifier__sample_weight=weights,
        )
        self.rank_combined_extra.fit(rank_combined, labels, sample_weight=weights)
        self.rank_combined_logistic.fit(
            rank_combined,
            labels,
            logisticregression__sample_weight=weights,
        )
        return self

    def raw_branch_scores(self, x: np.ndarray) -> np.ndarray:
        combined, base, coherent = self._views(x)
        columns = (
            self.coherent_extra.predict_proba(coherent)[:, 1],
            self.coherent_forest.predict_proba(coherent)[:, 1],
            self.coherent_hist.predict_proba(coherent)[:, 1],
            self.combined_extra.predict_proba(combined)[:, 1],
            self.combined_logistic.predict_proba(combined)[:, 1],
            self.base_prototype.predict_proba(base)[:, 1],
        )
        return np.clip(np.column_stack(columns), 1e-6, 1.0 - 1e-6)

    def rank_branch_scores(self, x: np.ndarray) -> np.ndarray:
        combined, _, _ = self._views(x)
        rank_combined = percentile_feature_matrix(combined)
        _, _, rank_coherent = self._views(rank_combined)
        columns = (
            self.rank_coherent_hist.predict_proba(rank_coherent)[:, 1],
            self.rank_combined_extra.predict_proba(rank_combined)[:, 1],
            self.rank_combined_logistic.predict_proba(rank_combined)[:, 1],
        )
        return np.clip(np.column_stack(columns), 1e-6, 1.0 - 1e-6)

    def branch_scores(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack((self.raw_branch_scores(x), self.rank_branch_scores(x)))

    def request_rebased_branches(
        self,
        x: np.ndarray,
        precomputed: np.ndarray,
    ) -> np.ndarray:
        """Reuse raw predictions and recompute rank branches for this request."""
        branches = np.asarray(precomputed, dtype=float).copy()
        if branches.shape != (len(x), len(BRANCH_NAMES)):
            raise ValueError("precomputed branches do not match request rows")
        branches[:, RAW_BRANCH_COUNT:] = self.rank_branch_scores(x)
        return branches

    def probability_score(self, x: np.ndarray) -> np.ndarray:
        branches = self.branch_scores(x)
        weights = np.asarray(self.branch_weights_, dtype=float)
        weights = np.clip(weights, 0.0, None)
        if weights.size != branches.shape[1] or float(weights.sum()) <= 0:
            weights = np.ones(branches.shape[1], dtype=float)
        return np.clip(branches @ (weights / weights.sum()), 1e-6, 1.0 - 1e-6)


def percentile_columns(
    branches: np.ndarray,
    tie_keys: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert each branch to deterministic [0,1] request-relative ranks."""
    values = np.asarray(branches, dtype=float)
    if values.ndim != 2:
        raise ValueError("branches must be a 2D matrix")
    n = values.shape[0]
    if n <= 1:
        return np.full_like(values, 0.5)
    keys = [f"{index:012d}" for index in range(n)] if tie_keys is None else [str(key) for key in tie_keys]
    if len(keys) != n:
        raise ValueError("tie_keys must match branch row count")
    ranked = np.empty_like(values)
    for column in range(values.shape[1]):
        order = sorted(range(n), key=lambda index: (float(values[index, column]), keys[index]))
        ranked[order, column] = np.arange(n, dtype=float) / (n - 1)
    return ranked


def blend_branches(
    branches: np.ndarray,
    weights: Sequence[float],
    mode: str,
    *,
    tie_keys: Sequence[str] | None = None,
) -> np.ndarray:
    values = np.asarray(branches, dtype=float)
    clean = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if clean.size != values.shape[1] or float(clean.sum()) <= 0:
        raise ValueError("invalid branch weights")
    if mode == "rank":
        values = percentile_columns(values, tie_keys=tie_keys)
    elif mode != "probability":
        raise ValueError("blend mode must be 'probability' or 'rank'")
    return np.clip(values @ (clean / clean.sum()), 1e-6, 1.0 - 1e-6)
