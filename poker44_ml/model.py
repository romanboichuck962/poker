"""The ensemble: decorrelated members combined by within-batch rank.

Why rank and not probability. Blending member probabilities has a failure mode
that is fatal here: if one member's scores collapse on shifted data (say every
output lands in [0.30, 0.32]), it drags the blend into that band and can hold the
whole batch below 0.5, tripping the zero-true-positive gate and zeroing the
reward. Converting each member to its within-batch percentile first makes the
blend immune to any member's score scale. Every competitive miner on this subnet
does this; at least one of them learned it the hard way and wrote it down.

Members are chosen for decorrelated errors, not individual strength:

    tree      monotone-constrained LightGBM. Monotone constraints on
              sign-stable features are a strong prior against fitting
              benchmark-specific noise.
    forest    ExtraTrees. Different variance profile, no boosting bias.
    neural    PCA -> MLP on the full feature vector. Sees linear structure the
              trees carve up axis-aligned.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def rank01(scores: Sequence[float]) -> np.ndarray:
    """Within-batch percentile in [0,1], ties averaged, stable ordering."""
    values = np.asarray(scores, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return values
    if n == 1:
        return np.array([0.5])

    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    start = 0
    for end in range(1, n + 1):
        if end == n or ordered[end] != ordered[start]:
            if end - start > 1:
                ranks[start:end] = (start + 1 + end) / 2.0
            start = end
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return (out - 0.5) / float(n)


class RankVoteEnsemble:
    """Picklable weighted-rank vote over member estimators.

    Members must expose ``predict_proba(X) -> (n, 2)``. The blend is a weighted
    mean of per-member within-batch percentiles, so the output is a rank score in
    [0,1] — NOT a probability. Do not calibrate it as one; the 0.5 line is placed
    downstream by poker44_ml.policy, which is the only thing that should decide it.
    """

    def __init__(
        self,
        members: Sequence[Any],
        weights: Sequence[float],
        names: Sequence[str] | None = None,
    ) -> None:
        if len(members) != len(weights):
            raise ValueError(f"{len(members)} members but {len(weights)} weights")
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members: List[Any] = list(members)
        self.weights: List[float] = [max(0.0, float(w)) for w in weights]
        self.names: List[str] = list(names or [f"member_{i}" for i in range(len(members))])
        if sum(self.weights) <= 0.0:
            raise ValueError("ensemble weights must not sum to zero")

    def member_columns(self, x: np.ndarray) -> np.ndarray:
        """Raw per-member P(bot), shape (n_rows, n_members). Diagnostics."""
        columns = []
        for member in self.members:
            proba = np.asarray(member.predict_proba(x))
            column = proba[:, 1] if proba.ndim == 2 else proba
            columns.append(np.clip(np.asarray(column, dtype=float), 0.0, 1.0))
        return np.column_stack(columns)

    def score(self, x: np.ndarray) -> np.ndarray:
        """Blended within-batch rank score for a whole request."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] == 0:
            return np.zeros(0, dtype=float)
        columns = self.member_columns(x)
        total = sum(self.weights)
        blended = sum(
            weight * rank01(columns[:, index])
            for index, weight in enumerate(self.weights)
        ) / total
        return np.clip(np.asarray(blended, dtype=float), 0.0, 1.0)

    # sklearn-shaped alias so the object survives generic tooling
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self.score(x)
        return np.column_stack([1.0 - scores, scores])

    def __repr__(self) -> str:
        pairs = ", ".join(f"{n}={w:g}" for n, w in zip(self.names, self.weights))
        return f"RankVoteEnsemble({pairs})"


# --------------------------------------------------------------------------- #
# member construction
# --------------------------------------------------------------------------- #

def monotone_constraints(
    x: np.ndarray,
    y: np.ndarray,
    *,
    min_abs_corr: float = 0.06,
) -> List[int]:
    """+1/-1 where a feature correlates stably with the label, else 0.

    Constraining only sign-stable features leaves the booster free where the
    relationship is genuinely non-monotone, while blocking it from inventing
    fold-back shapes that fit benchmark noise and invert on live data.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    y_centered = y - y.mean()
    y_std = y.std() + 1e-12

    out: List[int] = []
    for column in range(x.shape[1]):
        feature = x[:, column]
        std = float(feature.std())
        if std <= 1e-12:
            out.append(0)
            continue
        corr = float(np.mean((feature - feature.mean()) * y_centered) / (std * y_std))
        out.append((1 if corr > 0 else -1) if abs(corr) >= min_abs_corr else 0)
    return out


def build_stack(x: np.ndarray, y: np.ndarray, *, seed: int = 42) -> Any:
    """Leaf-wise gradient stack: four diverse boosters -> logistic meta on OOF.

    This replaces what was a single monotone booster. Stacking decorrelated
    learners is the one change measured to move AP (+0.01-0.04 elsewhere), and AP
    is 35% of the reward and the half no calibration can recover. Out-of-fold
    meta-training is what keeps the meta-learner from simply memorising base
    overfit.

    Deliberately UNCONSTRAINED: the monotone prior lives in its own member, so
    this one is free to fit interactions the constraints would forbid. Two
    members that disagree are worth more to a rank vote than two that agree.
    """
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    import lightgbm as lgb

    positive = max(1, int(np.sum(y == 1)))
    negative = max(1, int(np.sum(y == 0)))
    # Push the boosters to rank bots correctly; that is AP, which is what pays.
    scale_pos_weight = negative / positive

    estimators = [
        ("lgbm_leafwise", lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.03, num_leaves=127,
            min_data_in_leaf=20, feature_fraction=0.7, bagging_fraction=0.8,
            bagging_freq=1, reg_lambda=1.0, objective="binary",
            n_jobs=-1, random_state=seed, verbose=-1,
        )),
        ("xgb_lossguide", XGBClassifier(
            n_estimators=700, learning_rate=0.03, max_depth=0, max_leaves=63,
            grow_policy="lossguide", subsample=0.9, colsample_bytree=0.8,
            reg_lambda=2.0, tree_method="hist", eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, n_jobs=-1, random_state=seed + 1,
        )),
        ("catboost_deep", CatBoostClassifier(
            iterations=600, learning_rate=0.03, depth=7, l2_leaf_reg=3.0,
            loss_function="Logloss", random_seed=seed + 2, verbose=0,
            allow_writing_files=False,
        )),
        ("rf_deep", RandomForestClassifier(
            n_estimators=500, max_depth=18, min_samples_leaf=2,
            max_features="sqrt", n_jobs=-1, random_state=seed + 3,
        )),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=2000),
        cv=3,
        stack_method="predict_proba",
        n_jobs=1,          # base learners already use every core
        passthrough=False,
    ).fit(x, y)


def build_members(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    human_weight: float = 2.0,
    pca_components: int = 56,
    mlp_hidden: tuple[int, ...] = (80,),
) -> tuple[List[Any], List[float], List[str]]:
    """Fit the three members. Returns ``(members, weights, names)``.

    The trio is chosen for decorrelated errors, not individual strength:

        stack   unconstrained leaf-wise gradient stack (capacity)
        mono    monotone-constrained LightGBM (out-of-distribution prior)
        neural  PCA -> MLP (sees linear structure the trees carve axis-aligned)

    Humans are upweighted because false positives are the expensive error: they
    drive fpr@0.5 toward the tsq penalty, and the top of the ranking is where the
    reward is actually collected.
    """
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    import lightgbm as lgb

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    sample_weight = np.where(y == 0, human_weight, 1.0)

    stack = build_stack(x, y, seed=seed)

    tree = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=48,
        min_data_in_leaf=25,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_lambda=1.0,
        objective="binary",
        monotone_constraints=monotone_constraints(x, y),
        monotone_constraints_method="advanced",
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    tree.fit(x, y, sample_weight=sample_weight)

    # The MLP has no sample_weight, so humans are oversampled instead.
    repeats = max(0, int(round(human_weight)) - 1)
    if repeats and np.any(y == 0):
        mask = y == 0
        x_mlp = np.vstack([x] + [x[mask]] * repeats)
        y_mlp = np.concatenate([y] + [y[mask]] * repeats)
    else:
        x_mlp, y_mlp = x, y

    neural = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=int(min(pca_components, x.shape[1], x.shape[0])),
                    random_state=seed + 2)),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=tuple(int(h) for h in mlp_hidden),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1,
            random_state=seed + 3,
        )),
    ])
    neural.fit(x_mlp, y_mlp)

    # Weights follow the published rank-detector-b trio (0.35/0.30/0.35), which
    # was itself selected on walk-forward. Re-tune them the same way, not by eye.
    return (
        [stack, tree, neural],
        [0.35, 0.30, 0.35],
        ["gradient_stack", "mono_lgbm", "pca_mlp"],
    )


def select_features(
    x: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    *,
    top_k: int = 150,
    seed: int = 42,
) -> List[str]:
    """Keep the top-K features by LightGBM gain. Fitted on TRAIN rows only."""
    import lightgbm as lgb

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if top_k >= len(names):
        return sorted(names)

    probe = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=48,
        min_data_in_leaf=25,
        objective="binary",
        monotone_constraints=monotone_constraints(x, y),
        monotone_constraints_method="advanced",
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    probe.fit(x, y, sample_weight=np.where(y == 0, 2.0, 1.0))
    gain = probe.booster_.feature_importance(importance_type="gain")
    keep = np.argsort(gain)[::-1][:top_k]
    return sorted(str(names[i]) for i in keep)
