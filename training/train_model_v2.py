"""V2 training pipeline: stacked LGB/XGB/CatBoost/ExtraTrees/RandomForest.

Beats the current ``train_model.py`` by:

* Out-of-fold (K-fold) **stacking** with a logistic-regression meta-learner,
  trained against the same labels.
* **Isotonic calibration** on stacked OOF scores so the final ranking is
  monotone and well-suited to average precision (65% of the validator reward).
* **Conformal FPR control**: a logit shift picked on a held-out set so that
  chunk-level FPR stays under a target well below the validator's 10% cliff.
* **Asymmetric sample weights** that protect humans (the FPR penalty term is
  squared and binary-cliffed, so a missed human is much worse than a missed
  bot).
* **Post-hoc calibration** (optional remap + logit shift) tuned for ranking AP
  first, then bot recall, while keeping chunk FPR well below the 0.10 cliff
  (default cap 0.05). Matches the live leaderboard pattern: high AP, modest
  recall, very low FPR.

The artifact format stays compatible with :class:`poker44_ml.inference.Poker44Model`:
``models = [StackedEnsemble(...)]`` with ``model_weights = [1.0]``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Silence the benign LightGBM <-> sklearn 1.7 feature-name validation warning.
# It fires because LightGBM 4.x stores a feature signature on fit even for
# numpy input, which sklearn's validator then compares against later
# (also-numpy) predict-time input. Predictions are correct -- the warning is
# noise that drowns out useful training output. Filter is scoped to this
# specific message, so any other sklearn UserWarning still shows up.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import numpy as np

from poker44.score.scoring import reward, format_reward_breakdown
from poker44_ml.chunk_score_metrics import (
    human_bot_prob_bounds,
    print_chunk_score_diagnostics,
)
from poker44.utils.model_manifest import artifact_model_identity
from poker44_ml.calibration import BlendedIsotonicCalibrator
from poker44_ml.inference import Poker44Model
from poker44_ml.stacked import StackedEnsemble
from training.build_dataset import (
    load_benchmark_examples,
    resolve_benchmark_paths,
)
from training.robust_features import filter_robust_feature_names, summarize_robust_filter

try:
    import joblib
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("joblib is required to train Poker44 models.") from exc

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None

try:
    from poker44_ml.sequence_model import (
        SequenceModelConfig,
        SequenceModelWrapper,
    )
except Exception:  # pragma: no cover
    SequenceModelConfig = None  # type: ignore[assignment]
    SequenceModelWrapper = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------- helpers ---------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stacked Poker44 model (v2).")
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "models" / "poker44_stacked_v2.joblib"),
    )
    parser.add_argument("--holdout-latest-days", type=int, default=2)
    parser.add_argument("--holdout-source-dates", type=str, default=None)
    parser.add_argument(
        "--use-released-split",
        action="store_true",
        help=(
            "Use the benchmark's own per-date 'split' field (train vs "
            "validation) instead of a date-based holdout. CAVEAT: for this "
            "dataset train/validation for the same date share ~65-90% "
            "identical hand content (recombined into different batches with "
            "the bot/human twin swapped), so this is NOT a leakage-free "
            "holdout the way --holdout-source-dates is -- expect optimistic "
            "metrics relative to a true held-out-date eval."
        ),
    )
    parser.add_argument(
        "--exclude-train-source-dates",
        type=str,
        default=None,
        help=(
            "Comma-separated sourceDate values to remove from the training "
            "side only. Use this when a specific date in training causes "
            "negative transfer to the holdout date (i.e. the model "
            "generalizes worse when that date is included). The dates are "
            "removed AFTER the holdout split so they affect training only."
        ),
    )
    parser.add_argument(
        "--no-miner-visible-payload",
        action="store_true",
        help=(
            "Train on raw benchmark JSON instead of prepare_hand_for_miner() "
            "sanitized payloads. Not recommended for production miners."
        ),
    )
    parser.add_argument(
        "--train-latest-days",
        type=int,
        default=0,
        help=(
            "If >0, keep only the N most recent sourceDate values in the "
            "training split (after holdout). Use to match live eval recency."
        ),
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.25,
        help=(
            "Fraction of training rows held out for score_remap calibration "
            "(tuned on validator_reward, not on the test holdout)."
        ),
    )
    parser.add_argument(
        "--max-validator-fpr",
        type=float,
        default=0.05,
        help=(
            "Reject calibration configs with chunk FPR at or above this value "
            "(default 0.05 leaves headroom below the 0.10 reward cliff)."
        ),
    )
    parser.add_argument(
        "--calibration-objective",
        type=str,
        choices=("ap_first", "reward", "recall"),
        default="ap_first",
        help=(
            "How to pick score_remap / score_logit on OOF: ap_first maximizes "
            "PR-AUC then recall; reward maximizes validator_reward; recall "
            "maximizes bot_recall (legacy)."
        ),
    )
    parser.add_argument(
        "--stack-calibrator",
        type=str,
        choices=("auto", "passthrough", "isotonic"),
        default="isotonic",
        help=(
            "Post-stack calibrator family. isotonic fits a monotonic recalibration "
            "on OOF stacked scores; passthrough applies none. (The legacy quantile "
            "calibrator has been removed.)"
        ),
    )
    parser.add_argument(
        "--isotonic-calibration-blend",
        type=float,
        default=0.5,
        help=(
            "Dulling for the isotonic stack calibrator: "
            "out = blend*isotonic(raw) + (1-blend)*raw. 1.0 = pure (sharp, "
            "step-shaped) isotonic; lower = duller/smoother (preserves ranking "
            "resolution); 0.0 = passthrough. Default 0.5."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="K-fold splits for OOF meta-learner training. Set to 1 to use a single "
             "holdout split (no cross-validation); controlled by --holdout-frac.",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.20,
        help="Fraction of training data held out for OOF when --n-folds=1. Default 0.20.",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.04,
        help="Conformal target for chunk-level FPR. Stays well below the 0.10 cliff.",
    )
    parser.add_argument(
        "--human-weight-multiplier",
        type=float,
        default=2.0,
        help="Asymmetric sample weight ratio for human chunks (higher = safer).",
    )
    parser.add_argument(
        "--meta-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for the logistic meta-learner.",
    )
    parser.add_argument(
        "--meta-hard-bot-weight",
        type=float,
        default=2.5,
        help=(
            "Extra OOF hard-example weight multiplier for bot rows in the meta "
            "learner. Higher values push recall up without post-hoc remap."
        ),
    )
    parser.add_argument(
        "--meta-hard-bot-gamma",
        type=float,
        default=2.0,
        help=(
            "Hard-bot focusing exponent for meta reweighting. Bots with lower OOF "
            "probability get larger weights."
        ),
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="If > 0, keep only the top-K features by LightGBM importance.",
    )
    parser.add_argument(
        "--robust-features-only",
        action="store_true",
        help=(
            "Keep only validator-generalized features (action mix, signatures, "
            "entropy). Drops outcome/position/absolute-BB columns."
        ),
    )
    parser.add_argument(
        "--no-score-remap",
        action="store_true",
        help="Do not tune the legacy high-band score_remap on the calibration split.",
    )
    parser.add_argument(
        "--fixed-score-remap-threshold",
        type=float,
        default=0.0,
        help=(
            "If > 0, use a FIXED threshold_logit score_remap with this threshold "
            "(the 0.5 crossing lands where the calibrated score equals this value) "
            "instead of fitting one on the calibration split. Robust to the "
            "benchmark->live date shift that makes fitted thresholds fragile under "
            "the 0.1.34 reward. Typical: 0.70."
        ),
    )
    parser.add_argument(
        "--fixed-score-remap-temperature",
        type=float,
        default=0.25,
        help="Temperature for the fixed score_remap (only used with --fixed-score-remap-threshold).",
    )
    parser.add_argument(
        "--score-remap-temperature-grid",
        type=str,
        default="0.12,0.18,0.25,0.35,0.50,0.65,0.85,1.0,1.25",
        help=(
            "Comma-separated threshold_logit temperatures for score_remap tuning. "
            "Higher values = duller/smoother remap (less sharp jumps near threshold)."
        ),
    )
    parser.add_argument(
        "--no-score-remap-prefer-smooth",
        action="store_true",
        help=(
            "When calibration metrics tie, do not prefer higher remap temperature "
            "(default prefers smoother/duller remap)."
        ),
    )
    parser.add_argument(
        "--no-score-logit-tune",
        action="store_true",
        help="Skip score_logit bias/temperature grid search on the calibration split.",
    )
    parser.add_argument(
        "--score-logit-bias-grid",
        type=str,
        default="-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5",
        help="Comma-separated grid of additive logit biases to search.",
    )
    parser.add_argument(
        "--score-logit-temperature-grid",
        type=str,
        default="0.6,0.8,1.0,1.2",
        help="Comma-separated grid of logit temperatures to search.",
    )
    parser.add_argument(
        "--disable-lightgbm",
        action="store_true",
        help="Skip LightGBM base learner (useful for ablation / lib testing).",
    )
    parser.add_argument(
        "--disable-xgboost",
        action="store_true",
        help="Skip XGBoost base learner.",
    )
    parser.add_argument(
        "--disable-catboost",
        action="store_true",
        help="Skip CatBoost base learner.",
    )
    parser.add_argument(
        "--disable-extratrees",
        action="store_true",
        help="Skip ExtraTrees base learner.",
    )
    parser.add_argument(
        "--disable-randomforest",
        action="store_true",
        help="Skip RandomForest base learner.",
    )
    parser.add_argument(
        "--sequence-only",
        action="store_true",
        help=(
            "Use only the chunk-level sequence base learner (disables all tree "
            "base learners)."
        ),
    )
    parser.add_argument(
        "--enable-gpu-trees",
        action="store_true",
        help="Use GPU for XGBoost and CatBoost (LightGBM stays on CPU since "
        "the pip wheel does not include a GPU build).",
    )
    parser.add_argument(
        "--enable-sequence",
        action="store_true",
        help="Enable the chunk-level Set Transformer base learner.",
    )
    parser.add_argument(
        "--sequence-epochs",
        type=int,
        default=8,
        help="Number of training epochs for the sequence model.",
    )
    parser.add_argument(
        "--sequence-batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--sequence-learning-rate",
        type=float,
        default=1e-3,
        help=(
            "Default/fallback LR when --sequence-learning-rate-schedule is "
            "unset or shorter than --sequence-epochs."
        ),
    )
    parser.add_argument(
        "--sequence-learning-rate-schedule",
        type=str,
        default=None,
        help=(
            "Piecewise LR per epoch: lr:epochs segments, comma-separated. "
            "Example: 1.3e-3:4,1e-3:4 runs 1.3e-3 for epochs 1-4 then 1e-3 "
            "for 5-8. Extra epochs repeat the last segment LR."
        ),
    )
    parser.add_argument(
        "--sequence-d-model",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--sequence-heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--sequence-action-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--sequence-hand-layers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--sequence-max-hands",
        type=int,
        default=64,
        help="Max sampled hands per chunk for the sequence model tokenizer.",
    )
    parser.add_argument(
        "--sequence-max-actions",
        type=int,
        default=12,
        help="Max actions retained per hand for the sequence model tokenizer.",
    )
    parser.add_argument(
        "--sequence-dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--sequence-device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--sequence-verbose",
        action="store_true",
        help="Print per-epoch train/val loss for the sequence model.",
    )
    parser.add_argument(
        "--sequence-verbose-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During sequence fit: per-epoch val metrics plus train/val summary "
            "(prob_min/max, bot_recall, FPR, etc.). Default on."
        ),
    )
    parser.add_argument(
        "--oof-learner-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Each CV fold: print OOF prob_min/max, human_prob_max, bot_prob_min, "
            "validator bot_recall/FPR for every base learner and sequence. Default on."
        ),
    )
    parser.add_argument(
        "--per-source-date",
        action="store_true",
        help="Print per-source-date diagnostics on the holdout split.",
    )
    return parser.parse_args()


def _repo_metadata() -> Dict[str, str]:
    def run(args: List[str]) -> str:
        try:
            completed = subprocess.run(
                args,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return ""
        return completed.stdout.strip()

    return {
        "repo_commit": run(["git", "rev-parse", "HEAD"]),
        "repo_url": run(["git", "config", "--get", "remote.origin.url"]),
    }


def _feature_schema_hash(feature_names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


def _build_matrix(
    examples: Sequence[Dict[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    return np.asarray(
        [
            [float(example["features"].get(name, 0.0)) for name in feature_names]
            for example in examples
        ],
        dtype=np.float64,
    )


def _split_temporal(
    examples: Sequence[Dict[str, Any]],
    *,
    holdout_source_dates: str | None,
    holdout_latest_days: int,
    exclude_train_source_dates: str | None,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    dates = sorted(
        {
            str(example.get("source_date", "")).strip()
            for example in examples
            if str(example.get("source_date", "")).strip()
        }
    )
    requested = [
        item.strip()
        for item in str(holdout_source_dates or "").split(",")
        if item.strip()
    ]
    excluded_train_dates = [
        item.strip()
        for item in str(exclude_train_source_dates or "").split(",")
        if item.strip()
    ]
    excluded_train_set = set(excluded_train_dates)

    holdout_dates = requested or dates[-max(1, int(holdout_latest_days)) :]
    if holdout_dates and dates:
        holdout_set = set(holdout_dates)
        train = [
            example
            for example in examples
            if str(example.get("source_date", "")).strip() not in holdout_set
        ]
        test = [
            example
            for example in examples
            if str(example.get("source_date", "")).strip() in holdout_set
        ]
        # Apply the train-only exclusion *after* the holdout split so the
        # excluded dates only disappear from training, not from the test set.
        if excluded_train_set:
            train = [
                example
                for example in train
                if str(example.get("source_date", "")).strip()
                not in excluded_train_set
            ]
        if (
            train
            and test
            and len({int(example["label"]) for example in train}) >= 2
            and len({int(example["label"]) for example in test}) >= 2
        ):
            return train, test, {
                "split_strategy": "holdout_source_dates",
                "holdout_source_dates": list(holdout_dates),
                "excluded_train_source_dates": excluded_train_dates,
                "train_source_dates": [
                    d
                    for d in dates
                    if d not in holdout_set and d not in excluded_train_set
                ],
            }

    labels = [int(example["label"]) for example in examples]
    train, test = train_test_split(
        list(examples),
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    if excluded_train_set:
        train = [
            example
            for example in train
            if str(example.get("source_date", "")).strip()
            not in excluded_train_set
        ]
    return list(train), list(test), {
        "split_strategy": "random_stratified",
        "holdout_source_dates": [],
        "excluded_train_source_dates": excluded_train_dates,
        "train_source_dates": [d for d in dates if d not in excluded_train_set],
    }


def _split_released(
    examples: Sequence[Dict[str, Any]],
    *,
    exclude_train_source_dates: str | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Use the benchmark's own per-date split field (train vs validation)
    instead of a date-based holdout.

    CAVEAT (verified 2026-07-05): for this dataset, train/validation for the
    SAME date share ~65-90% identical hand content (the per-date bot/human
    mirror spans both splits, just recombined into different batches with the
    label twin swapped). So this is NOT a leakage-free holdout the way
    HOLDOUT_SOURCE_DATES (a disjoint date range) is -- expect optimistic
    metrics here relative to a true held-out-date eval.
    """
    excluded_train_dates = [
        item.strip()
        for item in str(exclude_train_source_dates or "").split(",")
        if item.strip()
    ]
    excluded_train_set = set(excluded_train_dates)
    dates = sorted(
        {
            str(example.get("source_date", "")).strip()
            for example in examples
            if str(example.get("source_date", "")).strip()
        }
    )
    train = [e for e in examples if e.get("released_split") == "train"]
    test = [e for e in examples if e.get("released_split") == "validation"]
    unlabeled = len(examples) - len(train) - len(test)
    if unlabeled:
        raise RuntimeError(
            f"--use-released-split: {unlabeled}/{len(examples)} examples have no "
            "released 'split' field (train/validation) -- benchmark file predates "
            "the split field or was built without it."
        )
    if excluded_train_set:
        train = [
            e
            for e in train
            if str(e.get("source_date", "")).strip() not in excluded_train_set
        ]
    return list(train), list(test), {
        "split_strategy": "released_split",
        "holdout_source_dates": [],
        "excluded_train_source_dates": excluded_train_dates,
        "train_source_dates": [d for d in dates if d not in excluded_train_set],
    }


def _split_released_holdout(
    examples: Sequence[Dict[str, Any]],
    *,
    holdout_source_dates: str | None,
    exclude_train_source_dates: str | None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Combine --use-released-split with --holdout-source-dates:

      test  = released 'validation'-tagged examples, HOLDOUT dates only.
      train = ALL examples (both 'train'- and 'validation'-tagged) from every
              NON-holdout date (maximizes training data, same as plain
              date-holdout).
      discarded = 'train'-tagged examples of the HOLDOUT dates -- used
              NOWHERE. This is deliberate: train/validation for the SAME date
              share ~65-90% identical hand content (the bot/human mirror spans
              both tags), so including the holdout dates' 'train'-tagged
              batches in training would leak eval-adjacent content back in
              and defeat the point of a date-disjoint holdout.

    Net effect: date-disjoint (leakage-safe) between train and test, AND the
    test set is restricted to the officially-tagged validation subset of the
    holdout window (rather than every batch in that window).
    """
    requested = [
        item.strip()
        for item in str(holdout_source_dates or "").split(",")
        if item.strip()
    ]
    if not requested:
        raise RuntimeError(
            "--use-released-split combined with an empty --holdout-source-dates "
            "has nothing to hold out; pass real dates or drop --holdout-source-dates."
        )
    holdout_set = set(requested)
    excluded_train_dates = [
        item.strip()
        for item in str(exclude_train_source_dates or "").split(",")
        if item.strip()
    ]
    excluded_train_set = set(excluded_train_dates)
    dates = sorted(
        {
            str(example.get("source_date", "")).strip()
            for example in examples
            if str(example.get("source_date", "")).strip()
        }
    )
    missing = holdout_set - set(dates)
    if missing:
        raise RuntimeError(
            f"--holdout-source-dates has dates not present in the loaded "
            f"benchmark: {sorted(missing)}. Available dates: "
            f"{dates[0]}..{dates[-1]} ({len(dates)} dates)."
        )

    test = [
        e
        for e in examples
        if e.get("released_split") == "validation"
        and str(e.get("source_date", "")).strip() in holdout_set
    ]
    train = [
        e
        for e in examples
        if str(e.get("source_date", "")).strip() not in holdout_set
    ]
    discarded = len(examples) - len(train) - len(test)
    if not test:
        raise RuntimeError(
            "--use-released-split + --holdout-source-dates: no released "
            f"'validation'-tagged examples found for holdout dates {sorted(holdout_set)}."
        )
    if excluded_train_set:
        train = [
            e
            for e in train
            if str(e.get("source_date", "")).strip() not in excluded_train_set
        ]
    if (
        len({int(e["label"]) for e in train}) < 2
        or len({int(e["label"]) for e in test}) < 2
    ):
        raise RuntimeError(
            "--use-released-split + --holdout-source-dates: train or test set "
            "does not contain both labels after the split."
        )
    return list(train), list(test), {
        "split_strategy": "released_split_holdout",
        "holdout_source_dates": sorted(holdout_set),
        "excluded_train_source_dates": excluded_train_dates,
        "train_source_dates": [
            d for d in dates if d not in holdout_set and d not in excluded_train_set
        ],
        "discarded_train_tagged_holdout_count": discarded,
    }


# ---------- base learners ---------------------------------------------------


def _make_base_models(
    *,
    seed: int,
    enable_lgb: bool,
    enable_xgb: bool,
    enable_cb: bool,
    enable_extratrees: bool = True,
    enable_randomforest: bool = True,
    enable_gpu_trees: bool = False,
) -> List[Tuple[str, Any]]:
    models: List[Tuple[str, Any]] = []
    if enable_lgb and lgb is not None:
        models.append(
            (
                "lightgbm",
                lgb.LGBMClassifier(
                    n_estimators=int(os.getenv("LGB_N_ESTIMATORS", "1500")),
                    learning_rate=0.02,
                    num_leaves=63,
                    min_data_in_leaf=20,
                    feature_fraction=0.7,
                    bagging_fraction=0.8,
                    bagging_freq=1,
                    reg_lambda=1.0,
                    objective="binary",
                    n_jobs=-1,
                    random_state=seed,
                    verbose=-1,
                ),
            )
        )
    if enable_xgb and xgb is not None:
        xgb_kwargs: Dict[str, Any] = dict(
            n_estimators=int(os.getenv("XGB_N_ESTIMATORS", "1200")),
            learning_rate=0.025,
            max_depth=7,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed + 1,
            verbosity=0,
        )
        if enable_gpu_trees:
            xgb_kwargs["device"] = "cuda"
        models.append(("xgboost", xgb.XGBClassifier(**xgb_kwargs)))
    if enable_cb and CatBoostClassifier is not None:
        cb_kwargs: Dict[str, Any] = dict(
            iterations=1500,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=3.0,
            random_seed=seed + 2,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights=None,
            allow_writing_files=False,
            verbose=False,
        )
        if enable_gpu_trees:
            cb_kwargs["task_type"] = "GPU"
            cb_kwargs["devices"] = "0"
        models.append(("catboost", CatBoostClassifier(**cb_kwargs)))
    if enable_extratrees:
        models.append(
            (
                "extratrees",
                ExtraTreesClassifier(
                    n_estimators=int(os.getenv("ET_N_ESTIMATORS", "900")),
                    max_depth=12,
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=seed + 3,
                    n_jobs=-1,
                ),
            )
        )
    if enable_randomforest:
        models.append(
            (
                "randomforest",
                RandomForestClassifier(
                    n_estimators=int(os.getenv("RF_N_ESTIMATORS", "700")),
                    max_depth=12,
                    min_samples_leaf=1,
                    class_weight="balanced_subsample",
                    random_state=seed + 4,
                    n_jobs=-1,
                ),
            )
        )
    return models


def _make_sequence_model(args: argparse.Namespace) -> Any:
    if SequenceModelWrapper is None or SequenceModelConfig is None:
        raise RuntimeError("Sequence model requested but PyTorch is unavailable.")
    config = SequenceModelConfig(
        d_model=int(args.sequence_d_model),
        n_heads=int(args.sequence_heads),
        n_action_layers=int(args.sequence_action_layers),
        n_hand_layers=int(args.sequence_hand_layers),
        dropout=float(args.sequence_dropout),
        max_hands_per_chunk=max(1, int(args.sequence_max_hands)),
        max_actions_per_hand=max(1, int(args.sequence_max_actions)),
    )
    schedule = str(args.sequence_learning_rate_schedule or "").strip() or None
    return SequenceModelWrapper(
        config=config,
        n_epochs=int(args.sequence_epochs),
        batch_size=int(args.sequence_batch_size),
        learning_rate=float(args.sequence_learning_rate),
        learning_rate_schedule=schedule,
        seed=int(args.seed),
        device=str(args.sequence_device),
        verbose=bool(args.sequence_verbose),
        verbose_metrics=bool(args.sequence_verbose_metrics),
    )


def _clone(model: Any) -> Any:
    from sklearn.base import clone as sk_clone

    try:
        return sk_clone(model)
    except Exception:
        pass
    if lgb is not None and isinstance(model, lgb.LGBMClassifier):
        return lgb.LGBMClassifier(**model.get_params())
    if xgb is not None and isinstance(model, xgb.XGBClassifier):
        return xgb.XGBClassifier(**model.get_params())
    if CatBoostClassifier is not None and isinstance(model, CatBoostClassifier):
        return CatBoostClassifier(**model.get_all_params())
    raise RuntimeError(f"Cannot clone model of type {type(model).__name__}")


def _fit(model: Any, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
    if CatBoostClassifier is not None and isinstance(model, CatBoostClassifier):
        model.fit(x, y, sample_weight=weights)
    else:
        try:
            model.fit(x, y, sample_weight=weights)
        except TypeError:
            model.fit(x, y)


def _proba_pos(model: Any, x: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(x))
    return proba[:, 1] if proba.ndim == 2 else proba


def _hard_bot_meta_weights(
    base_weights: np.ndarray,
    labels: np.ndarray,
    oof_probs: np.ndarray,
    *,
    hard_bot_weight: float,
    gamma: float,
) -> np.ndarray:
    weights = np.asarray(base_weights, dtype=float).copy()
    if hard_bot_weight <= 0.0:
        return weights
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(oof_probs, dtype=float)
    gamma = max(float(gamma), 0.0)
    bot_mask = labels == 1
    if not np.any(bot_mask):
        return weights
    hardness = np.power(np.clip(1.0 - probs[bot_mask], 0.0, 1.0), gamma)
    weights[bot_mask] *= 1.0 + float(hard_bot_weight) * hardness
    return weights


# ---------- feature selection ----------------------------------------------


def _load_keep_only_names(path: str) -> list[str]:
    """Read a feature allowlist file (one name per line; ``#`` comments ok)."""
    names: list[str] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            name = line.split("#", 1)[0].strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    if not names:
        raise RuntimeError(f"ROBUST_KEEP_ONLY_FILE={path} contained no feature names.")
    return names


def _top_k_feature_indices(
    x: np.ndarray, y: np.ndarray, feature_names: Sequence[str], k: int, *, seed: int
) -> np.ndarray:
    if k <= 0 or k >= len(feature_names):
        return np.arange(len(feature_names), dtype=np.int64)
    if lgb is None:
        warnings.warn(
            "LightGBM not available; falling back to variance-based feature selection."
        )
        variances = np.var(x, axis=0)
        order = np.argsort(-variances)[:k]
        return np.sort(order).astype(np.int64)
    scout = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )
    scout.fit(x, y)
    importances = np.asarray(scout.feature_importances_, dtype=float)
    order = np.argsort(-importances)[:k]
    return np.sort(order).astype(np.int64)


# ---------- evaluation ------------------------------------------------------


def _binary_counts(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, float]:
    preds = [score >= 0.5 for score in scores]
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred)
    tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and not pred)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and not pred)
    positives = max(sum(1 for label in labels if label == 1), 1)
    negatives = max(sum(1 for label in labels if label == 0), 1)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "recall_at_0_5": float(tp / positives),
        "precision_at_0_5": float(tp / max(tp + fp, 1)),
        "fpr_at_0_5": float(fp / negatives),
    }


def _validator_metrics(
    labels: Sequence[int], scores: Sequence[float]
) -> Dict[str, float]:
    if not labels:
        return {}
    val_reward, details = reward(
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=int),
    )
    return {
        "validator_reward": float(val_reward),
        "validator_fpr": float(details.get("fpr", 1.0)),
        "validator_bot_recall": float(details.get("bot_recall", 0.0)),
        "validator_ap_score": float(details.get("ap_score", 0.0)),
        "validator_human_safety_penalty": float(
            details.get("human_safety_penalty", 0.0)
        ),
        "validator_base_score": float(details.get("base_score", 0.0)),
    }


def _reward_breakdown(metrics: Dict[str, float]) -> str:
    """Reward decomposition line from a ``validator_*`` metrics dict."""
    return format_reward_breakdown(
        float(metrics.get("validator_ap_score", 0.0)),
        float(metrics.get("validator_bot_recall", 0.0)),
        fpr=float(metrics.get("validator_fpr", 0.0)),
        reward=float(metrics.get("validator_reward", 0.0)),
    )


def _enrich_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    raw_scores: Sequence[float] | None = None,
) -> Dict[str, float]:
    from poker44_ml.chunk_score_metrics import enrich_chunk_metrics

    metrics = enrich_chunk_metrics(labels, scores, raw_scores=raw_scores)
    safe = [max(1e-6, min(1.0 - 1e-6, float(value))) for value in scores]
    if len(set(labels)) >= 2:
        metrics["mcc_at_0_5"] = float(
            matthews_corrcoef(labels, [value >= 0.5 for value in safe])
        )
    metrics["brier_score"] = float(brier_score_loss(labels, safe))
    metrics.update(_binary_counts(labels, safe))
    metrics.update(_validator_metrics(labels, safe))
    return metrics


# ---------- logit transforms (must match Poker44Model exactly) -------------


def _logit_shift(values: np.ndarray, bias: float, temperature: float) -> np.ndarray:
    if abs(float(bias)) < 1e-12 and abs(float(temperature) - 1.0) < 1e-12:
        return np.clip(values, 0.0, 1.0)
    temperature = max(float(temperature), 1e-6)
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = (np.log(clipped / (1.0 - clipped)) + float(bias)) / temperature
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _calibration_objective_key(
    metrics: Dict[str, float], objective: str
) -> Tuple[float, float, float]:
    """Lexicographic sort key for post-hoc calibration grid search."""
    ap = float(metrics.get("pr_auc", 0.0))
    recall = float(metrics.get("validator_bot_recall", 0.0))
    val_reward = float(metrics.get("validator_reward", 0.0))
    bot_min = float(metrics.get("bot_prob_min", 0.0))
    objective = str(objective).strip().lower()
    if objective == "recall":
        return (recall, bot_min, val_reward)
    if objective in ("reward", "threshold_sanity"):
        # 0.1.34-aware: primary = live reward (now includes the 0.5
        # threshold_sanity term). Break reward ties toward MORE bot chunks
        # crossing 0.5 (recall_at_0_5) -> pushes the remap threshold DOWN, which
        # is robust to the benchmark->live score shift (bots keep crossing even
        # when live scores compress). ap is rank-invariant so it never breaks a
        # tie; recall_at_0_5 does.
        recall_at_0_5 = float(metrics.get("recall_at_0_5", 0.0))
        return (val_reward, recall_at_0_5, recall)
    return (ap, recall, val_reward)


def _is_better_calibration_candidate(
    key: Tuple[float, float, float],
    best_key: Tuple[float, float, float],
    *,
    temperature: float,
    best_temperature: float,
    prefer_smooth_remap: bool,
) -> bool:
    if key > best_key:
        return True
    if not prefer_smooth_remap:
        return False
    # Only when the FULL objective key ties (not just the primary) does the
    # smoothness tie-break apply -- otherwise "prefer higher temperature" would
    # override the recall_at_0_5 secondary and re-collapse scores toward 0.5.
    if (
        all(abs(a - b) <= 1e-4 for a, b in zip(key, best_key))
        and temperature > best_temperature + 1e-9
    ):
        return True
    return False


def _passes_calibration_constraints(
    metrics: Dict[str, float], *, max_validator_fpr: float
) -> bool:
    if metrics.get("validator_fpr", 1.0) >= max_validator_fpr - 1e-9:
        return False
    # 0.1.34 threshold_sanity: humans may cross 0.5 up to FPR 0.10 (was: exactly
    # 0, via human_prob_max <= 0.495 -- which forced the remap threshold above
    # every human score and collapsed live scores below 0.5). Bots MUST cross
    # 0.5, else the reward zero-gates.
    if metrics.get("fpr_at_0_5", 1.0) > 0.10 + 1e-9:
        return False
    if metrics.get("recall_at_0_5", 0.0) <= 0.0:
        return False
    return True


def _grid(values: str) -> List[float]:
    out: List[float] = []
    for token in str(values).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out or [0.0]


def _select_score_logit_for_validator_reward(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    *,
    target_fpr: float,
    max_validator_fpr: float,
    bias_grid: Sequence[float],
    temp_grid: Sequence[float],
    calibration_objective: str = "ap_first",
) -> tuple[float, float, Dict[str, float]]:
    """Tune logit shift on stacked OOF/cal scores under FPR constraints."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(raw_scores, dtype=float)
    bot_scores = scores[labels == 1]
    humans = scores[labels == 0]
    conformal_bias = _conformal_bias_for_target_fpr(humans, target_fpr)
    raw_bot_p10 = float(np.quantile(bot_scores, 0.10)) if bot_scores.size else 0.5
    raw_bot_p50 = float(np.median(bot_scores)) if bot_scores.size else 0.5

    candidates = sorted(
        {
            float(value)
            for value in bias_grid
            if abs(float(value)) <= 5.0
        }
        | {conformal_bias}
        | {
            conformal_bias + delta
            for delta in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
        }
    )
    if calibration_objective == "recall" and (
        raw_bot_p50 < 0.35 or raw_bot_p10 < 0.15
    ):
        candidates = [value for value in candidates if value >= -0.25]

    baseline_metrics = _enrich_metrics(labels.tolist(), scores.tolist())
    best_key = _calibration_objective_key(baseline_metrics, calibration_objective)
    best_bias = 0.0
    best_temp = 1.0
    best_metrics = dict(baseline_metrics)
    prefer_minimal_shift = calibration_objective == "ap_first"

    for bias in candidates:
        for temperature in temp_grid:
            temperature = max(float(temperature), 1e-6)
            if abs(float(bias)) < 1e-12 and abs(temperature - 1.0) < 1e-12:
                continue
            shifted = _logit_shift(scores, bias, temperature)
            metrics = _enrich_metrics(labels.tolist(), shifted.tolist())
            if not _passes_calibration_constraints(
                metrics, max_validator_fpr=max_validator_fpr
            ):
                continue
            key = _calibration_objective_key(metrics, calibration_objective)
            replace = key > best_key
            if (
                not replace
                and prefer_minimal_shift
                and abs(key[0] - best_key[0]) < 1e-9
                and abs(key[1] - best_key[1]) < 1e-9
                and abs(key[2] - best_key[2]) < 1e-9
                and abs(float(bias)) < abs(best_bias)
            ):
                replace = True
            if replace:
                best_key = key
                best_bias = float(bias)
                best_temp = float(temperature)
                best_metrics = dict(metrics)
                best_metrics["calibration_validator_reward"] = float(
                    metrics.get("validator_reward", 0.0)
                )
                best_metrics["tune_raw_bot_p10"] = raw_bot_p10
                best_metrics["tune_raw_bot_p50"] = raw_bot_p50
                best_metrics["calibration_objective"] = calibration_objective

    return best_bias, best_temp, best_metrics


def _conformal_bias_for_target_fpr(
    human_scores: np.ndarray, target_fpr: float, *, max_abs_bias: float = 5.0
) -> float:
    """Return the logit bias that drops human-score quantile to ~0.5.

    The result is clipped to ``+- max_abs_bias`` because very large biases
    indicate the upstream calibrator collapsed the score distribution to
    {0, 1}. Selecting a huge bias to land humans on exactly 0.5 gives a
    misleading FPR (the validator uses ``np.round`` which is banker-rounded,
    so 0.5 rounds to 0) but the result is brittle to floating-point noise.
    """
    if human_scores.size == 0:
        return 0.0
    target_fpr = max(min(float(target_fpr), 0.5), 1e-4)
    quantile = float(np.quantile(human_scores, 1.0 - target_fpr))
    quantile = min(max(quantile, 1e-6), 1.0 - 1e-6)
    cur_logit = np.log(quantile / (1.0 - quantile))
    bias = -cur_logit
    return float(max(-abs(max_abs_bias), min(abs(max_abs_bias), bias)))


def _restrict_train_latest_dates(
    train_examples: List[Dict[str, Any]],
    latest_days: int,
) -> List[Dict[str, Any]]:
    if latest_days <= 0:
        return train_examples
    dates = sorted(
        {
            str(example.get("source_date", "")).strip()
            for example in train_examples
            if str(example.get("source_date", "")).strip()
        }
    )
    if len(dates) <= latest_days:
        return train_examples
    keep = set(dates[-latest_days:])
    filtered = [
        example
        for example in train_examples
        if str(example.get("source_date", "")).strip() in keep
    ]
    if len({int(example["label"]) for example in filtered}) < 2:
        return train_examples
    return filtered


def _split_fit_calibration(
    train_examples: List[Dict[str, Any]],
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    labels = [int(example["label"]) for example in train_examples]
    if len(train_examples) < 80 or calibration_fraction <= 0 or len(set(labels)) < 2:
        return train_examples, []
    fit, calibration = train_test_split(
        train_examples,
        test_size=min(max(float(calibration_fraction), 0.05), 0.4),
        random_state=seed,
        stratify=labels,
    )
    return list(fit), list(calibration)


def _apply_score_remap_np(
    scores: np.ndarray,
    remap: Dict[str, Any],
) -> np.ndarray:
    if not remap or remap.get("kind") != "threshold_logit_v1":
        return np.clip(scores, 0.0, 1.0)
    threshold = float(remap.get("threshold", 0.5))
    temperature = max(float(remap.get("temperature", 0.25)), 1e-6)
    clipped = np.clip(scores.astype(float), 1e-6, 1.0 - 1e-6)
    adjusted = (clipped - threshold) / temperature
    return 1.0 / (1.0 + np.exp(-np.clip(adjusted, -40.0, 40.0)))


def _select_score_remap_for_validator_reward(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    *,
    target_fpr: float,
    max_validator_fpr: float,
    calibration_objective: str = "ap_first",
    temperature_grid: Sequence[float] | None = None,
    prefer_smooth_remap: bool = True,
) -> tuple[Dict[str, Any] | None, Dict[str, float]]:
    """Pick threshold/temperature on a calibration set."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(raw_scores, dtype=float)
    humans = scores[labels == 0]
    bots = scores[labels == 1]
    if humans.size < 8 or bots.size < 8:
        return None, {}

    thresholds: set[float] = set()
    for quantile in np.linspace(0.40, 0.995, 24):
        thresholds.add(float(np.quantile(humans, quantile)))
    for quantile in np.linspace(0.005, 0.60, 20):
        thresholds.add(float(np.quantile(bots, quantile)))
    for anchor in (0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30):
        thresholds.add(anchor)

    temperatures = sorted(
        {max(float(t), 1e-6) for t in (temperature_grid or [])}
        or [0.12, 0.18, 0.25, 0.35, 0.50, 0.65, 0.85, 1.0, 1.25]
    )
    baseline_metrics = _enrich_metrics(labels.tolist(), scores.tolist())
    best_key = _calibration_objective_key(baseline_metrics, calibration_objective)
    best_remap: Dict[str, Any] | None = None
    best_metrics: Dict[str, float] = dict(baseline_metrics)
    best_temperature = 0.0

    for threshold in sorted(thresholds):
        for temperature in temperatures:
            remap_candidate = {
                "kind": "threshold_logit_v1",
                "threshold": float(threshold),
                "temperature": float(temperature),
            }
            remapped = _apply_score_remap_np(scores, remap_candidate)
            metrics = _enrich_metrics(labels.tolist(), remapped.tolist())
            if not _passes_calibration_constraints(
                metrics, max_validator_fpr=max_validator_fpr
            ):
                continue
            key = _calibration_objective_key(metrics, calibration_objective)
            if _is_better_calibration_candidate(
                key,
                best_key,
                temperature=float(temperature),
                best_temperature=best_temperature,
                prefer_smooth_remap=prefer_smooth_remap,
            ):
                best_key = key
                best_remap = remap_candidate
                best_metrics = metrics
                best_temperature = float(temperature)

    if best_remap is None and calibration_objective != "ap_first":
        human_cut = float(np.quantile(humans, 1.0 - max(min(target_fpr, 0.2), 0.01)))
        bot_cut = float(np.quantile(bots, 0.10))
        threshold = 0.5 * (human_cut + bot_cut)
        fallback_temp = float(max(temperatures) if temperatures else 0.25)
        best_remap = {
            "kind": "threshold_logit_v1",
            "threshold": threshold,
            "temperature": fallback_temp,
            "human_cutoff": human_cut,
            "bot_cutoff": bot_cut,
            "fallback": True,
        }
        remapped = _apply_score_remap_np(scores, best_remap)
        best_metrics = _enrich_metrics(labels.tolist(), remapped.tolist())
    elif best_remap is not None:
        best_remap["human_cutoff"] = float(
            np.quantile(humans, 1.0 - max(min(target_fpr, 0.2), 0.01))
        )
        best_remap["bot_cutoff"] = float(np.quantile(bots, 0.10))
        best_remap["fallback"] = False

    best_metrics["calibration_validator_reward"] = float(
        best_metrics.get("validator_reward", 0.0)
    )
    best_metrics["calibration_objective"] = calibration_objective
    return best_remap, best_metrics


# ---------- main training routine ------------------------------------------


def train(args: argparse.Namespace) -> Dict[str, Any]:
    benchmark_paths = resolve_benchmark_paths(args.benchmark_path)
    miner_visible = not bool(args.no_miner_visible_payload)
    examples = load_benchmark_examples(
        benchmark_paths,
        miner_visible=miner_visible,
    )
    labels_total = Counter(int(example["label"]) for example in examples)
    if len(labels_total) != 2:
        raise RuntimeError(
            f"Benchmark must contain both labels, got {dict(labels_total)}"
        )

    if args.use_released_split and str(args.holdout_source_dates or "").strip():
        train_examples, test_examples, split_info = _split_released_holdout(
            examples,
            holdout_source_dates=args.holdout_source_dates,
            exclude_train_source_dates=args.exclude_train_source_dates,
        )
    elif args.use_released_split:
        train_examples, test_examples, split_info = _split_released(
            examples,
            exclude_train_source_dates=args.exclude_train_source_dates,
        )
    else:
        train_examples, test_examples, split_info = _split_temporal(
            examples,
            holdout_source_dates=args.holdout_source_dates,
            holdout_latest_days=args.holdout_latest_days,
            exclude_train_source_dates=args.exclude_train_source_dates,
            seed=args.seed,
        )
    train_examples = _restrict_train_latest_dates(
        train_examples,
        int(args.train_latest_days),
    )
    fit_examples, cal_examples = _split_fit_calibration(
        train_examples,
        calibration_fraction=float(args.calibration_fraction),
        seed=int(args.seed) + 17,
    )
    print(
        f"Loaded {len(examples)} examples "
        f"({labels_total.get(1, 0)} bot / {labels_total.get(0, 0)} human). "
        f"miner_visible_payload={miner_visible} "
        f"fit={len(fit_examples)} cal={len(cal_examples)} "
        f"test={len(test_examples)} "
        f"split={split_info['split_strategy']} "
        f"holdout={split_info.get('holdout_source_dates')} "
        f"excluded_train_dates={split_info.get('excluded_train_source_dates')}"
    )
    train_examples = fit_examples

    all_feature_names = sorted(examples[0]["features"].keys())
    keep_only_path = os.getenv("ROBUST_KEEP_ONLY_FILE", "").strip()
    if keep_only_path:
        allowed = _load_keep_only_names(keep_only_path)
        available = set(all_feature_names)
        feature_names = sorted(name for name in allowed if name in available)
        missing = sorted(name for name in allowed if name not in available)
        if len(feature_names) < 16:
            raise RuntimeError(
                f"ROBUST_KEEP_ONLY_FILE={keep_only_path} matched only "
                f"{len(feature_names)} of {len(all_feature_names)} available "
                "features; check the allowlist file and dataset schema."
            )
        print(
            f"Keep-only feature filter ({keep_only_path}): "
            f"kept {len(feature_names)}/{len(all_feature_names)} "
            f"(file listed {len(allowed)}, {len(missing)} not in dataset)"
        )
        if missing:
            print(f"  missing sample: {missing[:12]}")
    elif args.robust_features_only:
        feature_names = filter_robust_feature_names(all_feature_names)
        summary = summarize_robust_filter(all_feature_names, feature_names)
        if len(feature_names) < 16:
            raise RuntimeError(
                f"--robust-features-only kept only {len(feature_names)} features; "
                "check training/robust_features.py allowlist."
            )
        print(
            "Robust feature filter: "
            f"kept {summary['kept']}/{summary['total']} "
            f"(dropped {summary['dropped']})"
        )
    else:
        feature_names = all_feature_names
    x_train = _build_matrix(train_examples, feature_names)
    y_train = np.asarray(
        [int(example["label"]) for example in train_examples], dtype=np.int64
    )
    x_test = _build_matrix(test_examples, feature_names)
    y_test = np.asarray(
        [int(example["label"]) for example in test_examples], dtype=np.int64
    )

    feature_indices = _top_k_feature_indices(
        x_train, y_train, feature_names, args.max_features, seed=args.seed
    )
    x_train_sel = x_train[:, feature_indices]
    x_test_sel = x_test[:, feature_indices]
    print(
        f"Using {len(feature_indices)}/{len(feature_names)} features after selection."
    )

    sequence_enabled = bool(args.enable_sequence)
    if args.sequence_only:
        sequence_enabled = True

    base_specs = _make_base_models(
        seed=args.seed,
        enable_lgb=(not args.disable_lightgbm) and (not args.sequence_only),
        enable_xgb=(not args.disable_xgboost) and (not args.sequence_only),
        enable_cb=(not args.disable_catboost) and (not args.sequence_only),
        enable_extratrees=(not args.disable_extratrees) and (not args.sequence_only),
        enable_randomforest=(not args.disable_randomforest) and (not args.sequence_only),
        enable_gpu_trees=bool(args.enable_gpu_trees),
    )
    base_names_initial = [name for name, _ in base_specs]
    if sequence_enabled and SequenceModelWrapper is None:
        warnings.warn(
            "--enable-sequence was passed but the sequence model could not be "
            "imported (likely missing PyTorch). Falling back to feature-only stack."
        )
        sequence_enabled = False
    if not base_specs and not sequence_enabled:
        raise RuntimeError(
            "No base learners enabled. Enable at least one tree model or pass --enable-sequence/--sequence-only."
        )
    column_names = list(base_names_initial)
    if sequence_enabled:
        column_names.append("sequence")
    print("Base learners:", ", ".join(column_names))
    if sequence_enabled and args.sequence_verbose_metrics:
        print(
            "Sequence training metrics: ON (per-epoch val + fold OOF; "
            "disable with --no-sequence-verbose-metrics)"
        )
    if args.oof_learner_metrics:
        print(
            "Per-fold OOF learner metrics: ON (prob_min/max, bot_recall, FPR; "
            "disable with --no-oof-learner-metrics)"
        )

    train_chunks = [example["chunk"] for example in train_examples]

    sample_weights = np.where(
        y_train == 0,
        float(args.human_weight_multiplier),
        1.0,
    ).astype(np.float64)

    # OOF predictions for the meta-learner.
    # n_folds<=1 → no internal split: train on ALL fit_examples, predict on
    #             cal_examples to get meta/calibration scores.
    # n_folds≥2 → standard k-fold OOF on fit_examples.
    n_folds = max(1, int(args.n_folds))
    fold_aps: List[float] = []
    # These are set inside the n_folds==1 block and used during final refit.
    y_train_fit = y_train
    sample_weights_fit = sample_weights
    train_chunks_fit = train_chunks
    x_train_sel_fit = x_train_sel

    if n_folds == 1:
        # Train on all fit_examples (full 6400), predict on cal_examples (1600).
        # This gives the most training data to the base model while still
        # providing clean held-out scores for meta-learner and calibration.
        if not cal_examples:
            raise RuntimeError(
                "N_FOLDS=1 requires a calibration split (--calibration-fraction > 0) "
                "to produce held-out scores for the meta-learner."
            )
        x_cal = _build_matrix(cal_examples, feature_names)[:, feature_indices]
        y_cal_meta = np.asarray(
            [int(e["label"]) for e in cal_examples], dtype=np.int64
        )
        cal_chunks = [e["chunk"] for e in cal_examples]
        cal_weights = np.where(
            y_cal_meta == 0, float(args.human_weight_multiplier), 1.0
        ).astype(np.float64)

        oof = np.zeros((len(y_cal_meta), len(column_names)), dtype=np.float64)
        for model_idx, (name, model_proto) in enumerate(base_specs):
            model = _clone(model_proto)
            _fit(model, x_train_sel, y_train, sample_weights)
            base_proba = _proba_pos(model, x_cal)
            oof[:, model_idx] = base_proba
            print_chunk_score_diagnostics(
                f"no-fold {name} cal",
                y_cal_meta.tolist(),
                base_proba.tolist(),
            )
        if sequence_enabled:
            seq_model = _make_sequence_model(args)
            seq_model.fit(
                train_chunks,
                y_train.tolist(),
                sample_weight=sample_weights.tolist(),
            )
            seq_proba = seq_model.predict_proba(cal_chunks)[:, 1]
            oof[:, len(base_specs)] = seq_proba
            print_chunk_score_diagnostics(
                "no-fold sequence cal",
                y_cal_meta.tolist(),
                seq_proba.tolist(),
            )
        fold_ap = float(average_precision_score(y_cal_meta, oof.mean(axis=1)))
        fold_aps.append(fold_ap)
        print(
            f"[no-fold] trained on {len(y_train)} fit rows, "
            f"cal={len(y_cal_meta)} rows, mean-base AP={fold_ap:.4f}"
        )
        # Keep original fit_examples data for the final refit below.
        # Only swap y_train/sample_weights for meta-learner and calibration
        # (meta_idx selects all cal rows which all have real predictions).
        y_train_fit = y_train
        sample_weights_fit = sample_weights
        train_chunks_fit = train_chunks
        x_train_sel_fit = x_train_sel
        y_train = y_cal_meta
        sample_weights = cal_weights
        meta_idx = np.arange(len(y_cal_meta))

    else:
        oof = np.zeros((len(y_train), len(column_names)), dtype=np.float64)
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        splits = list(kfold.split(x_train_sel, y_train))

        for fold_idx, (tr_idx, va_idx) in enumerate(splits):
            fold_label = f"fold {fold_idx + 1}/{n_folds}"
            x_tr, x_va = x_train_sel[tr_idx], x_train_sel[va_idx]
            y_tr = y_train[tr_idx]
            w_tr = sample_weights[tr_idx]
            for model_idx, (name, model_proto) in enumerate(base_specs):
                model = _clone(model_proto)
                _fit(model, x_tr, y_tr, w_tr)
                base_proba = _proba_pos(model, x_va)
                oof[va_idx, model_idx] = base_proba
                if args.oof_learner_metrics:
                    print_chunk_score_diagnostics(
                        f"{fold_label} {name} OOF",
                        y_train[va_idx].tolist(),
                        base_proba.tolist(),
                    )
            if sequence_enabled:
                seq_model = _make_sequence_model(args)
                seq_train_chunks = [train_chunks[i] for i in tr_idx]
                seq_model.fit(
                    seq_train_chunks,
                    y_tr.tolist(),
                    sample_weight=w_tr.tolist(),
                )
                seq_proba = seq_model.predict_proba(
                    [train_chunks[i] for i in va_idx]
                )[:, 1]
                oof[va_idx, len(base_specs)] = seq_proba
                if args.oof_learner_metrics:
                    print_chunk_score_diagnostics(
                        f"{fold_label} sequence OOF",
                        y_train[va_idx].tolist(),
                        seq_proba.tolist(),
                    )
            fold_ap = float(
                average_precision_score(y_train[va_idx], oof[va_idx].mean(axis=1))
            )
            fold_aps.append(fold_ap)
            print(f"  {fold_label} mean-base AP={fold_ap:.4f}")

        meta_idx = np.arange(len(y_train))
    meta = LogisticRegression(
        C=float(args.meta_c),
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,
    )
    meta.fit(oof[meta_idx], y_train[meta_idx], sample_weight=sample_weights[meta_idx])
    stacked_oof = np.asarray(meta.predict_proba(oof[meta_idx]))[:, 1]
    if float(args.meta_hard_bot_weight) > 0.0:
        hard_weights = _hard_bot_meta_weights(
            sample_weights[meta_idx],
            y_train[meta_idx],
            stacked_oof,
            hard_bot_weight=float(args.meta_hard_bot_weight),
            gamma=float(args.meta_hard_bot_gamma),
        )
        meta.fit(oof[meta_idx], y_train[meta_idx], sample_weight=hard_weights)
        stacked_oof = np.asarray(meta.predict_proba(oof[meta_idx]))[:, 1]
        bot_probs = stacked_oof[y_train[meta_idx] == 1]
        print(
            "Meta hard-bot focus: "
            f"weight={float(args.meta_hard_bot_weight):.2f} "
            f"gamma={float(args.meta_hard_bot_gamma):.2f} "
            f"bot_oof_q10={float(np.quantile(bot_probs, 0.10)) if bot_probs.size else 0.0:.4f} "
            f"bot_oof_q50={float(np.quantile(bot_probs, 0.50)) if bot_probs.size else 0.0:.4f}"
        )
    oof_ap = float(average_precision_score(y_train[meta_idx], stacked_oof))
    print(f"Stacked OOF AP={oof_ap:.4f} (mean-fold {np.mean(fold_aps):.4f})")
    oof_raw_bounds = human_bot_prob_bounds(y_train[meta_idx].tolist(), stacked_oof.tolist())
    print(
        "Stacked OOF score range: "
        f"raw=[{float(np.min(stacked_oof)):.4f}, {float(np.max(stacked_oof)):.4f}] "
        f"raw_human_prob_max={oof_raw_bounds['human_prob_max']:.4f} "
        f"raw_bot_prob_min={oof_raw_bounds['bot_prob_min']:.4f}"
    )

    # Calibration on OOF stacked scores.
    calibrator_mode = str(args.stack_calibrator).strip().lower()
    if calibrator_mode == "auto":
        calibrator_mode = "isotonic"

    stack_calibrator: Optional[Any] = None
    if calibrator_mode == "passthrough":
        calibrated_oof = stacked_oof.copy()
        print(f"Calibrator: passthrough (OOF AP={oof_ap:.4f})")
    elif calibrator_mode == "isotonic":
        iso = BlendedIsotonicCalibrator(blend=float(args.isotonic_calibration_blend))
        iso.fit(stacked_oof, y_train[meta_idx])
        stack_calibrator = iso
        calibrated_oof = np.asarray(iso.transform(stacked_oof), dtype=float)
        print(
            "Calibrator: isotonic "
            f"(OOF AP={oof_ap:.4f}, blend={float(args.isotonic_calibration_blend):.2f})"
        )
    else:
        raise RuntimeError(f"Unknown stack calibrator mode: {calibrator_mode}")
    oof_cal_bounds = human_bot_prob_bounds(y_train[meta_idx].tolist(), calibrated_oof.tolist())
    print(
        "Stacked OOF calibrated range: "
        f"[{float(np.min(calibrated_oof)):.4f}, {float(np.max(calibrated_oof)):.4f}] "
        f"human_prob_max={oof_cal_bounds['human_prob_max']:.4f} "
        f"bot_prob_min={oof_cal_bounds['bot_prob_min']:.4f}"
    )

    # Refit base models on the full training set (fit_examples, not cal_examples).
    # When n_folds=1, y_train/sample_weights were swapped to cal data for
    # meta-learner calibration above — use the preserved _fit variables here.
    _y_refit = y_train_fit if n_folds == 1 else y_train
    _w_refit = sample_weights_fit if n_folds == 1 else sample_weights
    _x_refit = x_train_sel_fit if n_folds == 1 else x_train_sel
    _chunks_refit = train_chunks_fit if n_folds == 1 else train_chunks

    base_models: List[Any] = []
    base_names: List[str] = []
    for name, model_proto in base_specs:
        model = _clone(model_proto)
        _fit(model, _x_refit, _y_refit, _w_refit)
        base_models.append(model)
        base_names.append(name)

    chunk_models: List[Any] = []
    chunk_names: List[str] = []
    if sequence_enabled:
        final_seq_model = _make_sequence_model(args)
        if final_seq_model.learning_rate_schedule:
            from poker44_ml.sequence_model import parse_learning_rate_schedule

            epoch_lrs = parse_learning_rate_schedule(
                final_seq_model.learning_rate_schedule,
                default_lr=float(final_seq_model.learning_rate),
                n_epochs=int(final_seq_model.n_epochs),
            )
            print(
                "Sequence learning-rate schedule: "
                f"{final_seq_model.learning_rate_schedule} -> "
                f"{[round(lr, 6) for lr in epoch_lrs]}"
            )
        final_seq_model.fit(
            _chunks_refit,
            _y_refit.tolist(),
            sample_weight=_w_refit.tolist(),
        )
        chunk_models.append(final_seq_model)
        chunk_names.append("sequence")
        if args.sequence_verbose_metrics:
            seq_train_proba = final_seq_model.predict_proba(_chunks_refit)[:, 1]
            print_chunk_score_diagnostics(
                "sequence final fit (full train)",
                _y_refit.tolist(),
                seq_train_proba.tolist(),
            )

    # Stacked ensemble assembly.
    stacked = StackedEnsemble(
        base_models=base_models,
        meta_model=meta,
        calibrator=stack_calibrator,
        feature_indices=feature_indices,
        score_shift=0.0,
        chunk_models=chunk_models,
    )

    def _stacked_raw_scores(
        examples: List[Dict[str, Any]], *, calibrated: bool = True
    ) -> np.ndarray:
        x_rows = _build_matrix(examples, feature_names)[:, feature_indices]
        if chunk_models:
            return np.asarray(
                stacked.predict_chunk_scores(
                    [example["chunk"] for example in examples],
                    feature_rows=x_rows,
                    apply_calibration=calibrated,
                ),
                dtype=float,
            )
        return stacked.predict_proba(x_rows, apply_calibration=calibrated)[:, 1]

    cal_objective = str(args.calibration_objective)
    use_score_remap = not bool(args.no_score_remap)
    prefer_smooth_remap = not bool(args.no_score_remap_prefer_smooth)
    score_remap_temp_grid = _grid(args.score_remap_temperature_grid)
    score_remap: Dict[str, Any] | None = None
    cal_metrics: Dict[str, float] = {}
    if use_score_remap and cal_examples:
        y_cal = np.asarray(
            [int(example["label"]) for example in cal_examples], dtype=np.int64
        )
        cal_raw = _stacked_raw_scores(cal_examples)
        _fixed_remap_thr = float(getattr(args, "fixed_score_remap_threshold", 0.0) or 0.0)
        if _fixed_remap_thr > 0.0:
            # FIXED (not fitted) threshold_logit remap: place the 0.5 crossing at
            # a constant calibrated-score value. Cal-fit thresholds don't
            # generalize across the day-to-day v1.13 distribution shift (verified),
            # so under the 0.1.34 reward a fixed upward crossing is the robust
            # choice: bots stay above 0.5, humans below, no zero-gate.
            score_remap = {
                "kind": "threshold_logit_v1",
                "threshold": _fixed_remap_thr,
                "temperature": float(
                    getattr(args, "fixed_score_remap_temperature", 0.25) or 0.25
                ),
                "fixed": True,
            }
            cal_metrics = _enrich_metrics(
                y_cal.tolist(),
                _apply_score_remap_np(cal_raw, score_remap).tolist(),
            )
            print(
                "score_remap FIXED (not fitted, robust to date-shift): "
                f"threshold={score_remap['threshold']:.4f} "
                f"temperature={score_remap['temperature']:.4f}"
            )
        else:
            score_remap, cal_metrics = _select_score_remap_for_validator_reward(
                y_cal,
                cal_raw,
                target_fpr=float(args.target_fpr),
                max_validator_fpr=float(args.max_validator_fpr),
                calibration_objective=cal_objective,
                temperature_grid=score_remap_temp_grid,
                prefer_smooth_remap=prefer_smooth_remap,
            )
        cal_raw_bounds = human_bot_prob_bounds(y_cal.tolist(), cal_raw.tolist())
        cal_remap_line = ""
        if score_remap:
            cal_remapped = _apply_score_remap_np(cal_raw, score_remap)
            cal_remap_bounds = human_bot_prob_bounds(
                y_cal.tolist(), cal_remapped.tolist()
            )
            cal_remap_line = (
                f" remap_range=[{float(np.min(cal_remapped)):.4f}, "
                f"{float(np.max(cal_remapped)):.4f}]"
                f" remap_human_prob_max={cal_remap_bounds['human_prob_max']:.4f}"
                f" remap_bot_prob_min={cal_remap_bounds['bot_prob_min']:.4f}"
            )
        print(
            "Calibration score_remap: "
            f"objective={cal_objective} "
            f"threshold={score_remap.get('threshold') if score_remap else None} "
            f"temperature={score_remap.get('temperature') if score_remap else None} "
            f"cal_raw_range=[{float(np.min(cal_raw)):.4f}, {float(np.max(cal_raw)):.4f}] "
            f"raw_human_prob_max={cal_raw_bounds['human_prob_max']:.4f} "
            f"raw_bot_prob_min={cal_raw_bounds['bot_prob_min']:.4f}"
            f"{cal_remap_line} "
            f"cal_ap={cal_metrics.get('pr_auc', 0.0):.4f} "
            f"cal_reward={cal_metrics.get('calibration_validator_reward', 0.0):.4f} "
            f"cal_fpr={cal_metrics.get('validator_fpr', 0.0):.4f} "
            f"cal_recall={cal_metrics.get('validator_bot_recall', 0.0):.4f}"
        )
    elif use_score_remap:
        _fixed_remap_thr = float(getattr(args, "fixed_score_remap_threshold", 0.0) or 0.0)
        if _fixed_remap_thr > 0.0:
            score_remap = {
                "kind": "threshold_logit_v1",
                "threshold": _fixed_remap_thr,
                "temperature": float(
                    getattr(args, "fixed_score_remap_temperature", 0.25) or 0.25
                ),
                "fixed": True,
            }
            print(
                "score_remap FIXED (not fitted, no cal split needed): "
                f"threshold={score_remap['threshold']:.4f} "
                f"temperature={score_remap['temperature']:.4f}"
            )
        else:
            print("WARN: no calibration split; score_remap disabled.")
    else:
        print("score_remap disabled (--no-score-remap).")

    # Match the inference order (calibrator -> remap -> logit): the logit tune
    # must see the SAME remapped scores the bias/temperature will be applied to
    # at scoring time. Without this, the bias is tuned on calibrated-but-not-
    # remapped scores and applied post-remap, leaving it slightly off-target.
    oof_for_logit = calibrated_oof
    if score_remap and use_score_remap:
        oof_for_logit = _apply_score_remap_np(calibrated_oof, score_remap)

    score_logit_bias = 0.0
    score_logit_temperature = 1.0
    logit_cal_metrics: Dict[str, float] = {}
    if not args.no_score_logit_tune:
        score_logit_bias, score_logit_temperature, logit_cal_metrics = (
            _select_score_logit_for_validator_reward(
                y_train[meta_idx],
                oof_for_logit,
                target_fpr=float(args.target_fpr),
                max_validator_fpr=float(args.max_validator_fpr),
                bias_grid=_grid(args.score_logit_bias_grid),
                temp_grid=_grid(args.score_logit_temperature_grid),
                calibration_objective=cal_objective,
            )
        )
        print(
            "OOF score_logit tune: "
            f"objective={cal_objective} "
            f"bias={score_logit_bias:.4f} "
            f"temperature={score_logit_temperature:.4f} "
            f"oof_ap={logit_cal_metrics.get('pr_auc', 0.0):.4f} "
            f"oof_reward={logit_cal_metrics.get('calibration_validator_reward', 0.0):.4f} "
            f"oof_fpr={logit_cal_metrics.get('validator_fpr', 0.0):.4f} "
            f"oof_recall={logit_cal_metrics.get('validator_bot_recall', 0.0):.4f} "
            f"oof_bot_prob_min={logit_cal_metrics.get('bot_prob_min', 0.0):.4f}"
        )

    # Explicit pipeline stages for logging:
    #   test_precal     = base -> meta, BEFORE any calibration (raw stacked score)
    #   test_calibrated = + stack (isotonic) calibrator      [input to remap/logit]
    #   test_mid        = + score_remap
    #   test_final      = + score_logit  (the score the validator rounds at 0.5)
    test_precal = _stacked_raw_scores(test_examples, calibrated=False)
    test_calibrated = _stacked_raw_scores(test_examples)
    test_mid = test_calibrated
    if score_remap and use_score_remap:
        test_mid = _apply_score_remap_np(test_calibrated, score_remap)
    test_final = _logit_shift(
        test_mid, score_logit_bias, score_logit_temperature
    )
    raw_humans = test_precal[y_test == 0]
    raw_bots = test_precal[y_test == 1]
    raw_separation = (
        float(np.quantile(raw_bots, 0.10) - np.quantile(raw_humans, 0.90))
        if raw_humans.size and raw_bots.size
        else 0.0
    )
    test_metrics = _enrich_metrics(
        y_test.tolist(),
        test_final.tolist(),
        raw_scores=test_precal.tolist(),
    )
    remap_bounds: Dict[str, float] = {}
    if score_remap and use_score_remap:
        remap_bounds = human_bot_prob_bounds(y_test.tolist(), test_mid.tolist())
    best = {
        "reward": float(test_metrics.get("validator_reward", 0.0)),
        "bias": float(score_logit_bias),
        "temperature": float(score_logit_temperature),
        "metrics": test_metrics,
    }
    holdout_remap_line = ""
    if remap_bounds:
        holdout_remap_line = (
            f" remap_human_prob_max={remap_bounds['human_prob_max']:.4f}"
            f" remap_bot_prob_min={remap_bounds['bot_prob_min']:.4f}"
        )
    print(
        "Holdout test (honest, not used for calibration): "
        f"pr_auc={best['metrics'].get('pr_auc', 0.0):.4f} "
        f"validator_reward={best['reward']:.4f} "
        f"validator_fpr={best['metrics'].get('validator_fpr', 0.0):.4f} "
        f"validator_bot_recall={best['metrics'].get('validator_bot_recall', 0.0):.4f} "
        f"raw_range=[{float(np.min(test_precal)):.4f}, {float(np.max(test_precal)):.4f}] "
        f"raw_human_prob_max={best['metrics'].get('raw_human_prob_max', 0.0):.4f} "
        f"raw_bot_prob_min={best['metrics'].get('raw_bot_prob_min', 1.0):.4f}"
        f"{holdout_remap_line} "
        f"calibrated_range=[{float(np.min(test_calibrated)):.4f}, {float(np.max(test_calibrated)):.4f}] "
        f"post_remap_range=[{float(np.min(test_mid)):.4f}, {float(np.max(test_mid)):.4f}] "
        f"final_range=[{float(np.min(test_final)):.4f}, {float(np.max(test_final)):.4f}] "
        f"raw_separation_q90_q10={raw_separation:.4f} "
        f"human_prob_max={best['metrics'].get('human_prob_max', 0.0):.4f} "
        f"bot_prob_min={best['metrics'].get('bot_prob_min', 0.0):.4f}"
    )
    print("  Holdout reward breakdown: " + _reward_breakdown(best["metrics"]))

    framework_models = base_names + chunk_names
    metadata: Dict[str, Any] = {
        "framework": "stacked-v3:" + "+".join(framework_models),
        "task_type": "supervised-benchmark-stacked-v3",
        **_repo_metadata(),
        "feature_schema_hash": _feature_schema_hash(feature_names),
        "selected_feature_count": int(len(feature_indices)),
        "total_feature_count": int(len(feature_names)),
        "benchmark_paths": [str(path) for path in benchmark_paths],
        "benchmark_rows": float(len(examples)),
        "benchmark_positive_rows": float(labels_total.get(1, 0)),
        "benchmark_negative_rows": float(labels_total.get(0, 0)),
        "train_rows": float(len(train_examples)),
        "test_rows": float(len(test_examples)),
        "n_folds": float(n_folds),
        "oof_pr_auc": float(oof_ap),
        "fold_pr_auc_mean": float(np.mean(fold_aps)),
        "base_learners": base_names,
        "chunk_learners": chunk_names,
        "sequence_enabled": bool(sequence_enabled),
        "sequence_config": (
            chunk_models[0].config.to_dict()
            if (sequence_enabled and chunk_models)
            else {}
        ),
        "human_weight_multiplier": float(args.human_weight_multiplier),
        "meta_c": float(args.meta_c),
        "meta_hard_bot_weight": float(args.meta_hard_bot_weight),
        "meta_hard_bot_gamma": float(args.meta_hard_bot_gamma),
        "target_fpr": float(args.target_fpr),
        "max_validator_fpr": float(args.max_validator_fpr),
        "calibration_objective": cal_objective,
        "stack_calibrator": str(calibrator_mode),
        "isotonic_calibration_blend": float(args.isotonic_calibration_blend),
        "calibration_fraction": float(args.calibration_fraction),
        "calibration_rows": float(len(cal_examples)),
        "miner_visible_payload": bool(miner_visible),
        "train_latest_days": float(args.train_latest_days),
        "robust_features_only": bool(args.robust_features_only),
        "robust_feature_count": float(
            len(feature_names) if args.robust_features_only else 0
        ),
        "no_score_remap": bool(args.no_score_remap),
        "no_score_logit_tune": bool(args.no_score_logit_tune),
        "score_remap": dict(score_remap) if score_remap else {},
        "calibration_metrics": {**cal_metrics, **logit_cal_metrics},
        "holdout_test_metrics": best["metrics"],
        "raw_separation_q90_q10": float(raw_separation),
        "score_logit_bias": float(score_logit_bias),
        "score_logit_temperature": float(score_logit_temperature),
        "model_weights": [1.0],
        "ensemble_combiner": (
            f"stacking_logreg+{calibrator_mode}+score_remap+score_logit"
            if (use_score_remap and score_remap)
            else f"stacking_logreg+{calibrator_mode}+score_logit"
        ),
        **split_info,
    }

    output_path = Path(args.output)
    artifact_identity = artifact_model_identity(output_path)
    metadata.update(
        {
            "model_name": artifact_identity["model_name"],
            "model_version": os.getenv(
                "POKER44_MODEL_VERSION",
                artifact_identity["model_version"],
            ),
            "artifact_filename": artifact_identity["artifact_filename"],
        }
    )
    if sequence_enabled and chunk_models and isinstance(metadata.get("sequence_config"), dict):
        metadata["sequence_config"]["n_epochs"] = int(args.sequence_epochs)
        metadata["sequence_config"]["learning_rate"] = float(args.sequence_learning_rate)
        schedule = str(args.sequence_learning_rate_schedule or "").strip()
        if schedule:
            metadata["sequence_config"]["learning_rate_schedule"] = schedule
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": [stacked],
            "model_weights": [1.0],
            "feature_names": feature_names,
            "metadata": metadata,
            "calibrator": None,
        },
        output_path,
    )
    print(f"Saved stacked model to {output_path}")

    # Sanity round-trip: load via the canonical inference path and rescore.
    loaded = Poker44Model(output_path)
    test_chunks = [example["chunk"] for example in test_examples]
    rt_scores = loaded.predict_chunk_scores(test_chunks)
    # raw_scores here = pre-calibration stacked score (consistent with the
    # holdout log). The inference debug 'raw_scores' is post stack-calibration
    # because the stack calibrator lives inside the saved StackedEnsemble, so we
    # reuse the training-side pre-calibration score (identical model, same order).
    rt_metrics = _enrich_metrics(
        y_test.tolist(),
        rt_scores,
        raw_scores=test_precal.tolist(),
    )
    rt_metrics["latency_per_chunk_ms"] = loaded.benchmark_latency(
        [example["chunk"] for example in test_examples[:4]]
    )["latency_per_chunk_ms"]
    print("Round-trip metrics:")
    for key in (
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "mcc_at_0_5",
        "validator_reward",
        "validator_fpr",
        "validator_bot_recall",
        "validator_ap_score",
        "validator_base_score",
        "recall_at_0_5",
        "precision_at_0_5",
        "fpr_at_0_5",
        "human_prob_max",
        "bot_prob_min",
        "score_gap_at_0_5",
        "raw_human_prob_max",
        "raw_bot_prob_min",
        "raw_score_gap_at_0_5",
        "latency_per_chunk_ms",
    ):
        if key in rt_metrics:
            print(f"  {key}={float(rt_metrics[key]):.6f}")
    print("  Reward breakdown: " + _reward_breakdown(rt_metrics))

    if args.per_source_date:
        source_dates = sorted(
            {
                str(example.get("source_date", "")).strip()
                for example in test_examples
            }
        )
        for source_date in source_dates:
            idx = [
                i
                for i, example in enumerate(test_examples)
                if str(example.get("source_date", "")).strip() == source_date
            ]
            if not idx:
                continue
            sub_scores = [rt_scores[i] for i in idx]
            sub_labels = [int(test_examples[i]["label"]) for i in idx]
            if len(set(sub_labels)) < 2:
                continue
            sub_metrics = _enrich_metrics(sub_labels, sub_scores)
            print(
                f"  [{source_date}] rows={len(idx)} "
                f"reward={sub_metrics['validator_reward']:.4f} "
                f"pr_auc={sub_metrics['pr_auc']:.4f} "
                f"fpr={sub_metrics['validator_fpr']:.4f} "
                f"recall={sub_metrics['validator_bot_recall']:.4f}"
            )

    return {"metadata": metadata, "metrics": rt_metrics}


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
