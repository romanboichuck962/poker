"""Train Super Poker 3 with chronological walk-forward evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

from poker44.validator.payload_view import prepare_hand_for_miner
from super_poker.dataset import Example, load_examples
from super_poker.ensemble import ProbabilityEnsemble
from super_poker.feature_policy import feature_policy_report, validator_stable_features
from super_poker.features import chunk_features
from super_poker.scoring import metrics

DEFAULT_DATA = Path("../Poker44-subnet/data/raw")
DEFAULT_ARTIFACT = Path("artifacts/super_poker_3.joblib")
LIVE_SCORE_HISTORY = Path("config/live_scores.json")
LIVE_CHUNK_RANGE = (90, 105)
LIVE_CHUNKS_PER_DATE_LABEL = 3
CALIBRATION_RELEASES = 3
CALIBRATION_SAFETY_MARGIN = 0.10
XGBOOST_SEEDS = (44, 144, 244)
ENSEMBLE_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
XGBOOST_ENSEMBLE_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)
CALIBRATION_TARGET_GRID = (0.01, 0.02, 0.035)
CALIBRATION_MARGIN_GRID = (0.05, 0.10, 0.15)
XGBOOST_PROFILES = {
    "baseline": {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
    },
    "robust": {
        "n_estimators": 320,
        "learning_rate": 0.025,
        "max_depth": 3,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.4,
        "reg_lambda": 3.0,
    },
}


def make_model(seed: int = 44, profile: str = "baseline") -> XGBClassifier:
    if profile not in XGBOOST_PROFILES:
        raise ValueError(f"Unknown XGBoost profile: {profile}")
    return XGBClassifier(
        **XGBOOST_PROFILES[profile],
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
    )


def make_ensemble(seed_offset: int = 0) -> ProbabilityEnsemble:
    models: list[object] = [make_model(seed + seed_offset) for seed in XGBOOST_SEEDS]
    models.append(ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.7,
        class_weight="balanced",
        n_jobs=4,
        random_state=544 + seed_offset,
    ))
    return ProbabilityEnsemble(models, ENSEMBLE_WEIGHTS)


def make_xgboost_ensemble(
    seed_offset: int = 0, profile: str = "baseline"
) -> ProbabilityEnsemble:
    return ProbabilityEnsemble(
        [make_model(seed + seed_offset, profile) for seed in XGBOOST_SEEDS],
        XGBOOST_ENSEMBLE_WEIGHTS,
    )


def build_model(
    model_family: str, seed_offset: int = 0, xgb_profile: str = "baseline"
) -> object:
    if model_family == "xgboost":
        return make_model(44 + seed_offset, xgb_profile)
    if model_family == "ensemble":
        return make_ensemble(seed_offset)
    if model_family == "xgb_ensemble":
        return make_xgboost_ensemble(seed_offset, xgb_profile)
    raise ValueError(f"Unknown model family: {model_family}")


def fit_model(model: object, frame: pd.DataFrame, labels: np.ndarray) -> None:
    if isinstance(model, ProbabilityEnsemble):
        for learner in model.models:
            learner.fit(frame, labels)
    else:
        model.fit(frame, labels)


def prediction_stability(model: object, frame: pd.DataFrame) -> dict[str, float]:
    if not isinstance(model, ProbabilityEnsemble):
        return {}
    predictions = np.stack(
        [np.asarray(learner.predict_proba(frame), dtype=float)[:, 1] for learner in model.models],
        axis=1,
    )
    dispersion = np.std(predictions, axis=1)
    return {
        "component_std_mean": float(np.mean(dispersion)),
        "component_std_max": float(np.max(dispersion)),
    }


def matrix(examples: list[Example], columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    # Train on exactly what validators expose to miners. Raw benchmark hands
    # contain outcomes, cards, original seats, and amounts that are removed or
    # canonicalized before a DetectionSynapse is sent.
    visible_chunks = [
        [prepare_hand_for_miner(hand) for hand in example.hands]
        for example in examples
    ]
    frame = pd.DataFrame([chunk_features(chunk) for chunk in visible_chunks]).fillna(0.0)
    if columns is None:
        columns = sorted(frame.columns)
    return frame.reindex(columns=columns, fill_value=0.0).astype(float), columns


def augment_live_size_chunks(examples: list[Example], seed: int = 481) -> list[Example]:
    """Pool same-date, same-label examples into validator-sized training groups."""
    rng = random.Random(seed)
    grouped: dict[tuple[str, int], list[Example]] = {}
    for example in examples:
        grouped.setdefault((example.source_date, example.label), []).append(example)

    augmented: list[Example] = []
    for (source_date, label), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        for index in range(LIVE_CHUNKS_PER_DATE_LABEL):
            target = rng.randint(*LIVE_CHUNK_RANGE)
            order = rng.sample(group, len(group))
            hands: list[dict] = []
            used = 0
            for example in order:
                hands.extend(example.hands)
                used += 1
                if len(hands) >= target:
                    break
            if used >= 2 and len(hands) >= LIVE_CHUNK_RANGE[0]:
                augmented.append(Example(
                    hands=hands[:target],
                    label=label,
                    source_date=source_date,
                    split="augmented-live-size",
                    chunk_hash=f"aug:{source_date}:{label}:{index}",
                ))
    return augmented


def live_score_context(path: Path = LIVE_SCORE_HISTORY) -> dict | None:
    if not path.is_file():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    return records[-1] if records else None


def threshold_for_fpr(human_scores: np.ndarray, target_fpr: float) -> float:
    if not len(human_scores):
        return 0.5
    return float(np.quantile(human_scores, 1.0 - target_fpr))


def conservative_release_threshold(
    scores: np.ndarray, labels: np.ndarray, dates: np.ndarray, target_fpr: float,
    safety_margin: float = CALIBRATION_SAFETY_MARGIN,
) -> float:
    """Use the strictest human threshold across recent calibration releases."""
    thresholds = [
        threshold_for_fpr(scores[(dates == date) & (labels == 0)], target_fpr)
        for date in sorted(set(dates))
    ]
    return min(1.0 - 1e-6, max(thresholds, default=0.5) + safety_margin)


def remap_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    threshold = min(max(float(threshold), 1e-6), 1 - 1e-6)
    return np.clip(
        np.where(
            scores >= threshold,
            0.5 + 0.5 * (scores - threshold) / (1.0 - threshold),
            0.5 * scores / threshold,
        ),
        0.0,
        1.0,
    )


def select_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    *,
    target_fprs: tuple[float, ...] = CALIBRATION_TARGET_GRID,
    safety_margins: tuple[float, ...] = CALIBRATION_MARGIN_GRID,
) -> dict[str, float]:
    """Choose from a bounded, predetermined grid on past releases only."""
    candidates = []
    for target_fpr in target_fprs:
        for safety_margin in safety_margins:
            threshold = conservative_release_threshold(
                scores, labels, dates, target_fpr, safety_margin
            )
            candidate_metrics = metrics(labels, remap_threshold(scores, threshold))
            candidates.append({
                "threshold": threshold,
                "target_fpr": target_fpr,
                "safety_margin": safety_margin,
                **candidate_metrics,
            })
    eligible = [
        candidate for candidate in candidates
        if candidate["fpr"] <= 0.05 and candidate["hard_fpr"] <= 0.06
    ] or candidates
    return max(
        eligible,
        key=lambda candidate: (
            candidate["reward"],
            candidate["average_precision"],
            candidate["hard_bot_recall"],
            -candidate["hard_fpr"],
            -candidate["threshold"],
        ),
    )


def train(
    data_dir: Path,
    artifact_path: Path,
    *,
    folds: int = 5,
    target_fpr: float = 0.035,
    model_family: str = "xgboost",
    xgb_profile: str = "baseline",
) -> dict:
    examples = load_examples(data_dir)
    dates = sorted({example.source_date for example in examples})
    if len(dates) < folds + 2:
        required = folds + 2
        raise ValueError(
            f"Training requires at least {required} distinct release dates, but "
            f"found {len(dates)} under {data_dir.resolve()}. Download the full "
            "history with automation --backfill, or select the existing "
            "historical data directory."
        )
    augmented = augment_live_size_chunks(examples)
    training_examples = examples + augmented
    all_frame, all_columns = matrix(training_examples)
    # The drift policy applies to every model family. Payload-unstable features
    # (hero_*, stack_*, showdown_*, player_count, seat_utilization, hand_count)
    # are dropped before fitting: showdown_* in particular is hardcoded False by
    # the validator's payload sanitizer, so it is constant at inference time.
    columns = validator_stable_features(all_columns)
    all_frame = all_frame.reindex(columns=columns, fill_value=0.0)
    labels = np.asarray([example.label for example in training_examples], dtype=int)
    date_array = np.asarray([example.source_date for example in training_examples])
    real_count = len(examples)
    real_labels = labels[:real_count]
    real_dates = date_array[:real_count]
    oof = np.full(len(examples), np.nan)
    fold_results = []

    for fold_index, test_date in enumerate(dates[-folds:]):
        train_mask = date_array < test_date
        test_mask = real_dates == test_date
        if train_mask.sum() < 60 or len(set(labels[train_mask])) < 2:
            continue
        model = build_model(model_family, fold_index * 1000, xgb_profile)
        fit_model(model, all_frame.loc[train_mask], labels[train_mask])
        raw_test = model.predict_proba(all_frame.iloc[:real_count].loc[test_mask])[:, 1]

        earlier_dates = sorted(set(date_array[train_mask]))
        calibration_dates = earlier_dates[-CALIBRATION_RELEASES:]
        inner_fit = date_array < calibration_dates[0]
        inner_cal = np.isin(real_dates, calibration_dates)
        inner_model = build_model(
            model_family, 10000 + fold_index * 1000, xgb_profile
        )
        fit_model(inner_model, all_frame.loc[inner_fit], labels[inner_fit])
        calibration_scores = inner_model.predict_proba(all_frame.iloc[:real_count].loc[inner_cal])[:, 1]
        calibration = select_calibration(
            calibration_scores,
            real_labels[inner_cal],
            real_dates[inner_cal],
            target_fprs=tuple(sorted(set((*CALIBRATION_TARGET_GRID, target_fpr)))),
        )
        threshold = calibration["threshold"]
        mapped = remap_threshold(raw_test, threshold)
        oof[test_mask] = mapped
        fold_results.append({
            "date": test_date,
            "threshold": threshold,
            "calibration_target_fpr": calibration["target_fpr"],
            "calibration_safety_margin": calibration["safety_margin"],
            **prediction_stability(model, all_frame.iloc[:real_count].loc[test_mask]),
            **metrics(real_labels[test_mask], mapped),
        })

    valid = np.isfinite(oof)
    if not valid.any():
        raise RuntimeError("Walk-forward evaluation produced no predictions")
    overall = metrics(real_labels[valid], oof[valid])

    deployment_calibration_dates = dates[-CALIBRATION_RELEASES:]
    deployment_calibration_date = deployment_calibration_dates[-1]
    deployment_fit = date_array < deployment_calibration_dates[0]
    calibration_mask = np.isin(real_dates, deployment_calibration_dates)
    calibration_model = build_model(model_family, 20000, xgb_profile)
    fit_model(calibration_model, all_frame.loc[deployment_fit], labels[deployment_fit])
    calibration_scores = calibration_model.predict_proba(all_frame.iloc[:real_count].loc[calibration_mask])[:, 1]
    deployment_calibration = select_calibration(
        calibration_scores,
        real_labels[calibration_mask],
        real_dates[calibration_mask],
        target_fprs=tuple(sorted(set((*CALIBRATION_TARGET_GRID, target_fpr)))),
    )
    deployment_threshold = deployment_calibration["threshold"]

    final_model = build_model(model_family, 30000, xgb_profile)
    fit_model(final_model, all_frame, labels)
    metadata = {
        "model_name": "super-poker-3-xgboost-enhanced",
        "model_version": time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
        "framework": (
            "multi-seed-xgboost+extratrees-fixed-blend"
            if model_family == "ensemble"
            else (
                "multi-seed-xgboost-fixed-blend"
                if model_family == "xgb_ensemble"
                else "xgboost+chronological-calibration-search"
            )
        ),
        "feature_version": (
            "super-poker-3.v6-ensemble-drift-policy"
            if model_family == "ensemble"
            else (
                "super-poker-3.v8-xgboost-ensemble-drift-policy"
                if model_family == "xgb_ensemble"
                else "super-poker-3.v8-xgboost-drift-policy"
            )
        ),
        "example_count": len(examples),
        "augmented_example_count": len(augmented),
        "augmentation": {
            "kind": "same-date-same-label-live-size-pooling",
            "hand_count_range": list(LIVE_CHUNK_RANGE),
            "examples_per_date_label": LIVE_CHUNKS_PER_DATE_LABEL,
        },
        "release_dates": dates,
        "walk_forward_dates": [result["date"] for result in fold_results],
        "walk_forward": fold_results,
        "walk_forward_overall": overall,
        "target_fpr": target_fpr,
        "deployment_threshold": deployment_threshold,
        "deployment_calibration": {
            "target_fpr": deployment_calibration["target_fpr"],
            "safety_margin": deployment_calibration["safety_margin"],
        },
        "calibration_release": deployment_calibration_date,
        "calibration_releases": deployment_calibration_dates,
        "feature_count": len(columns),
        "feature_policy": feature_policy_report(all_columns, columns),
        "model_family": model_family,
        "xgboost_profile": xgb_profile,
        "xgboost_parameters": XGBOOST_PROFILES[xgb_profile],
        "ensemble": (
            {
                "xgboost_seeds": list(XGBOOST_SEEDS),
                "extra_trees_seed": 544,
                "weights": list(ENSEMBLE_WEIGHTS),
                "strategy": "fixed-75pct-multiseed-xgboost-25pct-extratrees",
            }
            if model_family == "ensemble"
            else (
                {
                    "xgboost_seeds": list(XGBOOST_SEEDS),
                    "weights": list(XGBOOST_ENSEMBLE_WEIGHTS),
                    "strategy": "fixed-equal-weight-multiseed-xgboost",
                }
                if model_family == "xgb_ensemble"
                else {}
            )
        ),
        "feature_schema_sha256": hashlib.sha256("\n".join(columns).encode()).hexdigest(),
        "live_score_context": live_score_context(),
        "change_reason": (
            "Competition 6 R2 scored 0.527 with stable serving. Evaluate drift filtering, "
            "multi-seed heterogeneous ensembling, bounded calibration search, and component "
            "stability against the incumbent without treating the aggregate score as a label."
        ),
        "training_data": (
            "Poker44 public benchmark only, projected through the validator-visible "
            "payload sanitizer; no validator-private labels"
        ),
    }
    artifact = {"model": final_model, "feature_names": columns, "threshold": deployment_threshold, "metadata": metadata}
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = artifact_path.with_suffix(".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(artifact_path)
    artifact_path.with_suffix(".metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fpr", type=float, default=0.035)
    parser.add_argument(
        "--model-family",
        choices=("xgboost", "ensemble", "xgb_ensemble"),
        default="xgboost",
    )
    parser.add_argument("--xgb-profile", choices=tuple(XGBOOST_PROFILES), default="baseline")
    args = parser.parse_args()
    metadata = train(
        args.data_dir,
        args.artifact,
        folds=args.folds,
        target_fpr=args.target_fpr,
        model_family=args.model_family,
        xgb_profile=args.xgb_profile,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
