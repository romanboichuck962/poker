from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from poker44.score.scoring import reward
from poker44_ml.inference import Poker44Model
from training.build_dataset import (
    load_benchmark_examples,
    resolve_benchmark_paths,
)

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        matthews_corrcoef,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover
    ExtraTreesClassifier = None
    HistGradientBoostingClassifier = None
    RandomForestClassifier = None
    LogisticRegression = None
    average_precision_score = None
    brier_score_loss = None
    log_loss = None
    matthews_corrcoef = None
    roc_auc_score = None
    train_test_split = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a clean Poker44 benchmark model.")
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "models" / "poker44_clean_restart.joblib"))
    parser.add_argument("--holdout-latest-days", type=int, default=2)
    parser.add_argument("--holdout-source-dates", type=str, default=None)
    parser.add_argument(
        "--exclude-train-source-dates",
        type=str,
        default=None,
        help=(
            "Comma-separated sourceDate values to remove from the training side only. "
            "Useful for testing whether one date causes negative transfer to a holdout date."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--calibration-size", type=float, default=0.2)
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=None,
        help="Random seed for the calibration split. Defaults to --seed.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=("logistic", "threshold_logit", "none"),
        default="logistic",
    )
    parser.add_argument(
        "--threshold-calibration-temperature",
        type=float,
        default=0.08,
        help="Sigmoid temperature for threshold_logit calibration. Smaller is sharper.",
    )
    parser.add_argument(
        "--threshold-calibration-human-quantile",
        type=float,
        default=1.0,
        help="Human score quantile used to set the threshold_logit midpoint.",
    )
    parser.add_argument(
        "--threshold-calibration-bot-quantile",
        type=float,
        default=0.0,
        help="Bot score quantile used to set the threshold_logit midpoint.",
    )
    parser.add_argument(
        "--threshold-calibration-aggregation",
        choices=("quantile", "trimmed_mean", "tail_mean"),
        default="quantile",
        help=(
            "quantile uses the quantile point as anchor; trimmed_mean averages "
            "humans <= human quantile and bots >= bot quantile; tail_mean "
            "averages humans >= human quantile and bots <= bot quantile."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=700)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--extra-trees-estimators", type=int, default=None)
    parser.add_argument("--random-forest-estimators", type=int, default=None)
    parser.add_argument("--hist-gradient-estimators", type=int, default=None)
    parser.add_argument("--extra-trees-max-depth", type=int, default=None)
    parser.add_argument("--random-forest-max-depth", type=int, default=None)
    parser.add_argument("--hist-gradient-max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--extra-trees-weight", type=float, default=0.45)
    parser.add_argument("--random-forest-weight", type=float, default=0.25)
    parser.add_argument("--hist-gradient-weight", type=float, default=0.30)
    parser.add_argument(
        "--bot-sample-weight",
        type=float,
        default=1.0,
        help="Training sample weight multiplier for labeled bot chunks.",
    )
    parser.add_argument(
        "--human-sample-weight",
        type=float,
        default=1.0,
        help="Training sample weight multiplier for labeled human chunks.",
    )
    parser.add_argument(
        "--score-logit-bias",
        type=float,
        default=0.0,
        help="Add this value to calibrated score logits before human guard. Negative lowers scores.",
    )
    parser.add_argument(
        "--score-logit-temperature",
        type=float,
        default=1.0,
        help="Divide score logits by this positive value after adding score-logit-bias.",
    )
    return parser.parse_args()


def _repo_metadata() -> dict[str, str]:
    def run(args: list[str]) -> str:
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


def _feature_schema_hash(feature_names: list[str]) -> str:
    payload = "\n".join(feature_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_matrix(examples: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    return [
        [float(example["features"].get(name, 0.0)) for name in feature_names]
        for example in examples
    ]


def _split_benchmark(
    examples: list[dict[str, Any]],
    *,
    holdout_source_dates: str | None,
    exclude_train_source_dates: str | None,
    holdout_latest_days: int,
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dates = sorted(
        {
            str(example.get("source_date", "")).strip()
            for example in examples
            if str(example.get("source_date", "")).strip()
        }
    )
    requested = [item.strip() for item in str(holdout_source_dates or "").split(",") if item.strip()]
    excluded_train_dates = [
        item.strip()
        for item in str(exclude_train_source_dates or "").split(",")
        if item.strip()
    ]
    excluded_train_set = set(excluded_train_dates)

    def apply_train_exclusion(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not excluded_train_set:
            return train_rows
        return [
            example
            for example in train_rows
            if str(example.get("source_date", "")).strip() not in excluded_train_set
        ]

    holdout_dates = requested or dates[-max(1, int(holdout_latest_days)) :]
    if holdout_dates:
        holdout_set = set(holdout_dates)
        train = [example for example in examples if str(example.get("source_date", "")).strip() not in holdout_set]
        test = [example for example in examples if str(example.get("source_date", "")).strip() in holdout_set]
        train = apply_train_exclusion(train)
        if train and test and len({int(example["label"]) for example in test}) >= 2:
            if len({int(example["label"]) for example in train}) < 2:
                raise RuntimeError(
                    "Training set must contain both labels after applying "
                    f"--exclude-train-source-dates={exclude_train_source_dates!r}"
                )
            return train, test, {
                "split_strategy": "holdout_source_dates",
                "holdout_source_dates": holdout_dates,
                "excluded_train_source_dates": excluded_train_dates,
                "train_source_dates": [
                    date
                    for date in dates
                    if date not in holdout_set and date not in excluded_train_set
                ],
            }

    labels = [int(example["label"]) for example in examples]
    train, test = train_test_split(
        examples,
        test_size=min(max(float(test_size), 0.05), 0.45),
        random_state=seed,
        stratify=labels,
    )
    train = apply_train_exclusion(train)
    if len({int(example["label"]) for example in train}) < 2:
        raise RuntimeError(
            "Training set must contain both labels after applying "
            f"--exclude-train-source-dates={exclude_train_source_dates!r}"
        )
    return train, test, {
        "split_strategy": "random_stratified",
        "holdout_source_dates": [],
        "excluded_train_source_dates": excluded_train_dates,
        "train_source_dates": [date for date in dates if date not in excluded_train_set],
    }


def _split_calibration(
    examples: list[dict[str, Any]],
    *,
    calibration_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [int(example["label"]) for example in examples]
    if len(examples) < 30 or len(set(labels)) < 2 or calibration_size <= 0:
        return examples, []
    fit, calibration = train_test_split(
        examples,
        test_size=min(max(float(calibration_size), 0.05), 0.4),
        random_state=seed,
        stratify=labels,
    )
    return fit, calibration


def _model_scores(models: list[object], weights: list[float], rows: list[list[float]]) -> list[float]:
    per_model: list[list[float]] = []
    for model in models:
        probabilities = model.predict_proba(rows)
        per_model.append([float(row[1]) for row in probabilities])
    clean_weights = [max(0.0, float(weight)) for weight in weights[: len(per_model)]]
    if len(clean_weights) != len(per_model) or sum(clean_weights) <= 0.0:
        clean_weights = [1.0 for _ in per_model]
    total = sum(clean_weights)
    return [
        float(sum(weight * scores[index] for weight, scores in zip(clean_weights, per_model)) / total)
        for index in range(len(rows))
    ]


def _fit_logistic_calibrator(labels: list[int], scores: list[float]) -> object | None:
    if len(set(int(label) for label in labels)) < 2:
        return None
    calibrator = LogisticRegression(solver="lbfgs", random_state=42)
    calibrator.fit([[float(score)] for score in scores], labels)
    return calibrator


def _fit_threshold_logit_calibrator(
    labels: list[int],
    scores: list[float],
    *,
    temperature: float,
    human_quantile: float,
    bot_quantile: float,
    aggregation: str,
) -> dict[str, float] | None:
    humans = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    bots = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    if not humans or not bots:
        return None
    human_q = min(max(float(human_quantile), 0.0), 1.0)
    bot_q = min(max(float(bot_quantile), 0.0), 1.0)
    human_values = np.asarray(humans, dtype=float)
    bot_values = np.asarray(bots, dtype=float)
    human_cutoff = float(np.quantile(human_values, human_q))
    bot_cutoff = float(np.quantile(bot_values, bot_q))
    if aggregation == "trimmed_mean":
        trimmed_humans = human_values[human_values <= human_cutoff]
        trimmed_bots = bot_values[bot_values >= bot_cutoff]
        if len(trimmed_humans) == 0 or len(trimmed_bots) == 0:
            return None
        human_anchor = float(np.mean(trimmed_humans))
        bot_anchor = float(np.mean(trimmed_bots))
    elif aggregation == "tail_mean":
        tail_humans = human_values[human_values >= human_cutoff]
        tail_bots = bot_values[bot_values <= bot_cutoff]
        if len(tail_humans) == 0 or len(tail_bots) == 0:
            return None
        human_anchor = float(np.mean(tail_humans))
        bot_anchor = float(np.mean(tail_bots))
    else:
        human_anchor = human_cutoff
        bot_anchor = bot_cutoff
    threshold = 0.5 * (human_anchor + bot_anchor)
    return {
        "kind": "threshold_logit_v1",
        "threshold": threshold,
        "temperature": max(float(temperature), 1e-6),
        "human_anchor": human_anchor,
        "bot_anchor": bot_anchor,
        "human_cutoff": human_cutoff,
        "bot_cutoff": bot_cutoff,
        "human_quantile": human_q,
        "bot_quantile": bot_q,
        "aggregation": str(aggregation),
    }


def _apply_calibrator(calibrator: object | None, scores: list[float]) -> list[float]:
    if calibrator is None:
        return [max(0.0, min(1.0, float(score))) for score in scores]
    if isinstance(calibrator, dict) and calibrator.get("kind") == "threshold_logit_v1":
        threshold = float(calibrator.get("threshold", 0.5))
        temperature = max(float(calibrator.get("temperature", 0.08)), 1e-6)
        return [
            float(1.0 / (1.0 + np.exp(-((float(score) - threshold) / temperature))))
            for score in scores
        ]
    probabilities = calibrator.predict_proba([[float(score)] for score in scores])
    return [max(0.0, min(1.0, float(row[1]))) for row in probabilities]


def _apply_score_logit(scores: list[float], *, bias: float, temperature: float) -> list[float]:
    if not scores:
        return []
    temperature = max(float(temperature), 1e-6)
    if abs(float(bias)) < 1e-12 and abs(temperature - 1.0) < 1e-12:
        return [max(0.0, min(1.0, float(score))) for score in scores]
    output: list[float] = []
    for score in scores:
        value = max(1e-6, min(1.0 - 1e-6, float(score)))
        logit = np.log(value / (1.0 - value))
        output.append(float(1.0 / (1.0 + np.exp(-((logit + float(bias)) / temperature)))))
    return output


def _binary_counts(labels: list[int], scores: list[float]) -> dict[str, float]:
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


def _validator_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
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


def _enrich_probability_metrics(
    labels: list[int],
    probabilities: list[float],
    raw_probabilities: list[float] | None = None,
) -> dict[str, float]:
    safe = [max(1e-6, min(1.0 - 1e-6, float(value))) for value in probabilities]
    metrics: dict[str, float] = {}
    if len(set(labels)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, safe))
        metrics["pr_auc"] = float(average_precision_score(labels, safe))
        metrics["mcc_at_0_5"] = float(matthews_corrcoef(labels, [value >= 0.5 for value in safe]))
    metrics["log_loss"] = float(log_loss(labels, safe, labels=[0, 1]))
    metrics["brier_score"] = float(brier_score_loss(labels, safe))
    metrics.update(_binary_counts(labels, safe))
    metrics.update(_validator_metrics(labels, safe))

    humans = [score for label, score in zip(labels, safe) if label == 0]
    bots = [score for label, score in zip(labels, safe) if label == 1]
    metrics["prob_min"] = float(min(safe)) if safe else 0.0
    metrics["prob_max"] = float(max(safe)) if safe else 0.0
    metrics["prob_mean"] = float(sum(safe) / max(len(safe), 1))
    metrics["human_prob_max"] = float(max(humans)) if humans else 0.0
    metrics["bot_prob_min"] = float(min(bots)) if bots else 1.0
    metrics["human_clearance_to_0_5"] = float(0.5 - metrics["human_prob_max"])
    metrics["bot_clearance_to_0_5"] = float(metrics["bot_prob_min"] - 0.5)
    metrics["score_gap_at_0_5"] = float(metrics["bot_prob_min"] - metrics["human_prob_max"])
    metrics["threshold_margin_at_0_5"] = float(
        min(metrics["human_clearance_to_0_5"], metrics["bot_clearance_to_0_5"])
    )
    if raw_probabilities is not None:
        raw = [max(0.0, min(1.0, float(value))) for value in raw_probabilities]
        raw_humans = [score for label, score in zip(labels, raw) if label == 0]
        raw_bots = [score for label, score in zip(labels, raw) if label == 1]
        metrics["raw_human_prob_max"] = float(max(raw_humans)) if raw_humans else 0.0
        metrics["raw_bot_prob_min"] = float(min(raw_bots)) if raw_bots else 1.0
        metrics["raw_score_gap_at_0_5"] = metrics["raw_bot_prob_min"] - metrics["raw_human_prob_max"]
    return metrics


def train_model(args: argparse.Namespace) -> tuple[list[object], list[str], dict[str, float], dict[str, Any]]:
    required = [
        joblib,
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
        LogisticRegression,
        train_test_split,
    ]
    if any(item is None for item in required):
        raise RuntimeError("joblib and scikit-learn are required for training.")

    benchmark_paths = resolve_benchmark_paths(args.benchmark_path)
    benchmark_examples = load_benchmark_examples(benchmark_paths)
    labels = [int(example["label"]) for example in benchmark_examples]
    label_counts = Counter(labels)
    if len(label_counts) != 2:
        raise RuntimeError(f"Benchmark must contain both labels, got {dict(label_counts)}")

    train_examples, test_examples, split_info = _split_benchmark(
        benchmark_examples,
        holdout_source_dates=args.holdout_source_dates,
        exclude_train_source_dates=args.exclude_train_source_dates,
        holdout_latest_days=args.holdout_latest_days,
        test_size=args.test_size,
        seed=args.seed,
    )
    fit_examples, calibration_examples = _split_calibration(
        train_examples,
        calibration_size=args.calibration_size,
        seed=int(args.calibration_seed if args.calibration_seed is not None else args.seed),
    )

    benchmark_feature_rows = [example["features"] for example in benchmark_examples]
    feature_names = sorted(benchmark_feature_rows[0])
    benchmark_chunk_sizes = sorted({len(example["chunk"]) for example in benchmark_examples})

    X_fit = _build_matrix(fit_examples, feature_names)
    y_fit = [int(example["label"]) for example in fit_examples]
    bot_sample_weight = max(0.0, float(args.bot_sample_weight))
    human_sample_weight = max(0.0, float(args.human_sample_weight))
    if bot_sample_weight <= 0.0 and human_sample_weight <= 0.0:
        bot_sample_weight = 1.0
        human_sample_weight = 1.0
    weights = [
        bot_sample_weight if label == 1 else human_sample_weight
        for label in y_fit
    ]

    X_cal = _build_matrix(calibration_examples, feature_names)
    y_cal = [int(example["label"]) for example in calibration_examples]
    X_test = _build_matrix(test_examples, feature_names)
    y_test = [int(example["label"]) for example in test_examples]

    extra_trees_estimators = int(args.extra_trees_estimators or args.n_estimators)
    random_forest_estimators = int(args.random_forest_estimators or args.n_estimators)
    hist_gradient_estimators = int(args.hist_gradient_estimators or args.n_estimators)
    extra_trees_max_depth = int(args.extra_trees_max_depth or args.max_depth)
    random_forest_max_depth = int(args.random_forest_max_depth or args.max_depth)
    hist_gradient_max_depth = int(args.hist_gradient_max_depth or args.max_depth)

    models: list[object] = [
        ExtraTreesClassifier(
            n_estimators=extra_trees_estimators,
            max_depth=extra_trees_max_depth,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=random_forest_estimators,
            max_depth=random_forest_max_depth,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=args.seed + 7,
            n_jobs=1,
        ),
        HistGradientBoostingClassifier(
            learning_rate=float(args.learning_rate),
            max_iter=hist_gradient_estimators,
            max_depth=hist_gradient_max_depth,
            min_samples_leaf=2,
            random_state=args.seed + 13,
        ),
    ]
    for model in models:
        model.fit(X_fit, y_fit, sample_weight=weights)

    model_weights = [
        float(args.extra_trees_weight),
        float(args.random_forest_weight),
        float(args.hist_gradient_weight),
    ]
    raw_cal = _model_scores(models, model_weights, X_cal) if X_cal else []
    calibrator = None
    if raw_cal and len(set(y_cal)) >= 2:
        if args.calibration_method == "logistic":
            calibrator = _fit_logistic_calibrator(y_cal, raw_cal)
        elif args.calibration_method == "threshold_logit":
            calibrator = _fit_threshold_logit_calibrator(
                y_cal,
                raw_cal,
                temperature=args.threshold_calibration_temperature,
                human_quantile=args.threshold_calibration_human_quantile,
                bot_quantile=args.threshold_calibration_bot_quantile,
                aggregation=args.threshold_calibration_aggregation,
            )

    raw_test = _model_scores(models, model_weights, X_test)
    post_test_calibrator = _fit_threshold_logit_calibrator(
        y_test,
        raw_test,
        temperature=args.threshold_calibration_temperature,
        human_quantile=args.threshold_calibration_human_quantile,
        bot_quantile=args.threshold_calibration_bot_quantile,
        aggregation=args.threshold_calibration_aggregation,
    )
    calibrated_test = _apply_calibrator(calibrator, raw_test)
    logit_test = _apply_score_logit(
        calibrated_test,
        bias=args.score_logit_bias,
        temperature=args.score_logit_temperature,
    )
    metrics = _enrich_probability_metrics(y_test, logit_test, raw_probabilities=raw_test)

    metadata: dict[str, Any] = {
        "framework": "clean-restart:ExtraTrees+RandomForest+HistGradientBoosting",
        "task_type": "supervised-benchmark-clean-restart",
        **_repo_metadata(),
        "feature_schema_hash": _feature_schema_hash(feature_names),
        "benchmark_paths": [str(path) for path in benchmark_paths],
        "benchmark_rows": float(len(benchmark_examples)),
        "benchmark_positive_rows": float(label_counts.get(1, 0)),
        "benchmark_negative_rows": float(label_counts.get(0, 0)),
        "benchmark_chunk_sizes": ",".join(str(size) for size in benchmark_chunk_sizes),
        "train_rows": float(len(X_fit)),
        "pre_calibration_train_rows": float(len(train_examples)),
        "excluded_train_source_dates": split_info.get("excluded_train_source_dates", []),
        "train_source_dates": split_info.get("train_source_dates", []),
        "calibration_rows": float(len(X_cal)),
        "calibration_method": str(args.calibration_method),
        "calibration_seed": float(
            args.calibration_seed if args.calibration_seed is not None else args.seed
        ),
        "threshold_calibration_temperature": float(args.threshold_calibration_temperature),
        "threshold_calibration_human_quantile": float(args.threshold_calibration_human_quantile),
        "threshold_calibration_bot_quantile": float(args.threshold_calibration_bot_quantile),
        "threshold_calibration_aggregation": str(args.threshold_calibration_aggregation),
        "calibrator": calibrator if isinstance(calibrator, dict) else {},
        "post_test_calibrator": post_test_calibrator if isinstance(post_test_calibrator, dict) else {},
        "test_rows": float(len(X_test)),
        "score_logit_bias": float(args.score_logit_bias),
        "score_logit_temperature": float(args.score_logit_temperature),
        "bot_sample_weight": float(bot_sample_weight),
        "human_sample_weight": float(human_sample_weight),
        "model_weights": model_weights,
        "n_estimators": float(args.n_estimators),
        "max_depth": float(args.max_depth),
        "extra_trees_estimators": float(extra_trees_estimators),
        "random_forest_estimators": float(random_forest_estimators),
        "hist_gradient_estimators": float(hist_gradient_estimators),
        "extra_trees_max_depth": float(extra_trees_max_depth),
        "random_forest_max_depth": float(random_forest_max_depth),
        "hist_gradient_max_depth": float(hist_gradient_max_depth),
        "learning_rate": float(args.learning_rate),
        **split_info,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": models,
            "model_weights": model_weights,
            "feature_names": feature_names,
            "metadata": metadata,
            "calibrator": calibrator,
        },
        output_path,
    )

    loaded = Poker44Model(output_path)
    latency = loaded.benchmark_latency([example["chunk"] for example in test_examples[:4]])
    metrics["latency_per_chunk_ms"] = latency["latency_per_chunk_ms"]
    return models, feature_names, metrics, metadata


def main() -> None:
    args = parse_args()
    _, feature_names, metrics, metadata = train_model(args)
    print(f"Saved model to {args.output}")
    print(f"Feature count: {len(feature_names)}")
    print(
        "Selected config: "
        f"framework={metadata.get('framework')} "
        f"split_strategy={metadata.get('split_strategy')} "
        f"holdout_dates={metadata.get('holdout_source_dates')} "
        f"excluded_train_dates={metadata.get('excluded_train_source_dates')} "
        f"benchmark_rows={metadata.get('benchmark_rows')}"
    )
    calibrator = metadata.get("calibrator")
    if isinstance(calibrator, dict) and calibrator.get("kind") == "threshold_logit_v1":
        print(
            "Calibration threshold: "
            f"kind={calibrator.get('kind')} "
            f"threshold={float(calibrator.get('threshold', 0.0)):.6f} "
            f"temperature={float(calibrator.get('temperature', 0.0)):.6f} "
            f"human_anchor={float(calibrator.get('human_anchor', 0.0)):.6f} "
            f"bot_anchor={float(calibrator.get('bot_anchor', 0.0)):.6f} "
            f"human_cutoff={float(calibrator.get('human_cutoff', 0.0)):.6f} "
            f"bot_cutoff={float(calibrator.get('bot_cutoff', 0.0)):.6f} "
            f"human_quantile={float(calibrator.get('human_quantile', 0.0)):.6f} "
            f"bot_quantile={float(calibrator.get('bot_quantile', 0.0)):.6f} "
            f"aggregation={calibrator.get('aggregation', 'quantile')}"
        )
    post_test_calibrator = metadata.get("post_test_calibrator")
    if isinstance(post_test_calibrator, dict) and post_test_calibrator.get("kind") == "threshold_logit_v1":
        print(
            "Post-test threshold (evaluation only): "
            f"kind={post_test_calibrator.get('kind')} "
            f"threshold={float(post_test_calibrator.get('threshold', 0.0)):.6f} "
            f"temperature={float(post_test_calibrator.get('temperature', 0.0)):.6f} "
            f"human_anchor={float(post_test_calibrator.get('human_anchor', 0.0)):.6f} "
            f"bot_anchor={float(post_test_calibrator.get('bot_anchor', 0.0)):.6f} "
            f"human_cutoff={float(post_test_calibrator.get('human_cutoff', 0.0)):.6f} "
            f"bot_cutoff={float(post_test_calibrator.get('bot_cutoff', 0.0)):.6f} "
            f"human_quantile={float(post_test_calibrator.get('human_quantile', 0.0)):.6f} "
            f"bot_quantile={float(post_test_calibrator.get('bot_quantile', 0.0)):.6f} "
            f"aggregation={post_test_calibrator.get('aggregation', 'quantile')}"
        )
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
        "validator_human_safety_penalty",
        "validator_base_score",
        "recall_at_0_5",
        "precision_at_0_5",
        "fpr_at_0_5",
        "tp",
        "fp",
        "tn",
        "fn",
        "prob_min",
        "prob_max",
        "human_prob_max",
        "bot_prob_min",
        "human_clearance_to_0_5",
        "bot_clearance_to_0_5",
        "score_gap_at_0_5",
        "threshold_margin_at_0_5",
        "prob_mean",
        "raw_human_prob_max",
        "raw_bot_prob_min",
        "raw_score_gap_at_0_5",
        "latency_per_chunk_ms",
    ):
        if key in metrics:
            print(f"{key}={float(metrics[key]):.6f}")


if __name__ == "__main__":
    main()
