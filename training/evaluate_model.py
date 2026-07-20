from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from poker44.score.scoring import reward_eval, format_reward_breakdown
from poker44_ml.inference import Poker44Model
from training.build_dataset import load_benchmark_examples, resolve_benchmark_paths
from training.train_model import _enrich_probability_metrics


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Poker44 model on released benchmark chunks."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(REPO_ROOT / "models" / "poker44_benchmark_supervised.joblib"),
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=None,
        help=(
            "Single file, comma-separated files, or directory of "
            "training_benchmark*.txt."
        ),
    )
    parser.add_argument(
        "--source-dates",
        type=str,
        default=None,
        help="Optional comma-separated sourceDate filter.",
    )
    parser.add_argument(
        "--per-source-date",
        action="store_true",
        help="Print a compact metric summary for each sourceDate.",
    )
    parser.add_argument(
        "--validator-reward-mode",
        type=str,
        choices=("live", "base", "soft"),
        default="live",
        help=(
            "Retained for back-compat. Under the live rank-first reward "
            "(subnet >=0.1.25) there is no FPR penalty to vary, so live/base/"
            "soft all return the SAME reward = 0.75*AP + 0.25*recall@FPR<=0.05."
        ),
    )
    return parser.parse_args()


def _apply_validator_reward_mode(
    metrics: dict[str, float],
    labels: list[int],
    probabilities: list[float],
    *,
    reward_mode: str,
) -> dict[str, float]:
    if reward_mode == "live":
        return metrics

    safe = [max(1e-6, min(1.0 - 1e-6, float(value))) for value in probabilities]
    eval_reward, details = reward_eval(
        np.asarray(safe, dtype=float),
        np.asarray(labels, dtype=int),
        mode=reward_mode,
    )
    out = dict(metrics)
    out["validator_reward_live"] = float(out.get("validator_reward", 0.0))
    out["validator_reward"] = float(eval_reward)
    out["validator_reward_mode"] = reward_mode
    out["validator_human_safety_penalty"] = float(
        details.get("human_safety_penalty", 0.0)
    )
    out["validator_base_score"] = float(details.get("base_score", eval_reward))
    return out


def _evaluate_examples(
    model: Poker44Model,
    examples: list[dict[str, Any]],
    *,
    reward_mode: str = "live",
) -> dict[str, float]:
    chunks = [list(example["chunk"]) for example in examples]
    labels = [int(example["label"]) for example in examples]
    probabilities = model.predict_chunk_scores(chunks)
    metrics = _enrich_probability_metrics(labels, probabilities)
    return _apply_validator_reward_mode(
        metrics,
        labels,
        probabilities,
        reward_mode=reward_mode,
    )


def _filter_examples(
    examples: list[dict[str, Any]],
    requested_dates: str | None,
) -> list[dict[str, Any]]:
    if not requested_dates:
        return examples
    allowed = {item.strip() for item in requested_dates.split(",") if item.strip()}
    return [
        example
        for example in examples
        if str(example.get("source_date", "")).strip() in allowed
    ]


def _print_metric_block(title: str, metrics: dict[str, float], rows: int) -> None:
    print(title)
    print(f"rows={rows}")
    for key in (
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "mcc_at_0_5",
        "validator_reward_mode",
        "validator_reward",
        "validator_reward_live",
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
    ):
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, str):
            print(f"{key}={value}")
        else:
            print(f"{key}={float(value):.6f}")
    if "validator_ap_score" in metrics:
        print("reward_breakdown: " + format_reward_breakdown(
            float(metrics.get("validator_ap_score", 0.0)),
            float(metrics.get("validator_bot_recall", 0.0)),
            fpr=float(metrics.get("validator_fpr", 0.0)),
            reward=float(metrics.get("validator_reward", 0.0)),
        ))


def main() -> None:
    args = parse_args()
    benchmark_paths = resolve_benchmark_paths(args.benchmark_path)
    examples = load_benchmark_examples(benchmark_paths)
    examples = _filter_examples(examples, args.source_dates)
    if not examples:
        raise RuntimeError("No benchmark examples matched the requested filters.")

    label_counts = Counter(int(example["label"]) for example in examples)
    source_dates = sorted(
        {
            str(example.get("source_date", "")).strip()
            for example in examples
            if str(example.get("source_date", "")).strip()
        }
    )

    model = Poker44Model(args.model_path)
    reward_mode = str(args.validator_reward_mode)
    metrics = _evaluate_examples(model, examples, reward_mode=reward_mode)

    print(f"Model path: {args.model_path}")
    print(f"Validator reward mode: {reward_mode}")
    print(f"Benchmark files: {len(benchmark_paths)}")
    print(f"Source dates: {source_dates}")
    print(
        f"Label counts: human={label_counts.get(0, 0)} bot={label_counts.get(1, 0)}"
    )
    _print_metric_block("Overall metrics", metrics, len(examples))

    if args.per_source_date:
        for source_date in source_dates:
            date_examples = [
                example
                for example in examples
                if str(example.get("source_date", "")).strip() == source_date
            ]
            if not date_examples:
                continue
            date_metrics = _evaluate_examples(
                model,
                date_examples,
                reward_mode=reward_mode,
            )
            _print_metric_block(f"Per-source-date metrics | {source_date}", date_metrics, len(date_examples))


if __name__ == "__main__":
    main()
