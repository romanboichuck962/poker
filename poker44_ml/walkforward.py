"""The ship gate: train on the past, test the next unseen date. Repeat.

    python -m poker44_ml.walkforward --days 5

This produces the ONE number that decides whether an artifact ships. Nothing
else — not a random-split OOF, not a single holdout, not benchmark AP — is
allowed to make that call.

Why it has to be this: chunks from one release date share a generation batch, so
a random split leaks and flatters the model. Walk-forward is the only protocol
that answers the question you actually care about: "trained on everything up to
today, how does this do on tomorrow's data it has never seen?"

Each fold is scored through the real ``Detector`` and the real upstream reward,
in request-sized batches, because the score policy is batch-relative and a
pooled evaluation would misstate it.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import numpy as np

from poker44_ml.data import DEFAULT_DIR, load_corpus
from poker44_ml.reward import format_metrics, reward_metrics
from poker44_ml.upstream import check as check_upstream

MIN_TRAIN_DATES = 5


def run(
    *,
    data_dir: Path = DEFAULT_DIR,
    days: int = 3,
    min_train_dates: int = MIN_TRAIN_DATES,
    batch_chunks: int = 100,
    **train_kwargs: Any,
) -> Dict[str, Any]:
    """Walk forward over the last ``days`` release dates."""
    from poker44_ml.inference import Detector
    from poker44_ml.train import train as fit

    check_upstream(strict=True)
    corpus = load_corpus(data_dir)
    dates = corpus.dates
    if len(dates) < min_train_dates + 1:
        raise RuntimeError(
            f"need at least {min_train_dates + 1} release dates, have {len(dates)}. "
            "Run: python -m poker44_ml.data --fetch"
        )

    targets = [d for d in dates[-days:] if dates.index(d) >= min_train_dates]
    if not targets:
        targets = dates[min_train_dates:][-days:]
    print(f"walk-forward over {len(targets)} date(s): {targets}\n")

    folds: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as scratch:
        for position, test_date in enumerate(targets, start=1):
            index = dates.index(test_date)
            train_dates = dates[:index]
            print(f"[{position}/{len(targets)}] test={test_date} "
                  f"train={len(train_dates)} prior date(s)")

            # Leakage guard: the test date must never be in the training set.
            assert test_date not in train_dates, "test date leaked into training"

            artifact = Path(scratch) / f"wf_{test_date}.joblib"
            fit(
                data_dir=data_dir,
                out_path=artifact,
                holdout_dates=[test_date],
                **train_kwargs,
            )

            detector = Detector.load(artifact)
            test = corpus.by_date(include=[test_date])
            y = np.asarray(test.labels, dtype=int)

            rewards: List[float] = []
            pooled: List[float] = []
            for start in range(0, len(test), batch_chunks):
                group = list(range(start, min(start + batch_chunks, len(test))))
                scores = detector.predict_chunk_scores([test.chunks[i] for i in group])
                pooled.extend(scores)
                rewards.append(reward_metrics(y[group], scores)["reward"])

            metrics = reward_metrics(y, pooled)
            drift = detector.drift_report(test.chunks)
            fold = {
                "date": test_date,
                "n_train_dates": len(train_dates),
                "n_test_chunks": len(test),
                "reward": metrics["reward"],
                "reward_min_batch": float(np.min(rewards)),
                "ap": metrics.get("ap_score", 0.0),
                "bot_recall": metrics.get("bot_recall", 0.0),
                "tsq": metrics.get("threshold_sanity_quality", 0.0),
                "fpr_at_0_5": metrics.get("hard_fpr", 0.0),
                "positive_fraction": detector.positive_fraction,
                "drift_median_shift_iqr": drift.get("median_shift_iqr"),
            }
            folds.append(fold)
            print(f"    {format_metrics(metrics)}")
            if fold["tsq"] <= 0.0:
                print("    *** ZERO GATE TRIPPED — this fold pays nothing ***")
            print()

    summary = {
        "folds": folds,
        "mean_reward": mean(f["reward"] for f in folds),
        "min_reward": min(f["reward"] for f in folds),
        "mean_ap": mean(f["ap"] for f in folds),
        "zero_gate_folds": sum(1 for f in folds if f["tsq"] <= 0.0),
    }

    print("=" * 78)
    print(f"{'date':<12} {'reward':>8} {'min/req':>8} {'AP':>7} {'recall':>7} "
          f"{'tsq':>5} {'fpr':>6} {'frac':>6}")
    print("-" * 78)
    for fold in folds:
        print(
            f"{fold['date']:<12} {fold['reward']:>8.4f} {fold['reward_min_batch']:>8.4f} "
            f"{fold['ap']:>7.4f} {fold['bot_recall']:>7.4f} {fold['tsq']:>5.2f} "
            f"{fold['fpr_at_0_5']:>6.3f} {fold['positive_fraction']:>6.3f}"
        )
    print("-" * 78)
    print(f"MEAN REWARD {summary['mean_reward']:.4f}   "
          f"min {summary['min_reward']:.4f}   mean AP {summary['mean_ap']:.4f}")
    if summary["zero_gate_folds"]:
        print(f"*** {summary['zero_gate_folds']} fold(s) tripped the zero gate — DO NOT SHIP ***")
    print("=" * 78)
    return summary


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward validation (the ship gate)")
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--days", type=int, default=3, help="how many recent dates to test")
    parser.add_argument("--min-train-dates", type=int, default=MIN_TRAIN_DATES)
    parser.add_argument("--feature-top-k", type=int, default=150)
    parser.add_argument("--human-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", default=None, help="write the summary here")
    args = parser.parse_args()

    summary = run(
        data_dir=Path(args.dir),
        days=args.days,
        min_train_dates=args.min_train_dates,
        feature_top_k=args.feature_top_k,
        human_weight=args.human_weight,
        seed=args.seed,
    )
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"summary -> {args.json}")
    return 1 if summary["zero_gate_folds"] else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
