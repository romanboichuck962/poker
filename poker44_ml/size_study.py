"""Does size-invariance actually pay under a chunk-size shift?

    python -m poker44_ml.size_study

THE QUESTION. Benchmark chunks are 30-40 hands; live chunks are 80-100. Two
published miners answer this incompatibly:

  * the leader (rank-detector-b) feeds ``hand_count`` and ``hand_count_log`` to
    the model as features and pool-augments training chunks up to 90-105 hands,
    i.e. teaches the model the size axis explicitly;
  * we make the features size-INVARIANT instead -- counts become rates, and the
    duplication/diversity block is re-evaluated at a fixed 30-hand reference
    (``rep30_*``) because those statistics drift mechanically with chunk length.

Only one of those can be right, and neither has been measured at live size.

WHY WE CANNOT JUST TEST AT 90 HANDS. A genuine 90-hand chunk is 90 hands from
ONE entity. Concatenating three 35-hand chunks gives 90 hands from THREE
entities, which destroys the within-entity consistency that is the actual bot
signal -- so it measures dilution, not size.

WHAT WE CAN DO. Subsampling a chunk downward keeps exactly one entity, so it is
a valid size change. Train at k hands, test at full size, and we have a real
shift in the same DIRECTION as benchmark->live (train small, serve large) with
entity structure intact. The variant that loses least across that gap is the one
whose features actually transfer.

Variants are column subsets of one shared feature matrix, so the whole study
costs a handful of extra fits rather than a full retrain per cell.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from poker44_ml.data import DEFAULT_DIR, Corpus, load_corpus
from poker44_ml.features import chunk_features, feature_matrix
from poker44_ml.model import monotone_constraints, rank01
from poker44_ml.policy import rank_map
from poker44_ml.reward import reward_metrics
from poker44_ml.upstream import check as check_upstream

BATCH = 100

# Column subsets under test. Each is a predicate over feature names.
VARIANTS: Dict[str, Any] = {
    # everything we ship today
    "full": lambda name: True,
    # drop the size-debiased duplication block -> rely on the raw, size-biased
    # rep_* only. This is the direct test of whether rep30_* earns its place.
    "no_rep30": lambda name: not name.startswith("rep30_"),
    # hide chunk size from the model entirely
    "no_size_cols": lambda name: name not in {"hand_count", "hand_count_log"},
    # the leader's posture: no size-debiasing, size fed in raw
    "leader_style": lambda name: not name.startswith("rep30_"),
}


def subsample(chunk: Sequence[dict], k: int, rng: random.Random) -> List[dict]:
    """Keep k hands. Still one entity, so this is a valid size change."""
    if k <= 0 or len(chunk) <= k:
        return list(chunk)
    return [chunk[i] for i in sorted(rng.sample(range(len(chunk)), k))]


def _fit_probe(x: np.ndarray, y: np.ndarray, *, seed: int) -> Any:
    """One monotone LightGBM. A probe, not the shipped ensemble.

    The study compares feature sets, so the model is held fixed and kept cheap;
    absolute numbers here are not the artifact's numbers.
    """
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=48, min_data_in_leaf=25,
        feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
        objective="binary", monotone_constraints=monotone_constraints(x, y),
        monotone_constraints_method="advanced", n_jobs=-1, random_state=seed,
        verbose=-1,
    )
    model.fit(x, y, sample_weight=np.where(y == 0, 2.0, 1.0))
    return model


def _evaluate(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Score through the same rank-then-map path the miner serves."""
    raw = rank01(np.asarray(model.predict_proba(x_test))[:, 1])
    pooled: List[float] = []
    for start in range(0, len(raw), BATCH):
        group = slice(start, min(start + BATCH, len(raw)))
        pooled.extend(rank_map(raw[group], 0.10).tolist())
    return reward_metrics(y_test, pooled)


def run(
    *,
    data_dir: Path = DEFAULT_DIR,
    holdout_days: int = 3,
    sizes: Sequence[int] = (12, 18, 26),
    top_k: int = 150,
    seed: int = 42,
) -> Dict[str, Any]:
    check_upstream(strict=True)
    corpus = load_corpus(data_dir)
    dates = corpus.dates
    holdout = dates[-holdout_days:]
    train = corpus.by_date(exclude=holdout)
    test = corpus.by_date(include=holdout)
    print(f"\ntrain={len(train)} chunks / test={len(test)} chunks (holdout {holdout})")

    names = sorted(chunk_features(train.chunks[0]))
    y_train = np.asarray(train.labels, dtype=np.int64)
    y_test = np.asarray(test.labels, dtype=np.int64)

    def matrix(chunks: Sequence[Sequence[dict]]) -> np.ndarray:
        return np.nan_to_num(
            np.asarray(feature_matrix(chunks, names), dtype=np.float64)
        )

    # Test is always full size: that is what "serve larger than you trained" means.
    print("extracting test features (full size)...")
    x_test_full = matrix(test.chunks)
    median_test = int(np.median([len(c) for c in test.chunks]))

    conditions: Dict[str, np.ndarray] = {}
    rng = random.Random(seed)
    print(f"extracting train features at sizes {['full'] + list(sizes)}...")
    conditions["full"] = matrix(train.chunks)
    for size in sizes:
        conditions[str(size)] = matrix([subsample(c, size, rng) for c in train.chunks])

    rows: List[Dict[str, Any]] = []
    for variant, keep in VARIANTS.items():
        columns = [i for i, name in enumerate(names) if keep(name)]
        for condition, x_train_all in conditions.items():
            x_train = x_train_all[:, columns]
            x_test = x_test_full[:, columns]

            # Feature selection inside the variant's own column budget, on train
            # rows only, so no variant gets an information advantage.
            probe = _fit_probe(x_train, y_train, seed=seed)
            gain = probe.booster_.feature_importance(importance_type="gain")
            keep_idx = np.argsort(gain)[::-1][: min(top_k, len(columns))]
            model = _fit_probe(x_train[:, keep_idx], y_train, seed=seed)
            metrics = _evaluate(model, x_test[:, keep_idx], y_test)

            rows.append({
                "variant": variant,
                "train_hands": condition,
                "n_features": len(columns),
                "reward": metrics["reward"],
                "ap": metrics.get("ap_score", 0.0),
                "recall": metrics.get("bot_recall", 0.0),
                "tsq": metrics.get("threshold_sanity_quality", 0.0),
            })
            print(f"  {variant:<14} train@{condition:<5} -> test@{median_test}  "
                  f"reward={metrics['reward']:.4f} ap={metrics.get('ap_score', 0):.4f}")

    print()
    print("=" * 78)
    print(f"{'variant':<14} {'features':>9} " + " ".join(f"{c:>9}" for c in conditions))
    print("-" * 78)
    baseline: Dict[str, float] = {}
    for variant in VARIANTS:
        cells = {r["train_hands"]: r for r in rows if r["variant"] == variant}
        baseline[variant] = cells["full"]["ap"]
        print(f"{variant:<14} {cells['full']['n_features']:>9} "
              + " ".join(f"{cells[c]['ap']:>9.4f}" for c in conditions))
    print("-" * 78)
    print("AP above. Below: AP lost relative to that variant's own no-shift cell,")
    print("i.e. how much each feature set gives up when train and serve sizes differ.")
    print("-" * 78)
    smallest = min(str(s) for s in sizes)
    for variant in VARIANTS:
        cells = {r["train_hands"]: r for r in rows if r["variant"] == variant}
        drops = {c: baseline[variant] - cells[c]["ap"] for c in conditions if c != "full"}
        worst = max(drops.values())
        print(f"{variant:<14} " + " ".join(f"{c}:{drops[c]:+.4f}" for c in sorted(drops))
              + f"   worst {worst:+.4f}")
    print("=" * 78)

    winner = min(
        VARIANTS,
        key=lambda v: max(
            baseline[v] - r["ap"] for r in rows
            if r["variant"] == v and r["train_hands"] != "full"
        ),
    )
    print(f"\nSmallest shift penalty: {winner}")
    print("Read this as a DIRECTION, not a live estimate: it is measured by")
    print("shrinking chunks, while production grows them. It answers which feature")
    print("set survives a size gap, not what the reward will be at 90 hands.")
    return {"rows": rows, "winner": winner, "holdout": holdout}


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Chunk-size transfer study")
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--holdout-days", type=int, default=3)
    parser.add_argument("--sizes", default="12,18,26")
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        data_dir=Path(args.dir),
        holdout_days=args.holdout_days,
        sizes=[int(s) for s in args.sizes.split(",") if s.strip()],
        top_k=args.top_k,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
