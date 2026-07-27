"""Fit an artifact.

    python -m poker44_ml.train                        # holdout = latest 2 dates
    python -m poker44_ml.train --holdout 2026-07-24   # explicit holdout dates
    python -m poker44_ml.train --final                # fit on everything, no holdout

The holdout is always the LATEST dates, never a random split. Chunks from one
release date share a data-generation batch, so a random split leaks and reports
an AP you will not see live. The reference miners all learned this; one of them
wrote "NO trusting random-split OOF for ship decisions" into their do-not list.

Every number printed here comes from scoring the holdout through the real
``Detector`` and the real upstream ``reward()``. There is no second scoring path.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from poker44_ml import MODEL_NAME, __version__
from poker44_ml.data import DEFAULT_DIR, Corpus, load_corpus
from poker44_ml.features import chunk_features, feature_matrix
from poker44_ml.inference import Detector
from poker44_ml.model import RankVoteEnsemble, build_members, select_features
from poker44_ml.policy import DEFAULT_FRACTION_GRID, choose_fraction
from poker44_ml.reward import format_metrics, reward_metrics
from poker44_ml.upstream import check as check_upstream

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"

# Chunks per simulated validator request when tuning the positive fraction. The
# policy is batch-relative, so it must be tuned on realistic request sizes.
LIVE_BATCH_CHUNKS = 100

DEFAULTS = {
    "holdout_days": 2,
    "calibration_fraction": 0.22,
    "feature_top_k": 150,
    "human_weight": 2.0,
    "seed": 42,
}


def temporal_split(
    corpus: Corpus,
    *,
    holdout_dates: Sequence[str] | None,
    holdout_days: int,
) -> tuple[Corpus, Corpus, List[str]]:
    """Train on the past, test on the most recent dates."""
    dates = corpus.dates
    if not dates:
        raise RuntimeError("corpus has no source dates; cannot split temporally")

    chosen = list(holdout_dates) if holdout_dates else dates[-max(1, holdout_days):]
    train = corpus.by_date(exclude=chosen)
    test = corpus.by_date(include=chosen)

    if not train.examples or not test.examples:
        raise RuntimeError(f"empty split with holdout {chosen} (dates on disk: {dates})")
    for name, part in (("train", train), ("test", test)):
        if len(set(part.labels)) < 2:
            raise RuntimeError(f"{name} split has only one class; widen the holdout")
    return train, test, chosen


def _batches(n: int, size: int = LIVE_BATCH_CHUNKS) -> List[List[int]]:
    """Index groups emulating separate validator requests."""
    if n <= size:
        return [list(range(n))]
    return [list(range(start, min(start + size, n))) for start in range(0, n, size)]


def _feature_reference(rows: np.ndarray) -> Dict[str, List[float]]:
    """Training-distribution quantiles, used for serve-time drift detection."""
    return {
        "q01": np.quantile(rows, 0.01, axis=0).tolist(),
        "q25": np.quantile(rows, 0.25, axis=0).tolist(),
        "median": np.median(rows, axis=0).tolist(),
        "q75": np.quantile(rows, 0.75, axis=0).tolist(),
        "q99": np.quantile(rows, 0.99, axis=0).tolist(),
    }


def train(
    *,
    data_dir: Path = DEFAULT_DIR,
    out_path: Path | None = None,
    holdout_dates: Sequence[str] | None = None,
    holdout_days: int = DEFAULTS["holdout_days"],
    feature_top_k: int = DEFAULTS["feature_top_k"],
    human_weight: float = DEFAULTS["human_weight"],
    calibration_fraction: float = DEFAULTS["calibration_fraction"],
    fraction_grid: Sequence[float] = DEFAULT_FRACTION_GRID,
    seed: int = DEFAULTS["seed"],
    final: bool = False,
) -> Dict[str, Any]:
    """Fit, tune the score policy, evaluate through the serving path, save."""
    import joblib

    # A changed reward or sanitizer invalidates the run before it starts.
    check_upstream(strict=True)

    corpus = load_corpus(data_dir)
    if final:
        train_corpus, test_corpus, chosen = corpus, Corpus([]), []
        print("FINAL fit: training on every date, no holdout evaluation")
    else:
        train_corpus, test_corpus, chosen = temporal_split(
            corpus, holdout_dates=holdout_dates, holdout_days=holdout_days
        )
        print(f"split: train={len(train_corpus)} test={len(test_corpus)} holdout={chosen}")

    # ---- features ---------------------------------------------------------
    print("extracting features...")
    all_names = sorted(chunk_features(train_corpus.chunks[0]))
    x_full = np.nan_to_num(
        np.asarray(feature_matrix(train_corpus.chunks, all_names), dtype=np.float64)
    )
    y_train = np.asarray(train_corpus.labels, dtype=np.int64)

    names = select_features(x_full, y_train, all_names, top_k=feature_top_k, seed=seed)
    print(f"selected {len(names)}/{len(all_names)} features by gain")
    keep = [all_names.index(name) for name in names]
    x_train = x_full[:, keep]

    # ---- fit / calibrate split -------------------------------------------
    rng = np.random.RandomState(seed + 17)
    order = rng.permutation(len(train_corpus))
    n_calibration = int(len(order) * calibration_fraction)
    cal_idx, fit_idx = order[:n_calibration], order[n_calibration:]
    if len(set(y_train[cal_idx].tolist())) < 2 or len(fit_idx) < 32:
        print("  calibration slice unusable; reusing the full train split")
        cal_idx = fit_idx = np.arange(len(train_corpus))

    print(f"fitting members on {len(fit_idx)} chunks...")
    members, weights, member_names = build_members(
        x_train[fit_idx], y_train[fit_idx], seed=seed, human_weight=human_weight
    )
    ensemble = RankVoteEnsemble(members, weights, member_names)
    print(f"  {ensemble}")

    # ---- tune the 0.5 line on held-out calibration chunks ------------------
    cal_scores = ensemble.score(x_train[cal_idx])
    selection = choose_fraction(
        cal_scores,
        y_train[cal_idx],
        batches=_batches(len(cal_idx)),
        grid=fraction_grid,
    )
    fraction = selection["selected"]
    print(f"positive fraction: {fraction:.3f}  "
          f"(worst case over bot shares {list(selection['prevalences'])})")
    for row in selection["grid"]:
        flag = " <-" if row["fraction"] == fraction else ""
        by = " ".join(f"{p}:{v:.3f}" for p, v in row.get("by_prevalence", {}).items())
        print(
            f"    {row['fraction']:.2f}: worst_reward={row.get('reward_min', 0):.4f} "
            f"fpr_max={row.get('fpr_max', 0):.3f} "
            f"zero_gate={row.get('zero_gate_rate', 0):.0%} "
            f"@bot_share[{by}]{flag}"
        )
    if selection["all_candidates_tripped_gate"]:
        print("  WARNING: every fraction tripped the zero-TP gate on some batch. "
              "The model is not separated enough; fix the model, not this knob.")
    elif selection.get("degenerate"):
        print(f"  NOTE: the reward ties across {selection['tied_fractions']} even after "
              "the bot-share sweep, so the SMALLEST was taken -- it carries the most "
              "fpr headroom, since worst-case fpr@0.5 -> fraction/(1-bot_share). These "
              "are still 30-40 hand benchmark chunks; live ones are 80-100.")

    # ---- save -------------------------------------------------------------
    artifact = {
        "ensemble": ensemble,
        "feature_names": names,
        "metadata": {
            "model_name": MODEL_NAME,
            "model_version": __version__,
            "positive_fraction": float(fraction),
            "member_names": member_names,
            "member_weights": weights,
            "feature_top_k": int(feature_top_k),
            "human_weight": float(human_weight),
            "seed": int(seed),
            "n_train": int(len(fit_idx)),
            "n_calibration": int(len(cal_idx)),
            "train_dates": train_corpus.dates,
            "holdout_dates": list(chosen),
            "feature_reference": _feature_reference(x_train),
            "fraction_grid": selection["grid"],
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    out_path = Path(out_path) if out_path else ARTIFACT_DIR / "artos_v1.joblib"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"saved -> {out_path}")

    # ---- honest evaluation, through the serving path ----------------------
    result: Dict[str, Any] = {"artifact": str(out_path), "positive_fraction": fraction}
    if test_corpus.examples:
        detector = Detector.load(out_path)
        y_test = np.asarray(test_corpus.labels, dtype=int)
        per_batch: List[float] = []
        pooled: List[float] = []
        for group in _batches(len(test_corpus)):
            chunks = [test_corpus.chunks[i] for i in group]
            scores = detector.predict_chunk_scores(chunks)
            pooled.extend(scores)
            per_batch.append(reward_metrics(y_test[group], scores)["reward"])

        metrics = reward_metrics(y_test, pooled)
        print()
        print(f"HOLDOUT {chosen}")
        print(f"  {format_metrics(metrics)}")
        print(f"  per-request reward: mean={np.mean(per_batch):.4f} "
              f"min={np.min(per_batch):.4f} over {len(per_batch)} request(s)")
        drift = detector.drift_report(test_corpus.chunks)
        if drift:
            print(f"  holdout drift vs train: "
                  f"outside_q01_q99={drift['outside_q01_q99_rate']:.1%} "
                  f"median_shift={drift['median_shift_iqr']:.2f} IQR")
        result["holdout"] = metrics
        result["holdout_reward_min"] = float(np.min(per_batch))

    (out_path.with_suffix(".meta.json")).write_text(
        json.dumps(
            {k: v for k, v in artifact["metadata"].items() if k != "feature_reference"},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Train the p44 v1 detector")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="benchmark data dir")
    parser.add_argument("--out", default=None, help="artifact path")
    parser.add_argument("--holdout", default=None, help="comma-separated holdout dates")
    parser.add_argument("--holdout-days", type=int, default=DEFAULTS["holdout_days"])
    parser.add_argument("--feature-top-k", type=int, default=DEFAULTS["feature_top_k"])
    parser.add_argument("--human-weight", type=float, default=DEFAULTS["human_weight"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--final", action="store_true",
                        help="fit on all dates (ship only after walkforward passes)")
    args = parser.parse_args()

    holdout = [d.strip() for d in (args.holdout or "").split(",") if d.strip()] or None
    train(
        data_dir=Path(args.dir),
        out_path=Path(args.out) if args.out else None,
        holdout_dates=holdout,
        holdout_days=args.holdout_days,
        feature_top_k=args.feature_top_k,
        human_weight=args.human_weight,
        seed=args.seed,
        final=args.final,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
