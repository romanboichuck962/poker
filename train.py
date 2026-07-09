"""Train and select the Poker44 bot-detection model.

Loads cached benchmark releases (see Poker44-subnet/scripts/download_benchmark.py),
builds hero-centric group features, compares several algorithms with
date-grouped cross-validation, evaluates candidates on a held-out release with
the validator's own reward formula, and saves the winning calibrated pipeline
to artifacts/poker44_model.joblib.

Usage:
    python train.py --data /root/Poker44-subnet/data/benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import extract_group_features, recenter_scores

sys.path.insert(0, "/root/Poker44-subnet")
from poker44.score.scoring import reward  # noqa: E402  (validator formula)


def load_dataset(data_dir: Path):
    X, y, dates, splits = [], [], [], []
    for path in sorted(data_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        for chunk in payload["chunks"]:
            split = chunk.get("split") or "train"
            for group, label in zip(chunk["chunks"], chunk["groundTruth"]):
                X.append(extract_group_features(group))
                y.append(int(label))
                dates.append(path.stem)
                splits.append(split)
    return np.vstack(X), np.array(y), np.array(dates), np.array(splits)


def candidate_models(seed: int = 42):
    return {
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, random_state=seed)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=3, n_jobs=-1, random_state=seed
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=3, n_jobs=-1, random_state=seed
        ),
        "hist_gbdt": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed
        ),
        "gbdt": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=seed
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                          max_iter=1500, random_state=seed),
        ),
    }


def fpr_threshold(prob: np.ndarray, y: np.ndarray, max_fpr: float = 0.05) -> float:
    """Smallest score threshold whose false-positive rate is within budget."""
    neg = np.sort(prob[y == 0])
    if not len(neg):
        return 0.5
    return float(neg[int(np.ceil(len(neg) * (1 - max_fpr))) - 1]) + 1e-9


def oof_threshold(base, X, y, groups, cv, max_fpr: float = 0.05) -> float:
    """Operating threshold from out-of-fold predictions (transfers to unseen data)."""
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    oof = cross_val_predict(model, X, y, groups=groups, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    return fpr_threshold(oof, y, max_fpr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/root/Poker44-subnet/data/benchmark")
    parser.add_argument("--holdout-date", default=None,
                        help="Release date held out for final scoring (default: newest)")
    parser.add_argument("--out", default=str(Path(__file__).parent / "artifacts" / "poker44_model.joblib"))
    args = parser.parse_args()

    X, y, dates, splits = load_dataset(Path(args.data))
    holdout_date = args.holdout_date or sorted(set(dates))[-1]

    test_mask = (dates == holdout_date) | (splits == "validation")
    train_mask = ~test_mask
    Xtr, ytr, dtr = X[train_mask], y[train_mask], dates[train_mask]
    Xte, yte = X[test_mask], y[test_mask]
    print(f"dataset: {len(y)} groups, {X.shape[1]} features, "
          f"{len(set(dates))} release dates")
    print(f"train: {len(ytr)} groups ({ytr.sum()} bots) | "
          f"holdout({holdout_date} + validation split): {len(yte)} groups ({yte.sum()} bots)\n")

    cv = GroupKFold(n_splits=min(5, len(set(dtr))))
    results = []
    for name, base in candidate_models().items():
        cv_auc = cross_val_score(base, Xtr, ytr, groups=dtr, cv=cv,
                                 scoring="roc_auc", n_jobs=-1)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(Xtr, ytr)
        threshold = oof_threshold(base, Xtr, ytr, dtr, cv)
        prob = recenter_scores(model.predict_proba(Xte)[:, 1], threshold)
        rew, det = reward(prob, yte)
        auc = roc_auc_score(yte, prob)
        ap = average_precision_score(yte, prob)
        results.append((name, model, rew, auc, ap, cv_auc.mean(), det))
        print(f"{name:14s} cv_auc={cv_auc.mean():.4f}±{cv_auc.std():.3f}  "
              f"holdout: reward={rew:.4f} auc={auc:.4f} ap={ap:.4f} "
              f"recall@fpr5={det['bot_recall']:.3f} hard_fpr={det['hard_fpr']:.3f} sanity={det['threshold_sanity_quality']:.3f}")

    # Select by holdout validator reward, tie-broken by AUC.
    results.sort(key=lambda r: (r[2], r[3]), reverse=True)
    name, model, rew, auc, ap, cv_auc, det = results[0]
    print(f"\nselected: {name} (holdout reward {rew:.4f}, auc {auc:.4f})")

    # Refit the winner on ALL data (including holdout) for deployment.
    final = CalibratedClassifierCV(candidate_models()[name], method="sigmoid", cv=3)
    final.fit(X, y)
    threshold = oof_threshold(candidate_models()[name], X, y, dates,
                              GroupKFold(n_splits=min(5, len(set(dates)))))
    print(f"deployment operating threshold (OOF FPR<=5%): {threshold:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "threshold": threshold, "selected": name}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
