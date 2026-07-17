"""Robust model selection by cross-validated per-window validator reward.

The live validator scores each evaluation window independently, so the honest
selection metric is the validator reward averaged over held-out windows, not a
single pooled holdout. This script caches the feature matrix, then for each
candidate computes out-of-fold probabilities with date-grouped CV, recenters to
the OOF 5%-FPR operating point, and averages reward across release dates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/POKER44-SUBNET-1")
from model import extract_group_features, recenter_scores  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from train import candidate_models, fpr_threshold  # noqa: E402

DATA = Path("/root/POKER44-SUBNET-1/data/benchmark")
CACHE = Path("/root/poker/artifacts/features.npz")
CANDIDATES = ["catboost", "voting_soft", "stack", "random_forest", "extra_trees"]


def build_cache():
    X, y, dates = [], [], []
    for path in sorted(DATA.glob("*.json")):
        payload = json.loads(path.read_text())
        for chunk in payload["chunks"]:
            for group, label in zip(chunk["chunks"], chunk["groundTruth"]):
                X.append(extract_group_features(group))
                y.append(int(label))
                dates.append(path.stem)
    X, y, dates = np.vstack(X), np.array(y), np.array(dates)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y, dates=dates)
    return X, y, dates


def load_cache():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["X"], d["y"], d["dates"]
    return build_cache()


def per_window_reward(prob, y, dates, min_groups=20):
    rewards, weights = [], []
    for date in np.unique(dates):
        m = dates == date
        yd = y[m]
        if m.sum() < min_groups or yd.sum() == 0 or yd.sum() == len(yd):
            continue
        r, _ = reward(prob[m], yd)
        rewards.append(r)
        weights.append(m.sum())
    rewards, weights = np.array(rewards), np.array(weights)
    return float(np.average(rewards, weights=weights)), len(rewards)


def main():
    X, y, dates = load_cache()
    print(f"dataset: {len(y)} groups, {X.shape[1]} features, {len(set(dates))} dates")
    cv = GroupKFold(n_splits=5)
    models = candidate_models()

    results = []
    for name in CANDIDATES:
        base = models[name]
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        oof = cross_val_predict(clf, X, y, groups=dates, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        thr = fpr_threshold(oof, y)
        centered = recenter_scores(oof, thr)
        win_reward, n_windows = per_window_reward(centered, y, dates)
        auc = roc_auc_score(y, oof)
        ap = average_precision_score(y, oof)
        results.append((name, win_reward, auc, ap, n_windows))
        print(f"{name:14s} cv per-window reward={win_reward:.4f}  "
              f"oof_auc={auc:.4f} oof_ap={ap:.4f} (over {n_windows} windows)")

    results.sort(key=lambda r: (r[1], r[2]), reverse=True)
    best = results[0][0]
    print(f"\nselected (robust): {best} (per-window reward {results[0][1]:.4f})")

    # Refit winner on all data; threshold from its own OOF predictions.
    final = CalibratedClassifierCV(models[best], method="sigmoid", cv=3)
    oof = cross_val_predict(final, X, y, groups=dates, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    thr = fpr_threshold(oof, y)
    final.fit(X, y)
    out = Path("/root/poker/artifacts/poker44_model.joblib")
    joblib.dump({"pipeline": final, "threshold": thr, "selected": best}, out)
    print(f"deployment threshold (OOF FPR<=5%): {thr:.4f}\nsaved {out}")


if __name__ == "__main__":
    main()
