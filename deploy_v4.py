"""Deploy the tuned stacking ensemble (best full-OOF AUC/AP among v4 candidates).

Rebuilds the Optuna-tuned LightGBM/CatBoost/XGBoost from the recorded params,
stacks them with ExtraTrees under a logistic meta-learner, verifies OOF metrics,
takes an honest held-out operating threshold, and saves a compact artifact.
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/Poker44-subnet")
from model import recenter_scores  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from robust_select import load_cache, per_window_reward  # noqa: E402
from train import fpr_threshold  # noqa: E402

SEED = 42
NAME = "tuned_stack"

LGBM = dict(colsample_bytree=0.5380616120402459, learning_rate=0.01120842898157175,
            max_depth=7, min_child_samples=25, n_estimators=400, num_leaves=24,
            reg_alpha=0.008515043896198893, reg_lambda=1.0529781513129652,
            subsample=0.6514619832844296, n_jobs=1, verbose=-1, random_state=SEED)
CAT = dict(iterations=800, learning_rate=0.021428868603326833, depth=4,
           l2_leaf_reg=2.407476983357051, random_strength=1.4937723262247309,
           subsample=0.6933539510408397, thread_count=1, verbose=0,
           allow_writing_files=False, random_seed=SEED)
XGB = dict(colsample_bytree=0.8432445184319163, gamma=1.398650501746932,
           learning_rate=0.023073982700012968, max_depth=5, min_child_weight=4,
           n_estimators=900, reg_alpha=0.2853956448950912, reg_lambda=3.3499043572545175,
           subsample=0.7871624339606097, tree_method="hist", eval_metric="logloss",
           n_jobs=1, random_state=SEED)


def build_stack():
    return StackingClassifier(
        estimators=[("lgbm", LGBMClassifier(**LGBM)),
                    ("cat", CatBoostClassifier(**CAT)),
                    ("xgb", XGBClassifier(**XGB)),
                    ("extra", ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2,
                                                   n_jobs=4, random_state=SEED))],
        final_estimator=LogisticRegression(max_iter=2000, C=1.0),
        stack_method="predict_proba", cv=4, n_jobs=4)


def main():
    X, y, dates = load_cache()
    cv5 = GroupKFold(5)

    model = build_stack()
    oof = cross_val_predict(model, X, y, groups=dates, cv=cv5,
                            method="predict_proba", n_jobs=4)[:, 1]
    thr_oof = fpr_threshold(oof, y)
    rew, n_win = per_window_reward(recenter_scores(oof, thr_oof), y, dates)
    print(f"{NAME}: OOF AUC={roc_auc_score(y, oof):.4f} AP={average_precision_score(y, oof):.4f} "
          f"per-window reward={rew:.4f} ({n_win} windows)")

    latest = sorted(set(dates))[-1]
    te = dates == latest
    hold = CalibratedClassifierCV(build_stack(), method="sigmoid", cv=3, ensemble=False)
    hold.fit(X[~te], y[~te])
    prob_te = hold.predict_proba(X[te])[:, 1]
    thr = fpr_threshold(prob_te, y[te])
    r, det = reward(recenter_scores(prob_te, thr), y[te])
    print(f"holdout {latest}: reward={r:.4f} auc={roc_auc_score(y[te], prob_te):.4f} "
          f"ap={average_precision_score(y[te], prob_te):.4f} recall@fpr5={det['bot_recall']:.3f} thr={thr:.4f}")

    final = CalibratedClassifierCV(build_stack(), method="sigmoid", cv=3, ensemble=False)
    final.fit(X, y)
    out = Path("/root/poker/artifacts/poker44_model.joblib")
    joblib.dump({"pipeline": final, "threshold": thr, "selected": NAME}, out, compress=3)
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
