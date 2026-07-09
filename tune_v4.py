"""v4: Optuna-tuned boosting models + tuned ensembles, selected by per-window reward.

Tunes LightGBM, CatBoost, and XGBoost to grouped-CV AUC, then compares the tuned
singles against tuned voting/stacking ensembles by cross-validated per-window
validator reward (the metric the live validator optimizes). Deploys the winner
as a compact calibrated artifact.

Threads are pinned (model n_jobs=1, folds parallel) to avoid the nested-parallelism
slowdown on small core counts.
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
import optuna  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier, VotingClassifier  # noqa: E402
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

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
N_TRIALS = int(os.getenv("N_TRIALS", "50"))
CV5 = GroupKFold(n_splits=5)
CV4 = GroupKFold(n_splits=4)


def oof_auc(model, X, y, dates, cv=CV4):
    oof = cross_val_predict(model, X, y, groups=dates, cv=cv,
                            method="predict_proba", n_jobs=4)[:, 1]
    return roc_auc_score(y, oof), oof


def tune_lgbm(X, y, dates):
    def obj(t):
        m = LGBMClassifier(
            n_estimators=t.suggest_int("n_estimators", 300, 900, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.08, log=True),
            num_leaves=t.suggest_int("num_leaves", 15, 63),
            max_depth=t.suggest_int("max_depth", 3, 8),
            min_child_samples=t.suggest_int("min_child_samples", 5, 40),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 0.5, 8.0, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            n_jobs=1, verbose=-1, random_state=SEED)
        return oof_auc(m, X, y, dates)[0]
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"  lgbm best AUC={s.best_value:.4f}")
    return LGBMClassifier(**s.best_params, n_jobs=1, verbose=-1, random_state=SEED)


def tune_cat(X, y, dates):
    def obj(t):
        m = CatBoostClassifier(
            iterations=t.suggest_int("iterations", 300, 900, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            depth=t.suggest_int("depth", 3, 8),
            l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            random_strength=t.suggest_float("random_strength", 0.0, 2.0),
            thread_count=1, verbose=0, allow_writing_files=False, random_seed=SEED)
        return oof_auc(m, X, y, dates)[0]
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"  cat best AUC={s.best_value:.4f}")
    return CatBoostClassifier(**s.best_params, thread_count=1, verbose=0,
                              allow_writing_files=False, random_seed=SEED)


def tune_xgb(X, y, dates):
    def obj(t):
        m = XGBClassifier(
            n_estimators=t.suggest_int("n_estimators", 300, 900, step=100),
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=t.suggest_int("max_depth", 3, 8),
            min_child_weight=t.suggest_int("min_child_weight", 1, 10),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 0.5, 8.0, log=True),
            reg_alpha=t.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            gamma=t.suggest_float("gamma", 0.0, 3.0),
            tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=SEED)
        return oof_auc(m, X, y, dates)[0]
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    s.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"  xgb best AUC={s.best_value:.4f}")
    return XGBClassifier(**s.best_params, tree_method="hist", eval_metric="logloss",
                         n_jobs=1, random_state=SEED)


def evaluate(name, model, X, y, dates):
    auc, oof = oof_auc(model, X, y, dates, cv=CV5)
    thr = fpr_threshold(oof, y)
    centered = recenter_scores(oof, thr)
    win_reward, n_win = per_window_reward(centered, y, dates)
    ap = average_precision_score(y, oof)
    print(f"{name:16s} per-window reward={win_reward:.4f}  oof_auc={auc:.4f} oof_ap={ap:.4f}")
    return {"name": name, "reward": win_reward, "auc": auc, "ap": ap}


def main():
    X, y, dates = load_cache()
    print(f"dataset: {len(y)} groups, {X.shape[1]} features, {len(set(dates))} dates | trials={N_TRIALS}\n")

    print("tuning boosting models...")
    lgbm = tune_lgbm(X, y, dates)
    cat = tune_cat(X, y, dates)
    xgb = tune_xgb(X, y, dates)
    extra = ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, n_jobs=4, random_state=SEED)

    voting = VotingClassifier(
        estimators=[("lgbm", lgbm), ("cat", cat), ("xgb", xgb), ("extra", extra)],
        voting="soft", n_jobs=4)
    stack = StackingClassifier(
        estimators=[("lgbm", lgbm), ("cat", cat), ("xgb", xgb), ("extra", extra)],
        final_estimator=LogisticRegression(max_iter=2000, C=1.0),
        stack_method="predict_proba", cv=4, n_jobs=4)

    print("\nevaluating candidates by CV per-window reward...")
    candidates = {"tuned_lgbm": lgbm, "tuned_cat": cat, "tuned_xgb": xgb,
                  "extra_trees": extra, "tuned_voting": voting, "tuned_stack": stack}
    results = [evaluate(n, m, X, y, dates) for n, m in candidates.items()]
    results.sort(key=lambda r: (r["reward"], r["auc"]), reverse=True)
    best_name = results[0]["name"]
    best_model = candidates[best_name]
    print(f"\nselected: {best_name} (per-window reward {results[0]['reward']:.4f}, "
          f"auc {results[0]['auc']:.4f})")

    # Honest held-out threshold from the newest release.
    latest = sorted(set(dates))[-1]
    te = dates == latest
    hold = CalibratedClassifierCV(best_model, method="sigmoid", cv=3, ensemble=False)
    hold.fit(X[~te], y[~te])
    prob_te = hold.predict_proba(X[te])[:, 1]
    thr = fpr_threshold(prob_te, y[te])
    r, det = reward(recenter_scores(prob_te, thr), y[te])
    print(f"holdout {latest}: reward={r:.4f} auc={roc_auc_score(y[te], prob_te):.4f} "
          f"recall@fpr5={det['bot_recall']:.3f} thr={thr:.4f}")

    # Deploy: winner refit on all data, compact.
    final = CalibratedClassifierCV(best_model, method="sigmoid", cv=3, ensemble=False)
    final.fit(X, y)
    out = Path("/root/poker/artifacts/poker44_model.joblib")
    joblib.dump({"pipeline": final, "threshold": thr, "selected": best_name}, out, compress=3)
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB)")

    # persist tuned params for the record
    params = {n: (m.get_params() if hasattr(m, "get_params") else str(m))
              for n, m in [("lgbm", lgbm), ("cat", cat), ("xgb", xgb)]}
    Path("/root/poker/artifacts/tuned_params.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in params.items()))


if __name__ == "__main__":
    main()
