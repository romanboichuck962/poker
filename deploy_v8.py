"""Deploy v8: robustness-first model for the public->private distribution shift.

Diagnosis (2026-07-12): the v7 hybrid scored ~0.86 on random-fold CV but only
~0.48 on strict temporal forward-chaining -- matching the live 0.498. The
207-feature Optuna-tuned stack + 3-net ensemble over-fit the public benchmark
bot pool and its (seed-deterministic) bet-amount coarsening noise, and the
OOF-fit threshold leaked future info so the safety gate collapsed live.

v8 fixes the generalization gap directly:
  - drop the 50 bet-amount-keyed features (coarsening-corrupted, non-transferable)
  - single heavily-regularized LightGBM instead of a stack+neural (far lower
    capacity for ~1.6k samples)
  - threshold from grouped OOF on all data
Forward-chaining reward: 0.478 (v7) -> 0.751 (v8), std 0.218 -> 0.064, no gate
collapses (min 0.66).
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/Poker44-subnet")
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from model import extract_group_features, recenter_scores, FEATURE_NAMES  # noqa
from poker44.score.scoring import reward  # noqa
from train import fpr_threshold  # noqa

DATA = Path("/root/Poker44-subnet/data/benchmark")
CACHE = Path("/tmp/claude-0/-root-Poker44-subnet/e7fcc35d-0d27-44fa-9895-49d8694f9df1/scratchpad/x207.npz")
OUT = Path("/root/poker/artifacts/poker44_model.joblib")

AMOUNT_SUBSTR = ("size_bb", "pot_ratio", "roundness", "pot_hist", "pot_modal",
                 "distinct_size", "distinct_pot", "size_cv", "_size_", "total_pot")
ROBUST = [i for i, n in enumerate(FEATURE_NAMES) if not any(s in n for s in AMOUNT_SUBSTR)]


def load():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return d["X"], d["y"], d["dates"]
    X, y, dates = [], [], []
    for p in sorted(DATA.glob("*.json")):
        d = json.loads(p.read_text())
        for c in d["chunks"]:
            for g, lab in zip(c["chunks"], c["groundTruth"]):
                X.append(extract_group_features(g)); y.append(int(lab)); dates.append(p.stem)
    return np.vstack(X), np.array(y), np.array(dates)


def build_pipeline():
    # first step slices the full 207-dim vector down to the robust columns, so the
    # artifact stays drop-in for model.py (which feeds all 207 features).
    sel = ColumnTransformer([("robust", "passthrough", ROBUST)], remainder="drop")
    lgbm = LGBMClassifier(n_estimators=300, num_leaves=15, min_child_samples=40,
                          learning_rate=0.03, subsample=0.7, subsample_freq=1,
                          colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=1.0,
                          n_jobs=4, verbose=-1, random_state=42)
    return Pipeline([("select", sel), ("lgbm", lgbm)])


def forward_check(X, y, dates):
    rel = sorted(set(dates.tolist()))
    rews = []
    for r in rel[-6:]:
        tr, te = np.array([d < r for d in dates]), dates == r
        if te.sum() < 20 or y[te].sum() in (0, te.sum()):
            continue
        pipe = build_pipeline(); pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        thr = fpr_threshold(pipe.predict_proba(X[tr])[:, 1], y[tr])
        rew, _ = reward(recenter_scores(p, thr), y[te])
        rews.append((r, rew))
    print("forward-chaining reward (v8):")
    for r, v in rews:
        print(f"  {r}: {v:.4f}")
    print(f"  mean={np.mean([v for _, v in rews]):.4f}")


def main():
    X, y, dates = load()
    print(f"{len(y)} groups, {int(y.sum())} bots, {len(set(dates))} releases; robust feat={len(ROBUST)}")
    forward_check(X, y, dates)

    # deployment threshold from grouped OOF on ALL data
    oof = cross_val_predict(build_pipeline(), X, y, groups=dates, cv=GroupKFold(5),
                            method="predict_proba", n_jobs=1)[:, 1]
    thr = fpr_threshold(oof, y)
    print(f"\nOOF pooled AUC={roc_auc_score(y, oof):.4f} AP={average_precision_score(y, oof):.4f} "
          f"deploy threshold(5%FPR)={thr:.4f}")

    pipe = build_pipeline(); pipe.fit(X, y)
    artifact = {"kind": "tabular", "pipeline": pipe, "threshold": float(thr),
                "selected": "robust_reg_lgbm_v8 (157 coarsening-invariant feat)"}
    joblib.dump(artifact, OUT, compress=3)
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    import json  # noqa
    main()
