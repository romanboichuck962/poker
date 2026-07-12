"""Deploy v9: robustness-first ExtraTrees+Logistic soft-blend.

Winner-take-all under a public->private distribution shift. Selected purely by
STRICT temporal forward-chaining reward (train on releases < T, predict T,
threshold from training-pool preds). Search results (fwd-reward mean / floor):
  v7 hybrid 207-feat stack+neural ... 0.478 / 0.000   (random-CV illusion 0.86)
  v8 reg-LGBM robust-157 ............ 0.751 / 0.660
  v9 ET(msl12)+Logit robust+coarse .. 0.785 / 0.720   <-- deployed

v9 = 157 coarsening-robust behavioral/policy features + 5 coarsening-SURVIVABLE
bucket-snapped sizing-concentration features (added to model.py). Model = soft
blend of a regularized ExtraTrees (bagged, low-variance) and a Logistic
regressor (linear, high-floor) -> diverse -> transfers better than one family.
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/poker"); sys.path.insert(0, "/root/Poker44-subnet")

import json
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from model import extract_group_features, recenter_scores, FEATURE_NAMES  # noqa
from poker44.score.scoring import reward  # noqa
from train import fpr_threshold  # noqa

DATA = Path("/root/Poker44-subnet/data/benchmark")
OUT = Path("/root/poker/artifacts/poker44_model.joblib")
AMOUNT_SUBSTR = ("size_bb", "pot_ratio", "roundness", "pot_hist", "pot_modal",
                 "distinct_size", "distinct_pot", "size_cv", "_size_", "total_pot")
COLS = [i for i, n in enumerate(FEATURE_NAMES) if not any(s in n for s in AMOUNT_SUBSTR)]


def load():
    X, y, dates = [], [], []
    for p in sorted(DATA.glob("*.json")):
        d = json.loads(p.read_text())
        for c in d["chunks"]:
            for g, lab in zip(c["chunks"], c["groundTruth"]):
                X.append(extract_group_features(g)); y.append(int(lab)); dates.append(p.stem)
    return np.vstack(X), np.array(y), np.array(dates)


def build_estimator():
    et = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=12, max_features=0.5,
                              n_jobs=4, random_state=42)
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=0.3, max_iter=3000))
    vote = VotingClassifier([("et", et), ("lr", lr)], voting="soft", weights=[0.5, 0.5], n_jobs=1)
    sel = ColumnTransformer([("keep", "passthrough", COLS)], remainder="drop")
    return Pipeline([("select", sel), ("clf", vote)])


def forward_check(X, y, dates):
    rel = sorted(set(dates.tolist()))
    out = []
    for r in rel[-6:]:
        tr, te = np.array([d < r for d in dates]), dates == r
        if te.sum() < 20 or y[te].sum() in (0, te.sum()):
            continue
        m = build_estimator(); m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        thr = fpr_threshold(m.predict_proba(X[tr])[:, 1], y[tr])
        rew, _ = reward(recenter_scores(p, thr), y[te])
        out.append((r, rew, roc_auc_score(y[te], p)))
    print("forward-chaining (v9):")
    for r, v, a in out:
        print(f"  {r}: reward={v:.4f} auc={a:.4f}")
    print(f"  mean reward={np.mean([v for _, v, _ in out]):.4f} "
          f"min={min(v for _, v, _ in out):.4f}")


def main():
    X, y, dates = load()
    print(f"{len(y)} groups, {int(y.sum())} bots, {len(set(dates))} releases; model feat={len(COLS)}")
    forward_check(X, y, dates)

    oof = cross_val_predict(build_estimator(), X, y, groups=dates, cv=GroupKFold(5),
                            method="predict_proba", n_jobs=1)[:, 1]
    thr = fpr_threshold(oof, y)
    print(f"\nOOF pooled AUC={roc_auc_score(y, oof):.4f} AP={average_precision_score(y, oof):.4f} "
          f"deploy threshold(5%FPR)={thr:.4f}")

    pipe = build_estimator(); pipe.fit(X, y)
    artifact = {"kind": "tabular", "pipeline": pipe, "threshold": float(thr),
                "selected": "v9 ExtraTrees+Logistic soft-blend, 162 robust+coarse feat"}
    joblib.dump(artifact, OUT, compress=3)
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
