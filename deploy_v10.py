"""Deploy v10: size-invariant training + batch safety budget.

Root cause of the live<->local gap (found 2026-07-13 by studying the stable
top-10 miners' public repos): the public benchmark groups have ~35 hands but the
validator scores on ~100-hand groups (min/max_hands_per_chunk=100). Our aggregate
features sit in a different regime at serve time, so no public metric predicted
live. The stable winners (uid 134/138/33) all (a) train on size-resampled groups
and (b) apply a per-batch positive-call budget to protect the safety gate.

v10:
  - size-resampled training: pool hands from same-label groups within a release
    into larger groups (sizes 40..105) so features are stable at the 100-hand
    live size. Native 35-hand groups kept too.
  - batch safety budget (in model.py score_chunks) caps the >=0.5 fraction,
    securing the safety/calibration gate on the shifted live distribution.
  - model: ExtraTrees + Logistic soft-blend on the 162 robust+coarse features.
Validated on 100-hand eval groups (live regime) with the full serving scoring.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/poker"); sys.path.insert(0, "/root/POKER44-SUBNET-1")
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from model import (extract_group_features, recenter_scores, FEATURE_NAMES,
                   _apply_batch_safety_budget, _MAX_POS_FRAC)
from poker44.score.scoring import reward
from train import fpr_threshold

DATA = Path("/root/POKER44-SUBNET-1/data/benchmark")
OUT = Path("/root/poker/artifacts/poker44_model.joblib")
AMOUNT_SUBSTR = ("size_bb","pot_ratio","roundness","pot_hist","pot_modal","distinct_size","distinct_pot","size_cv","_size_","total_pot")
COLS = [i for i,n in enumerate(FEATURE_NAMES) if not any(s in n for s in AMOUNT_SUBSTR)]
RNG = np.random.default_rng(11)
TRAIN_SIZES = [40, 60, 80, 100, 105]


def load_groups():
    by_rel = {}
    for p in sorted(DATA.glob("*.json")):
        gs = []
        d = json.loads(p.read_text())
        for c in d["chunks"]:
            for g, lab in zip(c["chunks"], c["groundTruth"]):
                gs.append((g, int(lab)))
        by_rel[p.stem] = gs
    return by_rel


def pool_to(pool, target, rng):
    order = rng.permutation(len(pool)); hands = []; i = 0
    while len(hands) < target and i < len(order) * 3:
        hands += list(pool[order[i % len(order)]]); i += 1
    return hands[:target]


def make_sized(gs, sizes, per, rng):
    out = []
    bylab = {0: [g for g, l in gs if l == 0], 1: [g for g, l in gs if l == 1]}
    for lab in (0, 1):
        pool = bylab[lab]
        if len(pool) < 2:
            continue
        for sz in sizes:
            for _ in range(per):
                out.append((pool_to(pool, sz, rng), lab))
    return out


def resampled_training(by_rel, releases, per=8):
    groups, y, dates = [], [], []
    for r in releases:
        for g, l in by_rel[r]:                       # keep native 35-hand groups
            groups.append(g); y.append(l); dates.append(r)
        for g, l in make_sized(by_rel[r], TRAIN_SIZES, per, RNG):
            groups.append(g); y.append(l); dates.append(r)
    return groups, np.array(y), np.array(dates)


def feats(groups):
    return np.vstack([extract_group_features(g) for g in groups])


def build_estimator():
    et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=12, max_features=0.5,
                              n_jobs=4, random_state=42)
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(C=0.3, max_iter=3000))
    vote = VotingClassifier([("et", et), ("lr", lr)], voting="soft", weights=[0.5, 0.5], n_jobs=1)
    sel = ColumnTransformer([("keep", "passthrough", COLS)], remainder="drop")
    return Pipeline([("select", sel), ("clf", vote)])


def serve_scores(prob, thr):
    s = recenter_scores(np.asarray(prob), thr)
    return _apply_batch_safety_budget(s, _MAX_POS_FRAC)


def forward_check(by_rel):
    rels = sorted(by_rel)
    rows = []
    for R in rels[-5:]:
        past = [r for r in rels if r < R]
        gtr, ytr, dtr = resampled_training(by_rel, past, per=6)
        Xtr = feats(gtr)
        test100 = make_sized(by_rel[R], [100], per=40, rng=RNG)
        Xte = feats([g for g, _ in test100]); yte = np.array([l for _, l in test100])
        pipe = build_estimator(); pipe.fit(Xtr, ytr)
        thr = fpr_threshold(pipe.predict_proba(Xtr)[:, 1], ytr)
        sc = serve_scores(pipe.predict_proba(Xte)[:, 1], thr)
        rew, det = reward(sc, yte)
        rows.append((R, rew, det["ap_score"], det["bot_recall"], det["human_safety_penalty"],
                     roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])))
    print("size-aware forward-chaining (train size-resampled, eval on 100-hand groups, full serving scoring):")
    for R, rew, ap, rec, saf, auc in rows:
        print(f"  {R}: reward={rew:.4f}  ap={ap:.4f} recall={rec:.4f} safety={saf:.3f} auc={auc:.4f}")
    print(f"  MEAN reward={np.mean([r[1] for r in rows]):.4f}  (safety>=0.999 confirms budget secures the gate)")


def main():
    by_rel = load_groups()
    rels = sorted(by_rel)
    print(f"{len(rels)} releases; MAX_POS_FRAC={_MAX_POS_FRAC}; model feat={len(COLS)}\n")
    forward_check(by_rel)

    # deployment fit on ALL releases (size-resampled)
    gtr, ytr, dtr = resampled_training(by_rel, rels, per=8)
    X = feats(gtr)
    print(f"\ndeployment training set: {len(ytr)} groups ({int(ytr.sum())} bot) incl. size-resamples")
    oof = cross_val_predict(build_estimator(), X, ytr, groups=dtr, cv=GroupKFold(5),
                            method="predict_proba", n_jobs=1)[:, 1]
    thr = fpr_threshold(oof, ytr)
    print(f"OOF AUC={roc_auc_score(ytr, oof):.4f} deploy threshold={thr:.4f}")

    pipe = build_estimator(); pipe.fit(X, ytr)
    artifact = {"kind": "tabular", "pipeline": pipe, "threshold": float(thr),
                "selected": "v10 ET+Logistic blend, size-resampled training + batch safety budget"}
    joblib.dump(artifact, OUT, compress=3)
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
