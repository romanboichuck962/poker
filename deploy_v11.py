"""Deploy v11: the top-miner recipe, adapted.

Code-level analysis of the stable top-10 miners (uid 134/33/138) + our own tests
identified why v7-v10 plateaued at ~0.47 while they hold ~0.54:
  1. TRAIN != SERVE: we trained on raw benchmark hands, but the validator sends
     every hand through prepare_hand_for_miner (seat re-alias, button=0, amount
     bucket+noise, 5-8 action window -> hero often absent). => train on SANITIZED.
  2. HERO-CENTRIC features collapse when the hero is windowed out. => added 50
     hero-free / all-actor features (turn-taking rhythm, action/hand-replay
     signatures, pot-flow dynamics, stack depth, exact validator bb-bucket grid).
  3. Single ExtraTrees+LogReg. => rank-blend of DIVERSE learners (boosted + bagged
     + PCA->MLP) fused by in-batch rank (calibration-free, optimizes AP/recall).
  4. FPR-anchored threshold from the human-score quantile; size-augment to live
     ~100-hand groups; batch safety budget (already in model.py).
Validated by walk-forward-by-date on SANITIZED 100-hand eval groups (live proxy).
"""
import json, sys, warnings, math
from pathlib import Path
import numpy as np
import joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/poker"); sys.path.insert(0, "/root/Poker44-subnet")
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import importlib, model as M; importlib.reload(M)
from model import (extract_group_features, recenter_scores, _rank01,
                   _apply_batch_safety_budget, _MAX_POS_FRAC, FEATURE_NAMES)
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44.score.scoring import reward

DATA = Path("/root/Poker44-subnet/data/benchmark")
OUT = Path("/root/poker/artifacts/poker44_model.joblib")
AMOUNT_SUBSTR = ("size_bb","pot_ratio","roundness","pot_hist","pot_modal","distinct_size","distinct_pot","size_cv","_size_","total_pot")
COLS = [i for i, n in enumerate(FEATURE_NAMES) if not any(s in n for s in AMOUNT_SUBSTR)]
RNG = np.random.default_rng(1201)
TARGET_FPR = 0.05
WEIGHTS = np.array([0.40, 0.25, 0.35])   # lgbm, extratrees, pca-mlp


def load_sanitized():
    by = {}
    for p in sorted(DATA.glob("*.json")):
        gs = []
        d = json.loads(p.read_text())
        for c in d["chunks"]:
            for g, l in zip(c["chunks"], c["groundTruth"]):
                gs.append(([prepare_hand_for_miner(h) for h in g], int(l)))
        by[p.stem] = gs
    return by


def pool(pool_, tgt, rng):
    o = rng.permutation(len(pool_)); h = []; i = 0
    while len(h) < tgt and i < len(o) * 3:
        h += list(pool_[o[i % len(o)]]); i += 1
    return h[:tgt]


def sized(gs, sizes, per, rng):
    out = []; by = {0: [g for g, l in gs if l == 0], 1: [g for g, l in gs if l == 1]}
    for lab in (0, 1):
        pl = by[lab]
        if len(pl) < 2:
            continue
        for sz in sizes:
            for _ in range(per):
                out.append((pool(pl, sz, rng), lab))
    return out


def training_set(by, releases, per=5):
    G, y, dates = [], [], []
    for r in releases:
        for g, l in by[r]:
            G.append(g); y.append(l); dates.append(r)
        for g, l in sized(by[r], [50, 75, 90, 105], per, RNG):
            G.append(g); y.append(l); dates.append(r)
    X = np.vstack([extract_group_features(g) for g in G])
    return X, np.array(y), np.array(dates)


def make_members():
    lgbm = LGBMClassifier(n_estimators=500, num_leaves=31, min_child_samples=30,
                          learning_rate=0.03, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=3.0, n_jobs=4, verbose=-1, random_state=42)
    et = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=10, max_features=0.5,
                              n_jobs=4, random_state=42)
    mlp = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        PCA(n_components=50, random_state=42),
                        MLPClassifier((64,), alpha=2.0, max_iter=700, early_stopping=True,
                                      validation_fraction=0.15, n_iter_no_change=15, random_state=42))
    return [lgbm, et, mlp]


def fit_members(X, y):
    ms = make_members()
    for m in ms:
        m.fit(X[:, COLS], y)
    return ms


def blend_prob(members, X):
    agg = np.zeros(X.shape[0])
    for m, w in zip(members, WEIGHTS):
        agg += w * _rank01(m.predict_proba(X[:, COLS])[:, 1])
    return agg / WEIGHTS.sum()


def serve(members, X, thr):
    return _apply_batch_safety_budget(recenter_scores(blend_prob(members, X), thr), _MAX_POS_FRAC)


def walk_forward(by):
    rels = sorted(by)
    rows = []
    for R in rels[-4:]:
        past = [r for r in rels if r < R]
        Xtr, ytr, _ = training_set(by, past, per=4)
        ms = fit_members(Xtr, ytr)
        ptr = blend_prob(ms, Xtr)
        thr = float(np.quantile(ptr[ytr == 0], 1 - TARGET_FPR))
        te = sized(by[R], [100], 40, RNG)
        Xte = np.vstack([extract_group_features(g) for g, _ in te]); yte = np.array([l for _, l in te])
        sc = _apply_batch_safety_budget(recenter_scores(blend_prob(ms, Xte), thr), _MAX_POS_FRAC)
        rew, det = reward(sc, yte)
        rows.append((R, rew, det["ap_score"], det["bot_recall"], det["human_safety_penalty"]))
    print("walk-forward (train SANITIZED, eval SANITIZED 100-hand groups):")
    for R, rew, ap, rec, saf in rows:
        print(f"  {R}: reward={rew:.4f} ap={ap:.4f} recall={rec:.4f} safety={saf:.3f}")
    print(f"  MEAN reward={np.mean([r[1] for r in rows]):.4f}\n")


def main():
    print("loading + sanitizing benchmark...")
    by = load_sanitized()
    rels = sorted(by)
    print(f"{len(rels)} releases; features={len(FEATURE_NAMES)} model cols={len(COLS)}\n")
    walk_forward(by)

    # OOF threshold (grouped by date) on full training set
    X, y, dates = training_set(by, rels, per=5)
    print(f"deployment training set: {len(y)} groups ({int(y.sum())} bot) incl. size-resamples")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups=dates):
        ms = fit_members(X[tr], y[tr])
        # rank-blend within the held-out fold (mirrors per-batch serving)
        agg = np.zeros(len(te))
        for m, w in zip(ms, WEIGHTS):
            agg += w * _rank01(m.predict_proba(X[te][:, COLS])[:, 1])
        oof[te] = agg / WEIGHTS.sum()
    thr = float(np.quantile(oof[y == 0], 1 - TARGET_FPR))
    print(f"OOF rank-blend AUC={roc_auc_score(y, oof):.4f}  FPR-anchored threshold={thr:.4f}")

    members = fit_members(X, y)
    artifact = {"kind": "rank_blend",
                "members": [{"est": m, "cols": COLS} for m in members],
                "weights": WEIGHTS.tolist(), "threshold": thr,
                "selected": "v11 rank-blend [LGBM,ExtraTrees,PCA-MLP] on sanitized+hero-free features"}
    joblib.dump(artifact, OUT, compress=3)
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
