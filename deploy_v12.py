"""Deploy v12: robust pure-tree rank-blend + approximate-redundancy features.

Full top-10 deep-diff (2026-07-14) showed the benchmark->live SHIFT is the whole
game and the winners are the most ROBUST/de-overfit, not the highest benchmark AP:
  - uid89 (#1, live 0.639): pure-tree soft-vote ET(700,d9)+RF(700,d9)+HGB(700,d9,
    lr.03) weights .45/.25/.30, trained BALANCED, NO MLP. Its edge is pure rank
    ordering; it even generalizes from OLD training data -> the recipe doesn't
    overfit. Depth-9 trees are the regularizer.
  - uid188: rp_* APPROXIMATE-redundancy features (gzip ratio, LZ76 complexity,
    Vendi diversity, pairwise Jaccard, entropy-rate) catch bots that replay
    NEAR-identical lines -- beyond our exact-match signature top/uniq shares.
    (now added to model.py extract_group_features, FEATURE_DIM 262->269.)
  - uid57 / uid26: de-overfitting prior; drop OOD magnitude/table-size features.

v12 vs v11 (LGBM+ET+PCA-MLP): swap to a regularized pure-tree rank-blend copying
uid89's proven engine, on our richer 269-dim sanitized+resampled features (which
now include rp_*). Trees are more robust OOD than the MLP. Batch budget tightened
to 0.125 (uid89). Candidates are compared by honest walk-forward-by-date on
SANITIZED 100-hand eval groups; the best mean-forward-reward config is deployed.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/poker"); sys.path.insert(0, "/root/Poker44-subnet")
from lightgbm import LGBMClassifier
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import importlib, model as M; importlib.reload(M)
from model import (extract_group_features, recenter_scores, _rank01,
                   _apply_batch_safety_budget, FEATURE_NAMES)
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44.score.scoring import reward

DATA = Path("/root/Poker44-subnet/data/benchmark")
OUT = Path("/root/poker/artifacts/poker44_model.joblib")
AMOUNT_SUBSTR = ("size_bb","pot_ratio","roundness","pot_hist","pot_modal","distinct_size","distinct_pot","size_cv","_size_","total_pot")
COLS = [i for i, n in enumerate(FEATURE_NAMES) if not any(s in n for s in AMOUNT_SUBSTR)]
RNG = np.random.default_rng(1212)
TARGET_FPR = 0.05
BUDGET = 0.125   # tightened from v11's 0.16 (uid89 uses 0.125)


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


# ---- candidate ensembles (name -> (build members fn, weights)) --------------
def _et():  return ExtraTreesClassifier(n_estimators=700, max_depth=9, n_jobs=4, random_state=42)
def _rf():  return RandomForestClassifier(n_estimators=700, max_depth=9, n_jobs=4, random_state=42)
def _hgb(): return HistGradientBoostingClassifier(max_iter=700, learning_rate=0.03, max_depth=9,
                                                   l2_regularization=1.0, random_state=42)
def _lgbm():return LGBMClassifier(n_estimators=500, num_leaves=31, min_child_samples=30, learning_rate=0.03,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0,
                                  n_jobs=4, verbose=-1, random_state=42)
def _mlp(): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 PCA(n_components=50, random_state=42),
                                 MLPClassifier((64,), alpha=2.0, max_iter=700, early_stopping=True,
                                               validation_fraction=0.15, n_iter_no_change=15, random_state=42))

CANDIDATES = {
    "uid89_trio":  (lambda: [_et(), _rf(), _hgb()],          np.array([0.45, 0.25, 0.30])),
    "trio+lgbm":   (lambda: [_et(), _rf(), _hgb(), _lgbm()], np.array([0.35, 0.20, 0.30, 0.15])),
    "v11_style":   (lambda: [_lgbm(), _et(), _mlp()],        np.array([0.40, 0.25, 0.35])),
}


def fit_members(build, X, y):
    ms = build()
    for m in ms:
        m.fit(X[:, COLS], y)
    return ms


def blend_prob(members, weights, X):
    agg = np.zeros(X.shape[0])
    for m, w in zip(members, weights):
        agg += w * _rank01(m.predict_proba(X[:, COLS])[:, 1])
    return agg / weights.sum()


def build_folds(by):
    """Precompute per-fold train/eval feature matrices ONCE (shared by candidates)."""
    rels = sorted(by); folds = []
    for R in rels[-4:]:
        past = [r for r in rels if r < R]
        Xtr, ytr, _ = training_set(by, past, per=3)
        te = sized(by[R], [100], 40, RNG)
        Xte = np.vstack([extract_group_features(g) for g, _ in te]); yte = np.array([l for _, l in te])
        folds.append((R, Xtr, ytr, Xte, yte))
    return folds


def eval_candidate(folds, build, weights):
    rows = []
    for R, Xtr, ytr, Xte, yte in folds:
        ms = fit_members(build, Xtr, ytr)
        thr = float(np.quantile(blend_prob(ms, weights, Xtr)[ytr == 0], 1 - TARGET_FPR))
        sc = _apply_batch_safety_budget(recenter_scores(blend_prob(ms, weights, Xte), thr), BUDGET)
        rew, det = reward(sc, yte)
        rows.append((R, rew, det["ap_score"], det["bot_recall"], det["human_safety_penalty"]))
    return rows


def main():
    print("loading + sanitizing benchmark...")
    by = load_sanitized(); rels = sorted(by)
    print(f"{len(rels)} releases; features={len(FEATURE_NAMES)} model cols={len(COLS)} budget={BUDGET}\n")

    print("precomputing walk-forward fold matrices (shared across candidates)...")
    folds = build_folds(by)

    results = {}
    for name, (build, weights) in CANDIDATES.items():
        rows = eval_candidate(folds, build, weights)
        mean = float(np.mean([r[1] for r in rows]))
        results[name] = mean
        print(f"[{name}] walk-forward (sanitized 100-hand eval):")
        for R, rew, ap, rec, saf in rows:
            print(f"    {R}: reward={rew:.4f} ap={ap:.4f} recall={rec:.4f} safety={saf:.3f}")
        print(f"    MEAN reward={mean:.4f}\n")

    best = max(results, key=results.get)
    print(f"=== best candidate: {best} (mean {results[best]:.4f}); v11 baseline was 0.8585 ===\n")
    build, weights = CANDIDATES[best]

    # full-data OOF threshold + AUC
    X, y, dates = training_set(by, rels, per=5)
    print(f"deployment training set: {len(y)} groups ({int(y.sum())} bot) incl. size-resamples")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups=dates):
        ms = fit_members(build, X[tr], y[tr])
        oof[te] = blend_prob(ms, weights, X[te])
    thr = float(np.quantile(oof[y == 0], 1 - TARGET_FPR))
    print(f"OOF rank-blend AUC={roc_auc_score(y, oof):.4f}  FPR-anchored threshold={thr:.4f}")

    members = fit_members(build, X, y)
    artifact = {"kind": "rank_blend",
                "members": [{"est": m, "cols": COLS} for m in members],
                "weights": weights.tolist(), "threshold": thr,
                "selected": f"v12 pure-tree rank-blend [{best}] on sanitized+hero-free+rp features, budget {BUDGET}"}
    joblib.dump(artifact, OUT, compress=3)
    print(f"saved {OUT} ({OUT.stat().st_size/1e6:.1f} MB) -- selected={best}")


if __name__ == "__main__":
    main()
