"""Train D0Draco (UID172 poker44-benchmark-huge-2 method) on our benchmark.

UID172's repo ships the exact serving code (d0/ is a verbatim port: two feature
views, the 4-member rank-blend, remap + batch budget) but no trainer, so this
trainer builds the members to their published spec strings:

  stack: 56-leaf benchmark-supervised StackingClassifier, cv=4
         (d0-family base mix: LGBM+XGB+CatBoost+ExtraTrees+RF -> LogisticRegression)
  mono : 3-seed depth-5 monotone XGBoost committee, signs mined per-date
  mlp  : 4-seed PCA-44 MLP(64,32) committee on the v2+phasberg UNION
  drse : DRSE(n=8, feature-fraction 0.75) on the v2 view
  blend: fixed rank-average weights {stack .28, mono .24, mlp .28, drse .20}

Protocol: sanitize every hand (train == serve), hold out the last two release
dates, fit members on earlier dates, evaluate holdout reward on live-realistic
20%-bot 100-chunk request windows (each window rank-blended independently, as
serving does), pick the deploy threshold from holdout human scores at
target FPR 4% (their conformal target), then refit on all dates and save the
artifact with that threshold.

Run: PYTHONPATH=/root/poker242:/root/POKER44-SUBNET-1 python train_d0.py
"""

from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from poker44.score.scoring import reward as validator_reward
from poker44.validator.payload_view import prepare_hand_for_miner

from d0.drse import DRSE
from d0.ensemble import D0Draco, W
from d0.views import phasberg_dict, v2_dict

DATA = Path("/root/POKER44-SUBNET-1/data/benchmark")
ART = Path(__file__).resolve().parent / "artifacts"
NJ = 4
SEED = 172
TARGET_FPR = 0.04          # their stacked.py conformal target
MAX_POS_FRAC = 0.16        # their infer.py default budget
HOLDOUT_DAYS = 2
WINDOW_BOT_FRAC = 0.20     # live-realistic request composition
MONO_MIN_DATES, MONO_MIN_RHO, MONO_MIN_AGREE = 4, 0.04, 0.70

STACK_CFG = dict(lgb_n=600, lgb_lr=0.03, lgb_leaves=56,
                 xgb_n=550, xgb_lr=0.04, xgb_depth=5,
                 cat_n=600, cat_lr=0.035, cat_depth=6,
                 et_n=600, et_depth=14, rf_n=500, rf_depth=14,
                 meta_c=0.5, cv=4)
MONO_CFG = dict(k=3, n=500, lr=0.03, depth=5, min_child_weight=5,
                subsample=0.8, colsample=0.7, reg_lambda=2.0, gamma=0.3)
MLP_CFG = dict(k=4, pca=44, hidden=(64, 32), alpha=1.0, max_iter=400)
DRSE_CFG = dict(n=8, ff=0.75, seed=SEED)


def load_sanitized():
    by = {}
    for path in sorted(DATA.glob("*.json")):
        groups = []
        payload = json.loads(path.read_text())
        for chunk in payload["chunks"]:
            for group, label in zip(chunk["chunks"], chunk["groundTruth"]):
                groups.append(([prepare_hand_for_miner(h) for h in group], int(label)))
        by[path.stem] = groups
    return by


def featurize(groups_by_date):
    cache = ART / "d0_features.npz"
    dates_all, y_all = [], []
    chunks = []
    for date in sorted(groups_by_date):
        for hands, label in groups_by_date[date]:
            chunks.append(hands)
            dates_all.append(date)
            y_all.append(label)
    probe_ph = sorted(phasberg_dict(chunks[0]).keys())
    probe_v2 = sorted(v2_dict(chunks[0]).keys())
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if (list(z["cols_ph"]) == probe_ph and list(z["cols_v2"]) == probe_v2
                and z["PH"].shape[0] == len(chunks)):
            print(f"feature cache hit: {cache}")
            return (z["PH"], z["V2"], np.asarray(y_all), np.asarray(dates_all),
                    probe_ph, probe_v2)
    t0 = time.time()
    PH = np.array([[float(d.get(c, 0.0)) for c in probe_ph]
                   for d in (phasberg_dict(c) for c in chunks)], dtype=float)
    V2 = np.array([[float(d.get(c, 0.0)) for c in probe_v2]
                   for d in (v2_dict(c) for c in chunks)], dtype=float)
    PH = np.nan_to_num(PH, nan=0.0, posinf=0.0, neginf=0.0)
    V2 = np.nan_to_num(V2, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"featurized {len(chunks)} chunks in {time.time()-t0:.0f}s "
          f"PH={PH.shape} V2={V2.shape}")
    np.savez_compressed(cache, PH=PH, V2=V2, cols_ph=probe_ph, cols_v2=probe_v2)
    return PH, V2, np.asarray(y_all), np.asarray(dates_all), probe_ph, probe_v2


def mine_monotone_signs(PH, y, dates, unique_dates):
    signs = []
    for j in range(PH.shape[1]):
        rhos = []
        for d in unique_dates:
            m = dates == d
            if m.sum() < 8 or len(set(y[m])) < 2:
                continue
            rho = spearmanr(PH[m, j], y[m]).correlation
            if not np.isnan(rho):
                rhos.append(rho)
        if (len(rhos) >= MONO_MIN_DATES
                and abs(np.mean(rhos)) >= MONO_MIN_RHO
                and (np.sign(rhos) == np.sign(np.mean(rhos))).mean() >= MONO_MIN_AGREE):
            signs.append(int(np.sign(np.mean(rhos))))
        else:
            signs.append(0)
    return signs


def make_stack():
    c = STACK_CFG
    base = [
        ("lgb", lgb.LGBMClassifier(n_estimators=c["lgb_n"], learning_rate=c["lgb_lr"],
                                   num_leaves=c["lgb_leaves"], n_jobs=NJ,
                                   random_state=SEED, verbose=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=c["xgb_n"], learning_rate=c["xgb_lr"],
                                  max_depth=c["xgb_depth"], tree_method="hist",
                                  n_jobs=NJ, random_state=SEED, eval_metric="logloss")),
        ("cat", cb.CatBoostClassifier(iterations=c["cat_n"], learning_rate=c["cat_lr"],
                                      depth=c["cat_depth"], verbose=0,
                                      thread_count=NJ, random_seed=SEED)),
        ("et", ExtraTreesClassifier(n_estimators=c["et_n"], max_depth=c["et_depth"],
                                    n_jobs=NJ, random_state=SEED,
                                    class_weight="balanced_subsample")),
        ("rf", RandomForestClassifier(n_estimators=c["rf_n"], max_depth=c["rf_depth"],
                                      n_jobs=NJ, random_state=SEED,
                                      class_weight="balanced_subsample")),
    ]
    return StackingClassifier(base,
                              final_estimator=LogisticRegression(C=c["meta_c"], max_iter=1000),
                              cv=c["cv"], n_jobs=1)


def make_mono(signs):
    c = MONO_CFG
    constraints = "(" + ",".join(str(int(s)) for s in signs) + ")"
    return VotingClassifier(
        [(f"x{i}", xgb.XGBClassifier(
            n_estimators=c["n"], learning_rate=c["lr"], max_depth=c["depth"],
            min_child_weight=c["min_child_weight"], subsample=c["subsample"],
            colsample_bytree=c["colsample"], reg_lambda=c["reg_lambda"],
            gamma=c["gamma"], tree_method="hist", monotone_constraints=constraints,
            n_jobs=NJ, random_state=SEED + i, eval_metric="logloss"))
         for i in range(c["k"])],
        voting="soft", n_jobs=1)


def make_mlp():
    c = MLP_CFG
    return VotingClassifier(
        [(f"m{i}", Pipeline([
            ("s", StandardScaler()),
            ("p", PCA(c["pca"], random_state=SEED)),
            ("m", MLPClassifier(c["hidden"], alpha=c["alpha"], max_iter=c["max_iter"],
                                early_stopping=True, validation_fraction=0.15,
                                n_iter_no_change=15, random_state=SEED + i)),
        ])) for i in range(c["k"])],
        voting="soft", n_jobs=1)


def fit_draco(PH, V2, y, dates, rows, cols_ph, cols_v2):
    UN = np.hstack([V2, PH])
    unique = sorted(set(dates[rows]))
    signs = mine_monotone_signs(PH[rows], y[rows], dates[rows], unique)
    print(f"  mono signs: +{sum(1 for s in signs if s > 0)} "
          f"-{sum(1 for s in signs if s < 0)} 0:{sum(1 for s in signs if s == 0)}")
    t0 = time.time()
    stack = make_stack().fit(PH[rows], y[rows]); print(f"  stack fit {time.time()-t0:.0f}s")
    t0 = time.time()
    mono = make_mono(signs).fit(PH[rows], y[rows]); print(f"  mono fit {time.time()-t0:.0f}s")
    t0 = time.time()
    mlp = make_mlp().fit(UN[rows], y[rows]); print(f"  mlp fit {time.time()-t0:.0f}s")
    t0 = time.time()
    drse = DRSE(**DRSE_CFG).fit(V2[rows], y[rows]); print(f"  drse fit {time.time()-t0:.0f}s")
    return D0Draco(stack, mono, mlp, drse, cols_ph, cols_v2, weights=W)


def remap_to_threshold(p, t):
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


def apply_budget(scores, max_frac=MAX_POS_FRAC):
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0 or max_frac >= 1.0:
        return s
    k = max(1, int(np.floor(max_frac * n)))
    positive = np.flatnonzero(s >= 0.5)
    if positive.size <= k:
        return s
    order = positive[np.argsort(-s[positive], kind="stable")]
    squeeze = order[k:]
    below = s[s < 0.5]
    lo = min(float(below.max()) if below.size else 0.45, 0.499)
    span = 0.5 - lo
    out = s.copy()
    m = squeeze.size
    for rank, idx in enumerate(squeeze):
        out[idx] = lo + span * (m - rank) / (m + 1.0)
    return np.clip(out, 0.0, 1.0)


def window_rewards(ens, PH, V2, y, idx_pool, thr, n_windows=100, size=100, seed=0):
    """Score sampled request windows exactly as serving does (rank within window)."""
    rng = np.random.default_rng(seed)
    pos = idx_pool[y[idx_pool] == 1]
    neg = idx_pool[y[idx_pool] == 0]
    k_pos = max(1, int(round(size * WINDOW_BOT_FRAC)))
    rewards = []
    for _ in range(n_windows):
        p = rng.choice(pos, k_pos, replace=len(pos) < k_pos)
        n = rng.choice(neg, size - k_pos, replace=len(neg) < size - k_pos)
        idx = np.concatenate([p, n]); rng.shuffle(idx)
        raw = ens.score(PH[idx], V2[idx])
        served = apply_budget(remap_to_threshold(np.asarray(raw), thr))
        value, _ = validator_reward(served, y[idx])
        rewards.append(float(value))
    r = np.asarray(rewards)
    return dict(mean=float(r.mean()), p10=float(np.quantile(r, 0.10)),
                minimum=float(r.min()), std=float(r.std()), zeros=int((r == 0).sum()))


def main():
    ART.mkdir(exist_ok=True)
    print("loading + sanitizing benchmark (train == serve)...", flush=True)
    by = load_sanitized()
    PH, V2, y, dates, cols_ph, cols_v2 = featurize(by)
    unique_dates = sorted(set(dates))
    holdout_dates = unique_dates[-HOLDOUT_DAYS:]
    train_rows = np.flatnonzero(~np.isin(dates, holdout_dates))
    hold_rows = np.flatnonzero(np.isin(dates, holdout_dates))
    print(f"{len(y)} chunks, {len(unique_dates)} dates | holdout {holdout_dates} "
          f"({hold_rows.size} chunks)", flush=True)

    print("fitting pre-holdout D0Draco...", flush=True)
    ens_h = fit_draco(PH, V2, y, dates, train_rows, cols_ph, cols_v2)

    # Deploy threshold: per-date rank-blend scores on the holdout, human
    # quantile at 1 - TARGET_FPR (their conformal 4% target). Rank output is
    # batch-relative, so score each date as its own request batch.
    hold_scores = np.empty(hold_rows.size)
    for d in holdout_dates:
        m = dates[hold_rows] == d
        idx = hold_rows[m]
        hold_scores[m] = ens_h.score(PH[idx], V2[idx])
    thr = float(np.quantile(hold_scores[y[hold_rows] == 0], 1 - TARGET_FPR))
    print(f"deploy threshold (human q{1-TARGET_FPR:.2f} on holdout): {thr:.4f}", flush=True)

    stats = window_rewards(ens_h, PH, V2, y, hold_rows, thr, seed=SEED)
    print(f"HOLDOUT reward @100-chunk {int(WINDOW_BOT_FRAC*100)}%-bot windows: "
          f"mean={stats['mean']:.4f} p10={stats['p10']:.4f} min={stats['minimum']:.4f} "
          f"zeros={stats['zeros']}/100", flush=True)

    print("fitting FINAL D0Draco on all dates...", flush=True)
    ens = fit_draco(PH, V2, y, dates, np.arange(len(y)), cols_ph, cols_v2)

    artifact = {
        "kind": "d0_draco",
        "ens": ens,
        "threshold": thr,
        "max_pos_frac": MAX_POS_FRAC,
        "weights": dict(W),
        "cols_ph": cols_ph,
        "cols_v2": cols_v2,
        "holdout_dates": holdout_dates,
        "holdout_window_rewards": stats,
        "target_fpr": TARGET_FPR,
        "window_bot_frac": WINDOW_BOT_FRAC,
        "member_configs": dict(stack=STACK_CFG, mono=MONO_CFG, mlp=MLP_CFG, drse=DRSE_CFG),
        "training_dates": unique_dates,
        "training_count": int(len(y)),
    }
    out = ART / "poker44_model.joblib"
    with out.open("wb") as fh:
        pickle.dump(artifact, fh, protocol=4)
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB)")
    report = {k: v for k, v in artifact.items() if k not in ("ens",)}
    (ART / "d0_train_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("saved artifacts/d0_train_report.json")


if __name__ == "__main__":
    main()
