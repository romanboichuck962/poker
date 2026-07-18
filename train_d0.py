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
CAP_DIR = Path("/root/poker/captures")
ART = Path(__file__).resolve().parent / "artifacts"
NJ = 4
SEED = 172
TARGET_FPR = 0.04          # their stacked.py conformal target
MAX_POS_FRAC = 0.16        # their infer.py default budget
HOLDOUT_DAYS = 2
WINDOW_BOT_FRAC = 0.20     # live-realistic request composition
MONO_MIN_DATES, MONO_MIN_RHO, MONO_MIN_AGREE = 4, 0.04, 0.70
Z_MAX = 5.0                # live-OOD ablation threshold (pillar 2)
WF_DATES = 4               # walk-forward dates for blend-weight selection
W_SELECT_MARGIN = 0.003    # reward gain required to abandon UID172's prior

# UID172's published prior + a grid of decorrelated variations around it.
W_PRIOR = dict(W)          # {stack .28, mono .24, mlp .28, drse .20}
W_GRID = [
    W_PRIOR,
    {"stack": 0.30, "mono": 0.20, "mlp": 0.30, "drse": 0.20},
    {"stack": 0.26, "mono": 0.22, "mlp": 0.30, "drse": 0.22},
    {"stack": 0.24, "mono": 0.18, "mlp": 0.34, "drse": 0.24},
    {"stack": 0.30, "mono": 0.26, "mlp": 0.26, "drse": 0.18},
    {"stack": 0.32, "mono": 0.22, "mlp": 0.28, "drse": 0.18},
    {"stack": 0.22, "mono": 0.20, "mlp": 0.32, "drse": 0.26},
    {"stack": 0.28, "mono": 0.20, "mlp": 0.32, "drse": 0.20},
]

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


def fit_members(PH, V2, y, dates, rows, verbose=True):
    UN = np.hstack([V2, PH])
    unique = sorted(set(dates[rows]))
    signs = mine_monotone_signs(PH[rows], y[rows], dates[rows], unique)
    if verbose:
        print(f"  mono signs: +{sum(1 for s in signs if s > 0)} "
              f"-{sum(1 for s in signs if s < 0)} 0:{sum(1 for s in signs if s == 0)}")
    t0 = time.time()
    stack = make_stack().fit(PH[rows], y[rows])
    mono = make_mono(signs).fit(PH[rows], y[rows])
    mlp = make_mlp().fit(UN[rows], y[rows])
    drse = DRSE(**DRSE_CFG).fit(V2[rows], y[rows])
    if verbose:
        print(f"  members fit {time.time()-t0:.0f}s")
    return {"stack": stack, "mono": mono, "mlp": mlp, "drse": drse}


def member_probs(members, PH, V2, rows):
    UN = np.hstack([V2, PH])
    return {
        "stack": members["stack"].predict_proba(PH[rows])[:, 1],
        "mono": members["mono"].predict_proba(PH[rows])[:, 1],
        "mlp": members["mlp"].predict_proba(UN[rows])[:, 1],
        "drse": members["drse"].predict_proba(V2[rows])[:, 1],
    }


def fit_draco(PH, V2, y, dates, rows, cols_ph, cols_v2, weights):
    members = fit_members(PH, V2, y, dates, rows)
    return D0Draco(members["stack"], members["mono"], members["mlp"], members["drse"],
                   cols_ph, cols_v2, weights=weights)


PARTS = ("stack", "mono", "mlp", "drse")


def _rank01(s):
    s = np.asarray(s, dtype=float)
    n = s.size
    if n <= 1:
        return np.zeros(n)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (n - 1)


def rank_blend(part_probs, weights):
    """Replicate D0Draco.score's rank-average for a set of per-member probs."""
    w = weights
    r = sum(w[p] * _rank01(part_probs[p]) for p in PARTS)
    return r / sum(w[p] for p in PARTS)


def load_captures():
    caps = []
    for p in sorted(CAP_DIR.glob("*.json")):
        h = json.loads(p.read_text())
        if isinstance(h, dict):
            h = h.get("hands") or h.get("chunk") or []
        if h:
            caps.append(h)
    return caps


def compute_ood_masks(cols_ph, cols_v2):
    """Per-view boolean masks of columns with live-vs-benchmark z > Z_MAX.

    live = captured validator chunks; bench = size-matched pooled sanitized
    benchmark groups. Uses captures (pillar 2) to drop structurally-OOD columns
    from both views in training AND serving."""
    caps = load_captures()
    if not caps:
        print("no captures; skipping OOD ablation", flush=True)
        return np.zeros(len(cols_ph), bool), np.zeros(len(cols_v2), bool)
    rng = np.random.default_rng(20260718)
    target = int(np.median([len(c) for c in caps]))
    bylab = {0: [], 1: []}
    for path in sorted(DATA.glob("*.json")):
        payload = json.loads(path.read_text())
        for chunk in payload["chunks"]:
            for group, label in zip(chunk["chunks"], chunk["groundTruth"]):
                bylab[int(label)].append([prepare_hand_for_miner(h) for h in group])

    def pool_to(pool, t):
        order = rng.permutation(len(pool)); hands, i = [], 0
        while len(hands) < t and i < len(order) * 3:
            hands += list(pool[order[i % len(order)]]); i += 1
        return hands[:t]

    bench = [pool_to(bylab[l], target) for l in (0, 1) for _ in range(200)]

    def mat(chunks, fn, cols):
        return np.array([[float(d.get(c, 0.0)) for c in cols]
                         for d in (fn(x) for x in chunks)], dtype=float)

    masks = {}
    for name, fn, cols in (("ph", phasberg_dict, cols_ph), ("v2", v2_dict, cols_v2)):
        L = mat(caps, fn, cols); B = mat(bench, fn, cols)
        z = np.abs(L.mean(0) - B.mean(0)) / (B.std(0) + 1e-9)
        masks[name] = z > Z_MAX
        print(f"  OOD ablation {name}: {int(masks[name].sum())}/{len(cols)} cols z>{Z_MAX}",
              flush=True)
    return masks["ph"], masks["v2"]


def select_weights(oof_parts, y_oof, covered_idx, seed=SEED):
    """Walk-forward-select blend weights on live-realistic request windows.

    Each candidate is scored exactly as it serves: rank-blend within a
    100-chunk 20%-bot window, deploy-threshold remap (per-candidate human
    quantile at 1-TARGET_FPR on the pooled OOF), 16% budget, our reward()."""
    rng = np.random.default_rng(seed + 999)
    pos = covered_idx[y_oof == 1]
    neg = covered_idx[y_oof == 0]
    k_pos = max(1, int(round(100 * WINDOW_BOT_FRAC)))
    windows = []
    for _ in range(150):
        p = rng.choice(pos, k_pos, replace=len(pos) < k_pos)
        n = rng.choice(neg, 100 - k_pos, replace=len(neg) < 100 - k_pos)
        idx = np.concatenate([p, n]); rng.shuffle(idx)
        windows.append(idx)
    pos_set = set(pos.tolist())
    results = []
    for cand in W_GRID:
        pooled_blend = rank_blend({p: oof_parts[p] for p in PARTS}, cand)
        human = pooled_blend[np.isin(covered_idx, neg)]
        thr = float(np.quantile(human, 1 - TARGET_FPR)) if human.size else 0.5
        rewards = []
        for idx in windows:
            parts = {p: oof_parts[p][np.searchsorted(covered_idx, idx)] for p in PARTS}
            raw = rank_blend(parts, cand)
            served = apply_budget(remap_to_threshold(raw, thr))
            labels = np.array([1 if i in pos_set else 0 for i in idx])
            value, _ = validator_reward(served, labels)
            rewards.append(float(value))
        r = np.asarray(rewards)
        results.append((cand, float(r.mean()), float(np.quantile(r, 0.10))))
    prior_reward = results[0][1]
    best = max(results, key=lambda t: t[1])
    for cand, mean, p10 in results:
        tag = "prior" if cand is W_GRID[0] else "     "
        print(f"    {tag} {{{', '.join(f'{k}:{cand[k]:.2f}' for k in PARTS)}}} "
              f"reward={mean:.4f} p10={p10:.4f}", flush=True)
    if best[0] is not W_GRID[0] and best[1] > prior_reward + W_SELECT_MARGIN:
        print(f"    weights: prior {prior_reward:.4f} -> selected {best[1]:.4f} "
              f"(+{best[1]-prior_reward:.4f} > {W_SELECT_MARGIN})", flush=True)
        return dict(best[0])
    print(f"    weights: keeping prior ({prior_reward:.4f}); best rival {best[1]:.4f} "
          f"did not clear {W_SELECT_MARGIN}", flush=True)
    return dict(W_GRID[0])


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

    # PILLAR 2 — measured live-OOD ablation from the captures: zero the
    # structurally out-of-distribution columns in BOTH views, in train and (via
    # the stored masks) serve, so no member can rank live chunks on them.
    print("measuring live-OOD from captures...", flush=True)
    mask_ph, mask_v2 = compute_ood_masks(cols_ph, cols_v2)
    PH = PH.copy(); PH[:, mask_ph] = 0.0
    V2 = V2.copy(); V2[:, mask_v2] = 0.0

    holdout_dates = unique_dates[-HOLDOUT_DAYS:]
    train_rows = np.flatnonzero(~np.isin(dates, holdout_dates))
    hold_rows = np.flatnonzero(np.isin(dates, holdout_dates))
    print(f"{len(y)} chunks, {len(unique_dates)} dates | holdout {holdout_dates} "
          f"({hold_rows.size} chunks)", flush=True)

    # PILLAR 4 / weight control — walk-forward member OOF, then select blend
    # weights on live-realistic windows scored exactly as they serve.
    print(f"walk-forward member OOF over last {WF_DATES} dates...", flush=True)
    wf_dates = unique_dates[-WF_DATES:]
    oof_parts = {p: np.full(len(y), np.nan) for p in PARTS}
    for td in wf_dates:
        tr = np.flatnonzero(dates < td)
        te = np.flatnonzero(dates == td)
        if tr.size < 60 or len(set(y[tr])) < 2:
            continue
        members = fit_members(PH, V2, y, dates, tr, verbose=False)
        probs = member_probs(members, PH, V2, te)
        for p in PARTS:
            oof_parts[p][te] = probs[p]
        print(f"  wf {td} (train={tr.size} test={te.size})", flush=True)
    covered = np.flatnonzero(~np.isnan(oof_parts["stack"]))
    y_oof = y[covered]
    oof_cov = {p: oof_parts[p][covered] for p in PARTS}
    print("selecting blend weights (walk-forward, live-geometry windows):", flush=True)
    weights = select_weights(oof_cov, y_oof, covered)

    print("fitting pre-holdout D0Draco (selected weights)...", flush=True)
    ens_h = fit_draco(PH, V2, y, dates, train_rows, cols_ph, cols_v2, weights)
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

    print("fitting FINAL D0Draco on all dates (selected weights)...", flush=True)
    ens = fit_draco(PH, V2, y, dates, np.arange(len(y)), cols_ph, cols_v2, weights)

    artifact = {
        "kind": "d0_draco",
        "ens": ens,
        "threshold": thr,
        "max_pos_frac": MAX_POS_FRAC,
        "weights": dict(weights),
        "weights_prior": dict(W_PRIOR),
        "cols_ph": cols_ph,
        "cols_v2": cols_v2,
        "ood_mask_ph": mask_ph,
        "ood_mask_v2": mask_v2,
        "z_max": Z_MAX,
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
    report = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
              for k, v in artifact.items() if k not in ("ens",)}
    report["n_ood_ph"] = int(mask_ph.sum())
    report["n_ood_v2"] = int(mask_v2.sum())
    (ART / "d0_train_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("saved artifacts/d0_train_report.json")


if __name__ == "__main__":
    main()
