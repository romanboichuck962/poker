"""Train the "rocket" ensemble — same architecture as UID163's rocket-r2.

  python3 deploy_rocket.py        # writes artifacts_staging/, promotes if it
                                   # beats (or safely trades off vs) the live
                                   # serving artifact.

UID163 ranked #3 on the live leaderboard with a four-component ensemble fused
in LOG-ODDS space (rocket_variant.py has the full rationale). This script
reproduces that exact architecture on OUR feature set and OUR promote gates:

  * stack (StackingClassifier: LGBM+XGBoost+CatBoost+ExtraTrees+RF -> LR meta)
    and mono (soft-voting monotone-XGBoost committee) run on the PH view
    (hero + behavioral features, robust cols from deploy_lgbm_best.py).
  * mlp (soft-voting PCA->MLP committee) runs on the UN view (PH+V2 union).
  * drse (drift-robust subspace ensemble) runs on the V2 view (hero-free +
    cross-hand redundancy features — genuinely decorrelated from PH).

Differences from UID163's train_rocket.py, all deliberate:
  * Blend weights are selected against OUR real poker44.score.scoring.reward(),
    not an approximate/older formula, and against OUR safety-processed scores
    (recenter + min-positives + batch cap) so the number that gates promotion
    matches what actually gets served.
  * Every candidate/serving decision goes through promote_guard.py (never
    overwrites the live artifact unless it clears the FPR ceiling and does not
    regress reward vs what's currently serving).
  * Monotone signs are re-mined per walk-forward fold from that fold's past
    dates only (no label leakage into the held-out date), matching UID163's
    own fix over dragon-0.

Train == serve: every hand is pushed through prepare_hand_for_miner() before
featurization, so the model learns the sanitized distribution it is scored on.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/Poker44-subnet")

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

import rocket_variant as variant
from drse import DRSE
from model import (  # noqa: E402
    FEATURE_NAMES,
    _MAX_POS_FRAC,
    _apply_batch_safety_budget,
    _ensure_min_positives,
    extract_group_features,
    recenter_scores,
)
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402
from promote_guard import (  # noqa: E402
    DEFAULT_MAX_DEPLOY_FPR,
    DEFAULT_REWARD_EPSILON,
    PromoteMetrics,
    promote_candidate,
    read_meta,
)
from rocket_ensemble import PARTS, RocketEnsemble, blend

DATA = Path("/root/Poker44-subnet/data/benchmark")
SERVING_DIR = Path("/root/poker/artifacts")
STAGING_DIR = Path("/root/poker/artifacts_staging")
BACKUPS_DIR = Path("/root/poker/artifacts_backups")
ARTIFACT_NAME = "poker44_model.joblib"
OUT = SERVING_DIR / ARTIFACT_NAME
HISTORY = Path("/root/poker/promote_history.jsonl")

TARGET_FPR = float(os.environ.get("POKER44_TARGET_FPR", "0.05"))
MAX_DEPLOY_FPR = float(os.environ.get("POKER44_MAX_DEPLOY_FPR", str(DEFAULT_MAX_DEPLOY_FPR)))
REWARD_EPSILON = float(os.environ.get("POKER44_REWARD_EPSILON", str(DEFAULT_REWARD_EPSILON)))
FORCE_DEPLOY = os.environ.get("POKER44_FORCE_DEPLOY", "0").strip().lower() in {"1", "true", "yes"}
NJ = int(os.environ.get("POKER44_TRAIN_JOBS", "4"))
WF = int(os.environ.get("POKER44_WF_POINTS", "4"))

# --- two feature views, both already present inside extract_group_features -- #
# PH ("phasberg"-equivalent): hero-keyed + behavioral + policy features, minus
# the drift-prone size/pot columns deploy_lgbm_best.py already flags.
_EXCLUDE_SUBSTR = (
    "size_bb", "pot_ratio", "roundness", "pot_hist", "pot_modal",
    "distinct_size", "distinct_pot", "size_cv", "_size_", "total_pot",
    "stack_bb", "mean_size", "std_size", "min_size", "max_size",
    "coarse_bucket", "coarse_potfrac", "bucket_amt",
)
_EXCLUDE_EXACT = {
    "group_hands", "hand_count", "hf_hand_count", "hf_hand_count_log",
    "n_decisions", "mean_n_players", "std_n_players", "q25_n_players",
    "q75_n_players", "hf_mean_n_actions", "hf_std_n_actions",
}
COLS_PH = [
    i for i, name in enumerate(FEATURE_NAMES)
    if not name.startswith("hf_") and not name.startswith("rp_")
    and name not in _EXCLUDE_EXACT and not any(s in name for s in _EXCLUDE_SUBSTR)
]
# V2 ("hero-free"-equivalent): order-invariant hf_* aggregates + rp_* cross-hand
# redundancy signatures — sanitization-invariant, genuinely decorrelated from PH.
COLS_V2 = [i for i, name in enumerate(FEATURE_NAMES) if name.startswith("hf_") or name.startswith("rp_")]

RNG = np.random.default_rng(20260716)


def load_sanitized():
    by = {}
    for path in sorted(DATA.glob("*.json")):
        groups = []
        payload = json.loads(path.read_text())
        for chunk in payload["chunks"]:
            for group, label in zip(chunk["chunks"], chunk["groundTruth"]):
                groups.append(([prepare_hand_for_miner(hand) for hand in group], int(label)))
        by[path.stem] = groups
    return by


def mat(rows_features: np.ndarray, cols):
    return rows_features[:, cols]


def make_stack():
    cfg = variant.STACK
    base = [
        ("lgb", lgb.LGBMClassifier(n_estimators=cfg["lgb_n"], learning_rate=cfg["lgb_lr"],
                                    num_leaves=cfg["lgb_leaves"], n_jobs=NJ,
                                    random_state=variant.SEED, verbose=-1)),
        ("xgb", xgb.XGBClassifier(n_estimators=cfg["xgb_n"], learning_rate=cfg["xgb_lr"],
                                   max_depth=cfg["xgb_depth"], tree_method="hist", n_jobs=NJ,
                                   random_state=variant.SEED, eval_metric="logloss")),
        ("cat", cb.CatBoostClassifier(iterations=cfg["cat_n"], learning_rate=cfg["cat_lr"],
                                       depth=cfg["cat_depth"], verbose=0, thread_count=NJ,
                                       random_seed=variant.SEED)),
        ("et", ExtraTreesClassifier(n_estimators=cfg["et_n"], max_depth=cfg["et_depth"],
                                     n_jobs=NJ, random_state=variant.SEED,
                                     class_weight="balanced_subsample")),
        ("rf", RandomForestClassifier(n_estimators=cfg["rf_n"], max_depth=cfg["rf_depth"],
                                       n_jobs=NJ, random_state=variant.SEED,
                                       class_weight="balanced_subsample")),
    ]
    return StackingClassifier(
        base, final_estimator=LogisticRegression(C=cfg["meta_c"], max_iter=1000),
        cv=cfg["cv"], n_jobs=1,
    )


def make_mono(signs):
    cfg = variant.MONO
    constraints = "(" + ",".join(str(int(s)) for s in signs) + ")"
    return VotingClassifier(
        [(f"x{i}", xgb.XGBClassifier(
            n_estimators=cfg["n"], learning_rate=cfg["lr"], max_depth=cfg["depth"],
            min_child_weight=cfg["min_child_weight"], subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample"], reg_lambda=cfg["reg_lambda"],
            gamma=cfg["gamma"], tree_method="hist", monotone_constraints=constraints,
            n_jobs=NJ, random_state=variant.SEED + i, eval_metric="logloss"))
         for i in range(cfg["k"])],
        voting="soft", n_jobs=1,
    )


def make_mlp():
    cfg = variant.MLP
    return VotingClassifier(
        [(f"m{i}", Pipeline([
            ("s", StandardScaler()),
            ("p", PCA(cfg["pca"], random_state=variant.SEED)),
            ("m", MLPClassifier(cfg["hidden"], alpha=cfg["alpha"], max_iter=cfg["max_iter"],
                                 early_stopping=True, validation_fraction=0.15,
                                 n_iter_no_change=15, random_state=variant.SEED + i)),
        ])) for i in range(cfg["k"])],
        voting="soft", n_jobs=1,
    )


def make_drse():
    return DRSE(**variant.DRSE)


def fit_components(PH, V2, UN, y, signs, rows):
    return {
        "stack": make_stack().fit(PH[rows], y[rows]),
        "mono": make_mono(signs).fit(PH[rows], y[rows]),
        "mlp": make_mlp().fit(UN[rows], y[rows]),
        "drse": make_drse().fit(V2[rows], y[rows]),
    }


def predict_components(models, PH, V2, UN, rows):
    return {
        "stack": models["stack"].predict_proba(PH[rows])[:, 1],
        "mono": models["mono"].predict_proba(PH[rows])[:, 1],
        "mlp": models["mlp"].predict_proba(UN[rows])[:, 1],
        "drse": models["drse"].predict_proba(V2[rows])[:, 1],
    }


def mine_monotone_signs(PH, y, dates, unique_dates):
    """+1/-1 for PH columns whose per-date Spearman sign is stable, else 0."""
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
        if (len(rhos) >= variant.MONO_MIN_DATES
                and abs(np.mean(rhos)) >= variant.MONO_MIN_RHO
                and (np.sign(rhos) == np.sign(np.mean(rhos))).mean() >= variant.MONO_MIN_AGREE):
            signs.append(int(np.sign(np.mean(rhos))))
        else:
            signs.append(0)
    return signs


def served_scores(prob, threshold):
    return _apply_batch_safety_budget(_ensure_min_positives(recenter_scores(prob, threshold)), _MAX_POS_FRAC)


def select_weights(oof_parts, y_oof, thr):
    """Walk-forward-select blend weights from variant.W_GRID against OUR served
    reward(); the prior keeps its seat unless a rival clears it by the margin."""
    prior = variant.W_PRIOR
    scored = []
    for cand in variant.W_GRID:
        p = blend(oof_parts, cand)
        r, _ = reward(served_scores(p, thr), y_oof)
        scored.append((float(r), cand))
    prior_reward = scored[0][0]
    best_reward, best = max(scored, key=lambda item: item[0])

    for r, cand in scored:
        tag = "prior" if cand is prior else "    "
        print(f"    {tag} {{{', '.join(f'{k}:{cand[k]:.2f}' for k in PARTS)}}} reward={r:.4f}", flush=True)

    if best is not prior and best_reward > prior_reward + variant.W_SELECT_MARGIN:
        print(f"    weights: prior {prior_reward:.4f} -> selected {best_reward:.4f} "
              f"(+{best_reward - prior_reward:.4f} > margin {variant.W_SELECT_MARGIN})", flush=True)
        return dict(best), best_reward, prior_reward
    print(f"    weights: keeping prior ({prior_reward:.4f}); best rival {best_reward:.4f} "
          f"did not clear the {variant.W_SELECT_MARGIN} margin", flush=True)
    return dict(prior), prior_reward, prior_reward


def main() -> None:
    t0 = time.time()
    print("loading + sanitizing benchmark...", flush=True)
    by = load_sanitized()
    releases = sorted(by)
    groups, y_list, dates_list = [], [], []
    for release in releases:
        for group, label in by[release]:
            groups.append(group)
            y_list.append(label)
            dates_list.append(release)
    y = np.asarray(y_list)
    dates = np.asarray(dates_list)
    unique_dates = releases

    print(f"{len(releases)} releases ({releases[0]}..{releases[-1]}); "
          f"n={len(y)} bot={int(y.sum())} ({time.time() - t0:.0f}s)", flush=True)

    baseline = read_meta(SERVING_DIR)
    if baseline:
        print(f"serving baseline: cv_reward={float(baseline.get('cv_reward', -1)):.4f} "
              f"cv_fpr={float(baseline.get('cv_fpr', 1)):.4f} dates={baseline.get('n_dates', '?')}")
    else:
        print("serving baseline: none (first promote allowed with warnings)")
    print(f"gates: target_fpr={TARGET_FPR:.3f} max_deploy_fpr={MAX_DEPLOY_FPR:.3f} "
          f"reward_eps={REWARD_EPSILON:.3f} pos_cap={_MAX_POS_FRAC:.2f} "
          f"wf={WF} force={FORCE_DEPLOY}\n", flush=True)

    all_feats = np.vstack([extract_group_features(g) for g in groups])
    PH = mat(all_feats, COLS_PH)
    V2 = mat(all_feats, COLS_V2)
    UN = np.hstack([V2, PH])
    print(f"features: ph={PH.shape[1]} v2={V2.shape[1]} un={UN.shape[1]} "
          f"({time.time() - t0:.0f}s)\n", flush=True)

    # --- walk-forward: train on the past, predict the next unseen date ----- #
    oof_parts = {name: np.full(len(y), np.nan) for name in PARTS}
    for test_date in unique_dates[-WF:]:
        train_rows = dates < test_date
        test_rows = dates == test_date
        if train_rows.sum() < 60 or len(set(y[train_rows])) < 2:
            continue
        past_dates = [d for d in unique_dates if d < test_date]
        fold_signs = mine_monotone_signs(PH[train_rows], y[train_rows], dates[train_rows], past_dates)
        models = fit_components(PH, V2, UN, y, fold_signs, train_rows)
        preds = predict_components(models, PH, V2, UN, test_rows)
        for name in PARTS:
            oof_parts[name][test_rows] = preds[name]
        print(f"  wf {test_date} | {sum(1 for s in fold_signs if s)} monotone "
              f"(mined on {len(past_dates)} past dates) ({time.time() - t0:.0f}s)", flush=True)

    covered = ~np.isnan(oof_parts["stack"])
    if covered.sum() < 20 or len(set(y[covered])) < 2:
        raise SystemExit("walk-forward produced too little held-out data to fit a blend")
    pooled = {name: oof_parts[name][covered] for name in PARTS}
    y_oof = y[covered]

    # threshold used while selecting weights: fpr-target quantile of the prior
    # blend's human scores on the OOF pool (matches what actually deploys).
    prior_probe = blend(pooled, variant.W_PRIOR)
    thr_probe = float(np.quantile(prior_probe[y_oof == 0], 1 - TARGET_FPR)) if np.any(y_oof == 0) else 0.5

    print("selecting blend weights (walk-forward, scored on OUR reward()):", flush=True)
    weights, _, prior_reward = select_weights(pooled, y_oof, thr_probe)

    oof_prob = blend(pooled, weights)
    thr = float(np.quantile(oof_prob[y_oof == 0], 1 - TARGET_FPR)) if np.any(y_oof == 0) else 0.5
    oof_served = served_scores(oof_prob, thr)
    cv_ap = float(average_precision_score(y_oof, oof_prob))
    cv_auc = float(roc_auc_score(y_oof, oof_prob))
    cv_reward, cv_det = reward(oof_served, y_oof)
    print(f"\nWALK-FORWARD[{WF}d, n={int(covered.sum())}]: cv_ap={cv_ap:.4f} auc={cv_auc:.4f} "
          f"reward={cv_reward:.4f} recall={cv_det['bot_recall']:.3f} "
          f"hard_fpr={cv_det['hard_fpr']:.3f} pos@0.5={cv_det['positive_prediction_rate']:.3f} "
          f"safety={cv_det['human_safety_penalty']:.3f} ({time.time() - t0:.0f}s)\n", flush=True)

    # --- final fit on everything --------------------------------------------#
    all_rows = np.ones(len(y), dtype=bool)
    signs = mine_monotone_signs(PH, y, dates, unique_dates)
    print(f"final fit | {sum(1 for s in signs if s)} monotone constraints over all "
          f"{len(unique_dates)} dates ({time.time() - t0:.0f}s)", flush=True)
    models = fit_components(PH, V2, UN, y, signs, all_rows)
    ens = RocketEnsemble(models["stack"], models["mono"], models["mlp"], models["drse"],
                          COLS_PH, COLS_V2, weights=weights)
    print(f"final fit done ({time.time() - t0:.0f}s)", flush=True)

    import shutil

    import joblib

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "kind": "rocket_logit",
        "ensemble": ens,
        "threshold": thr,
        "selected": (
            f"rocket-{variant.SLUG}: {variant.FRAMEWORK}; weighted-logit fusion("
            + " + ".join(f"{k} {weights[k]:.2f}" for k in PARTS)
            + ") of stack/mono on PH + mlp on UN + drse on V2; walk-forward weights; "
            "targetFPR=5%; remap+smart-minpos+16%cap; adapted from UID163 rocket-r2"
        ),
        "feature_policy": {
            "n_total": len(FEATURE_NAMES),
            "n_ph": len(COLS_PH),
            "n_v2": len(COLS_V2),
        },
        "promote_policy": {
            "target_fpr": TARGET_FPR,
            "max_deploy_fpr": MAX_DEPLOY_FPR,
            "reward_epsilon": REWARD_EPSILON,
            "max_pos_frac": _MAX_POS_FRAC,
        },
    }
    staging_artifact = STAGING_DIR / ARTIFACT_NAME
    joblib.dump(artifact, staging_artifact, compress=3)

    metrics = PromoteMetrics(
        cv_reward=cv_reward,
        cv_fpr=float(cv_det["hard_fpr"]),
        cv_ap=cv_ap,
        cv_bot_recall=float(cv_det["bot_recall"]),
        cv_safety=float(cv_det["human_safety_penalty"]),
        positive_rate=float(cv_det["positive_prediction_rate"]),
        hard_bot_recall=float(cv_det["hard_bot_recall"]),
        n_dates=len(releases),
        walk_forward=[{
            "n_walkforward": int(covered.sum()),
            "weights": weights,
            "weights_prior": variant.W_PRIOR,
            "cv_reward_prior": prior_reward,
        }],
        selected=str(artifact["selected"]),
    )
    promoted, reasons = promote_candidate(
        staging_dir=STAGING_DIR,
        serving_dir=SERVING_DIR,
        artifact_name=ARTIFACT_NAME,
        candidate=metrics,
        backups_dir=BACKUPS_DIR,
        max_deploy_fpr=MAX_DEPLOY_FPR,
        reward_epsilon=REWARD_EPSILON,
        force=FORCE_DEPLOY,
        history_path=HISTORY,
    )
    if promoted:
        print(f"PROMOTED -> {OUT} (reward={metrics.cv_reward:.4f}, fpr={metrics.cv_fpr:.4f}, "
              f"size={OUT.stat().st_size / 1e6:.1f} MB) ({time.time() - t0:.0f}s)")
        if reasons:
            print("promote warnings: " + "; ".join(reasons))
    else:
        print("REJECTED: serving artifact unchanged")
        print("reject reasons: " + "; ".join(reasons))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
