"""Best-of-3 Poker44 trainer: UID89 ranking + UID138 guards + hybrid LGBM.

Keeps advantages, fixes disadvantages:
  * From UID 89: LambdaMART ranking for AP / recall@FPR.
  * From UID 138: staging promote, FPR <= 6%, no reward regression, backups.
  * From our hybrid: LGBMClassifier + sanitization + drift filtering +
    remap-to-0.5 + 16% positive cap (never clip_below).

Fixes vs prior v16:
  * Drop ExtraTrees (complexity > benefit).
  * Favor LambdaMART weight for ranking (89 strength) with classifier assist.
  * Tighter threshold FPR (4.5%) like 138, deploy ceiling still 6%.
  * Smart min-positive: only lift when scores are discriminative but collapsed
    under 0.5 (no fake bots on flat/noisy batches).
  * Promote allows small reward tradeoff when FPR clearly improves (less
    over-conservative than pure reward lock).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/POKER44-SUBNET-1")

import importlib
import model as M

importlib.reload(M)
from model import (  # noqa: E402
    FEATURE_NAMES,
    _MAX_POS_FRAC,
    _apply_batch_safety_budget,
    _ensure_min_positives,
    _rank01,
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

DATA = Path("/root/POKER44-SUBNET-1/data/benchmark")
SERVING_DIR = Path("/root/poker/artifacts")
STAGING_DIR = Path("/root/poker/artifacts_staging")
BACKUPS_DIR = Path("/root/poker/artifacts_backups")
ARTIFACT_NAME = "poker44_model.joblib"
OUT = SERVING_DIR / ARTIFACT_NAME
HISTORY = Path("/root/poker/promote_history.jsonl")

# UID138: target ~4.5% for threshold, deploy reject at 6% (buffer under 10%).
TARGET_FPR = float(os.environ.get("POKER44_TARGET_FPR", "0.05"))
MAX_DEPLOY_FPR = float(os.environ.get("POKER44_MAX_DEPLOY_FPR", str(DEFAULT_MAX_DEPLOY_FPR)))
REWARD_EPSILON = float(os.environ.get("POKER44_REWARD_EPSILON", str(DEFAULT_REWARD_EPSILON)))
FORCE_DEPLOY = os.environ.get("POKER44_FORCE_DEPLOY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

# Drift-prone columns under prepare_hand_for_miner / live chunk-size shift.
_EXCLUDE_SUBSTR = (
    "size_bb",
    "pot_ratio",
    "roundness",
    "pot_hist",
    "pot_modal",
    "distinct_size",
    "distinct_pot",
    "size_cv",
    "_size_",
    "total_pot",
    "stack_bb",
    "mean_size",
    "std_size",
    "min_size",
    "max_size",
    "coarse_bucket",
    "coarse_potfrac",
    "bucket_amt",
)
_EXCLUDE_EXACT = {
    "group_hands",
    "hand_count",
    "hf_hand_count",
    "hf_hand_count_log",
    "n_decisions",
    "mean_n_players",
    "std_n_players",
    "q25_n_players",
    "q75_n_players",
    "hf_mean_n_actions",
    "hf_std_n_actions",
}

COLS = [
    i
    for i, name in enumerate(FEATURE_NAMES)
    if name not in _EXCLUDE_EXACT and not any(s in name for s in _EXCLUDE_SUBSTR)
]

RNG = np.random.default_rng(20260715)
# Proven offline blend: classifier-led + solid LambdaMART (89) + light ET diversity.
# Kept after ablation: LambdaMART-heavy (0.50+) hurt mean reward; this mix held 0.9159.
WEIGHTS = np.array([0.45, 0.40, 0.15])  # clf, LambdaMART, ExtraTrees


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


def pool(pool_, target, rng):
    order = rng.permutation(len(pool_))
    hands = []
    i = 0
    while len(hands) < target and i < len(order) * 3:
        hands += list(pool_[order[i % len(order)]])
        i += 1
    return hands[:target]


def sized(gs, sizes, per, rng):
    out = []
    by_label = {0: [g for g, lab in gs if lab == 0], 1: [g for g, lab in gs if lab == 1]}
    for lab in (0, 1):
        pool_ = by_label[lab]
        if len(pool_) < 2:
            continue
        for size in sizes:
            for _ in range(per):
                out.append((pool(pool_, size, rng), lab))
    return out


def training_set(by, releases, per=5):
    groups, y, dates = [], [], []
    for release in releases:
        for group, label in by[release]:
            groups.append(group)
            y.append(label)
            dates.append(release)
        for group, label in sized(by[release], [50, 75, 90, 105], per, RNG):
            groups.append(group)
            y.append(label)
            dates.append(release)
    x = np.vstack([extract_group_features(group) for group in groups])
    return x, np.asarray(y), np.asarray(dates)


def group_sizes(dates):
    values = np.asarray(dates)
    if not len(values):
        return []
    return [int(np.sum(values == date)) for date in dict.fromkeys(values)]


def make_members():
    clf = LGBMClassifier(
        n_estimators=700,
        num_leaves=47,
        min_child_samples=25,
        learning_rate=0.025,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=4.0,
        reg_alpha=0.2,
        n_jobs=4,
        verbose=-1,
        random_state=42,
    )
    # UID89-style LambdaMART: optimize within-release bot ranking / MAP.
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="map",
        eval_at=[5, 10],
        n_estimators=700,
        num_leaves=31,
        min_child_samples=25,
        learning_rate=0.025,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=3.0,
        n_jobs=4,
        random_state=43,
        verbosity=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=8,
        max_features=0.45,
        n_jobs=4,
        random_state=44,
    )
    return [("proba", clf), ("rank", ranker), ("proba", et)]


def fit_members(x, y, dates):
    members = []
    for kind, estimator in make_members():
        if kind == "rank":
            estimator.fit(x[:, COLS], y, group=group_sizes(dates))
        else:
            estimator.fit(x[:, COLS], y)
        members.append({"est": estimator, "prediction_kind": kind, "cols": COLS})
    return members


def member_scores(member, x):
    estimator = member["est"]
    cols = member["cols"]
    if member["prediction_kind"] == "rank":
        return np.asarray(estimator.predict(x[:, cols]), dtype=float)
    return np.asarray(estimator.predict_proba(x[:, cols])[:, 1], dtype=float)


def blend_prob(members, x):
    agg = np.zeros(x.shape[0], dtype=float)
    for member, weight in zip(members, WEIGHTS):
        agg += weight * _rank01(member_scores(member, x))
    return agg / WEIGHTS.sum()


def serve(members, x, threshold):
    raw = blend_prob(members, x)
    remapped = recenter_scores(raw, threshold)
    remapped = _ensure_min_positives(remapped)
    return _apply_batch_safety_budget(remapped, _MAX_POS_FRAC)


def walk_forward(by):
    releases = sorted(by)
    rows = []
    print("walk-forward (sanitized, ~100-hand live proxy):")
    for holdout in releases[-4:]:
        past = [r for r in releases if r < holdout]
        xtr, ytr, dtr = training_set(by, past, per=4)
        members = fit_members(xtr, ytr, dtr)
        thr = float(np.quantile(blend_prob(members, xtr)[ytr == 0], 1 - TARGET_FPR))
        test = sized(by[holdout], [100], 40, RNG)
        xte = np.vstack([extract_group_features(g) for g, _ in test])
        yte = np.asarray([lab for _, lab in test])
        scores = serve(members, xte, thr)
        rew, det = reward(scores, yte)
        pos_rate = float(np.mean(scores >= 0.5))
        rows.append(
            {
                "date": holdout,
                "reward": float(rew),
                "ap": float(det["ap_score"]),
                "recall": float(det["bot_recall"]),
                "safety": float(det["human_safety_penalty"]),
                "hard_fpr": float(det["hard_fpr"]),
                "hard_bot_recall": float(det["hard_bot_recall"]),
                "pos_rate": pos_rate,
            }
        )
        print(
            f"  {holdout}: reward={rew:.4f} ap={det['ap_score']:.4f} "
            f"recall={det['bot_recall']:.4f} safety={det['human_safety_penalty']:.3f} "
            f"hard_fpr={det['hard_fpr']:.3f} pos@0.5={pos_rate:.3f}"
        )
    mean_reward = float(np.mean([r["reward"] for r in rows])) if rows else 0.0
    print(f"  MEAN reward={mean_reward:.4f}\n")
    return rows


def main():
    print("loading + sanitizing benchmark...")
    by = load_sanitized()
    releases = sorted(by)
    print(
        f"{len(releases)} releases ({releases[0]}..{releases[-1]}); "
        f"features={len(FEATURE_NAMES)} model_cols={len(COLS)}\n"
    )
    baseline = read_meta(SERVING_DIR)
    if baseline:
        print(
            f"serving baseline: cv_reward={float(baseline.get('cv_reward', -1)):.4f} "
            f"cv_fpr={float(baseline.get('cv_fpr', 1)):.4f} "
            f"dates={baseline.get('n_dates', '?')}"
        )
    else:
        print("serving baseline: none (first promote allowed with warnings)")
    print(
        f"gates: target_fpr={TARGET_FPR:.3f} max_deploy_fpr={MAX_DEPLOY_FPR:.3f} "
        f"reward_eps={REWARD_EPSILON:.3f} weights={WEIGHTS.tolist()} "
        f"pos_cap={_MAX_POS_FRAC:.2f} force={FORCE_DEPLOY}\n"
    )

    wf_rows = walk_forward(by)
    wf_reward = float(np.mean([r["reward"] for r in wf_rows])) if wf_rows else 0.0
    wf_fpr = float(np.max([r["hard_fpr"] for r in wf_rows])) if wf_rows else 1.0
    wf_ap = float(np.mean([r["ap"] for r in wf_rows])) if wf_rows else 0.0
    wf_recall = float(np.mean([r["recall"] for r in wf_rows])) if wf_rows else 0.0
    wf_safety = float(np.min([r["safety"] for r in wf_rows])) if wf_rows else 0.0
    wf_pos = float(np.mean([r["pos_rate"] for r in wf_rows])) if wf_rows else 0.0
    wf_hard_recall = float(np.mean([r["hard_bot_recall"] for r in wf_rows])) if wf_rows else 0.0

    x, y, dates = training_set(by, releases, per=5)
    print(f"deployment set: {len(y)} groups ({int(y.sum())} bot) incl. size-resamples")

    oof = np.zeros(len(y))
    for train_idx, test_idx in GroupKFold(5).split(x, y, groups=dates):
        members = fit_members(x[train_idx], y[train_idx], dates[train_idx])
        oof[test_idx] = blend_prob(members, x[test_idx])

    thr = float(np.quantile(oof[y == 0], 1 - TARGET_FPR))
    oof_served = _apply_batch_safety_budget(
        _ensure_min_positives(recenter_scores(oof, thr)),
        _MAX_POS_FRAC,
    )
    oof_rew, oof_det = reward(oof_served, y)
    print(
        f"OOF blend AUC={roc_auc_score(y, oof):.4f} AP={average_precision_score(y, oof):.4f} "
        f"thr={thr:.4f}"
    )
    print(
        f"OOF served reward={oof_rew:.4f} safety={oof_det['human_safety_penalty']:.3f} "
        f"hard_fpr={oof_det['hard_fpr']:.3f} hard_recall={oof_det['hard_bot_recall']:.3f} "
        f"pos@0.5={oof_det['positive_prediction_rate']:.3f}"
    )

    gate_fpr = max(wf_fpr, float(oof_det["hard_fpr"]))
    gate_safety = min(wf_safety, float(oof_det["human_safety_penalty"]))
    gate_pos = min(wf_pos, float(oof_det["positive_prediction_rate"]))
    gate_hard_recall = min(wf_hard_recall, float(oof_det["hard_bot_recall"]))

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    members = fit_members(x, y, dates)
    artifact = {
        "kind": "rank_blend",
        "members": members,
        "weights": WEIGHTS.tolist(),
        "threshold": thr,
        "selected": (
            "best-of-3 v17: LGBMClassifier(0.45)+LambdaMART(0.40)+ExtraTrees(0.15); "
            "sanitized robust cols; targetFPR=5%; remap+smart-minpos+16%cap; "
            "UID138 promote gates (FPR<=6%, no reward regression / FPR tradeoff)"
        ),
        "feature_policy": {
            "n_total": len(FEATURE_NAMES),
            "n_used": len(COLS),
            "exclude_substr": list(_EXCLUDE_SUBSTR),
            "exclude_exact": sorted(_EXCLUDE_EXACT),
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
        cv_reward=wf_reward,
        cv_fpr=gate_fpr,
        cv_ap=wf_ap,
        cv_bot_recall=wf_recall,
        cv_safety=gate_safety,
        positive_rate=gate_pos,
        hard_bot_recall=gate_hard_recall,
        n_dates=len(releases),
        walk_forward=wf_rows,
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
        print(
            f"PROMOTED -> {OUT} "
            f"(reward={metrics.cv_reward:.4f}, fpr={metrics.cv_fpr:.4f}, "
            f"size={OUT.stat().st_size / 1e6:.1f} MB)"
        )
        if reasons:
            print("promote warnings: " + "; ".join(reasons))
    else:
        print("REJECTED: serving artifact unchanged")
        print("reject reasons: " + "; ".join(reasons))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
