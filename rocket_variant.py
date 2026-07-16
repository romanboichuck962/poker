"""rocket-p44 "logit" — mirrors UID163's rocket-r2 architecture for this miner.

UID163 (neradmrinka/benchmark-rocket-poker44-2) ranked #3 on the live leaderboard
with a four-component ensemble fused in LOG-ODDS space rather than by rank. This
file holds the tuned hyper-parameters and blend prior that make that architecture
concrete for OUR feature set (see rocket_features.py for the PH/V2/UN views).

Why fuse logits rather than ranks (UID163's own reasoning, still true here):
 Rank fusion throws away magnitude on purpose — a component that is *certain* a
 chunk is a bot counts exactly as much as one that mildly prefers it. Averaging
 log-odds keeps that confidence, so a rarely-but-strongly-right component can
 carry a chunk over the line. The reward's recall@FPR<=0.05 term is decided at
 the very top of the ranking, precisely where confidence separates and rank
 saturates.

 The cost: logit fusion is only as good as the components' calibration. The
 stack's LR meta-learner is well-calibrated by construction (cv=5 out-of-fold);
 the subspace-averaged DRSE tends to be under-confident, so it carries the
 lowest weight of the four here — the same choice UID163 made.

deploy_rocket.py walk-forward-scores every candidate in W_GRID against OUR
scoring.reward() and only moves off the prior when the data pays for it by more
than W_SELECT_MARGIN.
"""

SLUG = "p44r1"
FAMILY = "logit-fusion"
VERSION = "1.0"
FRAMEWORK = "sklearn-stack+monotone-xgb+pca-mlp+drse/weighted-logit-fusion"
SUMMARY = (
    "Weighted log-odds fusion of four decorrelated components over two feature "
    "views (ph=hero+behavioral, v2=hero-free+redundancy) + their union. "
    "Blend weights selected by walk-forward. Adapted from UID163 rocket-r2."
)

# --- blend weights (UID163's published prior; re-selected by walk-forward) -- #
W_PRIOR = {"stack": 0.30, "mono": 0.22, "mlp": 0.28, "drse": 0.20}
W_GRID = [
    W_PRIOR,
    {"stack": 0.30, "mono": 0.20, "mlp": 0.30, "drse": 0.20},
    {"stack": 0.34, "mono": 0.24, "mlp": 0.26, "drse": 0.16},
    {"stack": 0.28, "mono": 0.18, "mlp": 0.32, "drse": 0.22},
    {"stack": 0.32, "mono": 0.26, "mlp": 0.24, "drse": 0.18},
    {"stack": 0.26, "mono": 0.22, "mlp": 0.30, "drse": 0.22},
    {"stack": 0.36, "mono": 0.20, "mlp": 0.28, "drse": 0.16},
]
# Reward gain required before abandoning the prior; the walk-forward pool is
# only a few dates deep, so a hair's-breadth win is noise, not evidence.
W_SELECT_MARGIN = 0.002

SEED = 2

STACK = dict(
    lgb_n=650, lgb_lr=0.025, lgb_leaves=95,
    xgb_n=550, xgb_lr=0.04, xgb_depth=5,
    cat_n=650, cat_lr=0.035, cat_depth=7,
    et_n=600, et_depth=16,
    rf_n=500, rf_depth=16,
    meta_c=0.5, cv=5,
)
MONO = dict(k=3, n=500, lr=0.03, depth=5, min_child_weight=3,
            subsample=0.85, colsample=0.75, reg_lambda=2.0, gamma=0.3)
MLP = dict(k=3, pca=60, hidden=(96,), alpha=1.0, max_iter=800)
DRSE = dict(n=10, ff=0.75, seed=22)

# --- monotone-constraint mining --------------------------------------------- #
MONO_MIN_DATES = 4
MONO_MIN_AGREE = 0.70
MONO_MIN_RHO = 0.04
