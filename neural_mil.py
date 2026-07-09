"""Attention-MIL sequence/set model vs the GBDT stack — fair CV comparison + blend.

A chunk group is a SET of hands. This model encodes each hand's per-hand feature
vector with a small MLP, pools hands with masked attention (learned aggregation
instead of fixed mean/std/quantile), and classifies. Evaluated by the same
date-grouped CV per-window reward as the tabular models, then blended with the
GBDT stack to test for complementary signal. Deploys nothing — reports metrics.
"""

from __future__ import annotations

import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier, StackingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/Poker44-subnet")
from model import _HAND_KEYS, _hand_features, recenter_scores  # noqa: E402
from robust_select import load_cache, per_window_reward  # noqa: E402
from train import fpr_threshold  # noqa: E402

torch.manual_seed(42)
torch.set_num_threads(4)
DATA = Path("/root/Poker44-subnet/data/benchmark")
MAXH = 40
FDIM = len(_HAND_KEYS)

LGBM = dict(colsample_bytree=0.5305244107522612, learning_rate=0.021083935905732085,
            max_depth=6, min_child_samples=30, n_estimators=500, num_leaves=60,
            reg_alpha=0.12172349144489844, reg_lambda=0.638055406926429,
            subsample=0.7074282442414331, n_jobs=1, verbose=-1, random_state=42)
CAT = dict(iterations=700, learning_rate=0.017484424311711072, depth=4,
           l2_leaf_reg=1.5516113747748745, random_strength=1.095373270842297,
           subsample=0.702404726795441, thread_count=1, verbose=0,
           allow_writing_files=False, random_seed=42)
XGB = dict(colsample_bytree=0.6399081830761646, gamma=0.756762652051251,
           learning_rate=0.016521123436765234, max_depth=6, min_child_weight=6,
           n_estimators=600, reg_alpha=0.247319089444343, reg_lambda=0.5056730965902335,
           subsample=0.887128901583661, tree_method="hist", eval_metric="logloss",
           n_jobs=1, random_state=42)


def build_stack():
    return StackingClassifier(
        estimators=[("lgbm", LGBMClassifier(**LGBM)), ("cat", CatBoostClassifier(**CAT)),
                    ("xgb", XGBClassifier(**XGB)),
                    ("extra", ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2,
                                                   n_jobs=4, random_state=42))],
        final_estimator=LogisticRegression(max_iter=2000, C=1.0),
        stack_method="predict_proba", cv=4, n_jobs=4)


def build_hand_cache():
    """Per-group padded per-hand feature tensor [N, MAXH, FDIM] + mask [N, MAXH]."""
    Xs, masks, y, dates = [], [], [], []
    for path in sorted(DATA.glob("*.json")):
        d = json.loads(path.read_text())
        for c in d["chunks"]:
            for group, label in zip(c["chunks"], c["groundTruth"]):
                rows = [f for f in (_hand_features(h) for h in group) if f is not None]
                mat = np.zeros((MAXH, FDIM), dtype=np.float32)
                msk = np.zeros(MAXH, dtype=np.float32)
                for i, r in enumerate(rows[:MAXH]):
                    mat[i] = [r[k] for k in _HAND_KEYS]
                    msk[i] = 1.0
                Xs.append(mat); masks.append(msk); y.append(int(label)); dates.append(path.stem)
    return np.stack(Xs), np.stack(masks), np.array(y), np.array(dates)


class AttnMIL(nn.Module):
    def __init__(self, fdim, hidden=64, p=0.35):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(fdim, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Dropout(p + 0.1), nn.Linear(hidden * 2, 1))

    def forward(self, x, mask):
        h = self.enc(x)                                    # [B,H,D]
        a = self.attn(h).squeeze(-1)                       # [B,H]
        a = a.masked_fill(mask == 0, -1e9).softmax(-1)     # masked attention
        attn_pool = (h * a.unsqueeze(-1)).sum(1)           # [B,D]
        mean_pool = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([attn_pool, mean_pool], -1)).squeeze(-1)


def train_fold(Xtr, Mtr, ytr, Xva, Mva, yva, epochs=120, patience=15):
    dev = "cpu"
    model = AttnMIL(FDIM).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr, Mtr = torch.tensor(Xtr), torch.tensor(Mtr)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xva, Mva = torch.tensor(Xva), torch.tensor(Mva)
    best_auc, best_state, bad = -1, None, 0
    n = len(Xtr); bs = 128
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xtr[idx], Mtr[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = torch.sigmoid(model(Xva, Mva)).numpy()
        auc = roc_auc_score(yva, va) if 0 < yva.sum() < len(yva) else 0.5
        if auc > best_auc:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def neural_oof(X, M, y, dates):
    oof = np.zeros(len(y))
    cv = GroupKFold(5)
    mean = X.reshape(-1, FDIM)[X.reshape(-1, FDIM).any(1)].mean(0)
    std = X.reshape(-1, FDIM)[X.reshape(-1, FDIM).any(1)].std(0) + 1e-6
    Xn = ((X - mean) / std) * (M[..., None])   # standardize, keep pads zero
    for tr, te in cv.split(Xn, y, groups=dates):
        # inner split of train by date for early stopping
        inner = GroupKFold(4)
        itr, iva = next(inner.split(Xn[tr], y[tr], groups=dates[tr]))
        model = train_fold(Xn[tr][itr], M[tr][itr], y[tr][itr],
                           Xn[tr][iva], M[tr][iva], y[tr][iva])
        model.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(model(torch.tensor(Xn[te]), torch.tensor(M[te]))).numpy()
    return oof


def report(name, prob, y, dates):
    thr = fpr_threshold(prob, y)
    rew, nw = per_window_reward(recenter_scores(prob, thr), y, dates)
    print(f"{name:22s} per-window reward={rew:.4f}  OOF_auc={roc_auc_score(y,prob):.4f} "
          f"OOF_ap={average_precision_score(y,prob):.4f}")
    return rew


def main():
    Xtab, y, dates = load_cache()
    print(f"tabular {Xtab.shape}, building per-hand tensors...", flush=True)
    Xh, M, y2, d2 = build_hand_cache()
    assert (y == y2).all() and (dates == d2).all(), "cache misalignment"
    print(f"hand tensor {Xh.shape}, training neural OOF (this takes a few min)...", flush=True)

    gbdt_oof = cross_val_predict(build_stack(), Xtab, y, groups=dates,
                                 cv=GroupKFold(5), method="predict_proba", n_jobs=4)[:, 1]
    neural = neural_oof(Xh, M, y, dates)

    print(f"\ncorrelation(GBDT, neural) = {np.corrcoef(gbdt_oof, neural)[0,1]:.3f}")
    report("GBDT stack (v5)", gbdt_oof, y, dates)
    report("neural attn-MIL", neural, y, dates)

    # blend candidates
    for w in (0.2, 0.3, 0.4, 0.5):
        blend = (1 - w) * gbdt_oof + w * neural
        report(f"blend {int((1-w)*100)}/{int(w*100)} gbdt/neural", blend, y, dates)

    # logistic meta-blend (stacked)
    from sklearn.linear_model import LogisticRegression as LR
    meta = LR(max_iter=1000)
    Z = np.column_stack([gbdt_oof, neural])
    meta_oof = cross_val_predict(meta, Z, y, groups=dates, cv=GroupKFold(5),
                                 method="predict_proba", n_jobs=4)[:, 1]
    report("logistic meta-blend", meta_oof, y, dates)
    print("\nDONE")


if __name__ == "__main__":
    main()
