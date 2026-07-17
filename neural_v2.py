"""Improved neural set model: hand-dropout augmentation + seed ensembling.

The neural attn-MIL was the weak, high-variance half of the hybrid (OOF AUC
~0.73). Two fixes attack that directly:
  - hand-dropout augmentation: each training step randomly drops a fraction of
    the hero's hands. A subset of a labeled hero's hands keeps the label, so
    this multiplies effective training data and regularizes the set encoder.
  - seed ensembling: average N nets per fold to cut the variance of a small net
    on 1,186 samples.
Reports neural OOF and the re-optimized blend vs the current hybrid (0.816 AUC).
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402

sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/POKER44-SUBNET-1")
from model import recenter_scores  # noqa: E402
from neural_mil import FDIM, build_hand_cache, build_stack  # noqa: E402
from robust_select import load_cache, per_window_reward  # noqa: E402
from train import fpr_threshold  # noqa: E402

torch.set_num_threads(4)
N_SEEDS = 3
DROP = 0.25  # fraction of hands randomly dropped per training step


class AttnMIL(nn.Module):
    def __init__(self, fdim, hidden=64, p=0.35):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(fdim, hidden), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Dropout(p + 0.1), nn.Linear(hidden * 2, 1))

    def forward(self, x, mask):
        h = self.enc(x)
        a = self.attn(h).squeeze(-1).masked_fill(mask == 0, -1e9).softmax(-1)
        attn_pool = (h * a.unsqueeze(-1)).sum(1)
        mean_pool = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat([attn_pool, mean_pool], -1)).squeeze(-1)


def _hand_dropout(mask, gen):
    """Randomly drop DROP of each row's valid hands; keep >=5 hands."""
    keep = (torch.rand(mask.shape, generator=gen) > DROP).float() * mask
    # ensure at least a few hands survive
    empty = keep.sum(1) < 5
    keep[empty] = mask[empty]
    return keep


def train_one(Xtr, Mtr, ytr, Xva, Mva, yva, seed, epochs=150, patience=18):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = AttnMIL(FDIM)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr, Mtr = torch.tensor(Xtr), torch.tensor(Mtr)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xva, Mva = torch.tensor(Xva), torch.tensor(Mva)
    n, bs = len(Xtr), 128
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            m_aug = _hand_dropout(Mtr[idx], gen)
            opt.zero_grad()
            loss = lossf(model(Xtr[idx], m_aug), yt[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = torch.sigmoid(model(Xva, Mva)).numpy()
        auc = roc_auc_score(yva, va) if 0 < yva.sum() < len(yva) else 0.5
        if auc > best:
            best, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def neural_oof(X, M, y, dates):
    mean = X.reshape(-1, FDIM)[X.reshape(-1, FDIM).any(1)].mean(0)
    std = X.reshape(-1, FDIM)[X.reshape(-1, FDIM).any(1)].std(0) + 1e-6
    Xn = (X - mean) / std * M[..., None]
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Xn, y, groups=dates):
        itr, iva = next(GroupKFold(4).split(Xn[tr], y[tr], groups=dates[tr]))
        preds = []
        for s in range(N_SEEDS):
            m = train_one(Xn[tr][itr], M[tr][itr], y[tr][itr],
                          Xn[tr][iva], M[tr][iva], y[tr][iva], seed=42 + s)
            m.eval()
            with torch.no_grad():
                preds.append(torch.sigmoid(m(torch.tensor(Xn[te]), torch.tensor(M[te]))).numpy())
        oof[te] = np.mean(preds, axis=0)
    return oof


def report(name, prob, y, dates):
    thr = fpr_threshold(prob, y)
    rew, _ = per_window_reward(recenter_scores(prob, thr), y, dates)
    print(f"{name:26s} reward={rew:.4f} OOF_auc={roc_auc_score(y,prob):.4f} OOF_ap={average_precision_score(y,prob):.4f}", flush=True)
    return roc_auc_score(y, prob)


def main():
    Xtab, y, dates = load_cache()
    Xh, M, _, _ = build_hand_cache()
    print("recomputing GBDT OOF + augmented neural OOF...", flush=True)
    gbdt = cross_val_predict(build_stack(), Xtab, y, groups=dates,
                             cv=GroupKFold(5), method="predict_proba", n_jobs=4)[:, 1]
    neural = neural_oof(Xh, M, y, dates)
    print(f"\ncorr(gbdt,neural)={np.corrcoef(gbdt,neural)[0,1]:.3f}")
    report("GBDT stack", gbdt, y, dates)
    report("neural (aug+ensemble)", neural, y, dates)
    best_w, best_auc = 0.0, 0.0
    for w in (0.2, 0.3, 0.35, 0.4, 0.45, 0.5):
        auc = report(f"blend {int((1-w)*100)}/{int(w*100)}", (1 - w) * gbdt + w * neural, y, dates)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"\nBEST blend w={best_w} AUC={best_auc:.4f}  (current hybrid v6 = 0.816)")
    np.savez("/root/poker/artifacts/oof_v2.npz", gbdt=gbdt, neural=neural, y=y, dates=dates, best_w=best_w)
    print("DONE")


if __name__ == "__main__":
    main()
