"""Deploy v7: GBDT stack + augmented 3-seed attention-MIL ensemble, blended 50/50."""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.filterwarnings("ignore")

from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/POKER44-SUBNET-1")
from model import recenter_scores  # noqa: E402
from neural_mil import FDIM, build_hand_cache, build_stack  # noqa: E402
from neural_v2 import N_SEEDS, train_one  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from robust_select import load_cache  # noqa: E402
from train import fpr_threshold  # noqa: E402

BLEND_W = 0.5


def std_stats(Xh):
    flat = Xh.reshape(-1, FDIM)
    nz = flat[flat.any(1)]
    return nz.mean(0).astype(np.float32), (nz.std(0) + 1e-6).astype(np.float32)


def train_ensemble(Xh, M, y, dates, val_date):
    mean, std = std_stats(Xh)
    Xn = (Xh - mean) / std * M[..., None]
    va = dates == val_date
    nets = [train_one(Xn[~va], M[~va], y[~va], Xn[va], M[va], y[va], seed=42 + s)
            for s in range(N_SEEDS)]
    return nets, mean, std


def ens_predict(nets, Xh, M, mean, std):
    Xn = (Xh - mean) / std * M[..., None]
    X, Mt = torch.tensor(Xn), torch.tensor(M)
    with torch.no_grad():
        return np.mean([torch.sigmoid(n(X, Mt)).numpy() for n in nets], axis=0)


def main():
    Xtab, y, dates = load_cache()
    Xh, M, _, _ = build_hand_cache()
    latest = sorted(set(dates))[-1]
    te = dates == latest

    # honest held-out threshold + reward
    g = CalibratedClassifierCV(build_stack(), method="sigmoid", cv=3, ensemble=False)
    g.fit(Xtab[~te], y[~te])
    g_te = g.predict_proba(Xtab[te])[:, 1]
    nets_h, mh, sh = train_ensemble(Xh[~te], M[~te], y[~te], dates[~te], sorted(set(dates[~te]))[-1])
    n_te = ens_predict(nets_h, Xh[te], M[te], mh, sh)
    blend = (1 - BLEND_W) * g_te + BLEND_W * n_te
    thr = fpr_threshold(blend, y[te])
    r, det = reward(recenter_scores(blend, thr), y[te])
    print(f"holdout {latest}: reward={r:.4f} auc={roc_auc_score(y[te],blend):.4f} "
          f"ap={average_precision_score(y[te],blend):.4f} recall@fpr5={det['bot_recall']:.3f} thr={thr:.4f}")

    # deployment fit on all data
    gbdt = CalibratedClassifierCV(build_stack(), method="sigmoid", cv=3, ensemble=False)
    gbdt.fit(Xtab, y)
    nets, mean, std = train_ensemble(Xh, M, y, dates, latest)

    artifact = {
        "kind": "hybrid_ensemble",
        "pipeline": gbdt,
        "neural_states": [{k: v.cpu() for k, v in n.state_dict().items()} for n in nets],
        "feat_mean": mean, "feat_std": std,
        "blend_w": BLEND_W, "threshold": float(thr),
        "selected": "gbdt_stack + 3x augmented attn_mil (0.5/0.5)",
    }
    out = Path("/root/poker/artifacts/poker44_model.joblib")
    joblib.dump(artifact, out, compress=3)
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB), neural ensemble size={len(nets)}")


if __name__ == "__main__":
    main()
