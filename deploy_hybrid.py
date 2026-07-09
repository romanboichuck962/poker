"""Deploy the hybrid GBDT-stack + attention-MIL model (v6).

Fits the tabular GBDT stack and the neural set model, blends 60/40, takes an
honest held-out operating threshold, and saves a single combined artifact that
model.Poker44Model serves.
"""

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
sys.path.insert(0, "/root/Poker44-subnet")
from model import recenter_scores  # noqa: E402
from neural_mil import FDIM, build_hand_cache, build_stack, train_fold  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from robust_select import load_cache  # noqa: E402
from train import fpr_threshold  # noqa: E402

BLEND_W = 0.4  # neural weight (60/40 gbdt/neural was best on OOF)


def std_stats(Xh):
    flat = Xh.reshape(-1, FDIM)
    nz = flat[flat.any(1)]
    return nz.mean(0).astype(np.float32), (nz.std(0) + 1e-6).astype(np.float32)


def neural_predict(model, Xh, M, mean, std):
    Xn = (Xh - mean) / std * M[..., None]
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(Xn), torch.tensor(M))).numpy()


def train_neural(Xh, M, y, dates, val_date):
    """Train attn-MIL on all groups except val_date (used for early stopping)."""
    mean, std = std_stats(Xh)
    Xn = (Xh - mean) / std * M[..., None]
    va = dates == val_date
    model = train_fold(Xn[~va], M[~va], y[~va], Xn[va], M[va], y[va])
    return model, mean, std


def main():
    Xtab, y, dates = load_cache()
    Xh, M, y2, d2 = build_hand_cache()
    assert (y == y2).all() and (dates == d2).all()
    latest = sorted(set(dates))[-1]
    te = dates == latest

    # ---- honest held-out threshold + reward (train on all prior releases) ----
    gbdt_h = CalibratedClassifierCV(build_stack(), method="sigmoid", cv=3, ensemble=False)
    gbdt_h.fit(Xtab[~te], y[~te])
    g_te = gbdt_h.predict_proba(Xtab[te])[:, 1]
    nmodel_h, mean_h, std_h = train_neural(Xh[~te], M[~te], y[~te], dates[~te],
                                           sorted(set(dates[~te]))[-1])
    n_te = neural_predict(nmodel_h, Xh[te], M[te], mean_h, std_h)
    blend_te = (1 - BLEND_W) * g_te + BLEND_W * n_te
    thr = fpr_threshold(blend_te, y[te])
    r, det = reward(recenter_scores(blend_te, thr), y[te])
    print(f"holdout {latest}: reward={r:.4f} auc={roc_auc_score(y[te], blend_te):.4f} "
          f"ap={average_precision_score(y[te], blend_te):.4f} recall@fpr5={det['bot_recall']:.3f} thr={thr:.4f}")

    # ---- deployment fit: GBDT on ALL data, neural on all-but-latest (needs a val) ----
    gbdt = CalibratedClassifierCV(build_stack(), method="sigmoid", cv=3, ensemble=False)
    gbdt.fit(Xtab, y)
    nmodel, mean, std = train_neural(Xh, M, y, dates, latest)

    artifact = {
        "kind": "hybrid",
        "pipeline": gbdt,
        "neural_state": {k: v.cpu() for k, v in nmodel.state_dict().items()},
        "feat_mean": mean,
        "feat_std": std,
        "blend_w": BLEND_W,
        "threshold": float(thr),
        "selected": "gbdt_stack + attn_mil blend (0.6/0.4)",
    }
    out = Path("/root/poker/artifacts/poker44_model.joblib")
    joblib.dump(artifact, out, compress=3)
    print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
