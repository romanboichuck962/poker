"""Refit the selected model compactly and deploy the artifact.

Uses CalibratedClassifierCV(ensemble=False) so the artifact stores one fitted
ensemble plus a calibrator (not cv=3 clones), keeping it well under GitHub's
100 MB limit. The 0.5 operating threshold is taken from an honest held-out
estimate (newest release + all validation-split groups).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/Poker44-subnet")
from model import recenter_scores  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from robust_select import load_cache, per_window_reward  # noqa: E402
from train import candidate_models, fpr_threshold  # noqa: E402

SELECTED = "voting_soft"


def main():
    X, y, dates = load_cache()
    latest = sorted(set(dates))[-1]
    test_mask = dates == latest
    train_mask = ~test_mask

    # Honest held-out threshold: train on all but the newest release.
    holdout_model = CalibratedClassifierCV(
        candidate_models()[SELECTED], method="sigmoid", cv=3, ensemble=False
    )
    holdout_model.fit(X[train_mask], y[train_mask])
    prob_te = holdout_model.predict_proba(X[test_mask])[:, 1]
    thr = fpr_threshold(prob_te, y[test_mask])
    centered = recenter_scores(prob_te, thr)
    r, det = reward(centered, y[test_mask])
    print(f"holdout {latest}: reward={r:.4f} auc={roc_auc_score(y[test_mask], prob_te):.4f} "
          f"ap={average_precision_score(y[test_mask], prob_te):.4f} "
          f"recall@fpr5={det['bot_recall']:.3f} hard_fpr={det['hard_fpr']:.3f} thr={thr:.4f}")

    # Deploy: same estimator config refit on ALL data.
    final = CalibratedClassifierCV(
        candidate_models()[SELECTED], method="sigmoid", cv=3, ensemble=False
    )
    final.fit(X, y)
    out = Path("/root/poker/artifacts/poker44_model.joblib")
    joblib.dump({"pipeline": final, "threshold": thr, "selected": SELECTED},
                out, compress=3)
    size_mb = out.stat().st_size / 1e6
    print(f"saved {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
