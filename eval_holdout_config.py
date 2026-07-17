"""Compare candidate (weights, mode, fraction) configs on the locked holdout.

Refits the pre-holdout model (700 trees, same protocol as train_v4.py) and
scores each config on the two locked holdout dates with balanced request
windows, so a robustness-motivated config override can be priced against the
benchmark-selected one before deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from train_v4 import (
    Chunk,
    ModelConfig,
    _balanced_windows,
    _batched_request_branches,
    _deduplicate,
    _evaluate_predictions,
    _feature_matrix,
    _fit_model,
    load_sanitized_chunks,
)
from v4.calibration import fit_fixed_mapper
from v4.mapping import chunk_tie_key
from v4.model import blend_branches

DATA_DIR = Path("/root/POKER44-SUBNET-1/data/benchmark")

CONFIGS = {
    "selected_raw": {
        "weights": np.array([0, 0, 0.4, 0.4, 0.2, 0, 0, 0, 0], float),
        "mode": "rank",
        "fraction": 0.10,
    },
    "uid176_published_prob": {
        "weights": np.array([0, 0, 0.2, 0, 0.2, 0, 0.2, 0.2, 0.2], float),
        "mode": "probability",
        "fraction": 0.10,
    },
    "uid176_published_rank": {
        "weights": np.array([0, 0, 0.2, 0, 0.2, 0, 0.2, 0.2, 0.2], float),
        "mode": "rank",
        "fraction": 0.10,
    },
    "pure_rank_prob": {
        "weights": np.array([0, 0, 0, 0, 0, 0, 1 / 3, 1 / 3, 1 / 3], float),
        "mode": "probability",
        "fraction": 0.10,
    },
}


def main() -> None:
    chunks = load_sanitized_chunks(DATA_DIR)
    chunks, _ = _deduplicate(chunks)
    y = np.asarray([int(c.label) for c in chunks], dtype=int)
    dates = np.asarray([str(c.source_date) for c in chunks])
    x = _feature_matrix(chunks, y, dates, cache_path=Path("artifacts/v4_features.npz"))
    unique_dates = sorted(set(dates))
    holdout_dates = unique_dates[-2:]
    pre = np.flatnonzero(dates < holdout_dates[0])
    print(f"refit pre-holdout model on {len(pre)} chunks (holdout {holdout_dates})", flush=True)
    model = _fit_model(
        x, y, chunks, pre,
        seed=44,
        config=ModelConfig(trees=700, hist_iterations=700, max_depth=9, learning_rate=0.03),
        date_power=0.5,
    )
    folds = []
    for date in holdout_dates:
        idx = np.flatnonzero(dates == date)
        folds.append(
            {
                "date": date,
                "indices": idx,
                "branches": model.branch_scores(x[idx]),
                "matrix": x[idx],
                "model": model,
                "labels": y[idx],
                "keys": [chunk_tie_key(chunks[i].hands) for i in idx],
            }
        )

    results = {}
    for name, config in CONFIGS.items():
        e40 = _evaluate_predictions(folds, config, window_size=40, windows_per_date=100, seed=44 + 40000)
        e100 = _evaluate_predictions(folds, config, window_size=100, windows_per_date=50, seed=44 + 50000)
        results[name] = {
            "w40_mean": e40["window_mean"], "w40_p10": e40["window_p10"],
            "w100_mean": e100["window_mean"], "w100_p10": e100["window_p10"],
            "pooled_reward": e100["pooled"]["reward"],
            "pooled_ap": e100["pooled"]["ap_score"],
            "pooled_recall": e100["pooled"]["bot_recall"],
        }
        print(f"{name:24s} " + " ".join(f"{k}={v:.4f}" for k, v in results[name].items()), flush=True)

        # Refit the small-batch fallback mapper for each config from holdout raw blends.
        raw = np.concatenate([
            blend_branches(f["branches"], config["weights"], config["mode"], tie_keys=f["keys"])
            for f in folds
        ])
        labels = np.concatenate([f["labels"] for f in folds])
        results[name]["mapper"] = fit_fixed_mapper(raw, labels, target_human_fpr=0.05)

    Path("artifacts/holdout_config_comparison.json").write_text(
        json.dumps(results, indent=2, default=float)
    )
    print("saved artifacts/holdout_config_comparison.json")


if __name__ == "__main__":
    main()
