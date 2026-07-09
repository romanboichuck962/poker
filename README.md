# Poker44 Miner Submission

Open-source miner model for the Poker44 subnet (Bittensor netuid 126).

## Model

- **Name:** poker44-neptune-stack
- **Version:** 4
- **Framework:** scikit-learn stacking ensemble — Optuna-tuned
  LightGBM + CatBoost + XGBoost + ExtraTrees, logistic-regression meta-learner
- **License:** MIT
- **Inference mode:** remote

The model receives `DetectionSynapse(chunks=...)` where each chunk is a list of
poker hand payloads sharing one focus ("hero") seat, and returns one bot-risk
probability in `[0, 1]` per chunk.

## Architecture

1. **Feature extraction** ([`model.py`](model.py)): **199 hero-centric features**
   per chunk group.
   - Per-hand behavior: action rates (VPIP, PFR, gap, fold/call/check/raise/bet),
     aggression, per-street aggression (preflop→river), fold-to-bet response,
     street progression, showdown/win, stack, position, bet sizing in bb and
     pot-relative terms, and a "roundness" score.
   - **Action-sequence patterns** (mechanical bot lines): check-raise, bet-fold,
     call-raise, limp-reraise counts.
   - Aggregated across the group as mean / std / 25th / 75th percentile.
   - **Group bet-sizing distribution**: pooled pot-ratio histogram, modal-size
     dominance, sizing entropy, distinct-size fraction — bots concentrate on a
     few exact sizes; humans spread.
2. **Classifier**: sigmoid-calibrated **stacking ensemble** of three
   Optuna-tuned gradient boosters plus extra trees, combined by logistic
   regression. Selected among 12+ candidates including tuned singles and a
   tuned soft-voting ensemble.
3. **Score recentering**: a monotone map places the 5%-FPR operating point at
   0.5 so hard predictions respect the validator's false-positive budget.

## Selection & Training

Trained **exclusively on the public Poker44 training benchmark**
(`https://api.poker44.net/api/v1/benchmark`, all 45 published releases, 1,186
chunk groups). Pipeline:

- [`train.py`](train.py) — 12-algorithm baseline comparison.
- [`tune_v4.py`](tune_v4.py) — Optuna tuning (50 trials each) of LightGBM,
  CatBoost, XGBoost, then candidate comparison by cross-validated per-window
  validator reward.
- [`deploy_v4.py`](deploy_v4.py) — builds the tuned stacking ensemble and
  deploys the compact calibrated artifact.

Out-of-fold generalization (date-grouped 5-fold CV over all 1,186 groups):

| metric | value |
|---|---|
| OOF ROC AUC | 0.788 |
| OOF average precision | 0.815 |
| CV per-window reward | 0.785 |

Held-out newest release (2026-07-09, trained on all prior releases):

| metric | value |
|---|---|
| validator reward | **0.802** |
| ROC AUC | 0.830 |
| average precision | 0.845 |
| bot recall @ 5% FPR | **0.521** |

The deployed artifact is the tuned stacking ensemble refit on all data.

## Files

- [`miner.py`](miner.py) — serving miner (implementation file in the manifest)
- [`model.py`](model.py) — features + serving wrapper (implementation file in the manifest)
- [`train.py`](train.py) — baseline comparison
- [`tune_v4.py`](tune_v4.py) — Optuna tuning + per-window-reward selection
- [`deploy_v4.py`](deploy_v4.py) — build + deploy tuned stack
- [`robust_select.py`](robust_select.py) — per-window-reward evaluation helpers
- `artifacts/poker44_model.joblib` — deployed model (sha256 published as
  `artifact_sha256` in the manifest)
- `artifacts/tuned_params.txt` — the Optuna-selected hyperparameters

## Reproduce

```bash
pip install -r requirements.txt
pip install -e /path/to/Poker44-subnet
python /path/to/Poker44-subnet/scripts/download_benchmark.py --out data/benchmark
python tune_v4.py      # tune + compare
python deploy_v4.py    # build + deploy tuned stack
```

## Training Data Statement

Trained exclusively on the public Poker44 training benchmark. No
validator-only evaluation data is used.
