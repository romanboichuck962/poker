# Poker44 Miner Submission

Open-source miner model for the Poker44 subnet (Bittensor netuid 126).

## Model

- **Name:** poker44-neptune-hybrid
- **Version:** 6
- **Framework:** hybrid — Optuna-tuned GBDT stacking
  (LightGBM + CatBoost + XGBoost + ExtraTrees, logistic meta) **blended 60/40
  with a PyTorch attention-MIL neural set model**
- **License:** MIT
- **Inference mode:** remote

The model receives `DetectionSynapse(chunks=...)` where each chunk is a list of
poker hand payloads sharing one focus ("hero") seat, and returns one bot-risk
probability in `[0, 1]` per chunk.

## Architecture

1. **Feature extraction** ([`model.py`](model.py)): **207 hero-centric features**
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
   - **Policy-determinism signals**: conditional action entropy given game
     context (street, facing-aggression, pot bucket), per-context repeat rate,
     pure-policy fraction, and action-bigram entropy — a bot applies a
     near-fixed context→action policy (low entropy), a human mixes.
2. **Classifier — hybrid of two complementary views**:
   - a sigmoid-calibrated **GBDT stacking ensemble** (Optuna-tuned LightGBM +
     CatBoost + XGBoost + ExtraTrees, logistic meta) over the 207 group features;
   - a **PyTorch attention-MIL set model** that encodes each hand's per-hand
     feature vector with an MLP and pools hands with masked attention (learned
     aggregation over the *set* of hands, instead of fixed moments).
   The two are blended 60/40 (GBDT/neural). The neural model is weaker alone but
   only ~0.39 correlated with the GBDT stack, so the blend adds real signal.
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

Out-of-fold generalization (date-grouped 5-fold CV over all 1,186 groups),
GBDT stack vs the hybrid blend:

| metric | GBDT stack | **hybrid (deployed)** |
|---|---|---|
| OOF ROC AUC | 0.793 | **0.816** |
| OOF average precision | 0.820 | **0.835** |
| CV per-window reward | 0.790 | **0.827** |

Held-out newest release (2026-07-09, trained on all prior releases):

| metric | value |
|---|---|
| validator reward | **0.825** |
| ROC AUC | 0.853 |
| average precision | 0.865 |
| bot recall @ 5% FPR | 0.575 |

The neural set model scores OOF AUC 0.727 alone but correlates only 0.39 with the
GBDT stack, so blending lifts OOF AUC from 0.793 to 0.816.

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
