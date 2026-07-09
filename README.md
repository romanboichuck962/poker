# Poker44 Miner Submission

Open-source miner model for the Poker44 subnet (Bittensor netuid 126).

## Model

- **Name:** poker44-neptune-ensemble
- **Version:** 3
- **Framework:** scikit-learn calibrated soft-voting ensemble
  (XGBoost + LightGBM + CatBoost + ExtraTrees + MLP)
- **License:** MIT
- **Inference mode:** remote

The model receives `DetectionSynapse(chunks=...)` where each chunk is a list of
poker hand payloads sharing one focus ("hero") seat, and returns one bot-risk
probability in `[0, 1]` per chunk.

## Architecture

1. **Feature extraction** ([`model.py`](model.py)): 172 hero-centric behavioral
   features per chunk group. Per-hand signals — action rates (VPIP, PFR,
   VPIP/PFR gap, fold/call/check/raise/bet), aggression factor, per-street
   aggression (preflop/flop/turn/river), fold-to-bet response, street
   progression, showdown/win rates, stack depth, position, and **bet-sizing
   regularity** (mean/std/CV/min/max in big blinds, pot-relative sizing, and a
   "roundness" score that fires on clean bb increments or canonical pot
   fractions — a strong bot tell). Aggregated across the group as mean, std,
   25th and 75th percentiles, plus group-level consistency signals (distinct-
   sizing fraction, action-mix entropy, global sizing coefficient of variation,
   aggression consistency).
2. **Classifier**: sigmoid-calibrated **soft-voting ensemble** selected among
   12 candidates (logistic regression, RBF SVM, random forest, extra trees, two
   sklearn GBDTs, XGBoost, LightGBM, CatBoost, MLP, a soft-voting ensemble, and
   a stacking ensemble).
3. **Score recentering**: a monotone map places the out-of-fold 5%-FPR
   operating point at 0.5, so hard predictions respect the validator's
   false-positive budget without changing rank metrics.

## Selection & Training

Two scripts, both trained **exclusively on the public Poker44 training
benchmark** (`https://api.poker44.net/api/v1/benchmark`, all 45 published
release dates, 1,186 chunk groups):

- [`train.py`](train.py) — 12-algorithm comparison with date-grouped
  cross-validation.
- [`robust_select.py`](robust_select.py) — final selection by
  cross-validated **per-window validator reward** (mirroring how the live
  validator scores each evaluation window), using the subnet's own `reward()`.

Out-of-fold generalization (date-grouped 5-fold CV over the full dataset):

| model | per-window reward | OOF AUC | OOF AP |
|---|---|---|---|
| **soft-voting (deployed)** | **0.7914** | **0.7882** | **0.8164** |
| extra trees | 0.7839 | 0.7792 | 0.8080 |
| catboost | 0.7838 | 0.7771 | 0.8082 |
| stacking | 0.7822 | 0.7818 | 0.8121 |
| random forest | 0.7634 | 0.7726 | 0.8004 |

The deployed artifact is the winner refit on all data.

## Files

- [`miner.py`](miner.py) — serving miner (implementation file in the manifest)
- [`model.py`](model.py) — features + serving wrapper (implementation file in the manifest)
- [`train.py`](train.py) — 12-algorithm comparison
- [`robust_select.py`](robust_select.py) — per-window-reward final selection
- `artifacts/poker44_model.joblib` — deployed model (its sha256 is published as
  `artifact_sha256` in the manifest)

## Reproduce

```bash
pip install -r requirements.txt
pip install -e /path/to/Poker44-subnet
python /path/to/Poker44-subnet/scripts/download_benchmark.py --out data/benchmark
python train.py           # broad comparison
python robust_select.py   # final selection + deploy artifact
```

## Training Data Statement

Trained exclusively on the public Poker44 training benchmark. No
validator-only evaluation data is used.
