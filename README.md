# Poker44 Miner Submission

Open-source miner model for the Poker44 subnet (Bittensor netuid 126).

## Model

- **Name:** poker44-neptune-hybrid
- **Version:** 14
- **Framework:** rank-blended ensemble: LightGBM, ExtraTrees, PCA-MLP, and a
  LightGBM LambdaMART member
- **License:** MIT
- **Inference mode:** remote

The model receives `DetectionSynapse(chunks=...)` where each chunk is a list of
poker hand payloads sharing one focus ("hero") seat, and returns one bot-risk
probability in `[0, 1]` per chunk.

## Architecture

1. **Feature extraction** ([`model.py`](model.py)): **269 behavioral features**
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
2. **Classifier — rank-blended complementary views**:
   - LightGBM, ExtraTrees, and PCA-MLP classifiers over 269 group features;
   - a LightGBM **LambdaMART** ranker trained with source-release groups,
     optimizing within-release bot ordering.
   Each member contributes its per-request rank, avoiding dependence on any
   individual model's probability calibration.
3. **Serving calibration**: a monotone map places the 5%-FPR operating point at
   0.5, then a rank-preserving positive-call budget protects the validator's
   threshold-safety requirement.

## Selection & Training

Trained **exclusively on the public Poker44 training benchmark**
(`https://api.poker44.net/api/v1/benchmark`, 51 releases through 2026-07-15).
Every hand is passed through `prepare_hand_for_miner` before feature extraction,
and chunks are size-resampled to the live ~100-hand regime. Pipeline:

- [`deploy_v11.py`](deploy_v11.py) — trains the sanitized rank blend, evaluates
  date-walk-forward performance, and writes the deployment artifact.

Current deployment evaluation uses sanitized, size-resampled 100-hand groups:

| metric | value |
|---|---|
| OOF ROC AUC | 0.8795 |
| Mean walk-forward validator reward (2026-07-12 to 2026-07-15) | 0.9407 |
| Latest walk-forward reward (2026-07-15) | 0.9613 |
| Latest walk-forward AP | 0.9751 |
| Latest recall at 5% FPR | 0.9000 |

These are public-benchmark proxy metrics, not live leaderboard results.

## Files

- [`miner.py`](miner.py) — serving miner (implementation file in the manifest)
- [`model.py`](model.py) — features + serving wrapper (implementation file in the manifest)
- [`deploy_v11.py`](deploy_v11.py) — train and evaluate the deployed rank blend
- `artifacts/poker44_model.joblib` — deployed model (sha256 published as
  `artifact_sha256` in the manifest)

## Reproduce

```bash
pip install -r requirements.txt
pip install -e /path/to/Poker44-subnet
python /path/to/Poker44-subnet/scripts/download_benchmark.py --out data/benchmark
python deploy_v11.py   # build the sanitized LambdaMART rank blend
```

## Training Data Statement

Trained exclusively on the public Poker44 training benchmark. No
validator-only evaluation data is used.
