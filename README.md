# Poker44 Miner Submission

Open-source miner model for the Poker44 subnet (Bittensor netuid 126).

## Model

- **Name:** poker44-neptune-mlp
- **Version:** 2
- **Framework:** scikit-learn — calibrated MLP (`CalibratedClassifierCV(MLPClassifier(64, 32), sigmoid)`)
- **License:** MIT
- **Inference mode:** remote

The model receives `DetectionSynapse(chunks=...)` where each chunk is a list of
poker hand payloads sharing one focus ("hero") seat, and returns one bot-risk
probability in `[0, 1]` per chunk.

## Architecture

1. **Feature extraction** ([`model.py`](model.py)): 48 hero-centric behavioral
   features per chunk group — per-hand action rates (VPIP, PFR, fold/call/
   check/raise/bet), aggression factor, street progression, showdown and win
   rates, stack depth, position, bet sizing in big blinds and pot-relative
   terms — aggregated as means and standard deviations across the group, plus
   group-level consistency signals (distinct-sizing fraction, action-mix
   entropy, VPIP variance) that separate uniform bot behavior from human
   variance.
2. **Classifier**: sigmoid-calibrated MLP selected among 6 candidates
   (logistic regression, random forest, extra trees, two GBDT variants, MLP)
   by validator reward on a held-out benchmark release.
3. **Score recentering**: a monotone map places the out-of-fold 5%-FPR
   operating point at 0.5, so hard predictions respect the validator's
   false-positive budget without changing rank metrics.

## Training

[`train.py`](train.py) is the full training procedure. Data: **all published
releases of the public Poker44 training benchmark**
(`https://api.poker44.net/api/v1/benchmark`) — 45 release dates, 1,186 chunk
groups. Model selection uses date-grouped cross-validation plus a held-out
newest release; the deployed artifact is refit on all data.

Held-out evaluation (release 2026-07-09 + all `validation`-split groups,
n=410, using the validator's `reward()` from the subnet repo):

| metric | value |
|---|---|
| validator reward | 0.7525 |
| ROC AUC | 0.7321 |
| average precision | 0.7780 |
| bot recall @ 5% FPR | 0.434 |
| hard FPR @ 0.5 | 0.088 |

## Files

- [`miner.py`](miner.py) — serving miner (implementation file in the manifest)
- [`model.py`](model.py) — features + serving wrapper (implementation file in the manifest)
- [`train.py`](train.py) — training / model-selection procedure
- `artifacts/poker44_model.joblib` — deployed model (its sha256 is published
  as `artifact_sha256` in the manifest)

## Reproduce

```bash
pip install -r requirements.txt
pip install -e /path/to/Poker44-subnet
python /path/to/Poker44-subnet/scripts/download_benchmark.py --out data/benchmark
python train.py --data data/benchmark
```

## Training Data Statement

Trained exclusively on the public Poker44 training benchmark. No
validator-only evaluation data is used.
