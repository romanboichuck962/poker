# Poker44 Miner Submission

Open-source miner model for the Poker44 subnet (Bittensor netuid 126).

## Model

- **Name:** poker44-neptune-heuristic
- **Version:** 1
- **Framework:** python-heuristic
- **License:** MIT
- **Inference mode:** remote

The model receives `DetectionSynapse(chunks=...)` where each chunk is a list of
poker hand payloads, and returns one bot-risk score in `[0, 1]` per chunk.

The current implementation is a deterministic behavioral heuristic. Per hand it
aggregates:

- street depth and showdown frequency
- call / check / fold / raise action ratios
- table-size signal from player count

Per chunk, hand scores are averaged into a single risk score.

## Implementation

- [`miner.py`](miner.py) — full miner implementation served on-chain
  (the `implementation_sha256` in the published `model_manifest` covers this file).

## Training Data Statement

This heuristic has no training step. It uses only runtime chunk features from
the miner-visible payload. Development iteration uses the public Poker44
training benchmark (`https://api.poker44.net/api/v1/benchmark`). No
validator-only evaluation data is used.

## Reproduce Local Evaluation

```bash
pip install requests scikit-learn numpy
python - <<'PY'
import requests
from miner import Miner

base = "https://api.poker44.net/api/v1/benchmark"
sd = requests.get(base, timeout=30).json()["data"]["latestSourceDate"]
data = requests.get(f"{base}/chunks", params={"sourceDate": sd, "limit": 24}, timeout=60).json()["data"]
for chunk in data["chunks"]:
    preds = [Miner.score_chunk(g) for g in chunk["chunks"]]
    print(chunk["chunkId"], preds)
PY
```
