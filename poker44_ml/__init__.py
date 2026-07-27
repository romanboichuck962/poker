"""p44 — Poker44 subnet 126 bot-detection miner, version 1.

Package layout:

    upstream.py    pinned-hash guard against silent subnet-side changes
    reward.py      the authoritative validator reward (imported, never copied)
    canonical.py   benchmark -> miner-visible projection + train/serve parity check
    data.py        public benchmark download + labeled example loading
    features.py    size- and scale-invariant chunk features
    policy.py      rank-preserving positive-fraction score map
    model.py       rank-vote ensemble
    inference.py   serving wrapper (the ONLY scoring path)
    train.py       fit an artifact
    walkforward.py the ship gate: train on the past, test the next unseen date
    drift.py       train-vs-live feature drift (PSI / KS)
"""

__version__ = "1.0.0"
MODEL_NAME = "poker44-artos"
