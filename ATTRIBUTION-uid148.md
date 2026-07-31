# Attribution — UID148 code-seqsig-rw-detector-1

`poker44_ml/luck_detector.py` is vendored from UID148's public MIT-licensed
repo:

- https://github.com/payizogu20/code-seqsig-rw-detector-1
- commit `5c5aaf34216f55d86ea3f0597609dfe016432eb4`
- announced identity: `code-seqsig-rw-detector-1` v3.3.1

Profile: `sequence-signature-sd-rw` (S1-RW) — training-free signature
concentration + winsorized size-dispersion deficit + street uniformity with
piecewise-linear anchors.

`model_luck.py` is our thin serving adapter. Relative to upstream we keep
batch-rank remap on by default (`POKER44_BATCH_RANK` / `POKER44_MAX_POS_FRAC`).

See `LICENSE-uid148` for the upstream MIT license text.
