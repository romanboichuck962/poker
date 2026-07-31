# Attribution — UID237 jet-markovpot-gb-detector-3

`poker44_ml/luck_detector.py` is vendored from UID237's public MIT-licensed
repo:

- https://github.com/eyyupkemer7/jet-detector-3
- commit `2df44d27845c44bd4a27f99c4afc0c791565314b`
- announced identity: `jet-markovpot-gb-detector-3` v3.7.3

Profile: `markov-pot-geometry-gb` (M3-GB) — training-free Markov transition
entropy deficit + pot-geometry regularity + signature concentration, fused
with a weighted geometric blend and smoothstep anchors.

`model_luck.py` is our thin serving adapter. Relative to upstream we keep
batch-rank remap on by default (`POKER44_BATCH_RANK` / `POKER44_MAX_POS_FRAC`)
to secure the live validator safety gate without changing ranking.

See `LICENSE-uid237` for the upstream MIT license text.
