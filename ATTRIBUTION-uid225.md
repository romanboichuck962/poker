# UID225 luck-detector attribution

`poker44_ml/luck_detector.py` is vendored verbatim from UID225's public
submission repo:

  https://github.com/mitsuiminoru000-pixel/poker44-luck-detector-2
  commit 8e32064020aec8b3992676d31bffd8863d86f648
  model manifest: luck-signature-detector v3.2.0

Licensed MIT (see LICENSE-uid225). It is a training-free sequence-signature
behavioral scorer. `model_luck.py` is our thin serving adapter that runs it
through our miner's score_chunks() interface and adds UID142's rank-preserving
batch-rank remap (env POKER44_BATCH_RANK / POKER44_MAX_POS_FRAC) to secure the
validator safety gate at live geometry — the remap is strictly order-preserving,
so UID225's ranking signal is unchanged.
