"""Poker44 miner serving a trained bot-detection model (see model.py / train.py).

Falls back to a neutral 0.5 score for any chunk the model fails to score.
Run with the Poker44-subnet package installed (pip install -e Poker44-subnet).
"""

# NOTE: do NOT enable `from __future__ import annotations` here — bittensor's
# axon.attach() introspects forward()'s annotation and calls issubclass() on it,
# which requires the real DetectionSynapse class, not a stringized annotation.

import hashlib
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

from model import MODEL_ARTIFACT, Poker44Model

REPO_ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Miner(BaseMinerNeuron):
    """Miner returning one calibrated bot-risk probability per chunk."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.model = Poker44Model()
        bt.logging.info(f"🤖 Poker44 trained-model miner started (artifact={MODEL_ARTIFACT.name})")

        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[REPO_ROOT / "miner.py", REPO_ROOT / "model.py"],
            defaults={
                "model_name": "poker44-neptune-hybrid",
                "model_version": "12",
                "framework": "In-batch rank-blend (weights 0.35/0.20/0.30/0.15) of an ExtraTrees, a RandomForest, a HistGradientBoosting, and a LightGBM classifier (all regularized, depth<=9) over ~219 sanitization-robust behavioral, policy-determinism, bucket-snapped sizing, hero-free action/hand-replay-signature, and approximate cross-hand redundancy (gzip/LZ76/Vendi/Jaccard/entropy-rate) features. Trained on validator-sanitized payloads (prepare_hand_for_miner) with size-resampling to live ~100-hand groups; FPR-anchored threshold from the human-score quantile plus a per-batch positive-call safety budget",
                "license": "MIT",
                "repo_url": "https://github.com/romanboichuck962/poker",
                "open_source": True,
                "inference_mode": "remote",
                "artifact_sha256": _sha256(MODEL_ARTIFACT),
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 training benchmark "
                    "(https://api.poker44.net/api/v1/benchmark), all published releases, "
                    "each hand passed through the public prepare_hand_for_miner sanitizer so "
                    "training matches serving. See deploy_v11.py for the full training procedure."
                ),
                "training_data_sources": ["https://api.poker44.net/api/v1/benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
                "data_attestation": (
                    "All training data comes from the public Poker44 benchmark API."
                ),
                "notes": "Rank-blend bot detector. Every training hand is passed through the public prepare_hand_for_miner sanitizer (seat re-alias, button=0, bb-bucketed amounts, 5-8 action window) so training matches the served feed, and groups are size-resampled to the live ~100-hand regime. Four diverse, regularized tree learners (ExtraTrees, RandomForest, HistGradientBoosting, LightGBM; depth<=9) are fused by in-batch rank so no member's calibration can distort the blend; scoring applies a monotone threshold remap plus a per-batch positive-call budget (0.125) that secures the validator's safety/calibration gate without reordering (AP and recall@FPR are pure ranking). Features are behavioral + policy-determinism + bucket-snapped sizing + hero-free action/hand-replay signatures + approximate cross-hand redundancy (compression ratio, Lempel-Ziv complexity, Vendi diversity, pairwise Jaccard, entropy-rate); no hole/board cards or identifiers.",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            f"Manifest status={self.manifest_compliance['status']} "
            f"missing={self.manifest_compliance['missing_fields']} "
            f"violations={self.manifest_compliance['policy_violations']} "
            f"digest={self.manifest_digest[:16]}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        started = time.monotonic()
        try:
            scores = self.model.score_chunks(chunks)
        except Exception as err:  # never fail the synapse on a scoring error
            bt.logging.error(f"model scoring failed, using neutral scores: {err}")
            scores = [0.5] * len(chunks)
        synapse.risk_scores = [float(s) for s in scores]
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"Scored {len(chunks)} chunks in {time.monotonic() - started:.3f}s "
            f"(flagged={sum(synapse.predictions)})"
        )
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 trained-model miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
