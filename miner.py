"""Poker44 miner serving a trained bot-detection model (see model.py / train.py).

Falls back to a neutral 0.5 score for any chunk the model fails to score.
Run with the Poker44-subnet package installed (pip install -e Poker44-subnet).
"""

# NOTE: do NOT enable `from __future__ import annotations` here — bittensor's
# axon.attach() introspects forward()'s annotation and calls issubclass() on it,
# which requires the real DetectionSynapse class, not a stringized annotation.

import hashlib
import os
import subprocess
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

from model_artos import MODEL_ARTIFACT, Poker44Model
from capture import capture_chunks

REPO_ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(args: list[str], repo_root: Path) -> str:
    """Run a git command in repo_root, returning stripped stdout or "" on failure."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:  # pragma: no cover - git missing / not a repo
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


def _git_commit(repo_root: Path) -> str:
    """Current HEAD commit hash (manifest policy requires a real git commit)."""
    return _git(["rev-parse", "HEAD"], repo_root)


class Miner(BaseMinerNeuron):
    """Miner returning one calibrated bot-risk probability per chunk."""

    def __init__(self, config=None):
        super().__init__(config=config)
        self.model = Poker44Model()
        bt.logging.info(f"🤖 Poker44 trained-model miner started (artifact={MODEL_ARTIFACT.name})")

        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[
                REPO_ROOT / "miner.py",
                REPO_ROOT / "model_artos.py",
                REPO_ROOT / "poker44_ml" / "inference.py",
                REPO_ROOT / "poker44_ml" / "features.py",
                REPO_ROOT / "poker44_ml" / "model.py",
                REPO_ROOT / "poker44_ml" / "policy.py",
            ],
            defaults={
                "model_name": "poker44-neptune-artos",
                "model_version": "1",
                "framework": "poker44-artos (UID225's method, vendored under poker44_ml/ from https://github.com/judev113/p44-artos-v1 @0976ec00f74c, MIT - see LICENSE-artos; UID225 is a current top-5 miner ~0.66): a within-batch RANK blend (weights gradient_stack 0.35 + mono_lgbm 0.30 + pca_mlp 0.35) of three decorrelated members - a 4-base StackingClassifier (leaf-wise LGBM + loss-guide XGBoost + deep CatBoost + deep RandomForest -> LogisticRegression meta, cv3), a monotone-constrained LGBM (1200 trees, sign-mined constraints, human weight 2.0), and a StandardScaler->PCA56->MLP(80) pipeline - over the top-150-by-gain of ~466 SIZE-INVARIANT chunk features (61 per-hand scalars x 7 order-stats + signature/compression/LZ76/Vendi redundancy + reference-30-hand resampling; every absolute bb magnitude divided by the chunk's median pot to bridge the benchmark 30-40-hand -> live 80-100-hand shift). Members fused by within-batch percentile (rank01) so no member can drag the whole batch below 0.5. Serving operating point = their own rank_map at positive_fraction 0.07 (gate-safe smallest of the tied fractions from a bot-share sweep) which flags exactly ~7% of each request and cannot trip the zero-true-positive gate; sanitized train==serve.",
                "license": "MIT",
                "repo_url": "https://github.com/romanboichuck962/poker",
                "repo_commit": os.getenv("POKER44_MODEL_REPO_COMMIT") or _git_commit(REPO_ROOT),
                "open_source": True,
                "inference_mode": "remote",
                "artifact_sha256": _sha256(MODEL_ARTIFACT),
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 training benchmark "
                    "(https://api.poker44.net/api/v1/benchmark), releases through "
                    "2026-07-27 (including v1.13), "
                    "each hand passed through the public prepare_hand_for_miner sanitizer so "
                    "training matches serving. Architecture adapted from UID225's public "
                    "p44-artos-v1 (see poker44_ml/train.py)."
                ),
                "training_data_sources": ["https://api.poker44.net/api/v1/benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
                "data_attestation": (
                    "All training data comes from the public Poker44 benchmark API."
                ),
                "notes": "uid167 v1: replaced the UID225 luck detector (scored 0.385) with UID225's CURRENT method, artos (they switched repos to judev113/p44-artos-v1). Trained their exact default config on the public benchmark through 2026-07-27 (63 releases, 3866 balanced chunks; train 3556 / date-disjoint holdout 310 on the last 2 dates). Honest holdout: reward 0.9543, AP 0.9746, recall@FPR<=0.05 0.8774, tsq 1.0, fpr@0.5 0.0 - the best holdout of any model we have built. Live check on 1320 captures: within-batch rank blend keeps STRONG live spread (raw std 0.240, 1201 distinct - vs cold's collapsed 0.04), rank_map flags 7.0% with 0 zero-gate, permutation-invariant. CAVEAT: drift_report shows 36.6% of features outside training q01-q99 on live (median shift 1.44 IQR) - live is OOD; artos's size-invariant design is meant to absorb this and the spread survives, but ranking correctness on live is unverifiable without labels, and every prior competitor-method port has landed ~0.39 live regardless of benchmark strength.",
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
        # Input-only, best-effort capture of the live eval distribution for
        # offline benchmark->live feature-shift analysis. Never affects scoring.
        capture_chunks(chunks)
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
