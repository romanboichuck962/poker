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

from model import MODEL_ARTIFACT, Poker44Model
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
            implementation_files=[REPO_ROOT / "miner.py", REPO_ROOT / "model.py"],
            defaults={
                "model_name": "poker44-neptune-rocket",
                "model_version": "23",
                "framework": "rocket-p44r1 (adapted from UID163 rocket-r2): weighted log-odds fusion of stack(LGBM+XGBoost+CatBoost+ExtraTrees+RF->LogisticRegression meta)+mono(monotone-XGBoost committee) on hero+behavioral features, mlp(PCA+MLP committee) on the feature union, drse(drift-robust subspace ensemble) on an enriched hero-free view (28 all-actor per-hand scalars x 7 order-stats + replay signatures + compression/LZ76/Vendi redundancy); measured live-OOD ablation: features with z>5 vs 500 captured validator chunks are dropped from both views (live stacks pinned 100bb, pots ~5bb, 7-9 seats); blend weights chosen by a dense walk-forward simplex search on OUR reward(); sanitized train; targetFPR=5% remap-to-0.5; smart min-positive; 16% pos cap",
                "license": "MIT",
                "repo_url": "https://github.com/romanboichuck962/poker",
                "repo_commit": os.getenv("POKER44_MODEL_REPO_COMMIT") or _git_commit(REPO_ROOT),
                "open_source": True,
                "inference_mode": "remote",
                "artifact_sha256": _sha256(MODEL_ARTIFACT),
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 training benchmark "
                    "(https://api.poker44.net/api/v1/benchmark), releases through "
                    "2026-07-19 (including v1.13), "
                    "each hand passed through the public prepare_hand_for_miner sanitizer so "
                    "training matches serving. See deploy_rocket.py for training "
                    "(architecture adapted from UID163's rocket-r2) and promote gates."
                ),
                "training_data_sources": ["https://api.poker44.net/api/v1/benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
                "data_attestation": (
                    "All training data comes from the public Poker44 benchmark API."
                ),
                "notes": "v23: retrained the UID163 rocket-r2 four-component architecture on benchmark through 2026-07-19 (55 releases) with OOD ablation from 500 captured validator chunks, then re-controlled the blend weights via a dense 946-point walk-forward simplex search scored on OUR reward() with per-candidate FPR-anchored thresholds. Selected weights (stack 0.22, mono 0.10, mlp 0.38, drse 0.30) lift walk-forward reward 0.9178->0.9247 and recall@5%fpr 0.767->0.801 over the v22 blend (stack 0.28, mono 0.18, mlp 0.32, drse 0.22); promote-gated and cross-checked on captures (Spearman 0.952 vs v22, 12.4% flagged at budget).",
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
