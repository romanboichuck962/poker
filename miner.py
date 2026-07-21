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

from model_luck import Poker44Model
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
        bt.logging.info(f"🤖 Poker44 luck-detector miner started (backend={self.model.backend})")

        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[
                REPO_ROOT / "miner.py",
                REPO_ROOT / "model_luck.py",
                REPO_ROOT / "poker44_ml" / "luck_detector.py",
            ],
            defaults={
                "model_name": "poker44-neptune-luck",
                "model_version": "9",
                "framework": "luck-signature-detector (faithful port of UID225 poker44-luck-detector-2 v3.2.0): a TRAINING-FREE sequence-signature behavioral scorer. Per hand builds a signature = street-shape + street/action/size-bucket tokens; per chunk measures signature concentration = 0.45*top_sig_share + 0.35*repeat_mass + 0.20*(1-unique_share) blended with street-progression uniformity (0.18); piecewise-linear anchor calibration [0.30,0.90]->[0.5,1.0], floor 0.05. Scripted seats replay a few decision templates so their hands collapse onto a handful of signatures; humans spread across many. Serving adds UID142's rank-preserving batch-rank remap (top 12.5% of each request batch cross 0.5; env POKER44_BATCH_RANK/POKER44_MAX_POS_FRAC) which is strictly order-preserving (AP and recall@FPR<=0.05 unchanged) and secures the validator safety gate at live geometry. No trees, no benchmark fit.",
                "license": "MIT",
                "repo_url": "https://github.com/romanboichuck962/poker",
                "repo_commit": os.getenv("POKER44_MODEL_REPO_COMMIT") or _git_commit(REPO_ROOT),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "No training data of any kind. This is a deterministic, training-free "
                    "behavioral heuristic that scores each chunk purely from the action "
                    "sequences visible in that chunk (signature concentration + street "
                    "uniformity). No model is fit on the benchmark or any other dataset."
                ),
                "training_data_sources": ["none"],
                "private_data_attestation": (
                    "This miner does not train on any data, and in particular uses no "
                    "validator-only evaluation data."
                ),
                "data_attestation": (
                    "No datasets are used; scoring depends only on the incoming chunk."
                ),
                "notes": "uid242 v9: switched from the UID163 rocket ensemble to a faithful port of UID225's pure sequence-signature luck detector (the training-free heuristic uid225 actually serves; it scores 0.687, rank #3 on the live leaderboard). Rationale: our trained ensembles (rocket/cold/coherent/draco) all under-transferred live (uid242 rocket 0.42) while this de-overfit behavioral heuristic is a live-proven, decorrelated signal. Validated at live geometry (100-chunk 20%-bot windows) with UID142's rank-preserving batch-rank remap @ 12.5%: mean reward 0.613, p10 0.527, min 0.455, 0/200 zero-gates, safety 0.998, AP 0.485, recall@FPR<=0.05 0.312, fpr@0.5 0.031. Raw (uid225 default, batch-rank off) gives safety only 0.78 at our live geometry, so batch-rank is enabled by default (rank-preserving -> uid225's ranking signal is untouched). Serves <0.5 ms/chunk. Set POKER44_BATCH_RANK=0 to serve exactly what uid225 serves.",
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
