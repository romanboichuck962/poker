"""Poker44 miner serving UID237's training-free M3-GB luck detector.

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

from model_luck import MODEL_ARTIFACT, Poker44Model
from capture import capture_chunks

REPO_ROOT = Path(__file__).resolve().parent


def _sha256(path: Path | None) -> str:
    if path is None or not Path(path).is_file():
        return ""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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
        bt.logging.info(
            f"🤖 Poker44 jet M3-GB miner started (backend={getattr(self.model, 'backend', 'luck')})"
        )

        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[
                REPO_ROOT / "miner.py",
                REPO_ROOT / "model_luck.py",
                REPO_ROOT / "poker44_ml" / "luck_detector.py",
            ],
            defaults={
                "model_name": os.getenv("POKER44_MODEL_NAME", "jet-markovpot-gb-detector-3"),
                "model_version": os.getenv("POKER44_MODEL_VERSION", "3.7.3"),
                "framework": (
                    "markov-pot-geometry-gb / M3-GB (UID237 jet-detector-3 @2df44d27, "
                    "MIT - see LICENSE-uid237 / ATTRIBUTION-uid237.md): training-free "
                    "chunk scorer combining Markov action-transition entropy deficit, "
                    "pot bet/pot CV regularity, signature concentration, and street "
                    "uniformity via weighted geometric blend + smoothstep anchors "
                    "[0.24, 0.80]. No joblib artifact. Serving adds rank-preserving "
                    "batch-rank remap (POKER44_BATCH_RANK) at POKER44_MAX_POS_FRAC so "
                    "the live validator threshold_sanity gate stays safe without "
                    "changing ranking."
                ),
                "license": "MIT",
                "repo_url": "https://github.com/romanboichuck962/poker",
                "repo_commit": os.getenv("POKER44_MODEL_REPO_COMMIT") or _git_commit(REPO_ROOT),
                "open_source": True,
                "inference_mode": "remote",
                "artifact_sha256": _sha256(MODEL_ARTIFACT),
                "training_data_statement": (
                    "Training-free heuristic (UID237 M3-GB). Operating point "
                    "(POKER44_MAX_POS_FRAC) calibrated on the public Poker44 "
                    "training benchmark (https://api.poker44.net/api/v1/benchmark) "
                    "releases through 2026-07-31 plus unlabeled live captures for "
                    "zero-gate checks. No supervised fit on labels."
                ),
                "training_data_sources": ["https://api.poker44.net/api/v1/benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
                "data_attestation": (
                    "Heuristic scoring; calibration uses the public Poker44 benchmark API."
                ),
                "notes": (
                    "uid242: switched from cold-v14 to UID237's jet-markovpot-gb-detector-3 "
                    "(M3-GB @2df44d27). Training-free; anchors kept at upstream 0.24/0.80. "
                    "Holdout 07-30/31 (n=304): raw AP 0.735, raw reward 0.455 (tsq-limited); "
                    "batch-rank@0.125 window mean ~0.58. Captures (n=1740): raw med 0.35, "
                    "std 0.09, ~10% >=0.5 — rank-map required. Prevalence sweep keeps "
                    "POKER44_MAX_POS_FRAC=0.125 (40/40 capture windows have >=1 positive)."
                ),
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
