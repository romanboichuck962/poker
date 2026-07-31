"""Poker44 miner serving UID157's ckmon stacked coherence model.

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

from model_ckmon import MODEL_ARTIFACT, Poker44Model
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
                REPO_ROOT / "model_ckmon.py",
                REPO_ROOT / "poker44_ml" / "inference.py",
                REPO_ROOT / "poker44_ml" / "features.py",
                REPO_ROOT / "poker44_ml" / "coherence.py",
                REPO_ROOT / "poker44_ml" / "stacked.py",
                REPO_ROOT / "poker44_ml" / "calibration.py",
            ],
            defaults={
                "model_name": "poker44-neptune-ckmon",
                "model_version": "3",
                "framework": (
                    "poker44-ckmon (UID157's method, vendored under poker44_ml/ from "
                    "https://github.com/TakashiAkio/poker44-first @d226c492, MIT - see "
                    "LICENSE-uid157): stacked-v2 of LightGBM+XGBoost+CatBoost+ExtraTrees+"
                    "RandomForest+HistGradientBoosting -> LogisticRegression meta with "
                    "blended isotonic calibration; features = hand order-stats + fixed "
                    "n-gram vocab + pd-coast V4 cross-hand coherence block. Serving "
                    "operating point = within-batch RANK map at 16% positives "
                    "(POKER44_MAX_POS_FRAC); sanitized train==serve."
                ),
                "license": "MIT",
                "repo_url": "https://github.com/romanboichuck962/poker",
                "repo_commit": os.getenv("POKER44_MODEL_REPO_COMMIT") or _git_commit(REPO_ROOT),
                "open_source": True,
                "inference_mode": "remote",
                "artifact_sha256": _sha256(MODEL_ARTIFACT),
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 training benchmark "
                    "(https://api.poker44.net/api/v1/benchmark), releases through "
                    "2026-07-31 (including v1.13), "
                    "each hand passed through the public prepare_hand_for_miner sanitizer so "
                    "training matches serving. Architecture adapted from UID157's public "
                    "poker44-ckmon / TakashiAkio/poker44-first (see training/train_model_v2.py)."
                ),
                "training_data_sources": ["https://api.poker44.net/api/v1/benchmark"],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
                "data_attestation": (
                    "All training data comes from the public Poker44 benchmark API."
                ),
                "notes": (
                    "uid167 v3: SWITCH from artos to UID157 ckmon (coherence stacked-v2 + "
                    "HistGradientBoosting 6-base stack). Retrained on benchmark through "
                    "2026-07-31 (67 releases, 1102 features). Honest holdout 07-30/31: "
                    "reward 0.9617, AP 0.9809, recall@FPR<=0.05 0.8947. Operating point: "
                    "batch-rank at 16% (sweep flat 0.05-0.10 for reward_min; keep 0.16 from "
                    "uid167 live lesson that 0.07 zero-gated). Capture check on 520 chunks: "
                    "40/40 windows have >=1 positive; live scores compressed (med 0.747, "
                    "std 0.005) so rank-map is required."
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
        except Exception as err:
            bt.logging.error(f"model.score_chunks failed: {err}; returning 0.5")
            scores = [0.5] * len(chunks)
        if len(scores) != len(chunks):
            bt.logging.error(
                f"score/chunk length mismatch ({len(scores)} vs {len(chunks)}); padding 0.5"
            )
            scores = list(scores)[: len(chunks)] + [0.5] * max(0, len(chunks) - len(scores))
        synapse.risk_scores = [float(max(0.0, min(1.0, s))) for s in scores]
        synapse.model_manifest = self.model_manifest
        elapsed_ms = (time.monotonic() - started) * 1000.0
        bt.logging.debug(
            f"scored {len(chunks)} chunks in {elapsed_ms:.1f}ms "
            f"digest={self.manifest_digest[:12]}"
        )
        # Input-only, best-effort capture of the live eval distribution for
        # offline diagnosis. Never affects the returned scores.
        capture_chunks(chunks)
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return await self.blacklist_fn(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return await self.priority_fn(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.incentive}")
            time.sleep(5 * 60)
