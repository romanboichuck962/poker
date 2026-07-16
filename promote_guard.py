"""Promote guards (UID 138–style) to stop bad models from going live.

Rules adapted from perturb-poker-2 autopilot:
  * Train into staging; never overwrite the serving artifact until accepted.
  * Reject if walk-forward hard FPR >= MAX_DEPLOY_FPR (default 6%).
  * Reject if walk-forward reward regresses vs the currently serving meta
    (within REWARD_EPSILON).
  * Snapshot the previous artifact before any promote.
  * Still allow first-ever deploy when nothing is serving (with warning).

These gates specifically reduce live reward=0.0 from:
  * no score >= 0.5 (threshold sanity), or
  * hard FPR past the 10% cliff.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_MAX_DEPLOY_FPR = 0.060
DEFAULT_REWARD_EPSILON = 0.002
DEFAULT_MIN_POS_RATE = 0.02  # must have some hard positives after serve transform
DEFAULT_FPR_IMPROVE_TRADEOFF = 0.010
DEFAULT_REWARD_TRADEOFF = 0.010


@dataclass
class PromoteMetrics:
    """Metrics a candidate must publish before it can replace the live artifact."""

    cv_reward: float
    cv_fpr: float
    cv_ap: float = 0.0
    cv_bot_recall: float = 0.0
    cv_safety: float = 0.0
    positive_rate: float = 0.0
    hard_bot_recall: float = 0.0
    n_dates: int = 0
    walk_forward: Optional[list] = None
    selected: str = ""
    trained_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trained_at"] = self.trained_at or datetime.now(timezone.utc).isoformat()
        return payload


def read_meta(art_dir: Path) -> Optional[dict[str, Any]]:
    path = art_dir / "meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_meta(art_dir: Path, metrics: PromoteMetrics) -> Path:
    art_dir.mkdir(parents=True, exist_ok=True)
    path = art_dir / "meta.json"
    path.write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def backup_artifacts(art_dir: Path, backups_dir: Path, keep: int = 10) -> Optional[Path]:
    if not art_dir.exists():
        return None
    backups_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backups_dir / stamp
    shutil.copytree(art_dir, dest)
    snaps = sorted(p for p in backups_dir.iterdir() if p.is_dir())
    for old in snaps[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return dest


def evaluate_promote_gates(
    candidate: PromoteMetrics,
    baseline: Optional[dict[str, Any]],
    *,
    max_deploy_fpr: float = DEFAULT_MAX_DEPLOY_FPR,
    reward_epsilon: float = DEFAULT_REWARD_EPSILON,
    min_pos_rate: float = DEFAULT_MIN_POS_RATE,
    fpr_improve_tradeoff: float = DEFAULT_FPR_IMPROVE_TRADEOFF,
    reward_tradeoff: float = DEFAULT_REWARD_TRADEOFF,
    force: bool = False,
) -> tuple[bool, list[str]]:
    """Return (accepted, reasons). Reasons non-empty => reject unless first/force."""
    reasons: list[str] = []

    if candidate.cv_safety <= 0.0:
        reasons.append(
            f"safety={candidate.cv_safety:.3f} (threshold sanity would live-score 0.0)"
        )
    if candidate.cv_fpr >= max_deploy_fpr:
        reasons.append(
            f"fpr {candidate.cv_fpr:.4f} >= ceiling {max_deploy_fpr:.4f}"
        )
    if candidate.positive_rate < min_pos_rate:
        reasons.append(
            f"pos@0.5={candidate.positive_rate:.4f} < min {min_pos_rate:.4f} "
            "(risk of no TP above 0.5)"
        )
    if candidate.hard_bot_recall <= 0.0:
        reasons.append("hard_bot_recall=0 (no bot score crossed 0.5 on eval)")

    if baseline is not None and not force:
        old_reward = float(baseline.get("cv_reward", -1.0))
        old_fpr = float(baseline.get("cv_fpr", 1.0))
        reward_ok = candidate.cv_reward >= old_reward - reward_epsilon
        safer = (old_fpr - candidate.cv_fpr) >= fpr_improve_tradeoff
        mild_reward_drop = candidate.cv_reward >= old_reward - reward_tradeoff
        if not reward_ok and not (safer and mild_reward_drop):
            reasons.append(
                f"reward {candidate.cv_reward:.4f} < baseline {old_reward:.4f} - eps "
                f"(and no enough FPR improvement from {old_fpr:.4f})"
            )

    if force:
        return True, reasons

    if reasons and baseline is None:
        # First artifact ever: allow with warning, matching UID 138 behavior.
        return True, reasons

    return (len(reasons) == 0), reasons


def promote_candidate(
    *,
    staging_dir: Path,
    serving_dir: Path,
    artifact_name: str,
    candidate: PromoteMetrics,
    backups_dir: Path,
    max_deploy_fpr: float = DEFAULT_MAX_DEPLOY_FPR,
    reward_epsilon: float = DEFAULT_REWARD_EPSILON,
    min_pos_rate: float = DEFAULT_MIN_POS_RATE,
    force: bool = False,
    history_path: Optional[Path] = None,
) -> tuple[bool, list[str]]:
    """Atomically promote staging artifact into serving_dir when gates pass."""
    staging_artifact = staging_dir / artifact_name
    if not staging_artifact.exists():
        return False, ["staging artifact missing"]

    write_meta(staging_dir, candidate)
    baseline = read_meta(serving_dir)
    accepted, reasons = evaluate_promote_gates(
        candidate,
        baseline,
        max_deploy_fpr=max_deploy_fpr,
        reward_epsilon=reward_epsilon,
        min_pos_rate=min_pos_rate,
        force=force,
    )

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "decision": "promoted" if accepted else "rejected",
        "old_reward": None if baseline is None else float(baseline.get("cv_reward", -1)),
        "new_reward": candidate.cv_reward,
        "fpr": candidate.cv_fpr,
        "safety": candidate.cv_safety,
        "positive_rate": candidate.positive_rate,
        "reasons": reasons,
        "force": force,
    }
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    if not accepted:
        return False, reasons

    serving_dir.mkdir(parents=True, exist_ok=True)
    backup_artifacts(serving_dir, backups_dir)
    final_artifact = serving_dir / artifact_name
    tmp_artifact = serving_dir / (artifact_name + ".tmp")
    shutil.copy2(staging_artifact, tmp_artifact)
    tmp_artifact.replace(final_artifact)
    write_meta(serving_dir, candidate)
    return True, reasons
