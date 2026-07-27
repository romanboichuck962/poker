"""Daily learning and guarded competition-cycle deployment for Super Poker 3."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from super_poker.dataset import load_examples
from super_poker.download import update_cache
from super_poker.train import train


@dataclass(frozen=True)
class Gates:
    min_reward: float = 0.85
    min_average_precision: float = 0.92
    max_fpr: float = 0.05
    max_hard_fpr: float = 0.06
    max_reward_regression: float = 0.002
    max_ap_regression: float = 0.002
    min_fold_reward: float = 0.80
    max_fold_hard_fpr: float = 0.10
    max_component_std_mean: float = 0.20
    min_hard_bot_recall: float = 0.60
    min_fold_hard_bot_recall: float = 0.45


@dataclass(frozen=True)
class CycleSchedule:
    anchor_utc: datetime
    duration_hours: int = 120
    lead_hours: int = 6

    @classmethod
    def from_env(cls) -> "CycleSchedule":
        raw = os.getenv("SUPER_POKER_CYCLE_ANCHOR_UTC", "2026-07-16T12:00:00+00:00")
        anchor = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        return cls(
            anchor_utc=anchor.astimezone(timezone.utc),
            duration_hours=max(1, int(os.getenv("SUPER_POKER_CYCLE_HOURS", "120"))),
            lead_hours=max(1, int(os.getenv("SUPER_POKER_CYCLE_LEAD_HOURS", "6"))),
        )


def cycle_status(
    now: datetime | None = None, schedule: CycleSchedule | None = None
) -> dict[str, Any]:
    """Describe the upcoming competition start and its pre-deployment window."""
    schedule = schedule or CycleSchedule.from_env()
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    duration = timedelta(hours=schedule.duration_hours)
    if current < schedule.anchor_utc:
        upcoming = schedule.anchor_utc
    else:
        elapsed = (current - schedule.anchor_utc).total_seconds()
        upcoming = schedule.anchor_utc + (int(elapsed // duration.total_seconds()) + 1) * duration
    window_start = upcoming - timedelta(hours=schedule.lead_hours)
    return {
        "cycle_id": upcoming.isoformat(),
        "competition_start_utc": upcoming.isoformat(),
        "deployment_window_start_utc": window_start.isoformat(),
        "checked_at_utc": current.isoformat(),
        "due": window_start <= current < upcoming,
        "duration_hours": schedule.duration_hours,
        "lead_hours": schedule.lead_hours,
    }


def inspect_cache(data_dir: Path) -> dict[str, Any]:
    examples = load_examples(data_dir)
    labels = {0: 0, 1: 0}
    dates: dict[str, int] = {}
    for example in examples:
        labels[example.label] += 1
        dates[example.source_date] = dates.get(example.source_date, 0) + 1
    return {
        "release_count": len(dates),
        "example_count": len(examples),
        "human_count": labels[0],
        "bot_count": labels[1],
        "latest_source_date": max(dates),
        "examples_per_release": dict(sorted(dates.items())),
    }


def _metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def assess_candidate(
    candidate: dict[str, Any], incumbent: dict[str, Any] | None, gates: Gates = Gates()
) -> dict[str, Any]:
    current = candidate["walk_forward_overall"]
    reasons = []
    if current["reward"] < gates.min_reward:
        reasons.append("reward_below_absolute_minimum")
    if current["average_precision"] < gates.min_average_precision:
        reasons.append("average_precision_below_absolute_minimum")
    if current["fpr"] > gates.max_fpr:
        reasons.append("fpr_above_limit")
    if current["hard_fpr"] > gates.max_hard_fpr:
        reasons.append("hard_fpr_above_limit")
    if current.get("hard_bot_recall", 1.0) < gates.min_hard_bot_recall:
        reasons.append("hard_bot_recall_below_limit")
    fold_rewards = [float(fold["reward"]) for fold in candidate.get("walk_forward") or []]
    if not fold_rewards or min(fold_rewards) < gates.min_fold_reward:
        reasons.append("unstable_or_missing_walk_forward_fold")
    fold_hard_fprs = [float(fold["hard_fpr"]) for fold in candidate.get("walk_forward") or []]
    if not fold_hard_fprs or max(fold_hard_fprs) > gates.max_fold_hard_fpr:
        reasons.append("walk_forward_fold_hard_fpr_above_limit")
    component_std_means = [
        float(fold["component_std_mean"])
        for fold in candidate.get("walk_forward") or []
        if "component_std_mean" in fold
    ]
    if component_std_means and max(component_std_means) > gates.max_component_std_mean:
        reasons.append("ensemble_component_disagreement_above_limit")
    fold_hard_bot_recalls = [
        float(fold["hard_bot_recall"])
        for fold in candidate.get("walk_forward") or []
        if "hard_bot_recall" in fold
    ]
    if fold_hard_bot_recalls and min(fold_hard_bot_recalls) < gates.min_fold_hard_bot_recall:
        reasons.append("walk_forward_fold_hard_bot_recall_below_limit")
    if incumbent:
        previous = incumbent["walk_forward_overall"]
        if current["reward"] < previous["reward"] - gates.max_reward_regression:
            reasons.append("reward_regressed")
        if current["average_precision"] < previous["average_precision"] - gates.max_ap_regression:
            reasons.append("average_precision_regressed")
    return {"approved": not reasons, "reasons": reasons, "candidate": current,
            "incumbent": incumbent.get("walk_forward_overall") if incumbent else None}


def promote(candidate: Path, incumbent: Path, backup_dir: Path) -> Path | None:
    backup = None
    if incumbent.is_file():
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        backup = backup_dir / f"super_poker_3-{stamp}.joblib"
        shutil.copy2(incumbent, backup)
        incumbent_metrics = incumbent.with_suffix(".metrics.json")
        if incumbent_metrics.is_file():
            shutil.copy2(incumbent_metrics, backup.with_suffix(".metrics.json"))
    temporary = incumbent.with_suffix(".promoting")
    shutil.copy2(candidate, temporary)
    temporary.replace(incumbent)
    shutil.copy2(candidate.with_suffix(".metrics.json"), incumbent.with_suffix(".metrics.json"))
    return backup


def distribute_approved(
    candidate: Path, approved_dir: Path, decision: dict[str, Any]
) -> Path:
    """Publish an immutable local copy only after all quality gates pass."""
    approved_dir.mkdir(parents=True, exist_ok=True)
    version = str(
        json.loads(candidate.with_suffix(".metrics.json").read_text(encoding="utf-8"))
        .get("model_version") or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    )
    approved = approved_dir / f"super_poker_3-{version}.joblib"
    temporary = approved.with_suffix(".approving")
    shutil.copy2(candidate, temporary)
    temporary.replace(approved)
    shutil.copy2(candidate.with_suffix(".metrics.json"), approved.with_suffix(".metrics.json"))
    approved.with_suffix(".approval.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    return approved


def run(
    mode: str, data_dir: Path, artifacts: Path, *, download: bool = True,
    train_daily: bool = False, backfill: bool = False,
) -> dict[str, Any]:
    artifacts.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"mode": mode, "started_at": int(time.time())}
    if download:
        result["download"] = update_cache(data_dir, backfill=backfill)
    result["data"] = inspect_cache(data_dir)
    state_path = artifacts / "automation-state.json"

    should_train = mode in {"weekly", "cycle"} or train_daily
    if should_train:
        date = result["data"]["latest_source_date"]
        candidate_dir = artifacts / "candidates"
        candidate = candidate_dir / f"candidate-{date}.joblib"
        model_family = os.getenv("SUPER_POKER_MODEL_FAMILY", "xgboost").strip().lower()
        xgb_profile = os.getenv("SUPER_POKER_XGB_PROFILE", "baseline").strip().lower()
        result["candidate_metadata"] = train(
            data_dir, candidate, model_family=model_family, xgb_profile=xgb_profile
        )
        incumbent = artifacts / "super_poker_3.joblib"
        decision = assess_candidate(
            result["candidate_metadata"], _metrics(incumbent.with_suffix(".metrics.json"))
        )
        result["decision"] = decision
        if mode in {"weekly", "cycle"} and decision["approved"]:
            approved = distribute_approved(candidate, artifacts / "approved", decision)
            backup = promote(candidate, incumbent, artifacts / "backups")
            result["deployed"] = True
            result["approved_artifact"] = str(approved)
            result["backup"] = str(backup) if backup else None
        else:
            result["deployed"] = False
    state_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_cycle(
    data_dir: Path, artifacts: Path, *, force: bool = False,
    download: bool = True, backfill: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run at most once for the upcoming 120-hour competition boundary."""
    status = cycle_status(now)
    cycle_state_path = artifacts / "cycle-state.json"
    previous = _metrics(cycle_state_path) or {}
    if not force and previous.get("completed_cycle_id") == status["cycle_id"]:
        return {"mode": "cycle", "cycle": status, "skipped": "already_completed"}
    if not force and not status["due"]:
        return {"mode": "cycle", "cycle": status, "skipped": "outside_deployment_window"}

    result = run(
        "cycle", data_dir, artifacts, download=download, backfill=backfill
    )
    result["cycle"] = status
    result["completed_cycle_id"] = status["cycle_id"]
    cycle_state_path.parent.mkdir(parents=True, exist_ok=True)
    cycle_state_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (artifacts / "automation-state.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("daily", "weekly", "cycle"))
    parser.add_argument("--data-dir", type=Path, default=Path("../Poker44-subnet/data/raw"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--train-candidate", action="store_true", help="Train during a daily run")
    parser.add_argument("--backfill", action="store_true", help="Download all missing release dates")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-cycle", action="store_true", help="Run cycle workflow immediately")
    args = parser.parse_args()
    if args.mode == "cycle":
        result = run_cycle(
            args.data_dir, args.artifacts, force=args.force_cycle,
            download=not args.no_download, backfill=args.backfill,
        )
    else:
        result = run(args.mode, args.data_dir, args.artifacts, download=not args.no_download,
                     train_daily=args.train_candidate, backfill=args.backfill)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
