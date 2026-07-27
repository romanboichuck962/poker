"""Explicit feature policy for validator-stable model candidates."""

from __future__ import annotations

from collections.abc import Sequence


# These fields depend on a stable hero seat, absolute table composition, or
# chunk length. Validator payloads and competition batch sizes can change all
# three independently of bot behavior.
DRIFT_PRONE_TOKENS = (
    "hero_",
    "player_count",
    "seat_utilization",
    "showdown",
    "stack_",
)
DRIFT_PRONE_EXACT = {"hand_count", "log_hand_count"}


def validator_stable_features(names: Sequence[str]) -> list[str]:
    """Return a deterministic allowlist that avoids known payload drift."""
    return [
        name
        for name in sorted(names)
        if name not in DRIFT_PRONE_EXACT
        and not any(token in name for token in DRIFT_PRONE_TOKENS)
    ]


def feature_policy_report(names: Sequence[str], kept: Sequence[str]) -> dict:
    kept_set = set(kept)
    dropped = sorted(name for name in names if name not in kept_set)
    return {
        "policy": "validator-stable-v1",
        "total": len(names),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_features": dropped,
    }
