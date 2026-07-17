"""Deterministic rank-preserving score mapping for V4."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


_METADATA_FIELDS = (
    "ante",
    "bb",
    "button_seat",
    "game_type",
    "hand_ended_on_street",
    "hero_seat",
    "limit_type",
    "max_seats",
    "sb",
)
_PLAYER_FIELDS = ("seat", "starting_stack")
_ACTION_FIELDS = (
    "action_type",
    "actor_seat",
    "amount",
    "call_to",
    "normalized_amount_bb",
    "pot_after",
    "pot_before",
    "raise_to",
    "street",
)


def _project(value: Any, fields: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {field: value.get(field) for field in fields if field in value}


def _behavior_payload(hand: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude IDs, cards, outcomes and commitments from deterministic ties."""
    players = [_project(value, _PLAYER_FIELDS) for value in (hand.get("players") or [])]
    players.sort(key=lambda value: json.dumps(value, sort_keys=True, default=str))
    return {
        "metadata": _project(hand.get("metadata"), _METADATA_FIELDS),
        "players": players,
        "actions": [
            _project(value, _ACTION_FIELDS) for value in (hand.get("actions") or [])
        ],
        "streets": [
            _project(value, ("street",)) for value in (hand.get("streets") or [])
        ],
    }


def chunk_tie_key(chunk: Sequence[Mapping[str, Any]]) -> str:
    """Return an order-invariant public-payload fingerprint for tie-breaking."""
    hands = []
    for hand in chunk:
        if not isinstance(hand, Mapping):
            continue
        raw = json.dumps(
            _behavior_payload(hand),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
        hands.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
    return hashlib.sha256("|".join(sorted(hands)).encode("ascii")).hexdigest()


def positive_count(size: int, fraction: float) -> int:
    """Number of scores placed above 0.5 (competitor-compatible floor rule)."""
    if size <= 0 or not 0.0 < float(fraction) < 1.0:
        return 0
    return max(1, min(size, int(math.floor(size * float(fraction)))))


def exact_rank_map(
    scores: Sequence[float] | np.ndarray,
    fraction: float,
    *,
    tie_keys: Sequence[str] | None = None,
    positive_floor: float = 0.501,
    positive_ceiling: float = 0.509,
    negative_floor: float = 0.05,
    negative_ceiling: float = 0.49,
) -> np.ndarray:
    """Map exactly the top fraction above 0.5 while preserving total order.

    Ties are resolved by a behavior-derived key so permuting request chunks
    cannot change which tied chunk crosses the operational threshold.
    """
    raw = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    n = int(raw.size)
    if n == 0:
        return raw
    k = positive_count(n, fraction)
    if k <= 0:
        return np.clip(raw, 0.01, 0.99)
    if tie_keys is None:
        keys = [f"{index:012d}" for index in range(n)]
    else:
        if len(tie_keys) != n:
            raise ValueError("tie_keys must match score count")
        keys = [str(value) for value in tie_keys]

    order = sorted(range(n), key=lambda index: (-float(raw[index]), keys[index]))
    out = np.empty(n, dtype=float)
    positives = order[:k]
    negatives = order[k:]

    for rank, index in enumerate(positives):
        relative = 1.0 if len(positives) <= 1 else 1.0 - rank / (len(positives) - 1)
        out[index] = positive_floor + relative * (positive_ceiling - positive_floor)
    for rank, index in enumerate(negatives):
        relative = 1.0 if len(negatives) <= 1 else 1.0 - rank / (len(negatives) - 1)
        out[index] = negative_floor + relative * (negative_ceiling - negative_floor)
    return np.round(np.clip(out, 0.01, 0.99), 8)


def mapping_metadata(fraction: float) -> dict[str, float | str]:
    return {
        "kind": "exact_rank_budget_v1",
        "top_fraction": float(fraction),
        "positive_floor": 0.501,
        "positive_ceiling": 0.509,
        "negative_floor": 0.05,
        "negative_ceiling": 0.49,
    }
