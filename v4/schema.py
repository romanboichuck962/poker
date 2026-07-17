"""Minimal chunk schema helpers, from pd-coast model_v3/schema.py."""

from __future__ import annotations

from typing import Any, Dict, List


def _action_key(action: Dict[str, Any], fallback: int) -> tuple[int, int]:
    try:
        return int(action.get("action_id")), fallback
    except (TypeError, ValueError):
        return fallback, fallback


def clean_hand(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Defensively normalize a miner-visible hand without inventing behavior."""
    out = dict(hand)
    raw_actions = hand.get("actions") if isinstance(hand.get("actions"), list) else []
    actions = [dict(a) for a in raw_actions if isinstance(a, dict)]
    actions = [a for _, a in sorted(enumerate(actions), key=lambda z: _action_key(z[1], z[0]))]
    out["actions"] = actions
    out["metadata"] = dict(hand.get("metadata") or {})
    out["players"] = [dict(p) for p in (hand.get("players") or []) if isinstance(p, dict)]
    out["streets"] = [dict(s) for s in (hand.get("streets") or []) if isinstance(s, dict)]
    return out
