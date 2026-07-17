"""Order-invariant behavioral distribution features for Poker44 v3."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ACTIONS = ("check", "call", "bet", "raise", "fold")
STREETS = ("preflop", "flop", "turn", "river")
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    p = np.asarray(list(counts.values()), dtype=float) / total
    return float(-(p * np.log(p + 1e-12)).sum() / math.log(max(2, len(counts))))


def _amount_bucket(x: float) -> int:
    bounds = (0.0, 0.25, 0.75, 1.25, 2.25, 4.5, 9.0, 18.0, 50.0)
    return int(np.searchsorted(bounds, max(0.0, x), side="right") - 1)


def hand_features(hand: Dict[str, Any]) -> Tuple[Dict[str, float], Counter, Counter]:
    meta = hand.get("metadata") or {}
    hero = int(_f(meta.get("hero_seat"), 0))
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]
    n = max(1, len(actions))
    types = [str(a.get("action_type") or "").lower() for a in actions]
    streets = [str(a.get("street") or "").lower() for a in actions]
    actors = [int(_f(a.get("actor_seat"), 0)) for a in actions]
    amounts = np.asarray([max(0.0, _f(a.get("normalized_amount_bb"))) for a in actions])
    pots = np.asarray([max(0.0, _f(a.get("pot_after")) / 0.02) for a in actions])
    ratios = amounts / np.maximum(pots, 0.25) if len(actions) else np.zeros(0)
    hero_mask = np.asarray([a == hero and hero > 0 for a in actors], dtype=bool)
    aggressive = np.asarray([t in {"bet", "raise"} for t in types], dtype=bool)
    passive = np.asarray([t in {"check", "call"} for t in types], dtype=bool)

    f: Dict[str, float] = {
        "n_actions": float(len(actions)),
        "n_players": float(len(hand.get("players") or [])),
        "n_actors": float(len(set(a for a in actors if a > 0))),
        "n_streets": float(len(set(s for s in streets if s))),
        "hero_share": float(hero_mask.mean()) if len(actions) else 0.0,
        "aggression": float(aggressive.mean()) if len(actions) else 0.0,
        "passivity": float(passive.mean()) if len(actions) else 0.0,
        "actor_switch_rate": float(np.mean(np.diff(actors) != 0)) if len(actors) > 1 else 0.0,
        "type_entropy": _entropy(types),
        "actor_entropy": _entropy(map(str, actors)),
        "amount_mean": float(amounts.mean()) if amounts.size else 0.0,
        "amount_max": float(amounts.max()) if amounts.size else 0.0,
        "amount_nonzero": float((amounts > 0).mean()) if amounts.size else 0.0,
        "amount_bucket_entropy": _entropy(map(str, map(_amount_bucket, amounts.tolist()))),
        "pot_mean": float(pots.mean()) if pots.size else 0.0,
        "ratio_mean": float(np.clip(ratios, 0, 20).mean()) if ratios.size else 0.0,
        "ratio_max": float(np.clip(ratios, 0, 20).max()) if ratios.size else 0.0,
        "first_is_hero": float(bool(actors) and actors[0] == hero and hero > 0),
        "last_fold": float(bool(types) and types[-1] == "fold"),
        "last_aggressive": float(bool(types) and types[-1] in {"bet", "raise"}),
    }
    for action in ACTIONS:
        f[f"action_{action}"] = types.count(action) / n
        f[f"hero_{action}"] = (
            sum(t == action and hm for t, hm in zip(types, hero_mask)) / max(1, int(hero_mask.sum()))
        )
    for street in STREETS:
        sm = np.asarray([s == street for s in streets], dtype=bool)
        f[f"street_{street}"] = float(sm.mean()) if len(actions) else 0.0
        f[f"street_{street}_agg"] = float(aggressive[sm].mean()) if sm.any() else 0.0

    bigrams = Counter(f"{types[i]}>{types[i+1]}" for i in range(len(types) - 1))
    joints = Counter(f"{s}:{t}" for s, t in zip(streets, types))
    return f, bigrams, joints


def _aggregate(name: str, values: Sequence[float], out: Dict[str, float]) -> None:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        x = np.zeros(1)
    out[f"{name}__mean"] = float(x.mean())
    out[f"{name}__std"] = float(x.std())
    out[f"{name}__mad"] = float(np.median(np.abs(x - np.median(x))))
    for q in QUANTILES:
        out[f"{name}__q{int(q*100):02d}"] = float(np.quantile(x, q))


def chunk_feature_dict(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Permutation-invariant features. Hand count is intentionally excluded."""
    rows: List[Dict[str, float]] = []
    bigram_total: Counter = Counter()
    joint_total: Counter = Counter()
    templates: Counter = Counter()
    for hand in hands:
        row, bigrams, joints = hand_features(hand)
        rows.append(row)
        bigram_total.update(bigrams)
        joint_total.update(joints)
        template = tuple(round(row[k], 2) for k in sorted(row) if k.startswith("action_"))
        templates[template] += 1

    # Stable row order makes floating reductions bit-reproducible while still
    # being a function of the set contents rather than received hand position.
    rows.sort(key=lambda row: hashlib.sha256(
        repr(tuple((k, round(row[k], 10)) for k in sorted(row))).encode()
    ).digest())

    out: Dict[str, float] = {}
    keys = sorted({k for row in rows for k in row})
    for key in keys:
        _aggregate(key, [row.get(key, 0.0) for row in rows], out)

    denom_bi = max(1, sum(bigram_total.values()))
    for a in ACTIONS:
        for b in ACTIONS:
            out[f"bigram__{a}>{b}"] = bigram_total[f"{a}>{b}"] / denom_bi
    denom_joint = max(1, sum(joint_total.values()))
    for street in STREETS:
        for action in ACTIONS:
            out[f"joint__{street}:{action}"] = joint_total[f"{street}:{action}"] / denom_joint

    n = max(1, len(rows))
    out["template_concentration"] = max(templates.values(), default=0) / n
    out["template_unique_rate"] = len(templates) / n
    out["template_entropy"] = _entropy(map(str, templates.elements()))

    # Difference between deterministic hash-independent halves after sorting by
    # feature signature. It measures within-chunk heterogeneity, not hand order.
    if len(rows) >= 4:
        matrix = np.asarray([[r.get(k, 0.0) for k in keys] for r in rows], dtype=float)
        signatures = np.asarray([
            int.from_bytes(hashlib.sha256(np.round(row, 4).tobytes()).digest()[:8], "big")
            for row in matrix
        ], dtype=np.uint64)
        order = np.argsort(signatures, kind="stable")
        left, right = matrix[order[::2]], matrix[order[1::2]]
        m = min(len(left), len(right))
        out["half_disagreement"] = float(np.mean(np.abs(left[:m].mean(0) - right[:m].mean(0))))
    else:
        out["half_disagreement"] = 0.0
    return out


def feature_names() -> List[str]:
    # Build the stable schema from a synthetic empty hand and explicit motif grid.
    return sorted(chunk_feature_dict([{"actions": [], "metadata": {}, "players": [], "streets": []}]))


FEATURE_NAMES = feature_names()


def matrix_for_chunks(chunks: Sequence[Sequence[Dict[str, Any]]]) -> np.ndarray:
    rows = []
    for hands in chunks:
        f = chunk_feature_dict(list(hands))
        rows.append([f.get(name, 0.0) for name in FEATURE_NAMES])
    x = np.asarray(rows, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
