"""Poker44 bot-detection model: hero-centric behavioral features + sklearn classifier.

A chunk group is a list of poker hands sharing one focus ("hero") seat. The
model aggregates the hero's behavior across the group into a fixed feature
vector and predicts the probability that the hero is a bot.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

MODEL_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "poker44_model.joblib"

STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}
AGGRESSIVE = ("bet", "raise")
PASSIVE = ("call", "check")


def _hand_features(hand: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Per-hand hero behavior signals. Returns None when the hero is unknown."""
    metadata = hand.get("metadata") or {}
    hero_seat = metadata.get("hero_seat")
    if hero_seat is None:
        return None

    actions = hand.get("actions") or []
    players = hand.get("players") or []
    outcome = hand.get("outcome") or {}
    bb = float(metadata.get("bb") or 0.02) or 0.02

    hero_actions = [a for a in actions if a.get("actor_seat") == hero_seat]
    counts = Counter(a.get("action_type") for a in hero_actions)
    n_hero = max(1, len(hero_actions))

    # street progression of the hero
    hero_streets = [STREET_ORDER.get(a.get("street"), 0) for a in hero_actions]
    max_street = max(hero_streets) if hero_streets else 0

    # preflop behavior
    pre = [a for a in hero_actions if a.get("street") == "preflop"]
    vpip = any(a.get("action_type") in ("call", "bet", "raise") for a in pre)
    pfr = any(a.get("action_type") in ("bet", "raise") for a in pre)
    folded_pre = any(a.get("action_type") == "fold" for a in pre)

    # sizing behavior (in big blinds)
    sizes = [
        float(a.get("normalized_amount_bb") or 0.0)
        for a in hero_actions
        if a.get("action_type") in AGGRESSIVE and (a.get("normalized_amount_bb") or 0) > 0
    ]
    # pot-relative sizing for aggressive actions
    pot_ratios = []
    for a in hero_actions:
        if a.get("action_type") in AGGRESSIVE:
            pot_before = float(a.get("pot_before") or 0.0)
            amt = float(a.get("amount") or 0.0)
            if pot_before > 0 and amt > 0:
                pot_ratios.append(amt / pot_before)

    n_aggr = sum(counts.get(k, 0) for k in AGGRESSIVE)
    n_pass = sum(counts.get(k, 0) for k in PASSIVE)

    hero_player = next((p for p in players if p.get("seat") == hero_seat), {})
    stack_bb = float(hero_player.get("starting_stack") or 0.0) / bb

    winners = outcome.get("winners") or []
    payouts = outcome.get("payouts") or {}
    hero_uid = hero_player.get("player_uid")
    won = (hero_seat in winners) or (hero_uid in winners) or (
        isinstance(payouts, dict) and str(hero_uid) in payouts and float(payouts.get(str(hero_uid)) or 0) > 0
    )

    button_seat = metadata.get("button_seat")
    seat_count = max(1, len(players))
    rel_pos = ((hero_seat - button_seat) % seat_count) / seat_count if button_seat is not None else 0.5

    return {
        "n_actions": float(len(hero_actions)),
        "vpip": float(vpip),
        "pfr": float(pfr),
        "folded_pre": float(folded_pre),
        "fold_rate": counts.get("fold", 0) / n_hero,
        "call_rate": counts.get("call", 0) / n_hero,
        "check_rate": counts.get("check", 0) / n_hero,
        "raise_rate": counts.get("raise", 0) / n_hero,
        "bet_rate": counts.get("bet", 0) / n_hero,
        "aggression": n_aggr / max(1, n_pass),
        "max_street": float(max_street),
        "saw_flop": float(max_street >= 1),
        "saw_showdown": float(bool(outcome.get("showdown"))),
        "won": float(won),
        "stack_bb": stack_bb,
        "rel_pos": rel_pos,
        "n_players": float(len(players)),
        "mean_size_bb": float(np.mean(sizes)) if sizes else 0.0,
        "std_size_bb": float(np.std(sizes)) if len(sizes) > 1 else 0.0,
        "mean_pot_ratio": float(np.mean(pot_ratios)) if pot_ratios else 0.0,
        "std_pot_ratio": float(np.std(pot_ratios)) if len(pot_ratios) > 1 else 0.0,
    }


_HAND_KEYS = [
    "n_actions", "vpip", "pfr", "folded_pre", "fold_rate", "call_rate",
    "check_rate", "raise_rate", "bet_rate", "aggression", "max_street",
    "saw_flop", "saw_showdown", "won", "stack_bb", "rel_pos", "n_players",
    "mean_size_bb", "std_size_bb", "mean_pot_ratio", "std_pot_ratio",
]


def extract_group_features(group: List[Dict[str, Any]]) -> np.ndarray:
    """Aggregate per-hand hero features over a chunk group into one vector."""
    rows = [f for f in (_hand_features(h) for h in (group or [])) if f is not None]
    if not rows:
        return np.zeros(len(_HAND_KEYS) * 2 + 6, dtype=float)

    mat = np.array([[r[k] for k in _HAND_KEYS] for r in rows], dtype=float)
    means = mat.mean(axis=0)
    stds = mat.std(axis=0)

    # group-level consistency signals (bots tend to be more uniform)
    all_sizes = mat[:, _HAND_KEYS.index("mean_size_bb")]
    nonzero_sizes = np.round(all_sizes[all_sizes > 0], 2)
    distinct_size_frac = (
        len(set(nonzero_sizes.tolist())) / len(nonzero_sizes) if len(nonzero_sizes) else 0.0
    )

    action_mix = mat[:, [_HAND_KEYS.index(k) for k in ("fold_rate", "call_rate", "check_rate", "raise_rate", "bet_rate")]].mean(axis=0)
    total = action_mix.sum()
    entropy = 0.0
    if total > 0:
        probs = action_mix / total
        entropy = float(-np.sum([p * math.log(p) for p in probs if p > 0]))

    vpip_series = mat[:, _HAND_KEYS.index("vpip")]
    extras = np.array([
        float(len(rows)),
        distinct_size_frac,
        entropy,
        float(vpip_series.mean()),
        float(vpip_series.std()),
        float(mat[:, _HAND_KEYS.index("n_actions")].sum()),
    ])
    return np.concatenate([means, stds, extras])


FEATURE_NAMES = (
    [f"mean_{k}" for k in _HAND_KEYS]
    + [f"std_{k}" for k in _HAND_KEYS]
    + ["group_hands", "distinct_size_frac", "action_entropy", "vpip_mean", "vpip_std", "total_actions"]
)


def recenter_scores(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Monotone map so `threshold` lands at 0.5.

    Rank metrics (AUC, AP, recall@FPR) are unchanged; only the operating point
    of the hard 0.5 decision moves, so the model flags bots at the calibrated
    false-positive budget instead of at raw probability 0.5.
    """
    prob = np.asarray(prob, dtype=float)
    threshold = float(min(max(threshold, 1e-6), 1 - 1e-6))
    low = 0.5 * prob / threshold
    high = 0.5 + 0.5 * (prob - threshold) / (1.0 - threshold)
    return np.clip(np.where(prob < threshold, low, high), 0.0, 1.0)


class Poker44Model:
    """Serving wrapper: joblib artifact {pipeline, threshold} over group features."""

    def __init__(self, artifact_path: Path | str = MODEL_ARTIFACT):
        import joblib

        artifact = joblib.load(artifact_path)
        if isinstance(artifact, dict):
            self.pipeline = artifact["pipeline"]
            self.threshold = float(artifact.get("threshold", 0.5))
        else:  # backward compatibility with bare-pipeline artifacts
            self.pipeline = artifact
            self.threshold = 0.5

    def score_chunk(self, group: List[Dict[str, Any]]) -> float:
        return self.score_chunks([group])[0]

    def score_chunks(self, groups: List[List[Dict[str, Any]]]) -> List[float]:
        if not groups:
            return []
        features = np.vstack([extract_group_features(g) for g in groups])
        prob = self.pipeline.predict_proba(features)[:, 1]
        return recenter_scores(prob, self.threshold).astype(float).tolist()
