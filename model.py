"""Poker44 bot-detection model: hero-centric behavioral features + classifier.

A chunk group is a list of poker hands sharing one focus ("hero") seat. The
model aggregates the hero's behavior across the group into a fixed feature
vector and predicts the probability that the hero is a bot.

Feature design targets the tells that separate bots from humans in a payload
with no timing data: bet-sizing regularity ("roundness" and low variance),
per-street aggression structure, response to aggression, and cross-hand
consistency of decisions.
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
# Common human pot-fraction bet sizes; bots often snap to an exact subset.
POT_FRACTIONS = np.array([0.25, 0.33, 0.4, 0.5, 0.6, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0])


def _roundness(amount_bb: float, pot_ratio: Optional[float]) -> float:
    """1.0 if a size looks 'clean' (round bb or a canonical pot fraction)."""
    score = 0.0
    if amount_bb > 0:
        # multiples of 0.5bb
        if abs(round(amount_bb * 2) / 2 - amount_bb) < 0.02:
            score = max(score, 1.0)
        # whole bb
        if abs(round(amount_bb) - amount_bb) < 0.02:
            score = max(score, 1.0)
    if pot_ratio and pot_ratio > 0:
        if float(np.min(np.abs(POT_FRACTIONS - pot_ratio))) < 0.02:
            score = max(score, 1.0)
    return score


def _hand_features(hand: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Per-hand hero behavior signals. Returns None when the hero is unknown."""
    metadata = hand.get("metadata") or {}
    hero_seat = metadata.get("hero_seat")
    if hero_seat is None:
        return None

    actions = hand.get("actions") or []
    players = hand.get("players") or []
    outcome = hand.get("outcome") or {}
    streets = hand.get("streets") or []
    bb = float(metadata.get("bb") or 0.02) or 0.02

    hero_actions = [a for a in actions if a.get("actor_seat") == hero_seat]
    counts = Counter(a.get("action_type") for a in hero_actions)
    n_hero = max(1, len(hero_actions))

    hero_streets = [STREET_ORDER.get(a.get("street"), 0) for a in hero_actions]
    max_street = max(hero_streets) if hero_streets else 0

    def street_rates(street: str) -> tuple[float, float]:
        acts = [a for a in hero_actions if a.get("street") == street]
        if not acts:
            return 0.0, 0.0
        c = Counter(a.get("action_type") for a in acts)
        aggr = sum(c.get(k, 0) for k in AGGRESSIVE) / len(acts)
        passive = sum(c.get(k, 0) for k in PASSIVE) / len(acts)
        return aggr, passive

    pre_aggr, pre_pass = street_rates("preflop")
    flop_aggr, flop_pass = street_rates("flop")
    turn_aggr, _ = street_rates("turn")
    river_aggr, _ = street_rates("river")

    pre = [a for a in hero_actions if a.get("street") == "preflop"]
    vpip = any(a.get("action_type") in ("call", "bet", "raise") for a in pre)
    pfr = any(a.get("action_type") in ("bet", "raise") for a in pre)
    folded_pre = any(a.get("action_type") == "fold" for a in pre)
    n_pre_raises = sum(1 for a in pre if a.get("action_type") == "raise")

    # response to aggression: did hero fold facing a bet/raise?
    faced_bet = folds_to_bet = 0
    prev_aggr_by_other = False
    for a in actions:
        seat = a.get("actor_seat")
        atype = a.get("action_type")
        if seat == hero_seat:
            if prev_aggr_by_other:
                faced_bet += 1
                if atype == "fold":
                    folds_to_bet += 1
            prev_aggr_by_other = False
        elif atype in AGGRESSIVE:
            prev_aggr_by_other = True
        elif atype in ("call", "check", "fold"):
            prev_aggr_by_other = False
    fold_to_bet = folds_to_bet / faced_bet if faced_bet else 0.0

    # action-sequence patterns (mechanical bot lines): scan hero actions in order
    check_raise = bet_fold = call_raise = limp_reraise = 0
    by_street: Dict[str, list] = {}
    for a in hero_actions:
        by_street.setdefault(a.get("street"), []).append(a.get("action_type"))
    for street, seq in by_street.items():
        for i in range(len(seq) - 1):
            if seq[i] == "check" and seq[i + 1] in AGGRESSIVE:
                check_raise += 1
            if seq[i] == "call" and seq[i + 1] == "raise":
                call_raise += 1
                if street == "preflop":
                    limp_reraise += 1
    hero_seq = [a.get("action_type") for a in hero_actions]
    for i in range(len(hero_seq) - 1):
        if hero_seq[i] == "bet" and "fold" in hero_seq[i + 1:]:
            bet_fold += 1
            break

    # bet-sizing statistics (hero aggressive actions)
    sizes_bb, pot_ratios, round_flags = [], [], []
    for a in hero_actions:
        if a.get("action_type") in AGGRESSIVE:
            amt_bb = float(a.get("normalized_amount_bb") or 0.0)
            pot_before = float(a.get("pot_before") or 0.0)
            amt = float(a.get("amount") or 0.0)
            pr = amt / pot_before if pot_before > 0 and amt > 0 else None
            if amt_bb > 0:
                sizes_bb.append(amt_bb)
                round_flags.append(_roundness(amt_bb, pr))
            if pr:
                pot_ratios.append(pr)

    sizes_arr = np.array(sizes_bb) if sizes_bb else np.array([0.0])
    pot_arr = np.array(pot_ratios) if pot_ratios else np.array([0.0])
    size_cv = float(sizes_arr.std() / sizes_arr.mean()) if sizes_arr.mean() > 0 else 0.0

    n_aggr = sum(counts.get(k, 0) for k in AGGRESSIVE)
    n_pass = sum(counts.get(k, 0) for k in PASSIVE)

    hero_player = next((p for p in players if p.get("seat") == hero_seat), {})
    stack_bb = float(hero_player.get("starting_stack") or 0.0) / bb

    winners = outcome.get("winners") or []
    payouts = outcome.get("payouts") or {}
    hero_uid = hero_player.get("player_uid")
    won = (hero_seat in winners) or (hero_uid in winners) or (
        isinstance(payouts, dict) and str(hero_uid) in payouts
        and float(payouts.get(str(hero_uid)) or 0) > 0
    )

    button_seat = metadata.get("button_seat")
    seat_count = max(1, len(players))
    rel_pos = ((hero_seat - button_seat) % seat_count) / seat_count if button_seat is not None else 0.5

    total_pot_bb = float(outcome.get("total_pot") or 0.0) / bb
    n_board_streets = sum(1 for s in streets if isinstance(s, dict) and s.get("board_cards"))

    return {
        "n_actions": float(len(hero_actions)),
        "vpip": float(vpip),
        "pfr": float(pfr),
        "vpip_pfr_gap": float(vpip) - float(pfr),
        "folded_pre": float(folded_pre),
        "n_pre_raises": float(n_pre_raises),
        "fold_rate": counts.get("fold", 0) / n_hero,
        "call_rate": counts.get("call", 0) / n_hero,
        "check_rate": counts.get("check", 0) / n_hero,
        "raise_rate": counts.get("raise", 0) / n_hero,
        "bet_rate": counts.get("bet", 0) / n_hero,
        "aggression": n_aggr / max(1, n_pass),
        "aggr_frac": n_aggr / n_hero,
        "pre_aggr": pre_aggr,
        "flop_aggr": flop_aggr,
        "turn_aggr": turn_aggr,
        "river_aggr": river_aggr,
        "postflop_aggr": (flop_aggr + turn_aggr + river_aggr) / 3.0,
        "fold_to_bet": fold_to_bet,
        "faced_bet": float(faced_bet),
        "max_street": float(max_street),
        "saw_flop": float(max_street >= 1),
        "saw_turn": float(max_street >= 2),
        "saw_river": float(max_street >= 3),
        "saw_showdown": float(bool(outcome.get("showdown"))),
        "n_board_streets": float(n_board_streets),
        "won": float(won),
        "stack_bb": stack_bb,
        "rel_pos": rel_pos,
        "n_players": float(len(players)),
        "total_pot_bb": total_pot_bb,
        "mean_size_bb": float(sizes_arr.mean()),
        "std_size_bb": float(sizes_arr.std()),
        "size_cv": size_cv,
        "min_size_bb": float(sizes_arr.min()),
        "max_size_bb": float(sizes_arr.max()),
        "mean_pot_ratio": float(pot_arr.mean()),
        "std_pot_ratio": float(pot_arr.std()),
        "roundness": float(np.mean(round_flags)) if round_flags else 0.0,
        "n_aggr_actions": float(n_aggr),
        "check_raise": float(check_raise),
        "bet_fold": float(bet_fold),
        "call_raise": float(call_raise),
        "limp_reraise": float(limp_reraise),
        "_pot_ratios": pot_ratios,   # popped for group-level pooling
        "_sizes_bb": sizes_bb,
    }


_HAND_KEYS = [
    "n_actions", "vpip", "pfr", "vpip_pfr_gap", "folded_pre", "n_pre_raises",
    "fold_rate", "call_rate", "check_rate", "raise_rate", "bet_rate",
    "aggression", "aggr_frac", "pre_aggr", "flop_aggr", "turn_aggr",
    "river_aggr", "postflop_aggr", "fold_to_bet", "faced_bet", "max_street",
    "saw_flop", "saw_turn", "saw_river", "saw_showdown", "n_board_streets",
    "won", "stack_bb", "rel_pos", "n_players", "total_pot_bb", "mean_size_bb",
    "std_size_bb", "size_cv", "min_size_bb", "max_size_bb", "mean_pot_ratio",
    "std_pot_ratio", "roundness", "n_aggr_actions",
    "check_raise", "bet_fold", "call_raise", "limp_reraise",
]

# pooled pot-ratio histogram buckets (bots concentrate; humans spread)
_POT_BUCKETS = [0.0, 0.4, 0.6, 0.8, 1.1, np.inf]

_EXTRA_KEYS = [
    "group_hands", "distinct_size_frac", "action_entropy", "vpip_mean",
    "vpip_std", "total_actions", "mean_roundness", "size_bb_global_cv",
    "pot_ratio_global_cv", "aggr_consistency", "showdown_rate", "win_rate",
    "pot_hist_0", "pot_hist_1", "pot_hist_2", "pot_hist_3", "pot_hist_4",
    "pot_modal_dominance", "pot_ratio_entropy", "distinct_pot_frac",
    "size_bb_entropy", "distinct_size_bb_frac", "n_aggr_pool",
]

FEATURE_NAMES = (
    [f"mean_{k}" for k in _HAND_KEYS]
    + [f"std_{k}" for k in _HAND_KEYS]
    + [f"q25_{k}" for k in _HAND_KEYS]
    + [f"q75_{k}" for k in _HAND_KEYS]
    + _EXTRA_KEYS
)
FEATURE_DIM = len(FEATURE_NAMES)


def extract_group_features(group: List[Dict[str, Any]]) -> np.ndarray:
    """Aggregate per-hand hero features over a chunk group into one vector."""
    rows = [f for f in (_hand_features(h) for h in (group or [])) if f is not None]
    if not rows:
        return np.zeros(FEATURE_DIM, dtype=float)

    # pool every aggressive-action size across the whole group before aggregating
    pooled_pr = [pr for r in rows for pr in r.pop("_pot_ratios")]
    pooled_bb = [s for r in rows for s in r.pop("_sizes_bb")]

    mat = np.array([[r[k] for k in _HAND_KEYS] for r in rows], dtype=float)
    means = mat.mean(axis=0)
    stds = mat.std(axis=0)
    q25 = np.quantile(mat, 0.25, axis=0)
    q75 = np.quantile(mat, 0.75, axis=0)

    # group-level consistency signals (bots tend to be more uniform)
    all_sizes = mat[:, _HAND_KEYS.index("mean_size_bb")]
    nonzero_sizes = np.round(all_sizes[all_sizes > 0], 2)
    distinct_size_frac = (
        len(set(nonzero_sizes.tolist())) / len(nonzero_sizes) if len(nonzero_sizes) else 0.0
    )

    action_mix = mat[:, [_HAND_KEYS.index(k) for k in
                         ("fold_rate", "call_rate", "check_rate", "raise_rate", "bet_rate")]].mean(axis=0)
    total = action_mix.sum()
    entropy = 0.0
    if total > 0:
        probs = action_mix / total
        entropy = float(-np.sum([p * math.log(p) for p in probs if p > 0]))

    def global_cv(key: str) -> float:
        col = mat[:, _HAND_KEYS.index(key)]
        col = col[col > 0]
        return float(col.std() / col.mean()) if len(col) and col.mean() > 0 else 0.0

    # pooled bet-sizing distribution over all aggressive actions in the group
    pr_arr = np.array(pooled_pr) if pooled_pr else np.array([])
    if pr_arr.size:
        hist, _ = np.histogram(pr_arr, bins=_POT_BUCKETS)
        hist = hist / hist.sum()
        modal_dominance = float(hist.max())
        pot_entropy = float(-np.sum([p * math.log(p) for p in hist if p > 0]))
        distinct_pot_frac = len(set(np.round(pr_arr, 1).tolist())) / len(pr_arr)
    else:
        hist = np.zeros(len(_POT_BUCKETS) - 1)
        modal_dominance = pot_entropy = distinct_pot_frac = 0.0

    bb_arr = np.array(pooled_bb) if pooled_bb else np.array([])
    if bb_arr.size:
        rounded_bb = np.round(bb_arr, 1)
        vals, cnts = np.unique(rounded_bb, return_counts=True)
        probs_bb = cnts / cnts.sum()
        size_bb_entropy = float(-np.sum(probs_bb * np.log(probs_bb)))
        distinct_size_bb_frac = len(vals) / len(bb_arr)
    else:
        size_bb_entropy = distinct_size_bb_frac = 0.0

    vpip_series = mat[:, _HAND_KEYS.index("vpip")]
    aggr_series = mat[:, _HAND_KEYS.index("aggr_frac")]
    extras = np.array([
        float(len(rows)),
        distinct_size_frac,
        entropy,
        float(vpip_series.mean()),
        float(vpip_series.std()),
        float(mat[:, _HAND_KEYS.index("n_actions")].sum()),
        float(mat[:, _HAND_KEYS.index("roundness")].mean()),
        global_cv("mean_size_bb"),
        global_cv("mean_pot_ratio"),
        1.0 - float(aggr_series.std()),  # high = very consistent aggression
        float(mat[:, _HAND_KEYS.index("saw_showdown")].mean()),
        float(mat[:, _HAND_KEYS.index("won")].mean()),
        float(hist[0]), float(hist[1]), float(hist[2]), float(hist[3]), float(hist[4]),
        modal_dominance,
        pot_entropy,
        distinct_pot_frac,
        size_bb_entropy,
        distinct_size_bb_frac,
        float(len(pooled_pr)),
    ])
    return np.concatenate([means, stds, q25, q75, extras])


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
        scores = recenter_scores(prob, self.threshold)
        # A group with no parseable hero hands yields an all-zero feature vector;
        # return a neutral 0.5 rather than whatever the classifier extrapolates.
        empty = ~features.any(axis=1)
        scores[empty] = 0.5
        return scores.astype(float).tolist()
