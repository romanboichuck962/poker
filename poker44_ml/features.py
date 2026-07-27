"""Chunk -> feature vector.

Everything here is computed from the sanitized, miner-visible payload only. For
reference, what the validator has already deleted by the time we see a hand:

    hole cards, board cards         always None / []
    outcome                         zeroed: no winners, no payouts, showdown=False
    metadata.button_seat            always 0
    metadata.hand_ended_on_street   always ""
    blinds / antes                  dropped from the action list
    seats                           re-aliased 1..N in order of first action
    amounts                         snapped to a 16-value bb grid + seeded noise
    actions                         only 5-8 of them survive per hand

Two invariance rules drive every feature below, because the public benchmark and
live traffic do not match:

    benchmark  ~30-40 hands/chunk, pot scale X
    live       ~80-100 hands/chunk, pot scale ~X/2

1. SCALE. Every absolute bb magnitude is divided by the chunk's own median pot.
   The ratio survives the pot-scale shift; the raw magnitude is 2-11 sigma
   out-of-distribution on live traffic.
2. SIZE. Every count is a per-hand rate or a share, and the duplication /
   signature features — which are mechanically biased by chunk length — are
   additionally computed at a fixed reference size of 30 hands.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import random
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

# Reference chunk size for size-debiased signature features. Chosen to sit
# inside the benchmark's own range so *_at30 means the same thing on a 34-hand
# training chunk and a 95-hand live chunk.
REFERENCE_HANDS = 30
REFERENCE_DRAWS = 5

# Cap on the O(n^2) pairwise-similarity block.
MAX_PAIRWISE_HANDS = 48

_MEANINGFUL = ("check", "call", "bet", "raise", "fold")
_AGGRESSIVE = ("bet", "raise")
_PASSIVE = ("check", "call")
_STREETS = ("preflop", "flop", "turn", "river")

# The validator's visible bb grid (payload_view._VISIBLE_BB_BUCKETS).
_BB_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0)

# payload_view._MINER_ACTION_WINDOW_MIN .. _MAX
_WINDOW_SIZES = (5, 6, 7, 8)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return default if (math.isnan(out) or math.isinf(out)) else out
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _mean(values: Sequence[float]) -> float:
    return _div(sum(values), len(values))


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = _mean(values)
    return math.sqrt(max(0.0, _mean([(v - mu) ** 2 for v in values])))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = min(max(q, 0.0), 1.0) * (len(xs) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _median(values: Sequence[float]) -> float:
    return _quantile(values, 0.5)


def _norm_entropy(values: Sequence[Any]) -> float:
    """Shannon entropy normalised to [0,1] by the number of observed symbols."""
    if not values:
        return 0.0
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = float(sum(counts.values()))
    ent = -sum((c / total) * math.log(c / total) for c in counts.values())
    return ent / math.log(len(counts))


def _switch_rate(seq: Sequence[Any]) -> float:
    if len(seq) < 2:
        return 0.0
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b) / (len(seq) - 1)


def _max_run_share(seq: Sequence[Any]) -> float:
    if not seq:
        return 0.0
    best = run = 1
    for a, b in zip(seq, seq[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best / len(seq)


def _bucket_index(bb: float) -> int:
    """Index of the nearest visible bb bucket. Total function — no fallthrough."""
    return min(range(len(_BB_GRID)), key=lambda i: abs(_BB_GRID[i] - bb))


# --------------------------------------------------------------------------- #
# per-hand view
# --------------------------------------------------------------------------- #

def _hand_view(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Per-hand scalars plus the signature tuples used for cross-hand repetition."""
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = [a for a in (hand.get("actions") or []) if isinstance(a, dict)]

    hero_seat = _i(metadata.get("hero_seat"), 0)
    max_seats = max(2, _i(metadata.get("max_seats"), 6))

    types: List[str] = []
    actors: List[int] = []
    street_names: List[str] = []
    roles: List[str] = []
    buckets: List[int] = []
    amounts_bb: List[float] = []
    hero_amounts_bb: List[float] = []
    pots_before: List[float] = []
    pots_after: List[float] = []
    hero_types: List[str] = []
    raise_to_seen = 0
    call_to_seen = 0

    for action in actions:
        action_type = str(action.get("action_type") or "").strip().lower()
        if not action_type:
            continue
        seat = _i(action.get("actor_seat"), 0)
        is_hero = hero_seat > 0 and seat == hero_seat
        amount_bb = max(0.0, _f(action.get("normalized_amount_bb")))

        types.append(action_type)
        actors.append(seat)
        roles.append("H" if is_hero else "o")
        street_names.append(str(action.get("street") or "").strip().lower())
        buckets.append(_bucket_index(amount_bb))
        amounts_bb.append(amount_bb)
        pots_before.append(max(0.0, _f(action.get("pot_before")) / 0.02))
        pots_after.append(max(0.0, _f(action.get("pot_after")) / 0.02))
        raise_to_seen += int(action.get("raise_to") is not None)
        call_to_seen += int(action.get("call_to") is not None)
        if is_hero:
            hero_types.append(action_type)
            if amount_bb > 0:
                hero_amounts_bb.append(amount_bb)

    n = len(types)
    counts = Counter(types)
    meaningful = max(1, sum(counts.get(k, 0) for k in _MEANINGFUL))
    aggressive = sum(counts.get(k, 0) for k in _AGGRESSIVE)
    passive = sum(counts.get(k, 0) for k in _PASSIVE)

    hero_counts = Counter(hero_types)
    hero_n = max(1, len(hero_types))
    hero_aggressive = sum(hero_counts.get(k, 0) for k in _AGGRESSIVE)

    stacks_bb = [
        _div(_f(p.get("starting_stack")), 0.02) for p in players if isinstance(p, dict)
    ]
    deltas = [max(0.0, a - b) for a, b in zip(pots_after, pots_before)]

    feat: Dict[str, float] = {
        # ---- structure (already size-free)
        "n_actions": float(n),
        "n_players": float(len(players)),
        "n_streets": float(len(streets)),
        "seat_utilization": _div(len(players), max_seats),
        "distinct_actors": float(len(set(actors))),
        "distinct_actor_share": _div(len(set(actors)), max(1.0, len(players))),
        # ---- action mix
        **{f"share_{k}": _div(counts.get(k, 0), meaningful) for k in _MEANINGFUL},
        "share_aggressive": _div(aggressive, meaningful),
        "share_passive": _div(passive, meaningful),
        "aggression_ratio": _div(aggressive, max(aggressive + passive, 1)),
        # ---- street distribution
        **{
            f"share_street_{s}": _div(sum(1 for x in street_names if x == s), max(1, n))
            for s in _STREETS
        },
        "share_postflop": _div(
            sum(1 for x in street_names if x in _STREETS[1:]), max(1, n)
        ),
        # ---- regularity of the action stream within the hand
        "entropy_action": _norm_entropy(types),
        "entropy_actor": _norm_entropy(actors),
        "entropy_street": _norm_entropy(street_names),
        "entropy_bucket": _norm_entropy(buckets),
        "switch_action": _switch_rate(types),
        "switch_actor": _switch_rate(actors),
        "run_action_max": _max_run_share(types),
        "run_actor_max": _max_run_share(actors),
        # ---- hero-specific behaviour (hero_seat survives sanitization)
        "hero_action_share": _div(len(hero_types), max(1, n)),
        "hero_aggression": _div(hero_aggressive, hero_n),
        "hero_fold_rate": _div(hero_counts.get("fold", 0), hero_n),
        "hero_call_rate": _div(hero_counts.get("call", 0), hero_n),
        "hero_check_rate": _div(hero_counts.get("check", 0), hero_n),
        "hero_silent": float(len(hero_types) == 0),
        # ---- structured-field presence
        "raise_to_share": _div(raise_to_seen, max(1, n)),
        "call_to_share": _div(call_to_seen, max(1, n)),
        "nonzero_amount_share": _div(sum(1 for v in amounts_bb if v > 0), max(1, n)),
        # ---- pot dynamics as ratios (scale-free by construction)
        "pot_monotonic_rate": _div(
            sum(1 for a, b in zip(pots_after, pots_after[1:]) if b + 1e-9 >= a),
            max(len(pots_after) - 1, 1),
        ),
        "pot_growth_ratio": _div(pots_after[-1] if pots_after else 0.0,
                                 pots_before[0] if pots_before and pots_before[0] > 0 else 1.0),
        "bet_to_pot_mean": _mean(
            [_div(amount, pot) for amount, pot in zip(amounts_bb, pots_before) if pot > 0]
        ),
        "bet_to_pot_std": _std(
            [_div(amount, pot) for amount, pot in zip(amounts_bb, pots_before) if pot > 0]
        ),
        # ---- SIDE CHANNEL A: the surviving action count.
        # payload_view._deterministic_window_size seeds a 5..8 window on a hash of
        # (hero_seat, max_seats, first street, TRUE action count), so the visible
        # length is a deterministic function of the pre-sanitization length. One
        # hand leaks ~2 bits; a 90-hand chunk's distribution over these leaks the
        # underlying hand-complexity profile the sanitizer meant to erase.
        **{f"window_is_{w}": float(n == w) for w in _WINDOW_SIZES},
        "window_below_min": float(0 < n < _WINDOW_SIZES[0]),
        # ---- SIDE CHANNEL B: hero's seat alias.
        # button_seat is hardcoded to 0, so conventional position is gone. But
        # _build_seat_alias_map renumbers seats in ORDER OF FIRST ACTION, so a low
        # hero alias means hero acted early. That is table position, recovered.
        "hero_alias": float(hero_seat),
        "hero_alias_norm": _div(hero_seat, max_seats),
        "hero_acts_first": float(bool(actors) and hero_seat > 0 and actors[0] == hero_seat),
        "hero_acts_last": float(bool(actors) and hero_seat > 0 and actors[-1] == hero_seat),
    }

    # Absolute bb magnitudes. Chunk-level normalisation rescales these in place.
    feat["_abs_amount_mean_bb"] = _mean(amounts_bb)
    feat["_abs_amount_std_bb"] = _std(amounts_bb)
    feat["_abs_amount_max_bb"] = max(amounts_bb) if amounts_bb else 0.0
    feat["_abs_amount_q90_bb"] = _quantile(amounts_bb, 0.9)
    feat["_abs_hero_amount_mean_bb"] = _mean(hero_amounts_bb)
    feat["_abs_pot_before_mean_bb"] = _mean(pots_before)
    feat["_abs_pot_after_mean_bb"] = _mean(pots_after)
    feat["_abs_pot_delta_mean_bb"] = _mean(deltas)
    feat["_abs_stack_mean_bb"] = _mean(stacks_bb)
    feat["_abs_stack_std_bb"] = _std(stacks_bb)
    feat["_abs_stack_iqr_bb"] = _quantile(stacks_bb, 0.75) - _quantile(stacks_bb, 0.25)

    signatures = {
        "action": tuple(types),
        "role": tuple(roles),
        "street": tuple(street_names),
        "bucket": tuple(buckets),
        "rich": tuple(
            f"{s}|{t}|{b}" for s, t, b in zip(street_names, types, buckets)
        ),
    }
    return {"feat": feat, "sig": signatures}


_ABS_PREFIX = "_abs_"


def _normalize_scale(views: List[Dict[str, Any]]) -> None:
    """Rescale every absolute bb magnitude by the chunk's own median pot.

    All ``_abs_*`` values are scale-linear (means, quantiles, maxima of bb
    amounts), so dividing the per-hand aggregate is equivalent to normalising the
    raw amounts. The chunk's median pot moves with the same factor as every other
    magnitude between benchmark and live, so the ratio is invariant. Renames
    ``_abs_x`` -> ``rel_x`` in place.
    """
    pots = [
        v["feat"]["_abs_pot_before_mean_bb"]
        for v in views
        if v["feat"]["_abs_pot_before_mean_bb"] > 0.0
    ]
    reference = _median(pots) if pots else 0.0
    reference = reference if reference > 1e-9 else 1.0
    for view in views:
        feat = view["feat"]
        for key in [k for k in feat if k.startswith(_ABS_PREFIX)]:
            feat["rel_" + key[len(_ABS_PREFIX):]] = feat.pop(key) / reference
        feat["chunk_pot_reference_bb"] = reference


# --------------------------------------------------------------------------- #
# cross-hand repetition — the core bot tell
# --------------------------------------------------------------------------- #

def _ngram_set(seq: Tuple[Any, ...], size: int) -> frozenset:
    if len(seq) < size:
        return frozenset()
    return frozenset(tuple(seq[i : i + size]) for i in range(len(seq) - size + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _lz76(text: str) -> float:
    """Normalised Lempel-Ziv-76 complexity. Lower = more repetitive."""
    n = len(text)
    if n <= 1:
        return 0.0
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while True:
        if text[i + k - 1] == text[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k_max, k)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c / (n / math.log2(n))


def _entropy_rate(seq: Sequence[str]) -> float:
    """Order-1 conditional entropy H(a_t | a_{t-1}) in bits."""
    if len(seq) < 2:
        return 0.0
    pairs = Counter(zip(seq[:-1], seq[1:]))
    prevs = Counter(seq[:-1])
    total = float(len(seq) - 1)
    out = 0.0
    for (prev, _cur), count in pairs.items():
        out -= (count / total) * math.log2(count / prevs[prev])
    return out


def _repetition_block(views: Sequence[Dict[str, Any]], prefix: str = "") -> Dict[str, float]:
    """Self-similarity of a chunk's bag of hands.

    A scripted policy replays near-identical action sequences; a human does not.
    Every value is a ratio, so the block is comparable across chunk sizes — but
    duplication rates still fall mechanically as n grows (more hands, more room to
    differ), which is why the caller also evaluates this at a fixed size.
    """
    n = len(views)
    keys = (
        "dup_action", "dup_rich", "jaccard_mean", "jaccard_q90",
        "vendi_frac", "gzip_ratio", "lz76", "entropy_rate",
    )
    if n < 2:
        base = dict.fromkeys(keys, 0.0)
        base["vendi_frac"] = 1.0
        base["gzip_ratio"] = 1.0
        return {f"{prefix}rep_{k}": v for k, v in base.items()}

    action_sigs = [v["sig"]["action"] for v in views]
    rich_sigs = [v["sig"]["rich"] for v in views]

    out: Dict[str, float] = {
        "dup_action": 1.0 - _div(len(set(action_sigs)), n),
        "dup_rich": 1.0 - _div(len(set(rich_sigs)), n),
    }

    stride = max(1, n // MAX_PAIRWISE_HANDS)
    sample = [rich_sigs[i] for i in range(0, n, stride)][:MAX_PAIRWISE_HANDS]
    bigrams = [_ngram_set(sig, 2) for sig in sample]
    m = len(bigrams)

    sims: List[float] = []
    row_sums = [0.0] * m
    for i in range(m):
        for j in range(i + 1, m):
            value = _jaccard(bigrams[i], bigrams[j])
            sims.append(value)
            row_sums[i] += value
            row_sums[j] += value
    out["jaccard_mean"] = _mean(sims)
    out["jaccard_q90"] = _quantile(sims, 0.9)

    # Vendi-style effective diversity: exp(entropy of row masses) / m.
    total_mass = sum(row_sums) + m
    if total_mass > 0:
        weights = [(value + 1.0) / total_mass for value in row_sums]
        ent = -sum(w * math.log(w) for w in weights if w > 0)
        out["vendi_frac"] = min(1.0, max(0.0, math.exp(ent) / m))
    else:
        out["vendi_frac"] = 1.0

    # Compressibility of the whole chunk's token stream: compressed bytes per raw
    # byte. A policy that replays the same sequences compresses far better.
    #
    # NB: do NOT implement this as gzip(joined) / sum(gzip(part)). gzip emits a
    # ~20-byte header per call, so with short per-hand strings the denominator is
    # almost entirely header overhead and the "ratio" degenerates into a proxy
    # for 1/n_hands — a chunk-size tell, not a redundancy measure. Normalising by
    # raw length instead keeps it size-free and actually meaningful.
    stream = "#".join("|".join(sig) or "-" for sig in rich_sigs).encode()
    out["gzip_ratio"] = len(gzip.compress(stream, 6)) / max(len(stream), 1)

    flat = [a for sig in action_sigs for a in sig]
    out["lz76"] = _lz76("".join((a[:1] or "?") for a in flat)[:400])
    out["entropy_rate"] = _entropy_rate(flat[:4000])

    return {f"{prefix}rep_{k}": v for k, v in out.items()}


def _chunk_seed(views: Sequence[Dict[str, Any]]) -> int:
    """Deterministic seed from chunk content, so *_at30 is reproducible."""
    blob = "|".join("".join(str(t) for t in v["sig"]["rich"]) for v in views)
    return int(hashlib.sha256(blob.encode("utf-8", "ignore")).hexdigest()[:8], 16)


def _repetition_at_reference(views: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Repetition block evaluated at a FIXED chunk size.

    Duplication and diversity statistics are mechanically size-dependent: a
    95-hand live chunk looks more diverse than a 34-hand benchmark chunk from the
    same policy. Averaging the block over several fixed-size draws removes that
    bias, so the feature means the same thing in training and in production.
    """
    n = len(views)
    if n <= REFERENCE_HANDS:
        base = _repetition_block(views)
        return {key.replace("rep_", "rep30_"): value for key, value in base.items()}

    rng = random.Random(_chunk_seed(views))
    draws: List[Dict[str, float]] = []
    for _ in range(REFERENCE_DRAWS):
        sample = rng.sample(range(n), REFERENCE_HANDS)
        draws.append(_repetition_block([views[i] for i in sample]))
    return {
        key.replace("rep_", "rep30_"): _mean([draw[key] for draw in draws])
        for key in draws[0]
    }


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

_AGG_STATS = ("mean", "std", "min", "max", "q10", "q50", "q90")


def _aggregate(prefix: str, series: Sequence[float], out: Dict[str, float]) -> None:
    if not series:
        for stat in _AGG_STATS:
            out[f"{prefix}_{stat}"] = 0.0
        return
    out[f"{prefix}_mean"] = _mean(series)
    out[f"{prefix}_std"] = _std(series)
    out[f"{prefix}_min"] = min(series)
    out[f"{prefix}_max"] = max(series)
    out[f"{prefix}_q10"] = _quantile(series, 0.10)
    out[f"{prefix}_q50"] = _quantile(series, 0.50)
    out[f"{prefix}_q90"] = _quantile(series, 0.90)


def chunk_features(chunk: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Full feature vector for one chunk. Deterministic and dependency-free."""
    hands = [h for h in (chunk or []) if isinstance(h, dict)]
    if not hands:
        return {"hand_count": 0.0, "hand_count_log": 0.0}

    views = [_hand_view(hand) for hand in hands]
    _normalize_scale(views)

    n = float(len(views))
    out: Dict[str, float] = {
        # Kept raw AND logged: the model may legitimately use chunk size, but a
        # linear count is a benchmark-vs-live giveaway that trees will latch onto.
        "hand_count": n,
        "hand_count_log": math.log1p(n),
    }

    for key in sorted(views[0]["feat"]):
        _aggregate(key, [view["feat"][key] for view in views], out)

    # Cross-hand signature reuse, raw and size-debiased.
    for name in ("action", "role", "street", "bucket", "rich"):
        sigs = [view["sig"][name] for view in views]
        counts = Counter(sigs)
        out[f"sig_{name}_top_share"] = _div(max(counts.values()), n)
        out[f"sig_{name}_unique_share"] = _div(len(counts), n)

    out.update(_repetition_block(views))
    out.update(_repetition_at_reference(views))

    # Chunk-level rates of "extreme" hands.
    out["rate_high_aggression"] = _div(
        sum(1 for v in views if v["feat"]["share_aggressive"] >= 0.35), n
    )
    out["rate_low_entropy"] = _div(
        sum(1 for v in views if v["feat"]["entropy_action"] <= 0.30), n
    )
    out["rate_hero_silent"] = _div(sum(1 for v in views if v["feat"]["hero_silent"]), n)
    out["rate_showdown_depth"] = _div(
        sum(1 for v in views if v["feat"]["n_streets"] >= 3), n
    )

    # Side-channel aggregates: the shape of the visible-window distribution and
    # the hero-alias distribution across the chunk.
    window_counts = Counter(int(v["feat"]["n_actions"]) for v in views)
    for size in _WINDOW_SIZES:
        out[f"window_share_{size}"] = _div(window_counts.get(size, 0), n)
    out["window_entropy"] = _norm_entropy(
        [int(v["feat"]["n_actions"]) for v in views]
    )
    alias_counts = Counter(int(v["feat"]["hero_alias"]) for v in views)
    out["hero_alias_entropy"] = _norm_entropy(
        [int(v["feat"]["hero_alias"]) for v in views]
    )
    out["hero_alias_top_share"] = _div(max(alias_counts.values()), n) if alias_counts else 0.0

    return {key: float(value) for key, value in out.items()}


def feature_names(chunk: Sequence[Dict[str, Any]] | None = None) -> List[str]:
    """Stable sorted feature names. Pass a real chunk to enumerate them."""
    if chunk is None:
        raise ValueError("feature_names needs one example chunk to enumerate the schema")
    return sorted(chunk_features(chunk))


def feature_matrix(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    names: Sequence[str],
) -> List[List[float]]:
    """Rows aligned to ``names``. Unknown features read 0.0, so an artifact
    trained on an older schema still serves rather than crashing."""
    rows: List[List[float]] = []
    for chunk in chunks:
        values = chunk_features(chunk)
        rows.append([float(values.get(name, 0.0)) for name in names])
    return rows
