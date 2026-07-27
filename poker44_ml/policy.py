"""Rank-preserving score policy: where to put the 0.5 line.

The reward splits cleanly into two halves that need completely different
treatment:

    0.35*AP + 0.30*recall@fpr<=0.05     pure ranking. Absolute values irrelevant.
    0.20*tsq + 0.10*tsq + 0.05          a gate on the 0.5 threshold.

``tsq`` (threshold_sanity_quality) is not a gradient, it is a cliff with a trap:

    zero chunks scored >= 0.5 that are truly bots  ->  tsq = 0  ->  reward = 0
    at least one true positive and fpr@0.5 <= 0.10 ->  tsq = 1  ->  full 0.35

So 35% of the reward is available for free to anyone who places the 0.5 line
deliberately, and is lost entirely by anyone who lets a raw model probability
decide it. Every competitive miner on this subnet does some version of what is
below; the differences are in the fraction and in tie handling.

The map here is *strictly order-preserving*, so AP and recall@FPR are provably
unchanged by it — it can only ever help.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

# Score bands. Positives are packed just above 0.5 and negatives spread below,
# so a rank-preserving map never manufactures overconfident-looking output.
POSITIVE_FLOOR = 0.501
POSITIVE_CEILING = 0.509
NEGATIVE_FLOOR = 0.010
NEGATIVE_CEILING = 0.490

# Fields that survive sanitization and describe behaviour (not identity). Used
# only for deterministic tie-breaking.
_META_FIELDS = ("game_type", "limit_type", "max_seats", "hero_seat", "sb", "bb", "ante")
_ACTION_FIELDS = (
    "street",
    "actor_seat",
    "action_type",
    "amount",
    "raise_to",
    "call_to",
    "normalized_amount_bb",
    "pot_before",
    "pot_after",
)


def _project(value: Any, fields: Sequence[str]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {field: value.get(field) for field in fields if field in value}


def chunk_tie_key(chunk: Sequence[Mapping[str, Any]]) -> str:
    """Order-invariant behavioural fingerprint of a chunk.

    Two chunks that tie on score must break that tie the same way regardless of
    the order the validator happened to pack them in, or a reordered request
    could flip which chunk crosses 0.5. Hashing hand content (sorted, so hand
    order does not matter) gives a stable key.
    """
    digests: List[str] = []
    for hand in chunk:
        if not isinstance(hand, Mapping):
            continue
        payload = {
            "metadata": _project(hand.get("metadata"), _META_FIELDS),
            "actions": [
                _project(action, _ACTION_FIELDS) for action in (hand.get("actions") or [])
            ],
            "streets": [
                _project(street, ("street",)) for street in (hand.get("streets") or [])
            ],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digests.append(hashlib.sha256(raw.encode("utf-8")).hexdigest())
    return hashlib.sha256("|".join(sorted(digests)).encode("ascii")).hexdigest()


def positive_count(size: int, fraction: float) -> int:
    """How many chunks to place above 0.5. Always >= 1 so the gate cannot fire."""
    if size <= 0:
        return 0
    if not 0.0 < float(fraction) < 1.0:
        return 0
    return max(1, min(size, int(math.floor(size * float(fraction)))))


def rank_map(
    scores: Sequence[float],
    fraction: float,
    *,
    tie_keys: Sequence[str] | None = None,
) -> np.ndarray:
    """Place exactly ``floor(n*fraction)`` scores above 0.5, order preserved.

    Ranking is untouched, so the 65% of reward that is AP + recall@FPR is
    bit-for-bit identical to the raw model's. Only the 0.5 crossing moves.
    """
    raw = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    n = int(raw.size)
    if n == 0:
        return raw
    k = positive_count(n, fraction)
    if k <= 0:
        # Policy disabled (fraction out of range). Pass the raw ranking through.
        return np.clip(raw, NEGATIVE_FLOOR, POSITIVE_CEILING)
    # NB: k == n is reachable (n == 1, where positive_count floors to 0 and is
    # then raised to 1) and MUST fall through to the banding below. Returning the
    # raw score here instead would leave that single chunk wherever the model put
    # it — below 0.5 more often than not — which is precisely the zero-true-
    # positive gate this whole function exists to prevent.

    if tie_keys is None:
        keys = [f"{index:012d}" for index in range(n)]
    elif len(tie_keys) != n:
        raise ValueError(f"tie_keys length {len(tie_keys)} != score count {n}")
    else:
        keys = [str(value) for value in tie_keys]

    order = sorted(range(n), key=lambda index: (-float(raw[index]), keys[index]))
    out = np.empty(n, dtype=float)

    positives, negatives = order[:k], order[k:]
    for rank, index in enumerate(positives):
        share = 1.0 if len(positives) <= 1 else 1.0 - rank / (len(positives) - 1)
        out[index] = POSITIVE_FLOOR + share * (POSITIVE_CEILING - POSITIVE_FLOOR)
    for rank, index in enumerate(negatives):
        share = 1.0 if len(negatives) <= 1 else 1.0 - rank / (len(negatives) - 1)
        out[index] = NEGATIVE_FLOOR + share * (NEGATIVE_CEILING - NEGATIVE_FLOOR)
    return np.round(out, 8)


# --------------------------------------------------------------------------- #
# Choosing the fraction.
#
# Reference miners hardcode this: 0.20, 0.16, 0.10, 0.05 across the four repos
# we studied, trending down over time, none of them derived. It is not a
# constant — the safe range is set by how precise your model is at the top of
# its ranking and by the human share of a live batch:
#
#     fpr@0.5  ~=  (fraction * false_positive_share_of_top_k) / human_share
#
# Too high and fpr crosses 0.10 and tsq decays. Too low and you risk zero true
# positives in the flagged set, which zeroes the entire reward. So sweep it
# against the real reward on held-out data instead of guessing.
# --------------------------------------------------------------------------- #

DEFAULT_FRACTION_GRID = (0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20)

# Bot shares to stress the fraction against. The benchmark is pinned at exactly
# 50% (every release ships 2500 human / 2500 syntheticBot), but live prevalence
# is unknown and a real table is not half bots. Sweeping only at 50% is what
# makes every fraction look equivalent -- at 50% it genuinely is.
#
# The arithmetic that matters: flagging fraction f when the true bot share is p
# puts at most f*n chunks above 0.5, of which at least (f - p)*n must be humans,
# so
#         worst-case fpr@0.5  ->  f / (1 - p)      as p -> 0
#
# tsq holds at 1.0 only while fpr <= 0.10. So f = 0.10 converges on the cliff
# EXACTLY, with zero margin, and f = 0.05 keeps ~2x headroom. Measured on the
# holdout: fpr@0.5 rose 0.0000 -> 0.0839 as the bot share fell 50% -> 2%.
LIVE_PREVALENCE_GRID = (0.50, 0.20, 0.10, 0.05, 0.02)

# Smallest batch that says anything useful about the fraction. Live requests
# carry ~100 chunks; anything much below this is dominated by the granularity of
# k = floor(n * fraction) rather than by the fraction itself.
MIN_INFORMATIVE_BATCH = 50


def evaluate_fraction(
    scores: Sequence[float],
    labels: Sequence[int],
    fraction: float,
    *,
    batches: Sequence[Sequence[int]] | None = None,
) -> Dict[str, float]:
    """Score one candidate fraction through the real reward.

    ``batches`` are index groups emulating separate validator requests. The map
    is batch-relative, so evaluating one giant pool overstates its stability —
    pass realistic request-sized groups.
    """
    from poker44_ml.reward import reward_metrics

    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    groups = [list(range(s.size))] if batches is None else [list(b) for b in batches]

    rewards: List[float] = []
    zero_gates = 0
    fprs: List[float] = []
    skipped = 0
    for group in groups:
        # A batch carries NO information about this knob unless it holds both
        # classes and is big enough to be representative:
        #
        #   single-class -> upstream short-circuits tsq to 1.0 with ap = recall =
        #                   0, so the reward is a constant 0.35 no matter what we
        #                   score. Including these makes every fraction tie at
        #                   0.35 and the sweep silently measures nothing.
        #   tiny         -> a 6-chunk tail batch has k = 1 and one misplaced
        #                   chunk swings fpr to 1.0. Live batches are ~100.
        if len(group) < MIN_INFORMATIVE_BATCH or len(set(y[group].tolist())) < 2:
            skipped += 1
            continue
        mapped = rank_map(s[group], fraction)
        metrics = reward_metrics(y[group], mapped)
        rewards.append(metrics["reward"])
        fprs.append(metrics.get("hard_fpr", 0.0))
        if metrics.get("threshold_sanity_quality", 1.0) <= 0.0:
            zero_gates += 1

    if not rewards:
        return {
            "fraction": float(fraction), "reward_mean": 0.0, "reward_min": 0.0,
            "zero_gate_rate": 1.0, "batches": 0, "skipped": skipped,
            "uninformative": True,
        }
    return {
        "fraction": float(fraction),
        "reward_mean": float(np.mean(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_std": float(np.std(rewards)),
        "fpr_mean": float(np.mean(fprs)),
        "fpr_max": float(np.max(fprs)),
        "zero_gate_rate": zero_gates / len(rewards),
        "batches": len(rewards),
        "skipped": skipped,
        "uninformative": False,
    }


def _prevalence_batches(
    labels: Sequence[int],
    prevalence: float,
    *,
    batch_size: int = 100,
    seed: int = 0,
) -> List[List[int]]:
    """Index batches resampled to a target bot share.

    Keeps every human and subsamples bots, so the ranking under test is never
    altered — only the class mix the reward sees.
    """
    y = np.asarray(labels, dtype=int)
    humans = np.flatnonzero(y == 0)
    bots = np.flatnonzero(y == 1)
    if humans.size == 0 or bots.size == 0:
        return [list(range(y.size))]

    wanted = int(round(humans.size * prevalence / max(1e-9, 1.0 - prevalence)))
    wanted = max(1, min(wanted, bots.size))
    rng = np.random.default_rng(seed)
    index = np.concatenate([humans, rng.choice(bots, size=wanted, replace=False)])
    rng.shuffle(index)
    return [
        index[start:start + batch_size].tolist()
        for start in range(0, index.size, batch_size)
    ]


def choose_fraction(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    batches: Sequence[Sequence[int]] | None = None,
    grid: Sequence[float] = DEFAULT_FRACTION_GRID,
    prevalences: Sequence[float] = LIVE_PREVALENCE_GRID,
    max_zero_gate_rate: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Pick the positive fraction that survives the worst plausible bot share.

    Scored as ``min`` over the prevalence grid, not mean: this knob's downside is
    a whole cycle paying zero, and we do not know the live class mix. Optimising
    the average across prevalences would let a great 50% score paper over a bad
    2% one, which is exactly backwards for a quantity we cannot observe.
    """
    rows: List[Dict[str, Any]] = []
    for fraction in grid:
        per_prevalence = {
            p: evaluate_fraction(
                scores, labels, fraction,
                batches=_prevalence_batches(labels, p, seed=seed),
            )
            for p in prevalences
        }
        usable = {p: r for p, r in per_prevalence.items() if not r.get("uninformative")}
        if not usable:
            rows.append({
                "fraction": float(fraction), "reward_min": 0.0, "reward_mean": 0.0,
                "fpr_max": 1.0, "zero_gate_rate": 1.0, "uninformative": True,
                "by_prevalence": {},
            })
            continue
        # Mean across the batches within a prevalence (single-class and tiny
        # batches already dropped), then MIN across prevalences: we do not know
        # the live class mix, so score each candidate by its worst plausible one.
        rows.append({
            "fraction": float(fraction),
            "reward_min": min(r["reward_mean"] for r in usable.values()),
            "reward_mean": float(np.mean([r["reward_mean"] for r in usable.values()])),
            "fpr_max": max(r.get("fpr_max", 0.0) for r in usable.values()),
            "zero_gate_rate": max(r.get("zero_gate_rate", 1.0) for r in usable.values()),
            "worst_prevalence": float(min(usable, key=lambda p: usable[p]["reward_mean"])),
            "batches": sum(r.get("batches", 0) for r in usable.values()),
            "uninformative": False,
            "by_prevalence": {
                str(p): round(r["reward_mean"], 4) for p, r in usable.items()
            },
        })
    safe = [row for row in rows if row.get("zero_gate_rate", 1.0) <= max_zero_gate_rate]
    pool = safe or rows

    def key(row: Dict[str, float]) -> tuple[float, float]:
        return (round(row.get("reward_min", 0.0), 9), round(row.get("reward_mean", 0.0), 9))

    best_key = max(key(row) for row in pool)
    tied = [row for row in pool if key(row) == best_key]

    # On a well-separated in-distribution slice EVERY fraction ties: AP and
    # recall@FPR are rank-based (the map cannot touch them) and tsq is already
    # 1.0 everywhere, so the reward is genuinely flat in this knob. The sweep has
    # told us it cannot decide — taking the grid's first entry would be an
    # accident, and the two directions are not symmetric:
    #
    #   too LOW  -> few chunks flagged, so under distribution shift the flagged
    #               set can contain zero real bots. That is the hard gate: the
    #               whole cycle pays 0.
    #   too HIGH -> fpr@0.5 creeps over 0.10 and tsq decays *gradually*
    #               (0.20 fpr costs ~0.033 reward, not everything).
    #
    # A bounded linear cost on one side against a total loss on the other. With
    # the prevalence sweep above, ties that survive to here are ties across every
    # bot share we tested -- so take the SMALLEST such fraction, which carries the
    # most fpr headroom (f/(1-p) is the binding constraint as p -> 0) at no
    # measured cost. Resolving this fully still needs live-shaped 80-100 hand
    # chunks; see `degenerate` in the return value.
    selected = sorted(tied, key=lambda row: row["fraction"])[0]

    return {
        "selected": selected["fraction"],
        "selected_metrics": selected,
        "grid": rows,
        "prevalences": list(prevalences),
        "all_candidates_tripped_gate": not safe,
        "degenerate": len(tied) > 1,
        "tied_fractions": [row["fraction"] for row in tied],
    }
