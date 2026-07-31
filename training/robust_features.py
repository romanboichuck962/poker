"""Feature allowlist for validator-generalized Poker44 training.

Keeps action-mix, signature, and entropy features that stay meaningful on
miner-visible validator payloads. Drops outcome/position fields cleared by
``prepare_hand_for_miner`` and absolute BB aggregates that drift across eval API
vs static benchmark.
"""

from __future__ import annotations

from typing import Iterable, Sequence

# Substrings that indicate a column is fragile or empty after sanitization.
# Verified 2026-07-01 (benchmark vs real live-format sample):
#   * button_action_share / hero_button_same: button_seat is always 0 on BOTH
#     benchmark and live, so poker44_ml/features.py hard-zeros these (dead const).
#   * *_bb absolute magnitudes (amount_*_bb, pot_*_bb, starting_stack_*_bb):
#     2-11 sigma OOD on the sanitized live feed (live pots/bets ~half benchmark
#     size) -> trees split on benchmark-scale thresholds that collapse on live.
_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "button_action_share",
    "hero_button_same",
    "_bb",
)

# Exact feature names measured z>5 out-of-distribution between the benchmark
# (724 batches, 05-26..07-06) and 300 unique captured LIVE validator chunks
# (2026-07-06/07 snapshots), where z = |mean_live - mean_bench| / std_bench.
# The shifts are STRUCTURAL, not behavioral, so trees split on benchmark-scale
# thresholds that misfire on live inputs:
#   * table size: live seats up to 9 players, benchmark hard-caps at 6
#     (player_count_*);
#   * chunk size: live chunks are 80-100 hands vs benchmark 30-40, shifting
#     hand_count and every min/max/q10/q50/q90 order statistic;
#   * passivity: live preflop call/check rates are 10-18x benchmark
#     (passive_share_*, call_share_*, call_to_share_*, check_share_*, and the
#     passive ngram tokens pCs/pK0/... );
#   * benchmark-degenerate near-constants (actor_entropy_max std~2.8e-13, etc.).
# Dropping the set does NOT cost benchmark AP (5-fold LightGBM 0.8830 -> 0.8842)
# while removing the splits most likely to collapse on live. Exact-match names
# (not substrings) so sibling features at other percentiles are untouched.
_EXCLUDE_EXACT: frozenset[str] = frozenset({
    "hand_count",
    "schema_action_entropy_min",
    "schema_action_run_max_share_max",
    "schema_actor_entropy_max",
    "schema_actor_entropy_q90",
    "schema_actor_switch_rate_q50",
    "schema_call_share_mean",
    "schema_call_share_q10",
    "schema_call_share_q50",
    "schema_call_to_share_mean",
    "schema_call_to_share_q10",
    "schema_call_to_share_q50",
    "schema_check_share_mean",
    "schema_check_share_q10",
    "schema_check_share_q50",
    "schema_fold_share_max",
    "schema_fold_share_q90",
    "schema_ngram_fbs",
    "schema_ngram_fcs",
    "schema_ngram_pcs",
    "schema_ngram_pcs__pcs",
    "schema_ngram_pcs__pf0",
    "schema_ngram_pcs__pk0",
    "schema_ngram_pk0",
    "schema_ngram_pk0__pcs",
    "schema_ngram_pos1c",
    "schema_ngram_pos1k",
    "schema_ngram_rcs",
    "schema_ngram_tcs",
    "schema_passive_share_mean",
    "schema_passive_share_min",
    "schema_passive_share_q10",
    "schema_passive_share_q50",
    "schema_player_count_max",
    "schema_player_count_q90",
    "schema_unique_actor_share_q90",
})

# At least one must appear in the feature name.
_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "hand_count",
    "schema_",
)


def is_robust_feature_name(name: str) -> bool:
    """Return True if ``name`` is safe for live-validator generalization."""
    lowered = str(name).strip().lower()
    if not lowered:
        return False
    # Exact live-OOD drops take priority over the include allowlist (several
    # of these, e.g. hand_count, would otherwise be re-admitted below).
    if lowered in _EXCLUDE_EXACT:
        return False
    if any(token in lowered for token in _EXCLUDE_SUBSTRINGS):
        return False
    return any(token in lowered for token in _INCLUDE_SUBSTRINGS)


def filter_robust_feature_names(names: Sequence[str]) -> list[str]:
    """Stable sorted allowlist intersected with available columns."""
    return sorted(name for name in names if is_robust_feature_name(name))


def summarize_robust_filter(
    all_names: Sequence[str],
    kept: Sequence[str],
) -> dict[str, int | list[str]]:
    dropped = [name for name in all_names if name not in set(kept)]
    return {
        "total": len(all_names),
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_sample": dropped[:12],
    }
