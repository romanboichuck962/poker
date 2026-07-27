"""Benchmark -> miner-visible projection, with a train/serve parity check.

The public benchmark is documented as exposing "the miner-visible chunk
payload". Measured against the real sanitizer, that is not quite true: raw
benchmark hands have their first action on seat 1 only ~24% of the time, while
every hand a validator actually sends has it on seat 1 100% of the time,
because ``payload_view._build_seat_alias_map`` renumbers seats in order of
first action.

Training on unprojected data therefore trains on a distribution the miner never
sees. The projection is one function call; the check that it worked is the
valuable part, so it is wired into the loader rather than left as a comment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from poker44.validator.payload_view import prepare_hand_for_miner

# Below this share of first-action-on-seat-1, the chunk did not go through the
# validator's aliasing and must not be trained on. Genuine miner-visible data
# sits at 1.0; a hand whose action list is empty contributes nothing either way.
SEAT1_MIN_SHARE = 0.98


def project_hand(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Project one raw hand into exactly what a validator would send."""
    return prepare_hand_for_miner(hand)


def project_chunk(chunk: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project a chunk, dropping non-dict and empty-after-projection hands."""
    out: List[Dict[str, Any]] = []
    for hand in chunk:
        if not isinstance(hand, dict):
            continue
        projected = project_hand(hand)
        if projected.get("actions"):
            out.append(projected)
    return out


def first_action_seat1_share(chunks: Sequence[Sequence[Dict[str, Any]]]) -> float:
    """Share of hands whose first visible action is on seat alias 1.

    This is the canonicalization tell. ~0.24 means raw benchmark data; 1.0 means
    the hand went through the validator's seat aliasing.
    """
    total = 0
    seat1 = 0
    for chunk in chunks:
        for hand in chunk or []:
            actions = (hand or {}).get("actions") or []
            if not actions:
                continue
            total += 1
            try:
                if int(actions[0].get("actor_seat") or 0) == 1:
                    seat1 += 1
            except (TypeError, ValueError):
                continue
    return seat1 / total if total else 0.0


@dataclass
class CanonicalReport:
    hands: int
    seat1_share_before: float
    seat1_share_after: float
    chunks_in: int
    chunks_out: int

    @property
    def ok(self) -> bool:
        return self.seat1_share_after >= SEAT1_MIN_SHARE

    def __str__(self) -> str:
        return (
            f"canonicalized {self.chunks_out}/{self.chunks_in} chunks, {self.hands} hands | "
            f"first-action-is-seat1: {self.seat1_share_before:.1%} -> "
            f"{self.seat1_share_after:.1%}"
        )


def canonicalize(
    chunks: Sequence[Sequence[Dict[str, Any]]],
    *,
    strict: bool = True,
) -> tuple[List[List[Dict[str, Any]]], CanonicalReport]:
    """Project every chunk and verify the projection actually took effect.

    Returns ``(projected_chunks, report)``. With ``strict`` (the default) a
    projection that did not reach the expected seat-1 share raises, because the
    only way that happens is a sanitizer change — in which case the artifact you
    were about to train is already wrong.
    """
    before = first_action_seat1_share(chunks)
    projected = [project_chunk(chunk) for chunk in chunks]
    kept = [chunk for chunk in projected if chunk]
    after = first_action_seat1_share(kept)

    report = CanonicalReport(
        hands=sum(len(chunk) for chunk in kept),
        seat1_share_before=before,
        seat1_share_after=after,
        chunks_in=len(chunks),
        chunks_out=len(kept),
    )
    if strict and kept and not report.ok:
        raise RuntimeError(
            f"canonicalization did not take effect ({report}). Expected the "
            f"post-projection seat-1 share to be >= {SEAT1_MIN_SHARE:.0%}. "
            "poker44/validator/payload_view.py has probably changed — read it "
            "before training anything."
        )
    return kept, report
