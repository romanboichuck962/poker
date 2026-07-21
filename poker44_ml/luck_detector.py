"""Sequence-signature luck detector (variant S).

Scores a miner-visible chunk by how much its hands share a common *action
sequence signature*. A scripted seat replays a small number of decision
templates, so its hands collapse onto a handful of street/action n-gram
signatures; human play spreads across many distinct sequences. This variant
builds a per-hand token sequence, hashes it into a signature, and measures
chunk-level signature concentration together with street-progression
uniformity.

Self-contained behavioral scorer used as the live serving backend and as the
calibration prior for the daily retrain. Its signal family (sequence-signature
concentration) and its rank-blended calibration are distinct from the sibling
variants so its per-chunk outputs do not coincide with theirs.

Contract: ``score_chunk(chunk) -> float in [0, 1]``, higher == more bot-like.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Any, Dict, List

PROFILE = "sequence-signature"
VARIANT_TAG = "S"

_ACTION_CODE = {
    "fold": "F",
    "check": "K",
    "call": "C",
    "bet": "B",
    "raise": "R",
    "allin": "A",
    "all_in": "A",
}
_STREET_CODE = {"preflop": "p", "flop": "f", "turn": "t", "river": "r"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _size_code(amt: float) -> str:
    if amt <= 0:
        return "0"
    if amt <= 1.0:
        return "s"
    if amt <= 3.0:
        return "m"
    if amt <= 8.0:
        return "l"
    return "x"


class LuckDetector:
    """Sequence-signature bot detector (variant S)."""

    PROFILE = PROFILE

    def __init__(
        self,
        *,
        low_anchor: float = 0.30,
        high_anchor: float = 0.90,
        street_weight: float = 0.18,
        floor: float = 0.05,
    ) -> None:
        self.low_anchor = low_anchor
        self.high_anchor = high_anchor
        self.street_weight = street_weight
        self.floor = floor

    @classmethod
    def from_env(cls) -> "LuckDetector":
        return cls(
            low_anchor=_num(os.getenv("LUCK_SIG_LOW_ANCHOR"), 0.30),
            high_anchor=_num(os.getenv("LUCK_SIG_HIGH_ANCHOR"), 0.90),
            street_weight=_num(os.getenv("LUCK_SIG_STREET_WEIGHT"), 0.18),
            floor=_num(os.getenv("LUCK_SIG_FLOOR"), 0.05),
        )

    def _hand_signature(self, hand: dict) -> str:
        actions = hand.get("actions") or []
        tokens: List[str] = []
        for a in actions:
            if not isinstance(a, dict):
                continue
            st = _STREET_CODE.get(str(a.get("street", "")).lower(), "?")
            ac = _ACTION_CODE.get(str(a.get("action_type", "")).lower(), "?")
            sz = _size_code(_num(a.get("normalized_amount_bb"), _num(a.get("amount"))))
            tokens.append(f"{st}{ac}{sz}")
        street_shape = "".join(
            _STREET_CODE.get(str(s.get("street", "")).lower(), "?")
            for s in (hand.get("streets") or [])
            if isinstance(s, dict)
        )
        return f"{street_shape}#{'.'.join(tokens)}"

    def _street_uniformity(self, hands: List[dict]) -> float:
        shapes = Counter(
            "".join(
                _STREET_CODE.get(str(s.get("street", "")).lower(), "?")
                for s in (h.get("streets") or [])
                if isinstance(s, dict)
            )
            for h in hands
        )
        if not shapes:
            return 0.0
        total = sum(shapes.values())
        return max(shapes.values()) / total

    def score_chunk(self, chunk: List[dict]) -> float:
        hands = [h for h in (chunk or []) if isinstance(h, dict)]
        if not hands:
            return 0.5
        n = len(hands)
        signatures = [self._hand_signature(h) for h in hands]
        sig_counts = Counter(signatures)

        top_share = max(sig_counts.values()) / n
        unique_share = len(sig_counts) / n
        # Repeat mass: fraction of hands that are NOT the sole instance of their
        # signature (i.e. part of a replayed template).
        repeat_mass = sum(c for c in sig_counts.values() if c >= 2) / n

        concentration = _clamp01(0.45 * top_share + 0.35 * repeat_mass + 0.20 * (1.0 - unique_share))
        street_uni = self._street_uniformity(hands)

        raw = _clamp01(
            (1.0 - self.street_weight) * concentration + self.street_weight * street_uni
        )
        # Piecewise-linear (anchor) calibration: distinct from the sibling
        # logistic/tanh curves. Maps [low_anchor, high_anchor] onto ~[0.5, 1.0]
        # so concentrated (replayed) chunks cross 0.5.
        if raw <= self.low_anchor:
            out = self.floor + (0.5 - self.floor) * (raw / max(self.low_anchor, 1e-6))
        elif raw >= self.high_anchor:
            out = 1.0
        else:
            out = 0.5 + 0.5 * (raw - self.low_anchor) / max(self.high_anchor - self.low_anchor, 1e-6)
        return round(_clamp01(out), 6)

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        return [self.score_chunk(list(c or [])) for c in (chunks or [])]

    def debug_components(self, chunks: List[List[dict]]) -> Dict[str, List[float]]:
        top_out, uniq_out = [], []
        for c in chunks or []:
            hands = [h for h in (c or []) if isinstance(h, dict)]
            if not hands:
                top_out.append(0.0)
                uniq_out.append(1.0)
                continue
            sig_counts = Counter(self._hand_signature(h) for h in hands)
            top_out.append(max(sig_counts.values()) / len(hands))
            uniq_out.append(len(sig_counts) / len(hands))
        return {"sig_top_share": top_out, "sig_unique_share": uniq_out}


def build_luck_detector() -> "LuckDetector":
    return LuckDetector.from_env()
