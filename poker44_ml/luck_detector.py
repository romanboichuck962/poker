"""Markov-predictability + pot-geometry luck detector (variant M3-GB).

Scores a miner-visible chunk by how *mechanically predictable* its decisions are.
Two orthogonal tells drive the primary signal:

  * **Markov transition-predictability deficit** — a scripted seat's next action
    is nearly deterministic given its previous action, so the first-order
    conditional entropy H(next action | current action), pooled over the chunk,
    collapses well below that of human play, whose transitions stay noisy.
  * **Pot-geometry regularity** — a bot sizes off a handful of fixed pot
    fractions, so the coefficient of variation of its bet-to-pot ratios is
    anomalously tight, whereas a human's bet/pot ratios stay dispersed.

A lighter signature-concentration term is retained so the strong ranking on
clearly-replayed chunks is preserved, but the Markov + pot-geometry terms
dominate, giving this fork a genuinely different chunk ordering from the
signature-first siblings. Outputs pass through a smoothstep anchor calibration
(distinct from the sibling linear / logistic / power curves).

Fork fine-tune (GB / geometric-blend): the term blend is a weighted geometric
mean instead of the arithmetic sum, so a chunk only scores high when every
telltale is simultaneously elevated; anchors are re-tuned downward to keep the
0.5 operating point, cleaning the extreme top of the ranking.

Contract: ``score_chunk(chunk) -> float in [0, 1]``, higher == more bot-like.
"""

from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

PROFILE = "markov-pot-geometry-gb"
VARIANT_TAG = "M3-GB"

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
_VOLUNTARY = {"bet", "raise", "allin", "all_in"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _geo_blend(terms: List[tuple], eps: float) -> float:
    """Weighted geometric blend of [0, 1] terms.

    Demands joint agreement: the blend only rises when every telltale is
    simultaneously elevated, which cleans the extreme top of the ranking that
    the recall@FPR reward term reads. ``eps`` keeps one dead term from zeroing
    the product outright; weights renormalize over the active terms.
    """
    eps = min(max(eps, 1e-4), 0.5)
    total_w = sum(w for w, _ in terms if w > 0.0)
    if total_w <= 0.0:
        return 0.0
    acc = 0.0
    for w, x in terms:
        if w <= 0.0:
            continue
        acc += (w / total_w) * math.log(eps + (1.0 - eps) * _clamp01(x))
    return _clamp01((math.exp(acc) - eps) / max(1.0 - eps, 1e-6))


def _smoothstep(t: float) -> float:
    t = _clamp01(t)
    return t * t * (3.0 - 2.0 * t)


class LuckDetector:
    """Markov-predictability + pot-geometry bot detector (variant M3-GB)."""

    PROFILE = PROFILE

    def __init__(
        self,
        *,
        low_anchor: float = 0.24,
        high_anchor: float = 0.80,
        conc_weight: float = 0.55,
        street_weight: float = 0.12,
        sec_weight: float = 0.33,
        markov_vocab: float = 6.0,
        pot_cv_ref: float = 0.70,
        floor: float = 0.05,
        geo_eps: float = 0.06,
    ) -> None:
        self.low_anchor = low_anchor
        self.high_anchor = high_anchor
        self.conc_weight = conc_weight
        self.street_weight = street_weight
        self.sec_weight = sec_weight
        self.markov_vocab = markov_vocab
        self.pot_cv_ref = pot_cv_ref
        self.floor = floor
        self.geo_eps = geo_eps

    @classmethod
    def from_env(cls) -> "LuckDetector":
        return cls(
            low_anchor=_num(os.getenv("LUCK_M_LOW_ANCHOR"), 0.24),
            high_anchor=_num(os.getenv("LUCK_M_HIGH_ANCHOR"), 0.80),
            conc_weight=_num(os.getenv("LUCK_M_CONC_WEIGHT"), 0.55),
            street_weight=_num(os.getenv("LUCK_M_STREET_WEIGHT"), 0.12),
            sec_weight=_num(os.getenv("LUCK_M_SEC_WEIGHT"), 0.33),
            markov_vocab=_num(os.getenv("LUCK_M_MARKOV_VOCAB"), 6.0),
            pot_cv_ref=_num(os.getenv("LUCK_M_POT_CV_REF"), 0.70),
            floor=_num(os.getenv("LUCK_M_FLOOR"), 0.05),
            geo_eps=_num(os.getenv("LUCK_M_GEO_EPS"), 0.06),
        )

    def _action_code(self, a: dict) -> str:
        return _ACTION_CODE.get(str(a.get("action_type", "")).lower(), "?")

    def _hand_signature(self, hand: dict) -> str:
        toks = []
        for a in hand.get("actions") or []:
            if not isinstance(a, dict):
                continue
            st = _STREET_CODE.get(str(a.get("street", "")).lower(), "?")
            toks.append(f"{st}{self._action_code(a)}")
        return ".".join(toks)

    def _concentration(self, hands: List[dict]) -> float:
        n = len(hands)
        sig_counts = Counter(self._hand_signature(h) for h in hands)
        top_share = max(sig_counts.values()) / n
        unique_share = len(sig_counts) / n
        repeat_mass = sum(c for c in sig_counts.values() if c >= 2) / n
        # M-variant concentration mix (0.40/0.40/0.20): distinct from siblings.
        return _clamp01(0.40 * top_share + 0.40 * repeat_mass + 0.20 * (1.0 - unique_share))

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
        return max(shapes.values()) / sum(shapes.values())

    def _markov_deficit(self, hands: List[dict]) -> float:
        """1 - normalized first-order conditional entropy of action transitions."""
        trans: Dict[str, Counter] = defaultdict(Counter)
        for h in hands:
            prev = None
            for a in h.get("actions") or []:
                if not isinstance(a, dict):
                    continue
                cur = self._action_code(a)
                if prev is not None:
                    trans[prev][cur] += 1
                prev = cur
        total = sum(sum(c.values()) for c in trans.values())
        if total <= 0:
            return 0.0
        cond_entropy = 0.0
        for prev, nexts in trans.items():
            row_total = sum(nexts.values())
            p_prev = row_total / total
            row_h = -sum((c / row_total) * math.log(c / row_total) for c in nexts.values())
            cond_entropy += p_prev * row_h
        norm = cond_entropy / math.log(max(self.markov_vocab, 1.0 + 1e-6))
        return _clamp01(1.0 - norm)

    def _pot_geometry_regularity(self, hands: List[dict]) -> float:
        ratios: List[float] = []
        for h in hands:
            for a in h.get("actions") or []:
                if not isinstance(a, dict):
                    continue
                if str(a.get("action_type", "")).lower() in _VOLUNTARY:
                    amt = _num(a.get("normalized_amount_bb"), _num(a.get("amount")))
                    pot = _num(a.get("pot_before"))
                    if amt > 0 and pot > 0:
                        ratios.append(amt / pot)
        if len(ratios) < 4:
            return 0.0
        mean = sum(ratios) / len(ratios)
        if mean <= 0:
            return 0.0
        var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        cv = math.sqrt(var) / mean
        return _clamp01(1.0 - cv / max(self.pot_cv_ref, 1e-6))

    def score_chunk(self, chunk: List[dict]) -> float:
        hands = [h for h in (chunk or []) if isinstance(h, dict)]
        if not hands:
            return 0.5

        concentration = self._concentration(hands)
        street_uni = self._street_uniformity(hands)
        secondary = _clamp01(
            0.60 * self._markov_deficit(hands) + 0.40 * self._pot_geometry_regularity(hands)
        )

        # GB fork: geometric blend demands joint agreement across the terms.
        raw = _geo_blend(
            [
                (self.conc_weight, concentration),
                (self.street_weight, street_uni),
                (self.sec_weight, secondary),
            ],
            self.geo_eps,
        )
        # Smoothstep anchor calibration (distinct curve family from siblings).
        if raw <= self.low_anchor:
            out = self.floor + (0.5 - self.floor) * _smoothstep(raw / max(self.low_anchor, 1e-6))
        elif raw >= self.high_anchor:
            out = 1.0
        else:
            out = 0.5 + 0.5 * _smoothstep(
                (raw - self.low_anchor) / max(self.high_anchor - self.low_anchor, 1e-6)
            )
        return round(_clamp01(out), 6)

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        return [self.score_chunk(list(c or [])) for c in (chunks or [])]

    def debug_components(self, chunks: List[List[dict]]) -> Dict[str, List[float]]:
        mk, pg = [], []
        for c in chunks or []:
            hands = [h for h in (c or []) if isinstance(h, dict)]
            if not hands:
                mk.append(0.0)
                pg.append(0.0)
                continue
            mk.append(self._markov_deficit(hands))
            pg.append(self._pot_geometry_regularity(hands))
        return {"markov_deficit": mk, "pot_geometry_regularity": pg}


def build_luck_detector() -> "LuckDetector":
    return LuckDetector.from_env()
