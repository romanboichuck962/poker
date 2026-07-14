"""Input-only live-chunk capture for measuring the benchmark->live feature shift.

The validator sanitizes every hand (prepare_hand_for_miner) and re-sends the same
daily snapshot on every query, so deduping incoming chunks by content hash
accumulates the REAL evaluation distribution within about a day (peers report
~21 unique chunks/day out of thousands of queries). We can then compute, per
feature, z = |mean_live - mean_bench| / std_bench and drop the structurally
out-of-distribution columns, and validate a model on the actual eval snapshot
instead of the benchmark proxy.

Contract: strictly best-effort and input-ONLY. It never raises, never blocks,
and never touches the scores the miner returns. Gated by env POKER44_CAPTURE
(default off). Writes one JSON per unique chunk under POKER44_CAPTURE_DIR,
capped at POKER44_CAPTURE_MAX unique chunks. The directory is gitignored.
"""

import hashlib
import json
import os
from pathlib import Path

_ENABLED = os.environ.get("POKER44_CAPTURE", "0") == "1"
_DIR = Path(os.environ.get("POKER44_CAPTURE_DIR", "/root/poker/captures"))
_MAX = int(os.environ.get("POKER44_CAPTURE_MAX", "3000"))

_seen = set()
_ready = False


def _init():
    """Lazily create the dir and load already-captured hashes (survive restarts)."""
    global _ready
    if _ready:
        return
    _DIR.mkdir(parents=True, exist_ok=True)
    for p in _DIR.glob("*.json"):
        _seen.add(p.stem)
    _ready = True


def _chunk_hash(chunk) -> str:
    try:
        payload = json.dumps(chunk, sort_keys=True, default=str)
    except Exception:
        payload = repr(chunk)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def capture_chunks(chunks) -> None:
    """Persist any not-yet-seen chunk. Best-effort; swallows all errors."""
    if not _ENABLED:
        return
    try:
        _init()
        for ch in chunks or []:
            if len(_seen) >= _MAX:
                return
            h = _chunk_hash(ch)
            if h in _seen:
                continue
            _seen.add(h)
            tmp = _DIR / f".{h}.tmp"
            tmp.write_text(json.dumps(ch, default=str))
            tmp.replace(_DIR / f"{h}.json")
    except Exception:
        # capture must never affect serving
        pass
