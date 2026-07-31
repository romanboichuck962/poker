"""Input-only capture of live validator queries for OOD diagnosis.

A live query carries NO ground-truth label, so nothing here is used to score or
train directly. Its sole purpose is to record what the live distribution looks
like (the served-score band, and optionally the raw chunk payloads) so we can:

* quantify how far the live 0.5 boundary drifts from the benchmark boundary, and
* rebuild features from real live chunks to recalibrate against them.

Two independent, env-gated modes (both off by default; enabling neither makes
every function a cheap no-op):

* ``POKER44_CAPTURE=1``       -> :func:`capture` writes ONE JSONL record per
  query containing the raw chunks + served scores (heavier; for recalibration).
* ``POKER44_CAPTURE_BATCH=1`` -> :func:`capture_batch` writes ONE compact JSONL
  record per query with only the score distribution summary (lightweight; for
  monitoring the live score_range / bot-rate over time).

Everything is best-effort and MUST NEVER raise into the miner hot path: the
caller already runs this in a worker thread, but we also guard internally so a
serialization or IO error can never drop a validator response.

Env vars
--------
POKER44_CAPTURE            enable full capture (``capture``)
POKER44_CAPTURE_BATCH      enable summary capture (``capture_batch``)
POKER44_CAPTURE_DIR        output directory (default ``<repo>/live_capture``)
POKER44_CAPTURE_MAX_MB     soft cap for the full-capture file per day (default 512)
POKER44_CAPTURE_SAMPLE     fraction of full-capture queries to keep (default 1.0)
"""

from __future__ import annotations

import json
import os
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Per-file locks: capture fires from asyncio.to_thread worker threads, so
# concurrent appends to the same file must be serialized to avoid interleaving.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

# One-time warning latch so a disk-cap / IO problem logs at most once.
_WARNED: set[str] = set()


def _truthy(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    """True when full per-query capture (raw chunks + scores) is on."""
    return _truthy("POKER44_CAPTURE")


def batch_enabled() -> bool:
    """True when compact per-query summary capture is on."""
    return _truthy("POKER44_CAPTURE_BATCH")


def _capture_dir() -> Path:
    return Path(os.getenv("POKER44_CAPTURE_DIR", str(_REPO_ROOT / "live_capture")))


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    try:
        import bittensor as bt

        bt.logging.warning(message)
    except Exception:
        print(f"[live_capture] {message}")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Serialize + append one JSONL record under the file's lock."""
    line = json.dumps(record, default=str, ensure_ascii=False)
    lock = _lock_for(path)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _score_summary(scores: Sequence[float]) -> dict[str, Any]:
    values = [float(s) for s in scores]
    if not values:
        return {
            "num_chunks": 0,
            "score_min": 0.0,
            "score_max": 0.0,
            "score_mean": 0.0,
            "score_median": 0.0,
            "pos_rate_at_0_5": 0.0,
            "bot_count": 0,
            "human_count": 0,
        }
    ordered = sorted(values)
    n = len(ordered)
    median = (
        ordered[n // 2]
        if n % 2
        else 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])
    )
    bot_count = sum(1 for v in values if v >= 0.5)
    return {
        "num_chunks": n,
        "score_min": min(values),
        "score_max": max(values),
        "score_mean": sum(values) / n,
        "score_median": median,
        "pos_rate_at_0_5": bot_count / n,
        "bot_count": bot_count,
        "human_count": n - bot_count,
    }


def capture(
    chunks: List[List[dict]],
    scores: Sequence[float],
    uid: Any = None,
    validator: Any = None,
) -> None:
    """Append the full raw chunks + served scores for one query (heavy path).

    Gated by ``POKER44_CAPTURE``. Honors ``POKER44_CAPTURE_SAMPLE`` (keep a
    fraction) and a per-day ``POKER44_CAPTURE_MAX_MB`` soft cap so an always-on
    miner cannot fill the disk. Never raises.
    """
    try:
        if not enabled():
            return
        try:
            sample = float(os.getenv("POKER44_CAPTURE_SAMPLE", "1.0"))
        except (TypeError, ValueError):
            sample = 1.0
        if sample < 1.0 and random.random() > max(0.0, sample):
            return

        path = _capture_dir() / f"capture_{_today()}.jsonl"
        try:
            max_mb = float(os.getenv("POKER44_CAPTURE_MAX_MB", "512"))
        except (TypeError, ValueError):
            max_mb = 512.0
        if path.exists() and path.stat().st_size >= max_mb * 1024 * 1024:
            _warn_once(
                f"cap:{path}",
                f"live_capture: {path.name} hit {max_mb:.0f}MB cap; "
                "pausing full capture for today (set POKER44_CAPTURE_MAX_MB to raise).",
            )
            return

        record = {
            "ts": _now_iso(),
            "uid": uid,
            "validator": str(validator) if validator is not None else None,
            "num_chunks": len(chunks),
            "scores": [round(float(s), 8) for s in scores],
            "chunks": chunks,
        }
        _append_jsonl(path, record)
    except Exception as err:  # pragma: no cover - never break the miner.
        _warn_once("capture-err", f"live_capture.capture failed (non-fatal): {err}")


def capture_batch(
    chunks: List[List[dict]],
    scores: Sequence[float],
    uid: Any = None,
    validator: Any = None,
) -> None:
    """Append a compact score-distribution summary for one query (light path).

    Gated by ``POKER44_CAPTURE_BATCH``. Keeps the full scores array (small) plus
    a summary so live score_range / bot-rate can be tracked cheaply over time.
    Never raises.
    """
    try:
        if not batch_enabled():
            return
        summary = _score_summary(scores)
        record = {
            "ts": _now_iso(),
            "uid": uid,
            "validator": str(validator) if validator is not None else None,
            "scores": [round(float(s), 8) for s in scores],
            **summary,
        }
        _append_jsonl(_capture_dir() / f"batch_{_today()}.jsonl", record)
    except Exception as err:  # pragma: no cover - never break the miner.
        _warn_once(
            "capture-batch-err",
            f"live_capture.capture_batch failed (non-fatal): {err}",
        )
