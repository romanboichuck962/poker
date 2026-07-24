"""Derive a live-robust feature allowlist for the cold model from OUR captures.

UID142's robust_features.py hardcodes a blocklist measured on THEIR captures from
2026-07-06/07. That snapshot is now weeks stale, and our deployed cold model shows
the damage: on 920 captured live chunks it scores median 0.834 while benchmark
humans top out at 0.753 -- i.e. it reads essentially all live traffic as more
bot-like than any benchmark human, so live discrimination (and the score) collapses.

This recomputes their z-score method on CURRENT data:
    z = |mean_live - mean_bench| / std_bench
per feature, over live captures vs SIZE-MATCHED sanitized benchmark chunks
(live chunks run ~90 hands, benchmark groups ~30-40, so benchmark hands are
pooled up to the live median before featurising -- comparing raw would flag
every size-sensitive column for the wrong reason).

Features with z <= Z_MAX are written one-per-line to an allowlist consumed by
cold-v1's ROBUST_KEEP_ONLY_FILE env var. cold-v2 dropped that hook, which is why
we train cold-v1 here.
"""
from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path

import numpy as np

from poker44.validator.payload_view import prepare_hand_for_miner
from poker44_ml.features import chunk_features

CAP_DIR = Path("/root/poker242/captures")
BENCH_DIR = Path("/root/POKER44-SUBNET-1/data/benchmark")
OUT = Path("/root/poker242/artifacts/cold_allowlist.txt")
Z_MAX = float(os.environ.get("COLD_Z_MAX", "5.0"))
N_BENCH = int(os.environ.get("COLD_N_BENCH", "400"))
SEED = 7


def featurize(chunk: list[dict]) -> dict[str, float]:
    feats = chunk_features(chunk)
    feats["hand_count"] = float(len(chunk))
    return feats


def load_live() -> list[list[dict]]:
    out = []
    for path in sorted(CAP_DIR.glob("*.json")):
        try:
            chunk = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(chunk, list) and chunk:
            out.append(chunk)
    return out


def load_bench_pooled(target_hands: int, n_groups: int) -> list[list[dict]]:
    """Sanitized benchmark chunks pooled up to ~target_hands, to size-match live."""
    hands_by_label: list[list[dict]] = []
    for path in sorted(BENCH_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        for group in payload["chunks"]:
            for chunk in group["chunks"]:
                visible = [prepare_hand_for_miner(h) for h in chunk]
                if visible:
                    hands_by_label.append(visible)
    rng = random.Random(SEED)
    rng.shuffle(hands_by_label)

    pooled: list[list[dict]] = []
    buf: list[dict] = []
    for chunk in hands_by_label:
        buf.extend(chunk)
        while len(buf) >= target_hands:
            pooled.append(buf[:target_hands])
            buf = buf[target_hands:]
            if len(pooled) >= n_groups:
                return pooled
    return pooled


def main() -> None:
    live_chunks = load_live()
    if not live_chunks:
        raise SystemExit("no captures found; cannot derive allowlist")
    target = int(np.median([len(c) for c in live_chunks]))
    print(f"live: {len(live_chunks)} chunks, median {target} hands")

    bench_chunks = load_bench_pooled(target, N_BENCH)
    print(f"bench: {len(bench_chunks)} size-matched pooled chunks @ {target} hands")

    live_rows = [featurize(c) for c in live_chunks]
    bench_rows = [featurize(c) for c in bench_chunks]
    names = sorted(set().union(*(set(r) for r in live_rows + bench_rows)))

    L = np.array([[r.get(n, 0.0) for n in names] for r in live_rows], dtype=float)
    B = np.array([[r.get(n, 0.0) for n in names] for r in bench_rows], dtype=float)
    print(f"featurized live={L.shape} bench={B.shape}")

    bench_std = B.std(axis=0)
    # A benchmark-constant column carries no information and its z is undefined;
    # treat it as maximally OOD so it never enters the allowlist.
    safe_std = np.where(bench_std > 1e-9, bench_std, np.nan)
    z = np.abs(L.mean(axis=0) - B.mean(axis=0)) / safe_std
    z = np.nan_to_num(z, nan=np.inf)

    keep = [n for n, zi in zip(names, z) if zi <= Z_MAX]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "# live-robust cold feature allowlist\n"
        f"# z = |mean_live-mean_bench|/std_bench <= {Z_MAX}\n"
        f"# live={len(live_chunks)} captures, bench={len(bench_chunks)} pooled @ {target} hands\n"
        + "\n".join(keep)
        + "\n"
    )
    print(f"kept {len(keep)}/{len(names)} features (z<= {Z_MAX}) -> {OUT}")

    worst = np.argsort(-z)[:12]
    print("most-OOD columns (dropped):")
    for i in worst:
        zi = z[i]
        print(f"  z={'inf' if np.isinf(zi) else f'{zi:.1f}'}  {names[i]}")


if __name__ == "__main__":
    main()
