"""Regenerate artifacts/live_z.npy — the per-feature live-OOD z-vector.

For each of the 433 columns of model.extract_group_features, measures
z = |mean_live - mean_bench| / std_bench, where:
  * live  = all captured validator chunks (/root/poker/captures),
  * bench = sanitized benchmark groups pooled to the live hand-count so the
    comparison is size-matched (deploy_rocket drops columns with z > Z_MAX).

Run whenever the capture pool or benchmark grows, or FEATURE_NAMES changes:
    PYTHONPATH=/root/poker:/root/POKER44-SUBNET-1 python gen_live_z.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/poker")
sys.path.insert(0, "/root/POKER44-SUBNET-1")

from model import FEATURE_NAMES, extract_group_features  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

CAP_DIR = Path("/root/poker/captures")
DATA = Path("/root/POKER44-SUBNET-1/data/benchmark")
OUT = Path("/root/poker/artifacts/live_z.npy")
RNG = np.random.default_rng(20260718)


def load_captures():
    chunks = []
    for p in sorted(CAP_DIR.glob("*.json")):
        h = json.loads(p.read_text())
        if isinstance(h, dict):
            h = h.get("hands") or h.get("chunk") or []
        if h:
            chunks.append(h)
    return chunks


def pool_to(pool, target):
    order = RNG.permutation(len(pool))
    hands, i = [], 0
    while len(hands) < target and i < len(order) * 3:
        hands += list(pool[order[i % len(order)]])
        i += 1
    return hands[:target]


def load_bench_sizematched(target_hands: int, n_per_label: int = 200):
    """Sanitized benchmark groups pooled to ~target_hands, both labels."""
    by_label = {0: [], 1: []}
    for path in sorted(DATA.glob("*.json")):
        payload = json.loads(path.read_text())
        for chunk in payload["chunks"]:
            for group, label in zip(chunk["chunks"], chunk["groundTruth"]):
                by_label[int(label)].append(
                    [prepare_hand_for_miner(hand) for hand in group]
                )
    pooled = []
    for label in (0, 1):
        pool = by_label[label]
        for _ in range(n_per_label):
            pooled.append(pool_to(pool, target_hands))
    return pooled


def main() -> None:
    live_chunks = load_captures()
    if not live_chunks:
        raise SystemExit("no captures found; cannot regenerate live_z")
    live_sizes = [len(c) for c in live_chunks]
    target = int(np.median(live_sizes))
    print(
        f"live: {len(live_chunks)} chunks, hands/chunk "
        f"median={target} p10={int(np.quantile(live_sizes, 0.1))} "
        f"p90={int(np.quantile(live_sizes, 0.9))}",
        flush=True,
    )

    x_live = np.vstack([extract_group_features(c) for c in live_chunks])
    bench = load_bench_sizematched(target)
    x_bench = np.vstack([extract_group_features(c) for c in bench])
    print(f"featurized live={x_live.shape} bench={x_bench.shape}", flush=True)

    mu_b = x_bench.mean(0)
    sd_b = x_bench.std(0) + 1e-9
    z = np.abs(x_live.mean(0) - mu_b) / sd_b
    if z.shape != (len(FEATURE_NAMES),):
        raise SystemExit(f"z has {z.shape} but FEATURE_NAMES={len(FEATURE_NAMES)}")

    OUT.parent.mkdir(exist_ok=True)
    np.save(OUT, z)
    n_ood = int((z > 5.0).sum())
    print(f"saved {OUT}: {len(z)} cols, {n_ood} OOD (z>5)", flush=True)
    worst = np.argsort(-z)[:12]
    for i in worst:
        print(f"  z={z[i]:8.1f}  {FEATURE_NAMES[i]}")


if __name__ == "__main__":
    main()
