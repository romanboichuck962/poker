"""Train-vs-target feature drift (PSI / KS).

The public benchmark is not the live distribution: benchmark chunks run ~30-40
hands at pot scale X, live chunks run ~80-100 hands at roughly X/2. You cannot
measure accuracy on live traffic (no labels), but you can measure how far each
feature has moved — which tells you which features to distrust.

    python -m poker44_ml.drift --target data/live/captured.json

High-PSI features are the ones most likely to mislead the model in production.
Anything above ~0.25 is a serious shift; the fix is to make the feature
invariant, not to hope the model generalises.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from poker44_ml.features import chunk_features, feature_matrix

PSI_MINOR = 0.10
PSI_MAJOR = 0.25


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index over quantile bins of the expected sample."""
    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 3:
        return 0.0
    e = np.clip(np.histogram(expected, bins=edges)[0] / max(expected.size, 1), 1e-6, None)
    a = np.clip(np.histogram(actual, bins=edges)[0] / max(actual.size, 1), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (no scipy dependency)."""
    if expected.size == 0 or actual.size == 0:
        return 0.0
    grid = np.sort(np.concatenate([expected, actual]))
    cdf_e = np.searchsorted(np.sort(expected), grid, side="right") / expected.size
    cdf_a = np.searchsorted(np.sort(actual), grid, side="right") / actual.size
    return float(np.max(np.abs(cdf_e - cdf_a)))


def standardized_shift(expected: np.ndarray, actual: np.ndarray) -> float:
    """Median shift in units of the expected sample's IQR. Sigma-like, robust."""
    q25, q75 = np.quantile(expected, [0.25, 0.75])
    return float(abs(np.median(actual) - np.median(expected)) / max(q75 - q25, 1e-9))


def compare(
    train_chunks: Sequence[Sequence[Dict[str, Any]]],
    target_chunks: Sequence[Sequence[Dict[str, Any]]],
    *,
    names: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Per-feature drift, sorted worst-first."""
    if names is None:
        names = sorted(chunk_features(train_chunks[0]))
    expected = np.asarray(feature_matrix(train_chunks, names), dtype=float)
    actual = np.asarray(feature_matrix(target_chunks, names), dtype=float)
    expected = np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0)
    actual = np.nan_to_num(actual, nan=0.0, posinf=0.0, neginf=0.0)

    rows: List[Dict[str, Any]] = []
    for index, name in enumerate(names):
        e, a = expected[:, index], actual[:, index]
        value = psi(e, a)
        rows.append({
            "feature": name,
            "psi": value,
            "ks": ks(e, a),
            "shift_iqr": standardized_shift(e, a),
            "train_median": float(np.median(e)),
            "target_median": float(np.median(a)),
            "severity": "major" if value >= PSI_MAJOR
                        else ("minor" if value >= PSI_MINOR else "ok"),
        })
    rows.sort(key=lambda row: row["psi"], reverse=True)
    return rows


def summarize(rows: Sequence[Dict[str, Any]], *, top: int = 25) -> str:
    major = sum(1 for row in rows if row["severity"] == "major")
    minor = sum(1 for row in rows if row["severity"] == "minor")
    lines = [
        f"{len(rows)} features | {major} major (PSI>={PSI_MAJOR}) | "
        f"{minor} minor (PSI>={PSI_MINOR})",
        "",
        f"{'feature':<44} {'PSI':>7} {'KS':>6} {'shift':>7}  train->target",
        "-" * 92,
    ]
    for row in rows[:top]:
        lines.append(
            f"{row['feature']:<44} {row['psi']:>7.3f} {row['ks']:>6.3f} "
            f"{row['shift_iqr']:>7.2f}  {row['train_median']:.4g} -> {row['target_median']:.4g}"
        )
    return "\n".join(lines)


def _load_chunks(path: Path) -> List[List[Dict[str, Any]]]:
    """Accept a bare list of chunks, or JSONL capture records with a 'chunk' key."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        out: List[List[Dict[str, Any]]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict) and isinstance(record.get("chunk"), list):
                out.append(record["chunk"])
            elif isinstance(record, dict) and isinstance(record.get("chunks"), list):
                out.extend(c for c in record["chunks"] if isinstance(c, list))
        return out
    payload = json.loads(text)
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        return payload
    raise ValueError(f"unrecognised chunk file layout: {path}")


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Feature drift: benchmark vs target")
    parser.add_argument("--target", required=True, help="JSON/JSONL of target chunks")
    parser.add_argument("--dir", default=None, help="benchmark dir (default poker44_ml.data)")
    parser.add_argument("--csv", default=None, help="write the full table here")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    from poker44_ml.data import DEFAULT_DIR, load_corpus

    corpus = load_corpus(Path(args.dir) if args.dir else DEFAULT_DIR)
    target = _load_chunks(Path(args.target))
    print(f"target: {len(target)} chunks, "
          f"{np.median([len(c) for c in target]):.0f} hands/chunk (median)")

    rows = compare(corpus.chunks, target)
    print()
    print(summarize(rows, top=args.top))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nfull table -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
