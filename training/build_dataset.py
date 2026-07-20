from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from poker44_ml.features import chunk_features
from poker44.validator.payload_view import prepare_hand_for_miner


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = (
    REPO_ROOT / "hands_generator" / "evaluation_datas" / "training_benchmark.txt"
)


def load_json_or_gz(path: str | Path) -> Any:
    file_path = Path(path)
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_benchmark_paths(path: str | Path | None) -> list[Path]:
    if path:
        raw = str(path).strip()
        if "," in raw:
            paths = [Path(item.strip()) for item in raw.split(",") if item.strip()]
        else:
            candidate = Path(raw)
            if candidate.is_dir():
                paths = sorted(candidate.glob("training_benchmark*.txt"))
            else:
                paths = [candidate]
    else:
        paths = [DEFAULT_BENCHMARK_PATH]
    existing = [candidate for candidate in paths if candidate.exists()]
    if not existing:
        raise FileNotFoundError(f"No benchmark files found for {path or DEFAULT_BENCHMARK_PATH}")
    return existing


def _as_root(payload: Any) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload


def miner_visible_chunk(chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match validator production path: same sanitization miners receive."""
    return [
        prepare_hand_for_miner(hand)
        for hand in chunk
        if isinstance(hand, dict)
    ]


def _feature_row(
    chunk: list[dict[str, Any]],
    *,
    miner_visible: bool = True,
) -> dict[str, float]:
    payload = miner_visible_chunk(chunk) if miner_visible else chunk
    if not payload:
        return {"hand_count": 0.0}
    row = chunk_features(payload)
    row["hand_count"] = float(len(payload))
    return row


def _iter_release_groups(payload: Any) -> list[dict[str, Any]]:
    root = _as_root(payload)
    if isinstance(root, dict) and isinstance(root.get("chunks"), list):
        return [group for group in root["chunks"] if isinstance(group, dict)]
    return []


def _load_labeled_benchmark_file(
    path: Path,
    *,
    miner_visible: bool = True,
) -> list[dict[str, Any]]:
    payload = load_json_or_gz(path)
    root = _as_root(payload)
    groups = _iter_release_groups(payload)
    if not groups:
        raise RuntimeError(f"Benchmark file has no labeled chunk groups: {path}")

    examples: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        chunks = group.get("chunks") or []
        labels = group.get("groundTruth") or group.get("groundTruthLabels") or []
        if len(chunks) != len(labels):
            raise RuntimeError(
                f"Benchmark group {group_index} has {len(chunks)} chunks but "
                f"{len(labels)} labels in {path}"
            )
        source_date = str(group.get("sourceDate") or root.get("sourceDate") or "")
        group_id = str(group.get("chunkId") or f"group_{group_index}")
        group_hash = str(group.get("chunkHash") or "")
        released_split = str(group.get("split") or "").strip().lower()
        for item_index, (chunk, label) in enumerate(zip(chunks, labels)):
            if not isinstance(chunk, list):
                continue
            hand_chunk = [hand for hand in chunk if isinstance(hand, dict)]
            if not hand_chunk:
                continue
            visible_chunk = (
                miner_visible_chunk(hand_chunk) if miner_visible else hand_chunk
            )
            if not visible_chunk:
                continue
            examples.append(
                {
                    "chunk": visible_chunk,
                    "label": int(label),
                    "source_date": source_date,
                    "group_id": group_id,
                    "group_hash": group_hash,
                    "released_split": released_split,
                    "item_index": item_index,
                    "source_path": str(path),
                    "features": _feature_row(visible_chunk, miner_visible=False),
                }
            )
    if not examples:
        raise RuntimeError(f"No usable labeled chunks found in {path}")
    return examples


def load_benchmark_examples(
    paths: str | Path | list[str | Path],
    *,
    miner_visible: bool = True,
) -> list[dict[str, Any]]:
    path_list = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in path_list:
        for example in _load_labeled_benchmark_file(path, miner_visible=miner_visible):
            # Key on chunk CONTENT identity (not source_path) so the same chunk
            # appearing in multiple benchmark files dedupes across files. Keeping
            # source_path here let cross-file duplicates survive -> train/test
            # leakage in random splits + date over-weighting.
            key = "|".join(
                [
                    str(example.get("group_hash", "")),
                    str(example.get("group_id", "")),
                    str(example.get("item_index", "")),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
    if not examples:
        raise RuntimeError("No benchmark examples loaded.")
    return examples
