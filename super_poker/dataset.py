"""Load the public benchmark cache into labeled chunk examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Example:
    hands: list[dict]
    label: int
    source_date: str
    split: str
    chunk_hash: str


def load_examples(data_dir: Path) -> list[Example]:
    examples: list[Example] = []
    seen_hashes: set[str] = set()
    for path in sorted(data_dir.glob("*/*.json")):
        publication = json.loads(path.read_text(encoding="utf-8"))
        chunk_hash = str(publication.get("chunkHash") or "")
        if chunk_hash and chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)
        chunks = publication.get("chunks") or []
        labels = publication.get("groundTruth") or []
        if len(chunks) != len(labels):
            raise ValueError(f"Chunk/label mismatch in {path}")
        for hands, label in zip(chunks, labels):
            numeric = int(label)
            if numeric not in (0, 1):
                raise ValueError(f"Invalid label {label!r} in {path}")
            examples.append(Example(
                hands=[hand for hand in hands if isinstance(hand, dict)],
                label=numeric,
                source_date=str(publication.get("sourceDate") or path.parent.name),
                split=str(publication.get("split") or "unspecified"),
                chunk_hash=chunk_hash,
            ))
    if not examples:
        raise FileNotFoundError(f"No benchmark publications found under {data_dir}")
    return examples
