"""Public benchmark download and labeled-example loading.

    python -m poker44_ml.data --fetch              # pull any releases not on disk
    python -m poker44_ml.data --fetch --force      # re-download everything
    python -m poker44_ml.data --stats              # what is on disk

Releases land in ``data/benchmark/release_<date>.json`` (gitignored). Run the
fetch daily: new dates are the only source of fresh supervision, and the
reference miners train across 45+ of them.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

from poker44_ml.canonical import canonicalize

API_BASE = "https://api.poker44.net/api/v1/benchmark"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = Path(os.environ.get("POKER44_BENCHMARK_DIR", str(ROOT / "data" / "benchmark")))


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #

def _get(url: str, timeout: int = 120) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not payload.get("success", True):
        raise RuntimeError(f"benchmark API error for {url}: {payload}")
    return payload.get("data", payload)


def status() -> Dict[str, Any]:
    return _get(API_BASE)


def release_dates(*, page_size: int = 100) -> List[str]:
    dates: List[str] = []
    before: str | None = None
    while True:
        params: Dict[str, Any] = {"limit": page_size}
        if before:
            params["before"] = before
        page = _get(f"{API_BASE}/releases?{urllib.parse.urlencode(params)}")
        releases = page.get("releases") or []
        if not releases:
            break
        dates.extend(
            str(item["sourceDate"])
            for item in releases
            if isinstance(item, dict) and item.get("sourceDate")
        )
        before = str(releases[-1].get("sourceDate") or "")
        if len(releases) < page_size:
            break
    return sorted(set(dates))


def fetch_release(source_date: str, *, page_limit: int = 24) -> Dict[str, Any]:
    groups: List[Dict[str, Any]] = []
    cursor: str | None = None
    meta: Dict[str, Any] = {}
    while True:
        params: Dict[str, Any] = {"sourceDate": source_date, "limit": page_limit}
        if cursor:
            params["cursor"] = cursor
        page = _get(f"{API_BASE}/chunks?{urllib.parse.urlencode(params)}")
        if not meta:
            meta = {
                key: page.get(key)
                for key in ("sourceDate", "releaseVersion", "schemaVersion", "releaseType")
                if key in page
            }
        groups.extend(item for item in (page.get("chunks") or []) if isinstance(item, dict))
        cursor = page.get("nextCursor")
        if not cursor:
            break
    return {**meta, "chunks": groups}


def fetch_all(
    output_dir: Path = DEFAULT_DIR,
    *,
    dates: Sequence[str] | None = None,
    force: bool = False,
) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    info = status()
    print(
        f"API: totalChunks={info.get('totalChunks')} "
        f"latestSourceDate={info.get('latestSourceDate')}"
    )

    wanted = list(dates) if dates else release_dates()
    if not wanted:
        raise RuntimeError("no benchmark release dates available")

    added = skipped = failed = 0
    for source_date in wanted:
        target = output_dir / f"release_{source_date}.json"
        if target.exists() and not force:
            skipped += 1
            continue
        try:
            payload = fetch_release(source_date)
            examples = sum(
                min(len(group.get("chunks") or []), len(group.get("groundTruth") or []))
                for group in payload["chunks"]
            )
            if examples <= 0:
                print(f"  skip {source_date}: no labeled examples")
                failed += 1
                continue
            target.write_text(json.dumps(payload), encoding="utf-8")
            added += 1
            print(f"  saved {target.name}: {examples} labeled chunks")
        except urllib.error.HTTPError as err:
            print(f"  skip {source_date}: HTTP {err.code}")
            failed += 1
        except Exception as err:
            print(f"  skip {source_date}: {err}")
            failed += 1

    print(f"done: {added} new, {skipped} present, {failed} failed -> {output_dir}")
    return {"added": added, "skipped": skipped, "failed": failed}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

@dataclass
class Example:
    chunk: List[Dict[str, Any]]
    label: int
    source_date: str
    group_id: str
    group_hash: str
    item_index: int

    @property
    def key(self) -> str:
        return f"{self.group_hash}|{self.group_id}|{self.item_index}"


@dataclass
class Corpus:
    examples: List[Example] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    @property
    def labels(self) -> List[int]:
        return [example.label for example in self.examples]

    @property
    def chunks(self) -> List[List[Dict[str, Any]]]:
        return [example.chunk for example in self.examples]

    @property
    def dates(self) -> List[str]:
        return sorted({example.source_date for example in self.examples if example.source_date})

    def by_date(self, *, include: Sequence[str] | None = None,
                exclude: Sequence[str] | None = None) -> "Corpus":
        keep = set(include) if include is not None else None
        drop = set(exclude or ())
        return Corpus([
            example for example in self.examples
            if (keep is None or example.source_date in keep)
            and example.source_date not in drop
        ])

    def summary(self) -> str:
        bots = sum(self.labels)
        sizes = [len(example.chunk) for example in self.examples]
        span = f"{min(sizes)}-{max(sizes)}" if sizes else "0"
        return (
            f"{len(self)} chunks ({bots} bot / {len(self) - bots} human) "
            f"across {len(self.dates)} dates, {span} hands/chunk"
        )


def _read(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload


def load_release(path: Path, *, strict: bool = True) -> List[Example]:
    """Load one release file, projecting every chunk to the miner-visible view."""
    root = _unwrap(_read(path))
    groups = [g for g in (root.get("chunks") or []) if isinstance(g, dict)]
    if not groups:
        raise RuntimeError(f"no labeled chunk groups in {path}")

    raw_chunks: List[List[Dict[str, Any]]] = []
    stubs: List[Dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        inner = group.get("chunks") or []
        labels = group.get("groundTruth") or group.get("groundTruthLabels") or []
        if len(inner) != len(labels):
            raise RuntimeError(
                f"{path.name} group {group_index}: {len(inner)} chunks vs {len(labels)} labels"
            )
        for item_index, (chunk, label) in enumerate(zip(inner, labels)):
            if not isinstance(chunk, list):
                continue
            raw_chunks.append([h for h in chunk if isinstance(h, dict)])
            stubs.append({
                "label": 1 if str(label).lower() in {"1", "bot", "true"} else int(bool(label)),
                "source_date": str(group.get("sourceDate") or root.get("sourceDate") or ""),
                "group_id": str(group.get("chunkId") or f"group_{group_index}"),
                "group_hash": str(group.get("chunkHash") or ""),
                "item_index": item_index,
            })

    # One canonicalization pass over the whole file so the seat-1 assertion has
    # enough hands to be meaningful.
    projected, report = canonicalize(raw_chunks, strict=strict)
    print(f"  {path.name}: {report}")

    # canonicalize drops empties; re-pair by walking the originals in order.
    out: List[Example] = []
    cursor = 0
    for raw, stub in zip(raw_chunks, stubs):
        if cursor < len(projected) and raw and projected[cursor]:
            out.append(Example(chunk=projected[cursor], **stub))
            cursor += 1
    return out


def load_corpus(
    directory: Path = DEFAULT_DIR,
    *,
    strict: bool = True,
    dates: Sequence[str] | None = None,
) -> Corpus:
    """Load every release on disk, deduped on chunk identity."""
    directory = Path(directory)
    paths = sorted(directory.glob("release_*.json")) if directory.is_dir() else [directory]
    paths = [p for p in paths if p.exists()]
    if dates:
        wanted = set(dates)
        paths = [p for p in paths if p.stem.replace("release_", "") in wanted]
    if not paths:
        raise FileNotFoundError(
            f"no benchmark releases under {directory}. Run: python -m poker44_ml.data --fetch"
        )

    seen: set[str] = set()
    examples: List[Example] = []
    for path in paths:
        for example in load_release(path, strict=strict):
            if example.key in seen:
                continue
            seen.add(example.key)
            examples.append(example)

    corpus = Corpus(examples)
    if not corpus.examples:
        raise RuntimeError("loaded zero usable examples")
    print(f"corpus: {corpus.summary()}")
    return corpus


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Poker44 benchmark data tool")
    parser.add_argument("--fetch", action="store_true", help="download missing releases")
    parser.add_argument("--force", action="store_true", help="re-download existing releases")
    parser.add_argument("--stats", action="store_true", help="summarise what is on disk")
    parser.add_argument("--dates", type=str, default=None, help="comma-separated YYYY-MM-DD")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_DIR))
    args = parser.parse_args()

    directory = Path(args.dir)
    dates = [d.strip() for d in (args.dates or "").split(",") if d.strip()] or None

    if args.fetch:
        fetch_all(directory, dates=dates, force=args.force)
    if args.stats:
        print(load_corpus(directory, dates=dates).summary())
    if not args.fetch and not args.stats:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
