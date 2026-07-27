"""Incrementally cache public Poker44 benchmark releases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://api.poker44.net/api/v1/benchmark"


def _data(session: requests.Session, path: str = "", **params: Any) -> dict[str, Any]:
    response = session.get(f"{BASE_URL}{path}", params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success") or not isinstance(payload.get("data"), dict):
        raise RuntimeError(f"Unexpected benchmark response from {path or '/'}")
    return payload["data"]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def releases(session: requests.Session) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    before = None
    while True:
        params = {"limit": 100}
        if before:
            params["before"] = before
        page = list(_data(session, "/releases", **params).get("releases") or [])
        output.extend(page)
        if len(page) < 100:
            break
        next_before = str(page[-1].get("sourceDate") or "")
        if not next_before or next_before == before:
            raise RuntimeError("Release pagination did not advance")
        before = next_before
    unique = {str(item.get("sourceDate")): item for item in output if item.get("sourceDate")}
    return [unique[key] for key in sorted(unique, reverse=True)]


def download_release(session: requests.Session, source_date: str, destination: Path) -> int:
    cursor = None
    count = 0
    while True:
        params: dict[str, Any] = {"sourceDate": source_date, "limit": 24}
        if cursor:
            params["cursor"] = cursor
        page = _data(session, "/chunks", **params)
        for publication in page.get("chunks") or []:
            chunk_id = str(publication.get("chunkId") or "")
            if not chunk_id:
                raise ValueError(f"Publication without chunkId for {source_date}")
            path = destination / source_date / f"{chunk_id}.json"
            if not path.exists():
                _write(path, publication)
            count += 1
        next_cursor = str(page.get("nextCursor") or "")
        if not next_cursor:
            break
        if next_cursor == cursor:
            raise RuntimeError(f"Repeated chunk cursor for {source_date}")
        cursor = next_cursor
    return count


def update_cache(destination: Path, *, backfill: bool = False) -> dict[str, Any]:
    """Download missing releases; normally only the latest published date."""
    with requests.Session() as session:
        status = _data(session)
        catalog = releases(session)
        latest = str(status.get("latestSourceDate") or "")
        wanted = catalog if backfill else [item for item in catalog if str(item.get("sourceDate")) == latest]
        downloaded = {}
        for release in wanted:
            source_date = str(release["sourceDate"])
            downloaded[source_date] = download_release(session, source_date, destination)
        _write(destination / "status.json", status)
        _write(destination / "releases.json", catalog)
    return {"latest_source_date": latest, "release_publications": downloaded}
