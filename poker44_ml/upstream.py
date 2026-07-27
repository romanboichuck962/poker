"""Pinned-hash guard against silent subnet-side changes.

Two upstream files decide whether this miner is competitive, and neither is
versioned in a way you can depend on:

* ``poker44/score/scoring.py``        — the reward. Changed >=3 times already.
* ``poker44/validator/payload_view.py`` — the sanitizer. Defines what a hand
  even looks like by the time it reaches us, and our whole feature set is built
  against its exact behaviour (bucket grid, 5-8 action window, seat aliasing).

A change to either silently invalidates a trained artifact. So we hash them at
import and shout when they move. This is deliberately noisy: a loud failure the
day the subnet changes is worth far more than a quiet 20% reward loss.

Refresh the pins with::

    python -m poker44_ml.upstream --update
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict

import poker44

# Hashes verified against poker44 VALIDATOR_DEPLOY_VERSION 0.1.36 (2026-07-26).
PINNED_VERSION = "0.1.36"
PINNED_SHA256: Dict[str, str] = {
    "score/scoring.py": "913839aa8da2e3e16ea5338b3e5b66a6086f6133395781bdeb203fcedb10150b",
    "validator/payload_view.py": "a59f7e22cfb300ddeb2d2400d7b257d7441fbad3211dd480e83b38e774d9d2ca",
    "validator/synapse.py": "b90eade78d85f5d72b9e7d67b1abe46640e4740c059190fa8319672287240c84",
    "utils/model_manifest.py": "5265f564698e25d7fd708cb8032b3fdacf11ba884b50b597257da0550a9cb33b",
}

# Which pins are severe enough to justify refusing to train.
CRITICAL = ("score/scoring.py", "validator/payload_view.py")


def _package_root() -> Path:
    return Path(poker44.__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def current_hashes() -> Dict[str, str]:
    root = _package_root()
    out: Dict[str, str] = {}
    for relative in PINNED_SHA256:
        target = root / relative
        out[relative] = _sha256(target) if target.exists() else ""
    return out


def drifted() -> Dict[str, tuple[str, str]]:
    """Map of ``relative_path -> (pinned, current)`` for every changed file."""
    current = current_hashes()
    return {
        relative: (expected, current[relative])
        for relative, expected in PINNED_SHA256.items()
        if current[relative] != expected
    }


def check(*, strict: bool = False) -> Dict[str, tuple[str, str]]:
    """Warn (or raise) when upstream has moved under us.

    ``strict=True`` raises if a CRITICAL file changed — use it in train.py and
    walkforward.py, where acting on a stale reward or sanitizer wastes a whole
    training run. Serving stays non-strict: a live miner must keep answering
    even if the subnet shipped a cosmetic change five minutes ago.
    """
    changes = drifted()
    if not changes:
        return {}

    installed = getattr(poker44, "VALIDATOR_DEPLOY_VERSION", "unknown")
    lines = [
        "",
        "=" * 72,
        "UPSTREAM DRIFT DETECTED",
        f"  pinned against poker44 {PINNED_VERSION}, installed is {installed}",
        "",
    ]
    for relative, (expected, actual) in sorted(changes.items()):
        mark = "CRITICAL" if relative in CRITICAL else "info    "
        state = "MISSING" if not actual else f"{actual[:16]}..."
        lines.append(f"  [{mark}] poker44/{relative}")
        lines.append(f"             pinned {expected[:16]}...  now {state}")
    lines += [
        "",
        "  Re-read the changed files before trusting any offline number.",
        "  Then refresh the pins:  python -m poker44_ml.upstream --update",
        "=" * 72,
        "",
    ]
    message = "\n".join(lines)

    critical = [name for name in changes if name in CRITICAL]
    if strict and critical:
        raise RuntimeError(message + f"\nRefusing to proceed: {', '.join(critical)} changed.")
    print(message, file=sys.stderr)
    return changes


def _update() -> int:
    current = current_hashes()
    missing = [name for name, value in current.items() if not value]
    if missing:
        print(f"cannot update, files not found: {', '.join(missing)}", file=sys.stderr)
        return 1

    source = Path(__file__)
    text = source.read_text(encoding="utf-8")
    for relative, value in current.items():
        old = PINNED_SHA256[relative]
        if old != value:
            text = text.replace(f'"{old}"', f'"{value}"')
            print(f"updated {relative}: {old[:12]}... -> {value[:12]}...")
    installed = getattr(poker44, "VALIDATOR_DEPLOY_VERSION", "unknown")
    text = text.replace(f'PINNED_VERSION = "{PINNED_VERSION}"', f'PINNED_VERSION = "{installed}"')
    source.write_text(text, encoding="utf-8")
    print(f"pins now track poker44 {installed}")
    print("REMINDER: read the diff of every changed file before retraining.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check or refresh upstream pins.")
    parser.add_argument("--update", action="store_true", help="rewrite the pins to current hashes")
    args = parser.parse_args()
    if args.update:
        raise SystemExit(_update())
    changes = check(strict=False)
    print("upstream pins OK" if not changes else f"{len(changes)} file(s) drifted")
    raise SystemExit(1 if changes else 0)
