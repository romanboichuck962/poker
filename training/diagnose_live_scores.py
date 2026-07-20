"""Diagnose a Poker44 model's calibration on a *live* score log.

Reads a miner's PM2 log line with ``raw_scores`` / ``risk_scores`` /
``predictions`` for a real validator query and reports:

* The raw-score distribution (mean, std, min, max, q10, q90).
* The predicted bot rate and the post-shift score histogram.
* A flag for catastrophic distribution-shift signatures: raw scores all
  near zero with predictions > 60% True is the exact failure mode that
  causes the validator FPR cliff (`validator_fpr >= 0.10 -> reward = 0`).

Example:
    python -m training.diagnose_live_scores \
        --log /home/administrator/.pm2/logs/wolf-miner-5-out.log

Or paste a single ``Detailed chunk scores`` json blob:
    python -m training.diagnose_live_scores --paste-json '{"chunk_sizes": ...}'
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DETAIL_RE = re.compile(r"Detailed chunk scores \| (\{.*\})")


def _coerce_json(payload: str) -> Dict[str, Any]:
    payload = payload.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # The miner logs python-repr (single quotes, True/False) which is not
        # JSON. Fall back to ast.literal_eval which handles that safely.
        return ast.literal_eval(payload)


def _find_records(log_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = DETAIL_RE.search(line)
            if not match:
                continue
            try:
                out.append(_coerce_json(match.group(1)))
            except (json.JSONDecodeError, ValueError, SyntaxError):
                continue
    return out


def _summarize(name: str, values: List[float]) -> str:
    arr = np.asarray(values, dtype=float)
    return (
        f"{name}: n={arr.size} mean={arr.mean():.4f} std={arr.std():.4f} "
        f"min={arr.min():.4f} q10={np.quantile(arr, 0.10):.4f} "
        f"q50={np.quantile(arr, 0.50):.4f} q90={np.quantile(arr, 0.90):.4f} "
        f"max={arr.max():.4f}"
    )


def _diagnose(record: Dict[str, Any], index: int) -> None:
    chunk_sizes = record.get("chunk_sizes") or []
    risk = list(record.get("risk_scores") or [])
    preds = list(record.get("predictions") or [])
    components = record.get("components") or {}
    raw = list(components.get("raw_scores") or [])
    calibrated = list(components.get("calibrated_scores") or [])
    remapped = list(components.get("remapped_scores") or [])
    final_components = list(
        components.get("final_scores") or components.get("logit_scores") or []
    )

    bot_count = sum(1 for p in preds if p)
    human_count = max(len(preds) - bot_count, 0)
    print(
        f"=== Record #{index} | {len(risk)} chunks | "
        f"bot_count={bot_count} human_count={human_count} | hand_size_range="
        f"[{min(chunk_sizes) if chunk_sizes else '-'},"
        f"{max(chunk_sizes) if chunk_sizes else '-'}] ==="
    )
    if raw:
        print(_summarize("raw         ", raw))
    if calibrated and calibrated != raw:
        print(_summarize("stack_calib ", calibrated))
    if remapped and remapped != calibrated and remapped != raw:
        print(_summarize("score_remap ", remapped))
    if (
        final_components
        and final_components != remapped
        and final_components != calibrated
        and final_components != raw
    ):
        print(_summarize("final_comp  ", final_components))
    if risk:
        print(_summarize("risk_out    ", risk))

    bot_rate = bot_count / max(len(preds), 1)
    above_05 = sum(1 for s in risk if s >= 0.5) / max(len(risk), 1)
    print(
        f"predicted bot rate: {bot_rate:.2%}  "
        f"final>=0.5 rate: {above_05:.2%}  "
        f"bot_count={bot_count} human_count={human_count}"
    )

    flags: List[str] = []
    if raw:
        raw_arr = np.asarray(raw, dtype=float)
        post_raw = remapped or final_components or risk
        if float(raw_arr.std()) < 0.02 and bot_rate > 0.6:
            flags.append(
                "CRITICAL: raw scores have ~no spread but final says "
                f"{bot_rate:.0%} bots. The stack is not separating live "
                "chunks; post-processing (score_remap or logit bias) is "
                "amplifying noise into false positives. Expect "
                "validator_fpr >= 0.10 -> reward = 0."
            )
        if float(raw_arr.max()) < 0.10 and bot_rate > 0.5:
            flags.append(
                "CRITICAL: raw max < 0.10 but many chunks flagged bot. "
                "High final scores come from calibration/remap, not model "
                "separation on live data."
            )
        if post_raw and float(np.asarray(post_raw).std()) < 0.03:
            flags.append(
                "WARN: post-calibration scores are nearly flat; threshold "
                "0.5 may be arbitrary. Retrain with miner-visible payloads "
                "or relax score_remap temperature."
            )
    if bot_rate > 0.85:
        flags.append(
            f"HIGH RISK: predicted bot rate is {bot_rate:.0%}. Likely "
            "validator_fpr above the 0.10 cliff unless the batch is bot-heavy."
        )
    if bot_rate < 0.15:
        flags.append(
            f"WARN: predicted bot rate is {bot_rate:.0%}. If the batch "
            "contains bots, recall is too low (scores stuck below 0.5)."
        )

    if flags:
        print("Flags:")
        for flag in flags:
            print(f"  - {flag}")
    else:
        print("No catastrophic flags. Calibration looks plausible.")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose live-eval calibration for a Poker44 miner.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to a PM2 miner stdout log (e.g. wolf-miner-5-out.log).",
    )
    parser.add_argument(
        "--paste-json",
        type=str,
        default=None,
        help="A single 'Detailed chunk scores | {...}' python dict literal.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=5,
        help="When --log is given, show this many most-recent records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log and not args.paste_json:
        print(
            "ERROR: provide --log /path/to/wolf-miner-N-out.log or "
            "--paste-json '<dict>'",
            file=sys.stderr,
        )
        return 2

    if args.paste_json:
        record = _coerce_json(args.paste_json)
        _diagnose(record, index=1)
        return 0

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log path not found: {log_path}", file=sys.stderr)
        return 2
    records = _find_records(log_path)
    if not records:
        print(
            f"No 'Detailed chunk scores' lines found in {log_path}. The "
            "miner must be running with POKER44_LOG_SCORE_COMPONENTS=1 "
            "and have served at least one validator query.",
            file=sys.stderr,
        )
        return 1
    keep = max(1, int(args.last))
    for offset, record in enumerate(records[-keep:], start=max(1, len(records) - keep + 1)):
        _diagnose(record, index=offset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
