#!/usr/bin/env python3
"""UID 138–style guarded retrain/promote for the Neptune miner.

Flow:
  1) optional benchmark refresh
  2) train candidate via deploy_lgbm_best.py (staging + promote gates)
  3) restart miner only when a candidate was promoted

Usage:
  python autopilot.py
  python autopilot.py --no-restart
  POKER44_FORCE_DEPLOY=1 python autopilot.py   # bypass reward/FPR rejection
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "autopilot.log"
MINER_PM2_NAME = "poker44_miner_neptune"


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def refresh_benchmark() -> None:
    script = Path("/root/POKER44-SUBNET-1/scripts/download_benchmark.py")
    if not script.exists():
        log("REFRESH: download_benchmark.py missing; using cached releases")
        return
    proc = subprocess.run(
        [sys.executable, str(script), "--out", "/root/POKER44-SUBNET-1/data/benchmark"],
        cwd="/root/POKER44-SUBNET-1",
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        log(f"REFRESH: failed rc={proc.returncode}: {proc.stderr[-500:]}")
        return
    log("REFRESH: " + (proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "ok"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-restart", action="store_true")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--force-deploy", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    log("=== AUTOPILOT START ===")
    if not args.no_refresh:
        refresh_benchmark()
    if args.force_deploy:
        import os

        os.environ["POKER44_FORCE_DEPLOY"] = "1"

    proc = subprocess.run(
        [sys.executable, "-u", str(HERE / "deploy_lgbm_best.py")],
        cwd=str(HERE),
    )
    promoted = proc.returncode == 0
    if proc.returncode == 2:
        log("DEPLOY: candidate rejected by promote gates; serving model unchanged")
    elif proc.returncode != 0:
        log(f"DEPLOY: trainer failed rc={proc.returncode}; serving model unchanged")
    else:
        log("DEPLOY: candidate promoted")

    if promoted and not args.no_restart:
        try:
            out = subprocess.run(
                ["pm2", "restart", MINER_PM2_NAME, "--update-env"],
                capture_output=True,
                text=True,
            )
            if out.returncode == 0:
                log(f"DEPLOY: restarted pm2 '{MINER_PM2_NAME}'")
            else:
                log(f"DEPLOY: pm2 restart failed: {out.stderr[-400:]}")
        except FileNotFoundError:
            log("DEPLOY: pm2 not found; restart miner manually")
    elif promoted:
        log("DEPLOY: promoted but --no-restart set")

    log(f"=== AUTOPILOT DONE in {time.time()-t0:.0f}s promoted={promoted} ===")
    return 0 if promoted or proc.returncode == 2 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
