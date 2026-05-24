#!/usr/bin/env python3
"""Run scheduled production jobs in a simple long-running worker loop."""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BIST production worker loop")
    parser.add_argument("--interval-hours", type=float, default=24.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_cycle() -> None:
    _run([sys.executable, "scripts/seed_indices.py"])
    _run([sys.executable, "scripts/update_indices.py", "--lookback-days", "365"])
    _run(
        [
            sys.executable,
            "scripts/run_pipeline.py",
            "--source",
            "yfinance",
            "--timeframe",
            "1d",
            "--horizons",
            "short,medium,long",
            "--use-market-regime",
            "--train-ml",
        ]
    )
    # Send daily email report to all registered users
    _run([sys.executable, "scripts/send_daily_report.py"])


def notify_failure(message: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/notify_webhook.py",
            "--title",
            "BIST worker failed",
            "--message",
            message,
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )


def main() -> int:
    args = parse_args()
    while True:
        try:
            run_cycle()
        except Exception as exc:
            print(f"{datetime.now(timezone.utc).isoformat()} worker cycle failed: {exc}", flush=True)
            notify_failure(str(exc))
        if args.once:
            break
        time.sleep(max(args.interval_hours, 0.1) * 60 * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
