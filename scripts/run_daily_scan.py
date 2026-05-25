#!/usr/bin/env python3
"""
Daily Scan Runner.

Executes the daily market scan and notifications.
Supports single mode or --mode ALL to run all three scans.
"""

import sys
import logging
import argparse
import os
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from training import Trainer
from models import create_model
from notifications import DailyScanner, generate_signal_report
from notifications.scanner import generate_dashboard_json
from notifications.email_service import EmailNotifier, EmailConfig

logger = logging.getLogger(__name__)

ALL_MODES = ["KISA", "ORTA", "UZUN"]

MODEL_FILES = {
    "KISA": "transformer_kisa.pt",
    "ORTA": "transformer_orta.pt",
    "UZUN": "transformer_uzun.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run daily BIST100 scan")
    parser.add_argument(
        "--mode", type=str, default="UZUN",
        choices=["KISA", "ORTA", "UZUN", "ALL"],
        help="Scanning mode (KISA/ORTA/UZUN/ALL). ALL runs all three modes.",
    )
    parser.add_argument("--email", type=str, default="false", help="Send email report (true/false)")
    parser.add_argument("--sender-email", type=str, help="Email sender address")
    parser.add_argument("--sender-password", type=str, help="Email sender password")
    parser.add_argument("--email-to", type=str, help="Email recipient address")
    parser.add_argument("--data-source", type=str, default="yfinance")
    parser.add_argument("--output-dir", type=str, default="output/scans")
    parser.add_argument("--lookback", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated list of symbols to scan")
    return parser.parse_args()


def run_single_scan(mode: str, args, config):
    """Run a scan for a single mode. Returns (result, text_report) or (None, None) on error."""
    model_name = MODEL_FILES.get(mode, "transformer_uzun.pt")
    model_path = PROJECT_ROOT / "models" / model_name

    if not model_path.exists():
        logger.error("Model file not found: %s. Please train the %s model first.", model_path, mode)
        return None, None

    logger.info("Using %s model: %s", mode, model_path)

    symbols_to_scan = None
    if args.symbols:
        symbols_to_scan = [s.strip() for s in args.symbols.split(",")]
        logger.info("Custom symbols to scan: %s", symbols_to_scan)

    try:
        scanner = DailyScanner(
            model_path=str(model_path),
            config=config,
            device=config.training.device,
        )
    except Exception as e:
        logger.error("Failed to initialize scanner for %s: %s", mode, e)
        return None, None

    logger.info("Running %s scan...", mode)
    try:
        result = scanner.scan_all(
            symbols=symbols_to_scan,
            lookback_days=args.lookback,
            limit=args.limit,
            mode=mode,
        )
    except Exception as e:
        logger.error("Scan failed for %s: %s", mode, e)
        return None, None

    text_report = generate_signal_report(result)
    print(text_report)

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"scan_{result.scan_date}_{mode}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(text_report)

    # Generate dashboard data (merges into combined file)
    docs_path = PROJECT_ROOT / "docs" / "dashboard_data.json"
    generate_dashboard_json(result, str(docs_path))

    return result, text_report


def _get_email_credentials(args):
    """Resolve email credentials from args or env vars."""
    return (
        args.sender_email or os.getenv("BIST_EMAIL_SENDER"),
        args.sender_password or os.getenv("BIST_EMAIL_PASSWORD") or os.getenv("EMAIL_PASSWORD"),
        args.email_to or os.getenv("BIST_EMAIL_RECIPIENTS") or os.getenv("EMAIL_RECIPIENT"),
    )


def send_email(mode_label: str, text_report: str, result, args):
    """Send email report for one or more scans."""
    sender_email, sender_password, recipient = _get_email_credentials(args)
    if not (sender_email and sender_password and recipient):
        logger.info("Email sending skipped (credentials missing).")
        return

    logger.info("Sending email to %s...", recipient)
    try:
        email_config = EmailConfig(
            sender_email=sender_email,
            sender_password=sender_password,
            recipient_emails=[r.strip() for r in recipient.split(",")],
        )
        notifier = EmailNotifier(email_config)
        dashboard_link = f"https://{os.getenv('GITHUB_REPOSITORY_OWNER', 'dundar98')}.github.io/bist"
        email_body = f"⏳ Tarama Modu: {mode_label}\n{text_report}\n\n📊 Web Dashboard: {dashboard_link}"
        notifier.send_signal_report(
            report_text=email_body,
            scan_date=str(result.scan_date),
            buy_count=len(result.buy_signals),
            sell_count=len(result.sell_signals),
        )
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error("Failed to send email: %s", e)


def main():
    import pandas as pd  # noqa: F811

    args = parse_args()
    mode = args.mode.upper()
    config = get_config()

    modes_to_run = ALL_MODES if mode == "ALL" else [mode]

    all_reports = []
    for run_mode in modes_to_run:
        result, text_report = run_single_scan(run_mode, args, config)
        if result is None:
            logger.warning("Skipping %s — scan failed or model missing.", run_mode)
            continue
        all_reports.append((run_mode, text_report, result))

    if not all_reports:
        logger.error("No scans completed successfully.")
        sys.exit(1)

    # Send email
    should_send = args.email.lower() == "true"
    if should_send or all(
        os.getenv(v) for v in ("BIST_EMAIL_SENDER", "BIST_EMAIL_RECIPIENTS")
    ):
        if len(all_reports) == 1:
            m, r, res = all_reports[0]
            send_email(m, r, res, args)
        else:
            combined_report = "\n\n".join(f"=== {m} ===\n{r}" for m, r, _ in all_reports)
            send_email("KISA+ORTA+UZUN", combined_report, all_reports[0][2], args)


if __name__ == "__main__":
    main()
