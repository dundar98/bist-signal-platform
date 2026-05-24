#!/usr/bin/env python3
"""Send daily signal report email to all registered users with email addresses."""

import argparse
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select

from database.models import Horizon, PortfolioSnapshot, Signal, SignalDirection, User
from database.repositories.portfolio import get_latest_portfolio_snapshot
from database.session import SessionLocal
from notifications.email_service import EmailNotifier, create_email_config_from_env

logger = logging.getLogger(__name__)


def _get_active_user_emails(db) -> list[str]:
    """Return email addresses of all active users who have an email set."""
    stmt = select(User.email).where(User.is_active.is_(True), User.email.isnot(None), User.email != "")
    return [row for (row,) in db.execute(stmt).all()]


def _build_html_report(db) -> str:
    """Build an HTML report from latest portfolio snapshots and open signals."""
    today = date.today().strftime("%d %B %Y")

    # Gather latest portfolio snapshots per horizon
    horizons_data = {}
    total_buy = 0
    for horizon in (Horizon.SHORT, Horizon.MEDIUM, Horizon.LONG):
        snapshot = get_latest_portfolio_snapshot(db, horizon=horizon)
        if snapshot is None:
            horizons_data[horizon] = {"items": [], "time": "-"}
            continue
        buy_items = []
        for item in snapshot.items:
            signal = item.signal
            if signal and signal.direction == SignalDirection.BUY:
                buy_items.append({
                    "ticker": item.symbol.ticker if item.symbol else str(item.symbol_id),
                    "score": f"{item.score:.0f}",
                    "entry": f"{signal.entry_price:.2f}" if signal.entry_price else "-",
                    "target": f"{signal.target_price:.2f}" if signal.target_price else "-",
                    "stop": f"{signal.stop_price:.2f}" if signal.stop_price else "-",
                })
        total_buy += len(buy_items)
        horizons_data[horizon] = {
            "items": buy_items,
            "time": snapshot.snapshot_time.strftime("%d.%m.%Y %H:%M") if snapshot else "-",
        }

    # Count open signals
    open_count = db.scalar(
        select(Signal).where(Signal.status == "open").with_only_columns(Signal.id).limit(500).subquery()
    )
    # Simpler: count directly
    from sqlalchemy import func
    open_count = db.scalar(select(func.count()).select_from(Signal).where(Signal.status == "open")) or 0

    horizon_labels = {Horizon.SHORT: "KISA (1-5 gün)", Horizon.MEDIUM: "ORTA (10-20 gün)", Horizon.LONG: "UZUN (1-3 ay)"}
    horizon_bg = {Horizon.SHORT: "#fff3e0", Horizon.MEDIUM: "#e8f5e9", Horizon.LONG: "#e3f2fd"}
    horizon_header = {Horizon.SHORT: "#f57c00", Horizon.MEDIUM: "#2e7d32", Horizon.LONG: "#1565c0"}

    def _horizon_section(horizon, data):
        label = horizon_labels[horizon]
        items = data["items"]
        if not items:
            return f"""
            <div style="background: {horizon_bg[horizon]}; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h3 style="color: {horizon_header[horizon]}; margin: 0 0 10px 0;">{label}</h3>
                <p style="color: #888; margin: 0;">Bu vade için aktif AL sinyali yok.</p>
            </div>"""

        rows = ""
        for item in items:
            rows += f"""
                <tr>
                    <td style="padding: 8px; border-bottom: 1px solid #e0e0e0; font-weight: bold;">{item['ticker']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e0e0e0; color: #2e7d32;">{item['score']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e0e0e0;">₺{item['entry']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e0e0e0;">₺{item['target']}</td>
                    <td style="padding: 8px; border-bottom: 1px solid #e0e0e0;">₺{item['stop']}</td>
                </tr>"""

        return f"""
        <div style="background: {horizon_bg[horizon]}; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: {horizon_header[horizon]}; margin: 0 0 10px 0;">{label} ({len(items)} sinyal)</h3>
            <p style="color: #666; font-size: 12px; margin: 0 0 10px 0;">Üretim: {data['time']}</p>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; border-radius: 6px;">
                    <thead>
                        <tr style="background: {horizon_bg[horizon]};">
                            <th style="padding: 8px; text-align: left;">Hisse</th>
                            <th style="padding: 8px; text-align: left;">Skor</th>
                            <th style="padding: 8px; text-align: left;">Giriş</th>
                            <th style="padding: 8px; text-align: left;">Hedef</th>
                            <th style="padding: 8px; text-align: left;">Stop</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; background: #f5f5f5; color: #333;">
    <div style="max-width: 640px; margin: 0 auto; padding: 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%); padding: 25px; border-radius: 12px 12px 0 0; text-align: center;">
            <h1 style="margin: 0; color: #fff; font-size: 22px;">BIST Signal Platform - Günlük Rapor</h1>
            <p style="margin: 8px 0 0 0; color: rgba(255,255,255,0.85); font-size: 14px;">{today}</p>
        </div>

        <!-- Summary -->
        <div style="background: #fff; padding: 20px; border-bottom: 1px solid #e0e0e0;">
            <table style="width: 100%;" cellpadding="0" cellspacing="0">
                <tr>
                    <td style="text-align: center; padding: 10px;">
                        <div style="background: #e8f5e9; border-radius: 8px; padding: 15px;">
                            <div style="font-size: 28px; font-weight: bold; color: #2e7d32;">{total_buy}</div>
                            <div style="font-size: 13px; color: #555; margin-top: 4px;">Aktif AL Sinyali</div>
                        </div>
                    </td>
                    <td style="text-align: center; padding: 10px;">
                        <div style="background: #e3f2fd; border-radius: 8px; padding: 15px;">
                            <div style="font-size: 28px; font-weight: bold; color: #1565c0;">{open_count}</div>
                            <div style="font-size: 13px; color: #555; margin-top: 4px;">Açık Sinyal</div>
                        </div>
                    </td>
                </tr>
            </table>
        </div>

        <!-- Signal Sections -->
        <div style="background: #fff; padding: 20px;">
            {_horizon_section(Horizon.SHORT, horizons_data[Horizon.SHORT])}
            {_horizon_section(Horizon.MEDIUM, horizons_data[Horizon.MEDIUM])}
            {_horizon_section(Horizon.LONG, horizons_data[Horizon.LONG])}
        </div>

        <!-- Footer -->
        <div style="background: #fafafa; padding: 20px; border-radius: 0 0 12px 12px; text-align: center; border-top: 1px solid #e0e0e0;">
            <p style="margin: 0; color: #999; font-size: 11px;">
                Bu rapor BIST Signal Platform tarafindan otomatik olarak olusturulmustur.<br>
                Yatirim tavsiyesi degildir. Kendi arastirmanizi yapin.
            </p>
        </div>
    </div>
</body>
</html>"""
    return html


def send_daily_report(smtp_server=None, smtp_port=None, sender=None, password=None, recipients=None) -> bool:
    """Generate and send daily signal report email to registered users."""

    # Override config with arguments if provided
    if smtp_server:
        os.environ["BIST_SMTP_SERVER"] = smtp_server
    if smtp_port is not None:
        os.environ["BIST_SMTP_PORT"] = str(smtp_port)
    if sender:
        os.environ["BIST_EMAIL_SENDER"] = sender
    if password:
        os.environ["BIST_EMAIL_PASSWORD"] = password
    if recipients:
        os.environ["BIST_EMAIL_RECIPIENTS"] = recipients

    config = create_email_config_from_env()

    if not config.sender_email or not config.sender_password:
        logger.warning("Email credentials not configured. Set BIST_EMAIL_SENDER and BIST_EMAIL_PASSWORD env vars.")
        return False

    # Collect recipients: explicit env recipients + active users from DB
    all_recipients = list(config.recipient_emails)
    with SessionLocal() as db:
        db_users = _get_active_user_emails(db)
        for email in db_users:
            if email not in all_recipients:
                all_recipients.append(email)

        if not all_recipients:
            logger.warning("No recipients found (no env recipients and no users with email in DB).")
            return False

        logger.info("Sending report to %d recipients: %s", len(all_recipients), ", ".join(all_recipients))

        # Build report
        html_body = _build_html_report(db)

    # Send
    notifier = EmailNotifier(config)
    today_str = date.today().strftime("%d.%m.%Y")
    subject = f"BIST Sinyal Raporu - {today_str}"

    plain_body = "BIST Signal Platform günlük sinyal raporu ektedir. HTML destekleyen bir e-posta istemcisi ile görüntüleyin."

    return notifier.send(
        subject=subject,
        body=plain_body,
        html_body=html_body,
        recipients=all_recipients,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Send daily signal report email")
    parser.add_argument("--smtp-server", default=os.getenv("BIST_SMTP_SERVER"))
    parser.add_argument("--smtp-port", type=int, default=os.getenv("BIST_SMTP_PORT"))
    parser.add_argument("--sender", default=os.getenv("BIST_EMAIL_SENDER"))
    parser.add_argument("--password", default=os.getenv("BIST_EMAIL_PASSWORD"))
    parser.add_argument("--recipients", default=os.getenv("BIST_EMAIL_RECIPIENTS"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    success = send_daily_report(
        smtp_server=args.smtp_server,
        smtp_port=args.smtp_port,
        sender=args.sender,
        password=args.password,
        recipients=args.recipients,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
