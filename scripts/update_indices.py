#!/usr/bin/env python3
"""Update Turkish index OHLCV bars without BIST100 symbol validation."""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
YFINANCE_CACHE_DIR = PROJECT_ROOT / "output" / "yfinance_cache"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))
except Exception:
    pass

from database.models import Symbol
from database.repositories.prices import get_last_price_timestamp, timeframe_from_string, upsert_price_bars
from database.session import SessionLocal

YFINANCE_TICKERS = {
    "XU100": "XU100.IS",
    "XU030": "XU030.IS",
    "XBANK": "XBANK.IS",
    "XUSIN": "XUSIN.IS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update BIST index price bars")
    parser.add_argument("--timeframe", default="1d", choices=["1d"])
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def _download(ticker: str, start: date, end: date) -> pd.DataFrame:
    raw = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), interval="1d", progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [column[0] for column in raw.columns]
    raw = raw.reset_index()
    # yfinance farkli surumlerde tarih sutununu 'Date', 'date' veya index adiyla donebilir
    date_col = "Date" if "Date" in raw.columns else raw.columns[0]
    close_col = "Close" if "Close" in raw.columns else "close"
    open_col = "Open" if "Open" in raw.columns else "open"
    high_col = "High" if "High" in raw.columns else "high"
    low_col = "Low" if "Low" in raw.columns else "low"
    vol_col = "Volume" if "Volume" in raw.columns else ("volume" if "volume" in raw.columns else None)
    return pd.DataFrame(
        {
            "timestamp": raw[date_col],
            "open": raw[open_col],
            "high": raw[high_col],
            "low": raw[low_col],
            "close": raw[close_col],
            "volume": raw[vol_col] if vol_col else 0,
        }
    )


def main() -> int:
    args = parse_args()
    timeframe = timeframe_from_string(args.timeframe)
    end = date.fromisoformat(args.end) if args.end else date.today()
    changed_total = 0
    errors: list[str] = []

    with SessionLocal() as db:
        for ticker, yahoo_ticker in YFINANCE_TICKERS.items():
            symbol = db.scalar(select(Symbol).where(Symbol.ticker == ticker))
            if symbol is None:
                errors.append(f"{ticker}: missing symbol. Run scripts\\seed_indices.py first.")
                continue
            try:
                last_ts = get_last_price_timestamp(db, symbol.id, timeframe)
                start = last_ts.date() + timedelta(days=1) if last_ts else end - timedelta(days=args.lookback_days)
                if start > end:
                    print(f"{ticker}: already up to date")
                    continue
                frame = _download(yahoo_ticker, start, end)
                changed = upsert_price_bars(db, symbol, timeframe, frame, source="yfinance_index")
                db.commit()
                changed_total += changed
                print(f"{ticker}: changed={changed}")
            except Exception as exc:
                db.rollback()
                errors.append(f"{ticker}: {exc}")

    print(f"Index update complete. changed_bars={changed_total} errors={len(errors)}")
    for error in errors[:20]:
        print(f"ERROR {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
