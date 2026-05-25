"""Repository helpers for symbol and price bar persistence."""

from datetime import datetime
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database.models import PriceBar, Symbol, Timeframe
from utils import normalize_ticker


REQUIRED_PRICE_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


class PriceDataError(ValueError):
    """Raised when OHLCV data cannot be safely persisted."""


def get_symbol_by_ticker(db: Session, ticker: str) -> Symbol | None:
    """Return a symbol by ticker, handling Turkish characters safely."""

    return db.scalar(select(Symbol).where(Symbol.ticker == normalize_ticker(ticker)))


def list_active_symbols(
    db: Session,
    *,
    limit: int | None = None,
    bist100_only: bool = True,
) -> list[Symbol]:
    """Return active symbols ordered by ticker."""

    stmt = select(Symbol).where(Symbol.is_active.is_(True)).order_by(Symbol.ticker)
    if bist100_only:
        stmt = stmt.where(Symbol.is_bist100.is_(True))
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt).all())


def get_last_price_timestamp(
    db: Session,
    symbol_id: int,
    timeframe: Timeframe,
) -> datetime | None:
    """Return the latest persisted timestamp for a symbol/timeframe."""

    return db.scalar(
        select(func.max(PriceBar.timestamp)).where(
            PriceBar.symbol_id == symbol_id,
            PriceBar.timeframe == timeframe,
        )
    )


def list_price_bars(
    db: Session,
    symbol_id: int,
    timeframe: Timeframe,
    *,
    limit: int = 200,
) -> list[PriceBar]:
    """Return recent bars in chronological order."""

    subquery = (
        select(PriceBar.id)
        .where(PriceBar.symbol_id == symbol_id, PriceBar.timeframe == timeframe)
        .order_by(PriceBar.timestamp.desc())
        .limit(limit)
        .subquery()
    )
    stmt = (
        select(PriceBar)
        .where(PriceBar.id.in_(select(subquery.c.id)))
        .order_by(PriceBar.timestamp.asc())
    )
    return list(db.scalars(stmt).all())


def validate_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize OHLCV data before persistence."""

    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        raise PriceDataError(f"Missing price columns: {sorted(missing)}")

    clean = df.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"]).dt.tz_localize(None)
    clean = clean.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    for column in ["open", "high", "low", "close", "volume"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    if clean[list(REQUIRED_PRICE_COLUMNS)].isna().any().any():
        raise PriceDataError("Price data contains NaN values after normalization")

    if (clean[["open", "high", "low", "close"]] <= 0).any().any():
        raise PriceDataError("Price data contains non-positive OHLC values")

    if (clean["volume"] < 0).any():
        raise PriceDataError("Price data contains negative volume values")

    invalid_range = (
        (clean["high"] < clean["low"])
        | (clean["high"] < clean["open"])
        | (clean["high"] < clean["close"])
        | (clean["low"] > clean["open"])
        | (clean["low"] > clean["close"])
    )
    if invalid_range.any():
        raise PriceDataError("Price data contains invalid high/low ranges")

    return clean


def upsert_price_bars(
    db: Session,
    symbol: Symbol,
    timeframe: Timeframe,
    df: pd.DataFrame,
    *,
    source: str = "yfinance",
) -> int:
    """Insert missing bars and update existing bars for a symbol/timeframe."""

    clean = validate_price_frame(df)
    if clean.empty:
        return 0

    timestamps = [row.to_pydatetime() for row in clean["timestamp"]]
    existing = {
        bar.timestamp: bar
        for bar in db.scalars(
            select(PriceBar).where(
                PriceBar.symbol_id == symbol.id,
                PriceBar.timeframe == timeframe,
                PriceBar.timestamp.in_(timestamps),
            )
        ).all()
    }

    changed = 0
    for row in clean.itertuples(index=False):
        timestamp = row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp
        bar = existing.get(timestamp)
        values = {
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "source": source,
        }

        if bar is None:
            db.add(
                PriceBar(
                    symbol_id=symbol.id,
                    timeframe=timeframe,
                    timestamp=timestamp,
                    **values,
                )
            )
            changed += 1
        else:
            for key, value in values.items():
                setattr(bar, key, value)
            changed += 1

    return changed


def timeframe_from_string(value: str) -> Timeframe:
    """Convert API/CLI timeframe strings into the database enum."""

    try:
        return Timeframe(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in Timeframe)
        raise PriceDataError(f"Unsupported timeframe '{value}'. Allowed: {allowed}") from exc
