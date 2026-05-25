"""Symbol endpoints."""

import logging
from datetime import date, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.auth import get_current_user
from app.schemas.analysis import SymbolAnalysisRead
from app.schemas.features import FeatureValueRead
from app.schemas.prices import PriceBarRead
from app.schemas.symbols import SymbolRead
from database.models import Symbol, Timeframe, User
from database.repositories.analysis import build_symbol_analysis
from database.repositories.features import compute_and_store_features, list_feature_values
from database.repositories.prices import (
    PriceDataError,
    get_symbol_by_ticker,
    list_price_bars,
    timeframe_from_string,
    upsert_price_bars,
)
from database.session import get_db
from utils import normalize_ticker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/symbols", tags=["symbols"])


def _run_symbol_pipeline(
    db: Session,
    symbol: Symbol,
    timeframe: Timeframe,
    lookback_days: int = 365 * 3,
    lookback_bars: int = 260,
    feature_set: str = "technical_v1",
) -> int:
    """Fetch prices and compute features for a single symbol. Returns number of feature rows changed."""
    from data import get_data_loader

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    loader = get_data_loader(source="yfinance")
    try:
        df = loader.load(symbol.ticker, start_date=start_date, end_date=end_date, interval=timeframe.value)
        upsert_price_bars(db, symbol, timeframe, df, source="yfinance")
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("Price fetch failed for %s: %s", symbol.ticker, exc)

    try:
        changed = compute_and_store_features(
            db, symbol, timeframe, lookback_bars=lookback_bars, feature_set=feature_set
        )
        db.commit()
        return changed
    except Exception as exc:
        db.rollback()
        logger.warning("Feature computation failed for %s: %s", symbol.ticker, exc)
        return 0


@router.get("/search", response_model=list[SymbolRead])
def search_symbols(
    q: str = Query(default="", description="Ticker veya isim ile arama"),
    active_only: bool = True,
    limit: int = Query(default=25, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[Symbol]:
    """Search symbols by ticker or name across ALL symbols (not just BIST100)."""

    stmt = select(Symbol).order_by(Symbol.ticker)

    if active_only:
        stmt = stmt.where(Symbol.is_active.is_(True))

    if q.strip():
        query = normalize_ticker(q)
        stmt = stmt.where(
            Symbol.ticker.contains(query) | Symbol.name.ilike(f"%{q.strip()}%")
        )

    stmt = stmt.limit(limit)
    return list(db.scalars(stmt).all())


@router.get("", response_model=list[SymbolRead])
def list_symbols(
    active_only: bool = True,
    bist100_only: bool = True,
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[Symbol]:
    """List tradable symbols known by the platform."""

    stmt = select(Symbol).order_by(Symbol.ticker).limit(limit)

    if active_only:
        stmt = stmt.where(Symbol.is_active.is_(True))

    if bist100_only:
        stmt = stmt.where(Symbol.is_bist100.is_(True))

    return list(db.scalars(stmt).all())


@router.get("/{ticker}/prices", response_model=list[PriceBarRead])
def get_symbol_prices(
    ticker: str,
    timeframe: str = Query(default="1d"),
    limit: int = Query(default=200, ge=1, le=5000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list:
    """Return recent OHLCV bars for one symbol."""

    symbol = get_symbol_by_ticker(db, ticker)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{ticker}' not found")

    try:
        tf = timeframe_from_string(timeframe)
    except PriceDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    bars = list_price_bars(db, symbol.id, tf, limit=limit)
    return bars


@router.get("/{ticker}/features", response_model=list[FeatureValueRead])
def get_symbol_features(
    ticker: str,
    timeframe: str = Query(default="1d"),
    feature_set: str = Query(default="technical_v1"),
    limit: int = Query(default=200, ge=1, le=5000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list:
    """Return recent computed features for one symbol."""

    symbol = get_symbol_by_ticker(db, ticker)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{ticker}' not found")

    try:
        tf = timeframe_from_string(timeframe)
    except PriceDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return list_feature_values(db, symbol.id, tf, feature_set=feature_set, limit=limit)


@router.get("/{ticker}/analysis", response_model=SymbolAnalysisRead)
def get_symbol_analysis(
    ticker: str,
    timeframe: str = Query(default="1d"),
    force: bool = Query(default=False, description="Force re-fetch even if data exists"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SymbolAnalysisRead:
    """Return a user-facing symbol analysis summary. Auto-fetches data if none exists."""

    symbol = get_symbol_by_ticker(db, ticker)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{ticker}' not found")

    try:
        tf = timeframe_from_string(timeframe)
    except PriceDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Check if the symbol has enough price data and features; if not, trigger pipeline
    existing_prices = list_price_bars(db, symbol.id, tf, limit=1)
    existing_features = list_feature_values(db, symbol.id, tf, limit=1)

    needs_pipeline = force or not existing_prices or not existing_features
    if needs_pipeline:
        reason = (
            "force flag set"
            if force
            else "no price data" if not existing_prices
            else "no feature data"
        )
        logger.info("Triggering auto-pipeline for %s (reason: %s)", symbol.ticker, reason)
        try:
            _run_symbol_pipeline(db, symbol, tf)
        except Exception as exc:
            logger.error("Auto-pipeline failed for %s: %s", symbol.ticker, exc)
        else:
            logger.info("Auto-pipeline completed for %s", symbol.ticker)

    return build_symbol_analysis(db, symbol, tf)


@router.post("/{ticker}/analyze", response_model=SymbolAnalysisRead)
def trigger_symbol_analysis(
    ticker: str,
    timeframe: str = Query(default="1d"),
    lookback_days: int = Query(default=365 * 3, ge=30, le=3650),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SymbolAnalysisRead:
    """Force-fetch prices and features for a symbol, then return full analysis."""

    symbol = get_symbol_by_ticker(db, ticker)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{ticker}' not found")

    try:
        tf = timeframe_from_string(timeframe)
    except PriceDataError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    changed = _run_symbol_pipeline(db, symbol, tf, lookback_days=lookback_days)
    logger.info("Pipeline for %s completed: %s feature rows changed", symbol.ticker, changed)

    return build_symbol_analysis(db, symbol, tf)
