"""Repository helpers for symbol-level analysis views."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.schemas.analysis import (
    DecisionLogRead,
    SymbolAnalysisRead,
    SymbolFeatureSummary,
    SymbolPriceSummary,
    SymbolSignalSummary,
)
from database.models import DecisionLog, FeatureValue, PriceBar, Signal, Symbol, Timeframe


def _pct_change(current: float, previous: float | None) -> float | None:
    if previous is None or previous == 0:
        return None
    return ((current - previous) / previous) * 100


def _latest_feature(db: Session, symbol_id: int, timeframe: Timeframe) -> FeatureValue | None:
    return db.scalar(
        select(FeatureValue)
        .where(FeatureValue.symbol_id == symbol_id, FeatureValue.timeframe == timeframe)
        .order_by(FeatureValue.timestamp.desc())
        .limit(1)
    )


def _recent_prices(db: Session, symbol_id: int, timeframe: Timeframe, limit: int = 25) -> list[PriceBar]:
    rows = list(
        db.scalars(
            select(PriceBar)
            .where(PriceBar.symbol_id == symbol_id, PriceBar.timeframe == timeframe)
            .order_by(PriceBar.timestamp.desc())
            .limit(limit)
        ).all()
    )
    return list(reversed(rows))


def _latest_signals(db: Session, symbol_id: int, timeframe: Timeframe, limit: int = 6) -> list[Signal]:
    return list(
        db.scalars(
            select(Signal)
            .where(Signal.symbol_id == symbol_id, Signal.timeframe == timeframe)
            .order_by(Signal.signal_time.desc(), Signal.created_at.desc())
            .limit(limit)
        ).all()
    )


def _latest_decision_logs(db: Session, symbol_id: int, timeframe: Timeframe, limit: int = 10) -> list[DecisionLog]:
    return list(
        db.scalars(
            select(DecisionLog)
            .where(DecisionLog.symbol_id == symbol_id, DecisionLog.timeframe == timeframe)
            .order_by(DecisionLog.decision_time.desc())
            .limit(limit)
        ).all()
    )


def _summary(symbol: Symbol, feature: FeatureValue | None, signals: list[Signal], prices: list[PriceBar]) -> str:
    if feature is None:
        if not prices:
            return (
                f"{symbol.ticker} için henüz fiyat verisi çekilemedi. "
                f"Analizi başlatmak için 'Analiz Et' butonuna tıklayın veya /symbols/{symbol.ticker}/analyze endpoint'ini kullanın."
            )
        return (
            f"{symbol.ticker} için {len(prices)} fiyat barı mevcut ancak teknik göstergeler henüz hesaplanmadı. "
            "Analizi başlatmak için 'Analiz Et' butonuna tıklayın."
        )

    parts = []
    if feature.trend_score is not None:
        if feature.trend_score >= 70:
            parts.append("trend güçlü")
        elif feature.trend_score <= 40:
            parts.append("trend zayıf")
        else:
            parts.append("trend nötr")

    if feature.momentum_score is not None:
        if feature.momentum_score >= 70:
            parts.append("momentum destekli")
        elif feature.momentum_score <= 40:
            parts.append("momentum zayıf")

    if feature.volume_score is not None and feature.volume_score >= 65:
        parts.append("hacim desteği var")
    elif feature.volume_score is not None and feature.volume_score <= 35:
        parts.append("hacim zayıf")

    latest_buy = next((signal for signal in signals if signal.direction.value == "BUY"), None)
    signal_text = f" Son kayıtlı sinyal {latest_buy.horizon.value} vadede AL." if latest_buy else " Güncel AL sinyali yok."
    return f"{symbol.ticker}: {', '.join(parts) if parts else 'teknik görünüm karışık'}.{signal_text}"


def build_symbol_analysis(db: Session, symbol: Symbol, timeframe: Timeframe) -> SymbolAnalysisRead:
    prices = _recent_prices(db, symbol.id, timeframe)
    latest_price = prices[-1] if prices else None
    feature = _latest_feature(db, symbol.id, timeframe)
    signals = _latest_signals(db, symbol.id, timeframe)
    logs = _latest_decision_logs(db, symbol.id, timeframe)

    price_summary = None
    if latest_price is not None:
        closes = [row.close for row in prices]
        price_summary = SymbolPriceSummary(
            timestamp=latest_price.timestamp,
            close=latest_price.close,
            return_1d=_pct_change(closes[-1], closes[-2] if len(closes) >= 2 else None),
            return_5d=_pct_change(closes[-1], closes[-6] if len(closes) >= 6 else None),
            return_20d=_pct_change(closes[-1], closes[-21] if len(closes) >= 21 else None),
        )

    feature_summary = None
    if feature is not None:
        feature_summary = SymbolFeatureSummary(
            timestamp=feature.timestamp,
            rsi=feature.rsi,
            macd=feature.macd,
            macd_signal=feature.macd_signal,
            atr_pct=feature.atr_pct,
            volatility=feature.volatility,
            volume_ratio=feature.volume_ratio,
            trend_score=feature.trend_score,
            volume_score=feature.volume_score,
            momentum_score=feature.momentum_score,
        )

    return SymbolAnalysisRead(
        symbol_id=symbol.id,
        ticker=symbol.ticker,
        name=symbol.name,
        sector=symbol.sector,
        timeframe=timeframe.value,
        price=price_summary,
        feature=feature_summary,
        latest_signals=[
            SymbolSignalSummary(
                id=signal.id,
                signal_time=signal.signal_time,
                horizon=signal.horizon.value,
                direction=signal.direction.value,
                status=signal.status.value,
                final_score=signal.final_score,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                reason=signal.reason,
            )
            for signal in signals
        ],
        decision_logs=[
            DecisionLogRead(
                id=log.id,
                decision_time=log.decision_time,
                signal_time=log.signal_time,
                timeframe=log.timeframe.value,
                horizon=log.horizon.value,
                strategy=log.strategy,
                direction=log.direction.value,
                entry_price=log.entry_price,
                stop_price=log.stop_price,
                target_price=log.target_price,
                final_score=log.final_score,
                reason=log.reason,
            )
            for log in logs
        ],
        summary=_summary(symbol, feature, signals, prices),
    )
