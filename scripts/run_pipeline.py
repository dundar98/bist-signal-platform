#!/usr/bin/env python3
"""Run the daily BIST data, feature, signal, and outcome pipeline."""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import get_data_loader
from database.models import Horizon, Timeframe
from database.repositories.features import compute_and_store_features
from database.repositories.jobs import finish_job, start_job
from database.repositories.market import build_market_radar
from database.repositories.outcomes import update_signal_outcomes
from database.repositories.portfolio import create_portfolio_snapshot
from database.repositories.prices import (
    get_last_price_timestamp,
    list_active_symbols,
    timeframe_from_string,
    upsert_price_bars,
)
from database.session import SessionLocal
from ml.training import train_baseline_models
from signals.scoring import horizon_from_string
from utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineSummary:
    symbols_processed: int = 0
    price_bars_changed: int = 0
    feature_rows_changed: int = 0
    portfolio_ids: list[int] = field(default_factory=list)
    outcome_rows_changed: int = 0
    model_run_id: int | None = None
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full BIST research pipeline")
    parser.add_argument("--timeframe", default=Timeframe.DAILY.value, choices=[item.value for item in Timeframe])
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic"])
    parser.add_argument("--symbols", help="Comma-separated ticker list. Defaults to active BIST100 symbols.")
    parser.add_argument("--limit", type=int, default=None, help="Limit symbols for data and feature processing.")
    parser.add_argument("--start", default=None, help="Initial backfill start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--lookback-days", type=int, default=365 * 3)
    parser.add_argument("--lookback-bars", type=int, default=260)
    parser.add_argument("--feature-set", default="technical_v1")
    parser.add_argument("--strategy", default="technical_selective_v1")
    parser.add_argument("--horizons", default="short,medium,long")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=55.0)
    parser.add_argument("--outcome-limit", type=int, default=500)
    parser.add_argument("--skip-prices", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--skip-portfolios", action="store_true")
    parser.add_argument("--skip-outcomes", action="store_true")
    parser.add_argument("--train-ml", action="store_true", help="Train cheap baseline ML model after feature updates.")
    parser.add_argument("--train-outcome-meta", action="store_true", help="Train outcome meta-model after outcome updates.")
    parser.add_argument("--train-deep", action="store_true", help="Train deep learning models (LSTM/GRU/Transformer) in addition to baseline ML.")
    parser.add_argument("--deep-epochs", type=int, default=50, help="Max epochs for deep learning training.")
    parser.add_argument("--deep-models", default="transformer", help="Comma-separated deep model types (lstm,gru,cnn_lstm,transformer).")
    parser.add_argument("--ml-horizon-bars", type=int, default=5)
    parser.add_argument("--ml-target-return", type=float, default=0.03)
    parser.add_argument("--use-market-regime", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--send-email", action="store_true", help="Send daily report email after pipeline completes.")
    return parser.parse_args()


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


def _intraday_cap_days(timeframe: Timeframe, lookback_days: int) -> int:
    if timeframe == Timeframe.FIFTEEN_MIN:
        return min(lookback_days, 59)
    if timeframe == Timeframe.HOURLY:
        return min(lookback_days, 729)
    return lookback_days


def _parse_horizons(value: str) -> list[Horizon]:
    horizons = []
    for item in value.split(","):
        item = item.strip()
        if item:
            horizons.append(horizon_from_string(item))
    return horizons


def _record_error(summary: PipelineSummary, message: str, fail_fast: bool) -> None:
    summary.errors.append(message)
    logger.error(message)
    if fail_fast:
        raise RuntimeError(message)


def main() -> int:
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    timeframe = timeframe_from_string(args.timeframe)
    end_date = _parse_date(args.end) or date.today()
    lookback_days = _intraday_cap_days(timeframe, args.lookback_days)
    horizons = _parse_horizons(args.horizons)
    loader = get_data_loader(source=args.source)
    summary = PipelineSummary()

    with SessionLocal() as db:
        job = start_job(
            db,
            "daily_pipeline",
            summary={"timeframe": timeframe.value, "source": args.source, "horizons": args.horizons},
        )
        market_risk_mode = None
        if args.use_market_regime:
            market_risk_mode = build_market_radar(db, timeframe=timeframe).risk_mode
            logger.info("Market regime enabled. risk_mode=%s", market_risk_mode)

        if args.symbols:
            requested = {ticker.strip().upper() for ticker in args.symbols.split(",") if ticker.strip()}
            symbols = [symbol for symbol in list_active_symbols(db, limit=None) if symbol.ticker in requested]
            missing = sorted(requested - {symbol.ticker for symbol in symbols})
            for ticker in missing:
                _record_error(summary, f"{ticker}: symbol not found or inactive", args.fail_fast)
        else:
            symbols = list_active_symbols(db, limit=args.limit)

        if args.limit is not None and args.symbols:
            symbols = symbols[: args.limit]

        summary.symbols_processed = len(symbols)
        logger.info("Pipeline started. symbols=%s timeframe=%s horizons=%s", len(symbols), timeframe.value, args.horizons)

        if not args.skip_prices:
            for symbol in symbols:
                last_ts = None
                try:
                    last_ts = get_last_price_timestamp(db, symbol.id, timeframe)
                    if last_ts is not None:
                        start_date = last_ts.date() + timedelta(days=1)
                    else:
                        start_date = _parse_date(args.start) or (end_date - timedelta(days=lookback_days))

                    if start_date > end_date:
                        logger.info("%s already up to date", symbol.ticker)
                        continue

                    df = loader.load(symbol.ticker, start_date=start_date, end_date=end_date, interval=timeframe.value)
                    changed = upsert_price_bars(db, symbol, timeframe, df, source=args.source)
                    db.commit()
                    summary.price_bars_changed += changed
                    logger.info("%s: persisted %s bars", symbol.ticker, changed)
                except Exception as exc:
                    db.rollback()
                    if last_ts is not None and "No data returned" in str(exc):
                        logger.info("%s: no new bars returned; keeping existing latest bar %s", symbol.ticker, last_ts.date())
                        continue
                    _record_error(summary, f"{symbol.ticker}: price update failed: {exc}", args.fail_fast)

        if not args.skip_features:
            for symbol in symbols:
                try:
                    changed = compute_and_store_features(
                        db,
                        symbol,
                        timeframe,
                        lookback_bars=args.lookback_bars,
                        feature_set=args.feature_set,
                    )
                    db.commit()
                    summary.feature_rows_changed += changed
                    logger.info("%s: persisted %s feature rows", symbol.ticker, changed)
                except Exception as exc:
                    db.rollback()
                    _record_error(summary, f"{symbol.ticker}: feature update failed: {exc}", args.fail_fast)

        if not args.skip_portfolios:
            # Try to load outcome meta-scorer for signal boosting
            outcome_scorer = None
            try:
                from ml.outcome_learning import get_latest_meta_scorer
                outcome_scorer = get_latest_meta_scorer()
                if outcome_scorer is not None:
                    logger.info("Outcome meta-scorer loaded for signal boosting")
            except Exception:
                pass

            for horizon in horizons:
                try:
                    snapshot = create_portfolio_snapshot(
                        db,
                        timeframe=timeframe,
                        horizon=horizon,
                        strategy=args.strategy,
                        feature_set=args.feature_set,
                        symbol_limit=args.limit,
                        max_positions=args.max_positions,
                        min_score=args.min_score,
                        market_risk_mode=market_risk_mode,
                        outcome_meta_scorer=outcome_scorer,
                    )
                    db.commit()
                    db.refresh(snapshot)
                    summary.portfolio_ids.append(snapshot.id)
                    logger.info(
                        "Portfolio built. id=%s timeframe=%s horizon=%s items=%s",
                        snapshot.id,
                        timeframe.value,
                        horizon.value,
                        len(snapshot.items),
                    )
                except Exception as exc:
                    db.rollback()
                    _record_error(summary, f"{horizon.value}: portfolio build failed: {exc}", args.fail_fast)

        if not args.skip_outcomes:
            try:
                summary.outcome_rows_changed = update_signal_outcomes(db, limit=args.outcome_limit, only_buy=True)
                db.commit()
                logger.info("Outcome rows changed=%s", summary.outcome_rows_changed)
            except Exception as exc:
                db.rollback()
                _record_error(summary, f"outcome update failed: {exc}", args.fail_fast)

        if args.train_ml:
            try:
                result = train_baseline_models(
                    db,
                    timeframe=timeframe,
                    feature_set=args.feature_set,
                    horizon_bars=args.ml_horizon_bars,
                    target_return=args.ml_target_return,
                )
                db.commit()
                summary.model_run_id = result.model_run.id
                logger.info("ML baseline completed. model_run_id=%s metrics=%s", result.model_run.id, result.metrics)
            except Exception as exc:
                db.rollback()
                _record_error(summary, f"ML training failed: {exc}", args.fail_fast)

        if args.train_outcome_meta:
            try:
                from ml.outcome_learning import train_outcome_meta_model
                meta_result = train_outcome_meta_model(db)
                if meta_result is not None:
                    logger.info(
                        "Outcome meta-model trained. type=%s AUC=%.4f precision=%.4f",
                        meta_result.model_type,
                        meta_result.auc,
                        meta_result.precision,
                    )
                else:
                    logger.info("Outcome meta-model skipped (insufficient data)")
            except Exception as exc:
                _record_error(summary, f"Outcome meta-training failed: {exc}", args.fail_fast)

        if args.train_deep:
            try:
                from ml.deep_training import train_deep_models
                deep_model_types = [m.strip() for m in args.deep_models.split(",") if m.strip()]
                deep_results = train_deep_models(
                    db,
                    timeframe=timeframe,
                    horizon_bars=args.ml_horizon_bars,
                    target_return=args.ml_target_return,
                    model_types=deep_model_types,
                    epochs=args.deep_epochs,
                )
                db.commit()
                for dr in deep_results:
                    logger.info(
                        "Deep model trained: %s AUC=%.4f run_id=%s",
                        dr["model_type"],
                        dr["metrics"].get("auc", 0),
                        dr["model_run_id"],
                    )
            except Exception as exc:
                db.rollback()
                _record_error(summary, f"Deep learning training failed: {exc}", args.fail_fast)

        finish_job(
            db,
            job,
            status="failed" if summary.errors else "success",
            summary={
                "symbols": summary.symbols_processed,
                "changed_bars": summary.price_bars_changed,
                "changed_features": summary.feature_rows_changed,
                "portfolios": summary.portfolio_ids,
                "changed_outcomes": summary.outcome_rows_changed,
                "model_run_id": summary.model_run_id,
                "errors": summary.errors,
                "market_risk_mode": market_risk_mode,
            },
            error="\n".join(summary.errors) if summary.errors else None,
        )

    print(
        "Pipeline complete. "
        f"symbols={summary.symbols_processed} "
        f"changed_bars={summary.price_bars_changed} "
        f"changed_features={summary.feature_rows_changed} "
        f"portfolios={summary.portfolio_ids} "
        f"changed_outcomes={summary.outcome_rows_changed} "
        f"model_run_id={summary.model_run_id} "
        f"errors={len(summary.errors)}"
    )
    if summary.errors:
        for error in summary.errors[:20]:
            print(f"ERROR {error}")
        return 1

    # Optionally send email report
    if args.send_email:
        import subprocess
        print("Sending daily report email...")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "send_daily_report.py")],
            cwd=str(PROJECT_ROOT),
            check=False,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
