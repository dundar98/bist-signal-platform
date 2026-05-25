#!/usr/bin/env python3
"""
Daily Signal Scanner.

Scans all BIST100 stocks, generates predictions, and identifies
trading opportunities based on model signals.
"""

import logging
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import BIST100Validator, get_data_loader, prepare_features
from models import BaseModel
from strategy.signals import SignalGenerator, SignalType, SignalResult
from analysis import NewsSentimentAnalyzer
from utils.signal_history import SignalHistoryTracker

logger = logging.getLogger(__name__)


@dataclass
class StockSignal:
    """Trading signal for a single stock."""
    symbol: str
    probability: float
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    current_price: float
    change_1d: float  # 1-day price change %
    rsi: float
    volatility: float
    sentiment_score: float = 0.0
    target_price: float = 0.0
    horizon_days: int = 0
    history_info: str = ""
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def priority(self) -> int:
        """Higher priority = more actionable signal."""
        if self.signal.lower() == SignalType.BUY.value and self.probability > 0.75:
            return 3
        elif self.signal.lower() == SignalType.BUY.value:
            return 2
        elif self.signal.lower() == SignalType.SELL.value:
            return 1
        return 0
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'probability': self.probability,
            'signal': self.signal,
            'confidence': self.confidence,
            'current_price': self.current_price,
            'sentiment_score': self.sentiment_score,
            'change_1d': self.change_1d,
            'rsi': self.rsi,
            'volatility': self.volatility,
            'target_price': self.target_price,
            'horizon_days': self.horizon_days,
            'history_info': self.history_info,
            'reason': self.reason,
            'timestamp': str(self.timestamp),
        }


@dataclass
class DailyScanResult:
    """Result of daily market scan."""
    scan_date: date
    buy_signals: List[StockSignal]
    sell_signals: List[StockSignal]
    hold_signals: List[StockSignal]
    errors: List[str]
    scan_duration: float
    mode: str = "UZUN"
    
    @property
    def total_scanned(self) -> int:
        return len(self.buy_signals) + len(self.sell_signals) + len(self.hold_signals)
    
    def get_top_signals(self, n: int = 5) -> List[StockSignal]:
        """Get top N buy signals by probability."""
        return sorted(self.buy_signals, key=lambda x: x.probability, reverse=True)[:n]
    
    def to_summary_dict(self) -> dict:
        return {
            'scan_date': str(self.scan_date),
            'total_scanned': self.total_scanned,
            'buy_count': len(self.buy_signals),
            'sell_count': len(self.sell_signals),
            'hold_count': len(self.hold_signals),
            'error_count': len(self.errors),
            'scan_duration_seconds': self.scan_duration,
        }


class DailyScanner:
    """
    Scans BIST100 stocks daily and generates trading signals.
    """
    
    def __init__(
        self,
        model_path: str,
        config: any,
        device: str = "cpu",
        history_file: str = "output/signal_history.json"
    ):
        """
        Initialize daily scanner.
        """
        self.config = config
        self.device = device
        self.validator = BIST100Validator()
        logger.info(f"Validator initialized with {len(self.validator.get_all_symbols())} symbols.")
        self.history_tracker = SignalHistoryTracker(history_file)
        
        # Load model
        self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.eval()
        
        # Metadata from model or config
        self.feature_columns = getattr(self.model, 'feature_columns', None)
        self.lookback = getattr(self.model, 'lookback', 60)
        
        # Initialize signal generator
        self.signal_generator = SignalGenerator(
            entry_threshold=self.config.backtest.entry_threshold,
            exit_threshold=self.config.backtest.exit_threshold,
            use_dynamic_threshold=True
        )
        
        self.sentiment_analyzer = NewsSentimentAnalyzer()
    
    def scan_all(
        self,
        symbols: Optional[List[str]] = None,
        limit: Optional[int] = 30,
        mode: str = "UZUN",
        lookback_days: int = 180
    ) -> DailyScanResult:
        """
        Scan symbols and generate signals based on mode.
        """
        import time
        start_time = time.time()

        intervals = {"KISA": "15m", "ORTA": "1h", "UZUN": "1d"}
        interval = intervals.get(mode, "1d")
        
        # Yahoo Finance limit: 15m data is only available for the last 60 days
        if mode == "KISA" and lookback_days > 59:
            logger.info(f"Capping {mode} lookback from {lookback_days} to 59 days for Yahoo compatibility.")
            lookback_days = 59
        
        if symbols is None:
            symbols = self.validator.get_all_symbols()
            if limit is not None:
                symbols = symbols[:limit]
        
        # Determine Horizons and TP Multipliers
        tp_mults = {"KISA": 0.02, "ORTA": 0.07, "UZUN": 0.15}
        horizons = {"KISA": 2, "ORTA": 10, "UZUN": 60}
        
        tp_mult = tp_mults.get(mode, 0.10)
        horizon = horizons.get(mode, 30)
        
        results = []
        errors = []
        
        loader = get_data_loader(source="yfinance")
        
        logger.info(f"Starting {mode} scan ({interval}) for {len(symbols)} symbols...")
        
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        for symbol in symbols:
            try:
                df = loader.load(symbol, start_date=start_date, end_date=date.today(), interval=interval)
                signal = self._scan_symbol(symbol, df)
                
                if signal.signal.lower() == SignalType.BUY.value:
                    signal.target_price = signal.current_price * (1 + tp_mult)
                    signal.horizon_days = horizon
                    signal.history_info = self.history_tracker.get_signal_performance(symbol, signal.current_price) or ""
                    self.history_tracker.save_signal(symbol, signal.signal, signal.current_price, signal.probability, mode)

                results.append(signal)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                errors.append(f"{symbol}: {str(e)}")
        
        duration = time.time() - start_time
        
        buy_signals = [s for s in results if s.signal.lower() == SignalType.BUY.value]
        sell_signals = [s for s in results if s.signal.lower() == SignalType.SELL.value]
        hold_signals = [s for s in results if s.signal.lower() == SignalType.HOLD.value]

        result = DailyScanResult(
            scan_date=end_date,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            hold_signals=hold_signals,
            errors=errors,
            scan_duration=duration,
            mode=mode
        )
        
        return result
    
    def _scan_symbol(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> StockSignal:
        """
        Scan a single symbol and generate prediction.
        """
        if len(df) < self.lookback:
            raise ValueError(f"Not enough data: {len(df)} < {self.lookback}")
        
        df_features, feature_names = prepare_features(df, normalize=True)
        df_features = df_features.dropna()
        
        # Dynamic feature column discovery if missing from model metadata
        if self.feature_columns is None:
            self.feature_columns = feature_names
            logger.info(f"Dynamically discovered {len(self.feature_columns)} feature columns for scanner.")
            
        if len(df_features) < self.lookback:
            raise ValueError(f"Insufficient features after NaN drop")
        
        # Get latest features for prediction
        latest_features = df_features[self.feature_columns].values[-self.lookback:]
        latest_features = np.nan_to_num(latest_features, nan=0.0)
        
        x = torch.from_numpy(latest_features.astype(np.float32)).unsqueeze(0)
        x = x.to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            prob = output[0].item() if isinstance(output, tuple) else output.item()
        
        signal_result = self.signal_generator.generate(prob, current_position=None)
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        change_1d = (current_price - prev_price) / prev_price * 100
        
        rsi = df_features['rsi'].iloc[-1] if 'rsi' in df_features.columns else 50
        volatility = df_features['volatility'].iloc[-1] if 'volatility' in df_features.columns else 0
        sentiment = 0.0
        try:
            sentiment = self.sentiment_analyzer.get_stock_sentiment(symbol)
        except Exception:
            pass
            
        return StockSignal(
            symbol=symbol,
            probability=prob,
            signal=signal_result.signal.value.upper(),
            confidence=signal_result.confidence,
            current_price=current_price,
            change_1d=change_1d,
            rsi=rsi,
            volatility=volatility,
            sentiment_score=sentiment,
            reason=signal_result.reason,
        )


def generate_signal_report(result: DailyScanResult) -> str:
    """
    Generate a formatted text report of scan results.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"📊 BIST100 GÜNLÜK SİNYAL RAPORU - {result.scan_date}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"📈 Taranan Hisse: {result.total_scanned}")
    lines.append(f"✅ AL Sinyali: {len(result.buy_signals)}")
    lines.append(f"❌ SAT Sinyali: {len(result.sell_signals)}")
    lines.append(f"⏸️ BEKLE: {len(result.hold_signals)}")
    lines.append("")
    
    if result.buy_signals:
        lines.append("-" * 60)
        lines.append("-" * 60)
        lines.append("🟢 TÜM AL SİNYALLERİ:")
        lines.append("-" * 60)
        # Sort by probability but show all
        for signal in sorted(result.buy_signals, key=lambda x: x.probability, reverse=True):
            emoji = "🔥" if signal.probability > 0.70 else ("✅" if signal.probability > 0.60 else "⚠️")
            desc = "GÜÇLÜ AL" if signal.probability > 0.70 else ("AL" if signal.probability > 0.60 else "SPEKÜLATİF")
            
            target_info = (
                f"   └─ 🎯 Hedef: {signal.target_price:.2f} TL | "
                f"⏳ Vade: {signal.horizon_days} Gün | "
                f"📜 {signal.history_info}"
            ) if signal.signal.lower() == SignalType.BUY.value else ""
            
            lines.append(
                f"{emoji} {signal.symbol:6} | Sinyal: {desc} | "
                f"Olasılık: {signal.probability:.1%} | Fiyat: {signal.current_price:.2f} TL"
            )
            if target_info: lines.append(target_info)
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("⚠️ DİKKAT: Bu sinyaller tavsiye niteliğinde değildir.")
    lines.append("=" * 60)
    return "\n".join(lines)


def generate_dashboard_json(result: DailyScanResult, output_path: str = "docs/dashboard_data.json"):
    """
    Generate JSON data for the web dashboard.

    Saves a mode-specific file (e.g. dashboard_data_KISA.json) and merges it
    into a combined dashboard_data.json that contains all available modes.
    """
    import json

    avg_vol = np.mean([s.volatility for s in (result.buy_signals + result.sell_signals + result.hold_signals)]) if (result.buy_signals + result.sell_signals + result.hold_signals) else 0
    vol_status = "Yüksek ⚠️" if avg_vol > 0.03 else ("Düşük 💤" if avg_vol < 0.01 else "Normal")

    from config import get_config
    conf = get_config()

    data = {
        "scan_date": str(result.scan_date),
        "mode": result.mode,
        "total_scanned": result.total_scanned,
        "buy_count": len(result.buy_signals),
        "sell_count": len(result.sell_signals),
        "hold_count": len(result.hold_signals),
        "error_count": len(result.errors),
        "market_volatility": vol_status,
        "buy_signals": [s.to_dict() for s in result.buy_signals],
        "sell_signals": [s.to_dict() for s in result.sell_signals],
        "hold_signals": [], 
        "portfolio": {
            "total_equity": 100000.0,
            "daily_pnl": 1250.0,
            "daily_pnl_pct": 1.25,
            "holdings": []
        },
        "config": {
            "entry_threshold": conf.backtest.entry_threshold,
            "stop_loss": conf.backtest.stop_loss_pct
        }
    }

    out_dir = Path(output_path).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    # Save mode-specific file
    mode_file = out_dir / f"dashboard_data_{result.mode}.json"
    with open(mode_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Dashboard data saved to {mode_file}")

    # Merge into combined dashboard_data.json
    combined_path = Path(output_path)
    combined: dict = {}
    if combined_path.exists():
        try:
            with open(combined_path, 'r', encoding='utf-8') as f:
                combined = json.load(f)
        except (json.JSONDecodeError, OSError):
            combined = {}

    # combined structure: { "modes": { "KISA": {...}, ... }, "active_mode": "KISA" }
    if "modes" not in combined:
        combined = {"modes": {}, "active_mode": result.mode}
    combined["modes"][result.mode] = data
    combined["active_mode"] = result.mode
    combined["scan_date"] = str(result.scan_date)

    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    logger.info(f"Combined dashboard data updated at {combined_path}")
