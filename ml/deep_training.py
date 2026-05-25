"""
Deep Learning Model Training for the BIST Signal Platform.

Trains PyTorch models (LSTM, GRU, CNN-LSTM, Transformer) on price bar
sequences from the database, stores results in model_runs and predictions.

Unlike ml/training.py (which uses flat feature vectors + sklearn),
this module uses sequence-based deep learning with proper walk-forward
time-series splitting.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.features import FeatureEngine, FeatureNormalizer
from database.models import (
    FeatureValue,
    Horizon,
    ModelRun,
    Prediction,
    PriceBar,
    Symbol,
    Timeframe,
)
from models.factory import create_model
from training.dataset import TradingDataset
from training.splitter import ChronologicalSplitter
from training.trainer import Trainer

logger = logging.getLogger(__name__)

# Default feature columns for sequence-based training
DL_FEATURE_COLUMNS = [
    "rsi",
    "rsi_normalized",
    "macd",
    "macd_signal",
    "macd_histogram",
    "macd_normalized",
    "atr_pct",
    "volatility",
    "volatility_ratio",
    "intraday_range",
    "price_to_sma_short",
    "price_to_sma_long",
    "trend_strength",
    "bb_position",
    "bb_bandwidth",
    "volume_ratio",
    "obv_normalized",
    "log_return",
    "return_5d",
    "return_10d",
    "return_20d",
]

DL_MODEL_TYPES = ["lstm", "gru", "cnn_lstm", "transformer"]

ARTIFACT_DIR = Path("output/models")


def _fetch_price_bars(
    db: Session, timeframe: Timeframe, symbol_limit: int | None = None
) -> pd.DataFrame:
    """Fetch price bars from database, joined with symbol info."""
    stmt = (
        select(
            PriceBar.symbol_id,
            PriceBar.timestamp,
            PriceBar.open,
            PriceBar.high,
            PriceBar.low,
            PriceBar.close,
            PriceBar.volume,
            Symbol.ticker,
        )
        .join(Symbol, PriceBar.symbol_id == Symbol.id)
        .where(
            PriceBar.timeframe == timeframe,
            Symbol.is_active.is_(True),
        )
        .order_by(PriceBar.symbol_id, PriceBar.timestamp)
    )

    rows = db.execute(stmt).all()
    df = pd.DataFrame(
        rows,
        columns=["symbol_id", "timestamp", "open", "high", "low", "close", "volume", "symbol"],
    )
    logger.info("Fetched %d price bars for deep learning training", len(df))
    return df


def _compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute technical features using FeatureEngine."""
    engine = FeatureEngine()
    df = engine.compute_all_features(df)
    feature_names = engine.get_feature_names()

    # Normalize
    normalizer = FeatureNormalizer(method="rolling", window=60)
    df = normalizer.transform(df)
    norm_feature_names = [f"{name}_norm" for name in feature_names]

    # Only keep features that actually exist
    available_features = [c for c in norm_feature_names if c in df.columns]

    return df, available_features


def _train_deep_model(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_size: int,
    device: str = "cpu",
    epochs: int = 50,
) -> tuple[object, dict]:
    """Train a single deep learning model.

    Returns:
        Tuple of (trained_model, metrics_dict).
    """
    model = create_model(
        model_type=model_type,
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    )

    trainer = Trainer(model=model, device=device, learning_rate=0.001)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=10,
    )

    test_metrics = trainer.evaluate(test_loader)
    return model, test_metrics


def _build_dataloaders(
    df: pd.DataFrame,
    feature_columns: list[str],
    lookback: int = 60,
    label_threshold: float = 0.02,
    label_horizon: int = 5,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader, TradingDataset]:
    """Build train/val/test dataloaders from a DataFrame."""
    splitter = ChronologicalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    # Split per symbol to maintain temporal ordering
    train_parts, val_parts, test_parts = [], [], []

    for symbol in df["symbol"].unique():
        sym_df = df[df["symbol"] == symbol].copy()
        if len(sym_df) < lookback + label_horizon + 50:
            continue
        try:
            t, v, te = splitter.split_dataframe(sym_df)
            train_parts.append(t)
            val_parts.append(v)
            test_parts.append(te)
        except Exception:
            continue

    if not train_parts:
        raise ValueError("Not enough data for any symbol")

    train_df = pd.concat(train_parts)
    val_df = pd.concat(val_parts)
    test_df = pd.concat(test_parts)

    train_ds = TradingDataset(
        train_df,
        feature_columns,
        lookback=lookback,
        label_threshold=label_threshold,
        label_horizon=label_horizon,
    )
    val_ds = TradingDataset(
        val_df,
        feature_columns,
        lookback=lookback,
        label_threshold=label_threshold,
        label_horizon=label_horizon,
    )
    test_ds = TradingDataset(
        test_df,
        feature_columns,
        lookback=lookback,
        label_threshold=label_threshold,
        label_horizon=label_horizon,
    )

    from training.dataset import create_data_loaders

    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds, batch_size=batch_size
    )

    return train_loader, val_loader, test_loader, train_ds


def train_deep_models(
    db: Session,
    *,
    timeframe: Timeframe = Timeframe.DAILY,
    horizon_bars: int = 5,
    target_return: float = 0.03,
    model_types: list[str] | None = None,
    epochs: int = 50,
    min_bars_per_symbol: int = 200,
    device: str | None = None,
) -> list[dict]:
    """Train deep learning models on price bar sequences.

    Args:
        db: Database session.
        timeframe: Timeframe to train on.
        horizon_bars: Number of bars to look ahead for labeling.
        target_return: Minimum return threshold for positive label.
        model_types: List of model types to train. Default: all.
        epochs: Maximum training epochs.
        min_bars_per_symbol: Minimum bars required per symbol.
        device: Torch device. Auto-detected if None.

    Returns:
        List of result dicts with keys: model_type, metrics, model_run_id, artifact_path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_types is None:
        model_types = DL_MODEL_TYPES

    # 1. Fetch and prepare data
    df = _fetch_price_bars(db, timeframe)
    if df.empty:
        logger.warning("No price bars available for deep learning training")
        return []

    # Filter symbols with enough bars
    symbol_counts = df.groupby("symbol").size()
    valid_symbols = symbol_counts[symbol_counts >= min_bars_per_symbol].index.tolist()
    df = df[df["symbol"].isin(valid_symbols)]

    if df.empty:
        logger.warning("No symbols with >= %d bars", min_bars_per_symbol)
        return []

    logger.info(
        "Training deep models on %d symbols, %d total bars",
        len(valid_symbols),
        len(df),
    )

    # 2. Compute features
    df, feature_columns = _compute_features(df)

    if len(feature_columns) < 5:
        logger.warning("Too few features (%d) for deep learning", len(feature_columns))
        return []

    logger.info("Using %d normalized features for deep learning", len(feature_columns))

    # 3. Build dataloaders
    try:
        train_loader, val_loader, test_loader, train_ds = _build_dataloaders(
            df,
            feature_columns,
            lookback=60,
            label_threshold=target_return,
            label_horizon=horizon_bars,
            batch_size=64,
        )
    except ValueError as exc:
        logger.warning("Failed to build dataloaders: %s", exc)
        return []

    logger.info(
        "Data: %d train samples, label rate %.2f%%",
        len(train_ds),
        train_ds.labels.mean() * 100,
    )

    # 4. Train each model type
    results = []
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for model_type in model_types:
        logger.info("Training %s model...", model_type)
        try:
            model, metrics = _train_deep_model(
                model_type=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                input_size=len(feature_columns),
                device=device,
                epochs=epochs,
            )

            # Save model
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            artifact_path = ARTIFACT_DIR / f"deep_{model_type}_{timeframe.value}_{horizon_bars}_{timestamp}.pt"
            model.save(str(artifact_path))

            # Store in model_runs
            train_start = df["timestamp"].min()
            train_end = df["timestamp"].max()

            model_run = ModelRun(
                name=f"deep_{model_type}_{timeframe.value}_{horizon_bars}",
                model_type=f"deep_{model_type}",
                timeframe=timeframe,
                horizon_bars=horizon_bars,
                train_start=train_start.to_pydatetime() if hasattr(train_start, "to_pydatetime") else train_start,
                train_end=train_end.to_pydatetime() if hasattr(train_end, "to_pydatetime") else train_end,
                metrics_json=json.dumps(metrics, sort_keys=True, default=str),
                artifact_path=str(artifact_path),
            )
            db.add(model_run)
            db.flush()

            # Generate predictions for latest bars
            latest_data = df.sort_values(["symbol", "timestamp"]).groupby("symbol").tail(60)
            # For each symbol, create a prediction using the last 60 bars
            for symbol in valid_symbols:
                sym_data = df[df["symbol"] == symbol].tail(60)
                if len(sym_data) < 60:
                    continue

                seq = sym_data[feature_columns].values[-60:].astype(np.float32)
                if seq.shape[0] < 60:
                    continue

                tensor = torch.from_numpy(seq).unsqueeze(0).to(device)  # (1, 60, features)
                prob = model.predict_proba(tensor, return_numpy=True).flatten()[0]

                prediction = Prediction(
                    model_run_id=model_run.id,
                    symbol_id=int(sym_data["symbol_id"].iloc[-1]),
                    prediction_time=sym_data["timestamp"].iloc[-1].to_pydatetime()
                    if hasattr(sym_data["timestamp"].iloc[-1], "to_pydatetime")
                    else sym_data["timestamp"].iloc[-1],
                    probability=float(prob),
                    raw_score=float(prob),
                )
                db.add(prediction)

            db.flush()

            results.append(
                {
                    "model_type": f"deep_{model_type}",
                    "metrics": metrics,
                    "model_run_id": model_run.id,
                    "artifact_path": str(artifact_path),
                }
            )

            logger.info(
                "%s: AUC=%.4f F1=%.4f Precision=%.4f",
                model_type,
                metrics.get("auc", 0),
                metrics.get("f1", 0),
                metrics.get("precision", 0),
            )

        except Exception as exc:
            logger.error("Failed to train %s: %s", model_type, exc, exc_info=True)
            continue

    return results
