"""
Signal Outcome Meta-Learning Module.

Learns from accumulated signal_outcomes data to predict which signal
characteristics lead to successful trades. This meta-model provides
a "success probability" that can be used to filter or boost future signals.

Architecture:
  1. Fetch historical signals + outcomes from the database
  2. Train a classifier to predict hit_target (or positive return)
     based on signal features (scores, market conditions, symbol sector, etc.)
  3. Use this meta-model to score future signal candidates
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import (
    Horizon,
    Signal,
    SignalDirection,
    SignalOutcome,
    SignalStatus,
    Timeframe,
)

logger = logging.getLogger(__name__)

# Features we extract from each historical signal for meta-learning
META_FEATURE_COLUMNS = [
    "final_score",
    "model_score",
    "trend_score",
    "volume_score",
    "momentum_score",
    "risk_score",
    "horizon_encoded",
    "hour_of_day",
    "day_of_week",
    "month",
]

ARTIFACT_DIR = Path("output/models")


@dataclass(frozen=True)
class OutcomeMetaResult:
    """Result of outcome meta-learning training."""

    model_type: str
    accuracy: float
    precision: float
    auc: float
    feature_importances: dict[str, float]
    artifact_path: str
    trained_at: datetime


def _build_meta_frame(db: Session, min_samples: int = 50) -> pd.DataFrame:
    """Build a training DataFrame from historical signal outcomes.

    Each row = one historical signal with its features and outcome label.

    Returns:
        DataFrame with META_FEATURE_COLUMNS + 'label' (1 = hit_target, 0 = not)
    """
    stmt = (
        select(Signal, SignalOutcome)
        .join(SignalOutcome, Signal.id == SignalOutcome.signal_id)
        .where(
            Signal.direction == SignalDirection.BUY,
            SignalOutcome.hit_target.isnot(None),
        )
        .order_by(Signal.signal_time.desc())
        .limit(2000)
    )

    rows = db.execute(stmt).all()
    if len(rows) < min_samples:
        logger.warning(
            "Not enough outcome data for meta-learning: %d < %d",
            len(rows),
            min_samples,
        )
        return pd.DataFrame()

    records = []
    for signal, outcome in rows:
        horizon_map = {Horizon.SHORT: 0, Horizon.MEDIUM: 1, Horizon.LONG: 2}
        records.append(
            {
                "final_score": signal.final_score,
                "model_score": signal.model_score,
                "trend_score": signal.trend_score,
                "volume_score": signal.volume_score,
                "momentum_score": signal.relative_strength_score,
                "risk_score": signal.risk_score,
                "horizon_encoded": horizon_map.get(signal.horizon, 1),
                "hour_of_day": signal.signal_time.hour,
                "day_of_week": signal.signal_time.weekday(),
                "month": signal.signal_time.month,
                "label": 1 if outcome.hit_target else 0,
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        "Meta-learning frame: %d rows, %.1f%% hit_target rate",
        len(df),
        df["label"].mean() * 100,
    )
    return df


def train_outcome_meta_model(
    db: Session,
    *,
    min_samples: int = 50,
) -> OutcomeMetaResult | None:
    """Train a meta-model on historical signal outcomes.

    The meta-model learns to predict which signals are likely to hit their
    target based on the signal's feature scores and temporal context.

    Returns:
        OutcomeMetaResult if successful, None if insufficient data.
    """
    df = _build_meta_frame(db, min_samples=min_samples)
    if df.empty:
        return None

    x = df[META_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    # Handle class imbalance
    pos_rate = y.mean()
    if pos_rate < 0.1 or pos_rate > 0.9:
        logger.warning(
            "Extreme class imbalance (%.1f%% positive). Meta-model may be unreliable.",
            pos_rate * 100,
        )

    # Scale features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Try multiple models, pick best by AUC
    candidates = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        ),
    }

    best_model = None
    best_name = ""
    best_metrics = {"auc": 0.0, "accuracy": 0.0, "precision": 0.0}
    best_importances: dict[str, float] = {}

    for name, model in candidates.items():
        model.fit(x_scaled, y)
        proba = model.predict_proba(x_scaled)[:, 1]
        pred = model.predict(x_scaled)

        metrics = {
            "auc": float(roc_auc_score(y, proba)),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
        }

        # Cross-validation for robustness (if enough samples)
        if len(y) >= 100:
            try:
                cv_auc = cross_val_score(
                    model, x_scaled, y, cv=3, scoring="roc_auc"
                ).mean()
                metrics["auc"] = float(cv_auc)
            except Exception:
                pass

        logger.info("Meta-model %s: AUC=%.4f Prec=%.4f", name, metrics["auc"], metrics["precision"])

        if metrics["auc"] > best_metrics["auc"]:
            best_metrics = metrics
            best_model = model
            best_name = name

            # Extract feature importances
            if hasattr(model, "feature_importances_"):
                best_importances = dict(
                    zip(META_FEATURE_COLUMNS, model.feature_importances_.tolist())
                )
            elif hasattr(model, "coef_"):
                best_importances = dict(
                    zip(META_FEATURE_COLUMNS, model.coef_[0].tolist())
                )

    if best_model is None:
        return None

    # Save artifact
    import joblib

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    artifact_path = ARTIFACT_DIR / f"outcome_meta_{best_name}_{timestamp}.joblib"
    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "feature_columns": META_FEATURE_COLUMNS,
            "metrics": best_metrics,
            "feature_importances": best_importances,
        },
        artifact_path,
    )

    logger.info(
        "Meta-model saved to %s (AUC=%.4f, features=%d)",
        artifact_path,
        best_metrics["auc"],
        len(best_importances),
    )

    return OutcomeMetaResult(
        model_type=best_name,
        accuracy=best_metrics["accuracy"],
        precision=best_metrics["precision"],
        auc=best_metrics["auc"],
        feature_importances=best_importances,
        artifact_path=str(artifact_path),
        trained_at=datetime.now(timezone.utc),
    )


class OutcomeMetaScorer:
    """Loads a trained outcome meta-model and scores new signal candidates.

    Provides a success_probability that can be used to filter or weight
    signals in the portfolio construction pipeline.
    """

    def __init__(self, artifact_path: str):
        """Load meta-model from artifact file.

        Args:
            artifact_path: Path to joblib file from train_outcome_meta_model()
        """
        import joblib

        bundle = joblib.load(artifact_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_columns = bundle["feature_columns"]
        self.feature_importances = bundle.get("feature_importances", {})

    def score_signal_candidate(
        self,
        final_score: float,
        model_score: float,
        trend_score: float,
        volume_score: float,
        momentum_score: float,
        risk_score: float,
        horizon: Horizon,
        signal_time: datetime,
    ) -> float:
        """Predict success probability for a signal candidate.

        Args:
            final_score: Combined final score (0-100).
            model_score: ML model score (0-100).
            trend_score: Trend component score.
            volume_score: Volume component score.
            momentum_score: Momentum component score.
            risk_score: Risk component score.
            horizon: Investment horizon.
            signal_time: Timestamp of the signal.

        Returns:
            Success probability in [0, 1]. Higher = more likely to hit target.
        """
        horizon_map = {Horizon.SHORT: 0, Horizon.MEDIUM: 1, Horizon.LONG: 2}
        features = np.array(
            [
                [
                    final_score,
                    model_score,
                    trend_score,
                    volume_score,
                    momentum_score,
                    risk_score,
                    horizon_map.get(horizon, 1),
                    signal_time.hour,
                    signal_time.weekday(),
                    signal_time.month,
                ]
            ],
            dtype=float,
        )

        features_scaled = self.scaler.transform(features)
        proba = self.model.predict_proba(features_scaled)[0, 1]
        return float(proba)

    def get_top_features(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Return the most important features for outcome prediction."""
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_features[:top_n]


# ---------------------------------------------------------------------------
# Pipeline integration helpers
# ---------------------------------------------------------------------------

def get_latest_meta_scorer() -> OutcomeMetaScorer | None:
    """Load the most recent outcome meta-model artifact.

    Returns None if no artifact exists.
    """
    if not ARTIFACT_DIR.exists():
        return None

    artifacts = sorted(ARTIFACT_DIR.glob("outcome_meta_*.joblib"), reverse=True)
    if not artifacts:
        return None

    try:
        return OutcomeMetaScorer(str(artifacts[0]))
    except Exception as exc:
        logger.warning("Failed to load meta-scorer: %s", exc)
        return None


def compute_outcome_boost(
    scorer: OutcomeMetaScorer,
    final_score: float,
    model_score: float,
    trend_score: float,
    volume_score: float,
    momentum_score: float,
    risk_score: float,
    horizon: Horizon,
    signal_time: datetime,
    *,
    max_boost: float = 8.0,
    min_prob: float = 0.3,
) -> float:
    """Compute a score boost/penalty based on outcome meta-learning.

    Positive boost for signals that historically perform well.
    Negative boost (penalty) for signal patterns that historically fail.

    Args:
        scorer: Loaded OutcomeMetaScorer.
        max_boost: Maximum score boost (in 0-100 scale points).
        min_prob: Below this probability, apply penalty instead of boost.
        Other args: Same as OutcomeMetaScorer.score_signal_candidate().

    Returns:
        Boost value in [-max_boost, +max_boost].
    """
    prob = scorer.score_signal_candidate(
        final_score=final_score,
        model_score=model_score,
        trend_score=trend_score,
        volume_score=volume_score,
        momentum_score=momentum_score,
        risk_score=risk_score,
        horizon=horizon,
        signal_time=signal_time,
    )

    if prob >= 0.5:
        # Successful patterns get a proportional boost
        return max_boost * (prob - 0.5) * 2.0
    elif prob < min_prob:
        # Very low probability patterns get penalized
        return -max_boost * (min_prob - prob) / min_prob
    else:
        return 0.0
