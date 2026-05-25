"""Rule-based scoring with ML probability integration for explainable signal selection."""

from dataclasses import dataclass
from typing import Optional

from database.models import FeatureValue, Horizon


@dataclass(frozen=True)
class ScoreResult:
    final_score: float
    model_score: float
    trend_score: float
    volume_score: float
    momentum_score: float
    risk_score: float
    direction: str
    reason: str
    ml_probability: Optional[float] = None
    ml_blend_weight: float = 0.0


@dataclass(frozen=True)
class ScoringProfile:
    trend_weight: float
    momentum_weight: float
    volume_weight: float
    risk_weight: float
    ml_weight: float
    min_buy_score: float
    min_risk_score: float
    min_volume_score: float
    max_buy_rsi: float
    min_trend_score: float
    min_momentum_score: float
    min_ml_probability: float


SCORING_PROFILES = {
    Horizon.SHORT: ScoringProfile(
        trend_weight=0.12,
        momentum_weight=0.33,
        volume_weight=0.25,
        risk_weight=0.12,
        ml_weight=0.18,
        min_buy_score=30,
        min_risk_score=15,
        min_volume_score=25,
        max_buy_rsi=85,
        min_trend_score=15,
        min_momentum_score=20,
        min_ml_probability=0.40,
    ),
    Horizon.MEDIUM: ScoringProfile(
        trend_weight=0.28,
        momentum_weight=0.24,
        volume_weight=0.16,
        risk_weight=0.12,
        ml_weight=0.20,
        min_buy_score=30,
        min_risk_score=15,
        min_volume_score=25,
        max_buy_rsi=85,
        min_trend_score=15,
        min_momentum_score=15,
        min_ml_probability=0.40,
    ),
    Horizon.LONG: ScoringProfile(
        trend_weight=0.40,
        momentum_weight=0.12,
        volume_weight=0.08,
        risk_weight=0.20,
        ml_weight=0.20,
        min_buy_score=30,
        min_risk_score=20,
        min_volume_score=15,
        max_buy_rsi=85,
        min_trend_score=20,
        min_momentum_score=15,
        min_ml_probability=0.40,
    ),
}


def horizon_from_string(value: str | Horizon) -> Horizon:
    if isinstance(value, Horizon):
        return value
    try:
        return Horizon(value.lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in Horizon)
        raise ValueError(f"Unsupported horizon '{value}'. Allowed: {allowed}") from exc


def _value(value: float | None, fallback: float = 50.0) -> float:
    return fallback if value is None else float(value)


def _bounded(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _risk_score(feature: FeatureValue) -> float:
    volatility = _value(feature.volatility, 0.25)
    atr_pct = _value(feature.atr_pct, 0.03)

    # Lower volatility and tighter ATR receive a higher risk score.
    score = 100 - volatility * 140 - atr_pct * 500
    return _bounded(score)


def _rsi_penalty(feature: FeatureValue) -> float:
    rsi = _value(feature.rsi, 50.0)
    if rsi >= 78:
        return 18.0
    if rsi >= 70:
        return 10.0
    if rsi <= 25:
        return 8.0
    return 0.0


def score_feature(
    feature: FeatureValue,
    *,
    horizon: Horizon = Horizon.MEDIUM,
    ml_probability: Optional[float] = None,
) -> ScoreResult:
    """Score one feature row and produce an explainable direction.

    Args:
        feature: FeatureValue row with technical indicators.
        horizon: Investment horizon (SHORT/MEDIUM/LONG).
        ml_probability: Optional ML model probability [0, 1]. When provided,
            it is blended into the final_score using the profile's ml_weight.
    """

    profile = SCORING_PROFILES[horizon]
    trend = _value(feature.trend_score)
    volume = _value(feature.volume_score)
    momentum = _value(feature.momentum_score)
    risk = _risk_score(feature)
    rsi = _value(feature.rsi, 50.0)
    rsi_penalty = _rsi_penalty(feature)

    # Base rule-based score (0-100 scale)
    rule_score = (
        trend * profile.trend_weight
        + momentum * profile.momentum_weight
        + volume * profile.volume_weight
        + risk * profile.risk_weight
    )

    # Blend ML probability into the score if available
    ml_blend_weight = 0.0
    model_score = rule_score  # default: fallback to rule score

    if ml_probability is not None and 0.0 <= ml_probability <= 1.0:
        ml_blend_weight = profile.ml_weight
        # Convert ML probability [0, 1] to 0-100 scale
        ml_score = ml_probability * 100.0
        model_score = ml_score
        # Blend: rule_score gets (1 - ml_weight), ML gets ml_weight
        # Then subtract RSI penalty
        final_score = (
            rule_score * (1.0 - ml_blend_weight)
            + ml_score * ml_blend_weight
            - rsi_penalty
        )
        final_score = _bounded(final_score)
    else:
        final_score = _bounded(rule_score - rsi_penalty)

    blocks = []
    if risk < profile.min_risk_score:
        blocks.append(f"risk below {profile.min_risk_score:.0f}")
    if volume < profile.min_volume_score:
        blocks.append(f"volume below {profile.min_volume_score:.0f}")
    if rsi > profile.max_buy_rsi:
        blocks.append(f"RSI above {profile.max_buy_rsi:.0f}")
    if (
        ml_probability is not None
        and ml_probability < profile.min_ml_probability
    ):
        blocks.append(
            f"ML prob {ml_probability:.2f} below {profile.min_ml_probability:.2f}"
        )

    if (
        final_score >= profile.min_buy_score
        and trend >= profile.min_trend_score
        and momentum >= profile.min_momentum_score
        and not blocks
    ):
        direction = "BUY"
    elif final_score <= 35:
        direction = "SELL"
    else:
        direction = "HOLD"

    reasons = []
    reasons.append(f"horizon {horizon.value}")
    reasons.append(f"trend {trend:.0f}")
    reasons.append(f"momentum {momentum:.0f}")
    reasons.append(f"volume {volume:.0f}")
    reasons.append(f"risk {risk:.0f}")
    reasons.append(f"rsi {rsi:.0f}")
    if ml_probability is not None:
        reasons.append(f"ML prob {ml_probability:.2f} (blend {ml_blend_weight:.0%})")
    if rsi_penalty:
        reasons.append(f"RSI penalty {rsi_penalty:.0f}")
    if blocks:
        reasons.append("blocked: " + "; ".join(blocks))
    if direction == "BUY":
        reasons.append("passed selective filters")
    elif direction == "SELL":
        reasons.append("weak composite profile")
    else:
        reasons.append("watchlist only")

    return ScoreResult(
        final_score=final_score,
        model_score=_bounded(model_score),
        trend_score=trend,
        volume_score=volume,
        momentum_score=momentum,
        risk_score=risk,
        direction=direction,
        reason=", ".join(reasons),
        ml_probability=ml_probability,
        ml_blend_weight=ml_blend_weight,
    )
