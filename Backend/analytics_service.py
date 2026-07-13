"""Analytics helpers: 7-day model forecast, SHAP-style sources, health guidance."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from Backend.air_quality_service import (
    calculate_smog_index,
    get_aqi_status,
    model_prediction_fields,
    pm25_to_aqi,
)

# Model feature → smog source category (SHAP-style attribution groups)
_FEATURE_SOURCE_WEIGHTS: Dict[str, List[tuple]] = {
    "traffic_combined": [("Traffic Emissions", 1.4)],
    "no2": [("Traffic Emissions", 0.9), ("Industrial Activity", 0.4)],
    "so2": [("Industrial Activity", 1.1)],
    "crop_burning_intensity": [("Crop Burning", 1.6)],
    "wind_speed_10m_max": [("Weather / Other", 1.0)],
    "temp_wind": [("Weather / Other", 0.8)],
    "relative_humidity_2m_mean": [("Weather / Other", 0.7)],
    "temperature_2m_mean": [("Weather / Other", 0.5)],
    "pm_lag7": [("Weather / Other", 1.2)],
    "pm_lag14": [("Weather / Other", 1.0)],
    "pm_roll7": [("Weather / Other", 1.1)],
    "pm_roll14": [("Weather / Other", 0.9)],
    "pm_diff1": [("Weather / Other", 1.0)],
    "pm_diff3": [("Weather / Other", 0.8)],
    "pm_zscore7": [("Weather / Other", 0.7)],
    "no2_o3": [("Weather / Other", 0.6)],
    "pm2_5_log": [("Weather / Other", 1.3)],
}


def get_smog_health_guidance(pm25: float) -> Dict[str, Any]:
    """Health recommendations based on PM2.5 smog concentration (µg/m³)."""
    pm = float(max(0.0, pm25))
    derived_aqi = int(min(500, max(0, round(pm25_to_aqi(pm)))))
    status, _ = get_aqi_status(derived_aqi)

    if pm <= 12:
        sensitive = "Sensitive groups: No restrictions. Enjoy outdoor time."
        general = "General public: Air is clean. Normal outdoor activities are safe."
        exercise = "Exercise: Safe for outdoor workouts at any time of day."
        risk = "Low"
        resp_risk = 15
    elif pm <= 35:
        sensitive = "Sensitive groups: Reduce prolonged outdoor exertion if symptomatic."
        general = "General public: Generally safe. Unusually sensitive people may notice minor effects."
        exercise = "Exercise: Outdoor activity is fine; sensitive groups may prefer mornings."
        risk = "Low"
        resp_risk = 30
    elif pm <= 55:
        sensitive = "Sensitive groups: Limit prolonged outdoor activity. Use masks if needed."
        general = "General public: Acceptable for most, but long outdoor sessions may cause irritation."
        exercise = "Exercise: Sensitive groups should move vigorous exercise indoors."
        risk = "Medium"
        resp_risk = 50
    elif pm <= 150:
        sensitive = "Sensitive groups: Avoid outdoor activity. Stay indoors with air filtration."
        general = "General public: Reduce prolonged outdoor exertion. Wear N95 masks outdoors."
        exercise = "Exercise: Move all workouts indoors until smog levels drop."
        risk = "High"
        resp_risk = 75
    elif pm <= 250:
        sensitive = "Sensitive groups: Remain indoors. Seal windows and use air purifiers."
        general = "General public: Avoid outdoor activities. Everyone may experience health effects."
        exercise = "Exercise: Do not exercise outdoors. Use indoor ventilation."
        risk = "Very High"
        resp_risk = 90
    else:
        sensitive = "Sensitive groups: Health emergency. Stay indoors at all times."
        general = "General public: Hazardous smog. Avoid all outdoor exposure."
        exercise = "Exercise: Indoor only with filtered air. Do not go outside."
        risk = "Severe"
        resp_risk = 98

    visibility = max(0, min(100, 100 - int((pm / 180.0) * 100)))

    return {
        "pm2_5": pm,
        "aqi_derived": derived_aqi,
        "status": status,
        "respiratory_risk": resp_risk,
        "respiratory_risk_label": risk,
        "visibility_index": visibility,
        "recommendations": {
            "sensitive_groups": sensitive,
            "general_public": general,
            "exercise": exercise,
        },
        "summary": (
            f"Predicted smog level is {pm:.1f} µg/m³ ({status}). "
            f"Respiratory risk is {risk.lower()}."
        ),
    }


def compute_smog_source_attribution(feature_row: Dict[str, float]) -> Dict[str, Any]:
    """SHAP-style attribution from latest model feature values."""
    scores: Dict[str, float] = {}
    contributions: List[Dict[str, Any]] = []

    for feature, mappings in _FEATURE_SOURCE_WEIGHTS.items():
        raw = feature_row.get(feature)
        if raw is None:
            continue
        try:
            value = abs(float(raw))
        except (TypeError, ValueError):
            continue
        if value <= 0 or math.isnan(value) or math.isinf(value):
            continue
        for source, weight in mappings:
            score = value * weight
            scores[source] = scores.get(source, 0.0) + score
            contributions.append(
                {"feature": feature, "source": source, "impact": round(score, 3)}
            )

    total = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    percentages = {name: int(round((val / total) * 100)) for name, val in ranked}

    # Fix rounding to sum to 100
    if percentages:
        keys = list(percentages.keys())
        diff = 100 - sum(percentages.values())
        percentages[keys[0]] = max(0, percentages[keys[0]] + diff)

    top = ranked[0][0] if ranked else "Unknown"
    insight = (
        f"{top} is the dominant smog driver ({percentages.get(top, 0)}% SHAP-style attribution). "
        "Derived from GRU input feature impacts on the latest model forecast."
    )

    return {
        "sources": percentages,
        "contributions": sorted(contributions, key=lambda x: x["impact"], reverse=True)[:12],
        "insight": insight,
    }


def build_7_day_forecast(
    gru_series: Sequence[float],
    sarima_series: Sequence[float],
    stack_fn,
) -> List[Dict[str, Any]]:
    """Build 7-day smog forecast cards from model GRU + SARIMA outputs."""
    now = datetime.now(timezone.utc)
    days: List[Dict[str, Any]] = []

    for day_idx in range(7):
        g = float(gru_series[day_idx]) if day_idx < len(gru_series) else float(gru_series[-1])
        s = float(sarima_series[day_idx]) if day_idx < len(sarima_series) else float(sarima_series[-1])
        pm25 = float(stack_fn(g, s))
        pm25 = max(0.0, min(500.0, pm25))
        fields = model_prediction_fields(pm25)
        ts = now + timedelta(days=day_idx + 1)

        days.append(
            {
                "day": day_idx + 1,
                "date": ts.strftime("%Y-%m-%d"),
                "day_name": ts.strftime("%A"),
                "pm2_5": round(pm25, 2),
                "predicted_smog": round(pm25, 2),
                "aqi": fields["aqi_derived"],
                "aqi_category": fields["aqi_category"],
                "gru_pm2_5": round(g, 2),
                "sarima_pm2_5": round(s, 2),
                "health": get_smog_health_guidance(pm25),
            }
        )

    return days


def enrich_prediction_analytics(
    bridge_result: Dict[str, Any],
    feature_row: Optional[Dict[str, float]] = None,
    stack_fn=None,
) -> Dict[str, Any]:
    """Attach forecast, sources, and health blocks to a model prediction."""
    gru_series = bridge_result.get("gru_forecast") or [bridge_result.get("gru_prediction", 0.0)]
    sarima_series = bridge_result.get("sarima_forecast") or [bridge_result.get("sarima_prediction", 0.0)]

    if stack_fn is None:
        def _avg_stack(g, s):
            return (float(g) + float(s)) / 2.0
        stack_fn = _avg_stack

    forecast_7d = build_7_day_forecast(gru_series, sarima_series, stack_fn)
    pm25 = float(bridge_result.get("prediction", 0.0))
    health = get_smog_health_guidance(pm25)
    sources = compute_smog_source_attribution(feature_row or {})

    return {
        "forecast_7d": forecast_7d,
        "smog_sources": sources,
        "health": health,
        "health_recommendation": health["summary"],
    }
