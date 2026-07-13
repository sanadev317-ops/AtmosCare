import os
import requests
from datetime import datetime, timedelta
from Backend.location_service import get_detailed_location
import math

# World Air Quality Index API Key (get free key from https://aqicn.org/api/)
WAQI_API_KEY = os.getenv("WAQI_API_KEY", "")


def _coerce_float(value):
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def get_aqi_status(aqi_value):
    """Convert AQI value to status text and color"""
    if aqi_value <= 50:
        return "Good", "Success"
    elif aqi_value <= 100:
        return "Moderate", "Warning"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "Error"
    elif aqi_value <= 200:
        return "Unhealthy", "Error"
    elif aqi_value <= 300:
        return "Very Unhealthy", "Error"
    else:
        return "Hazardous", "Error"

def calculate_smog_index(pm25, pm10, o3=None, no2=None, co=None):
    """
    Calculate smog index based on pollutant concentrations.
    Formula: Smog Index = (PM2.5 * 0.4) + (PM10 * 0.3) + (O3 * 0.15) + (NO2 * 0.1) + (CO * 0.05)
    Returns a value between 0-500 representing AQI equivalent.
    """
    pm25_value = _coerce_float(pm25)
    pm10_value = _coerce_float(pm10)
    o3_value = _coerce_float(o3)
    no2_value = _coerce_float(no2)
    co_value = _coerce_float(co)

    smog_index = 0.0

    if pm25_value is not None and pm25_value >= 0:
        smog_index += pm25_to_aqi(pm25_value) * 0.4

    if pm10_value is not None and pm10_value >= 0:
        smog_index += pm10_to_aqi(pm10_value) * 0.3

    if o3_value is not None and o3_value >= 0:
        smog_index += o3_to_aqi(o3_value) * 0.15

    if no2_value is not None and no2_value >= 0:
        smog_index += no2_to_aqi(no2_value) * 0.1

    if co_value is not None and co_value >= 0:
        smog_index += co_to_aqi(co_value) * 0.05

    return int(min(500, max(0, round(smog_index))))


def estimate_secondary_pollutants(pm25) -> dict:
    """Estimate O3, NO2, and CO from PM2.5 when live readings are unavailable."""
    try:
        pm = max(0.0, float(pm25))
    except (TypeError, ValueError):
        pm = 42.0
    return {
        "o3": round(pm * 0.10, 2),
        "no2": round(pm * 0.20, 2),
        "co": round(max(0.3, pm * 0.0167), 2),
    }


def model_prediction_fields(pm25_prediction: float) -> dict:
    """Map model PM2.5 output (µg/m³) to API/dashboard fields."""
    predicted = float(pm25_prediction)
    derived_aqi = int(min(500, max(0, round(pm25_to_aqi(predicted)))))
    category, _ = get_aqi_status(derived_aqi)
    return {
        "prediction": predicted,
        "predicted_pm2_5": predicted,
        "predicted_smog": predicted,
        "prediction_type": "pm2_5",
        "prediction_unit": "µg/m³",
        "aqi_derived": derived_aqi,
        "aqi": derived_aqi,
        "aqi_category": category,
    }


def pm25_to_aqi(pm25):
    """Convert PM2.5 (µg/m³) to AQI"""
    if pm25 <= 12:
        return (pm25 / 12) * 50
    elif pm25 <= 35.4:
        return 50 + ((pm25 - 12) / 23.4) * 50
    elif pm25 <= 55.4:
        return 100 + ((pm25 - 35.4) / 20) * 50
    elif pm25 <= 150.4:
        return 150 + ((pm25 - 55.4) / 95) * 50
    elif pm25 <= 250.4:
        return 200 + ((pm25 - 150.4) / 100) * 50
    else:
        return 300 + ((pm25 - 250.4) / 250) * 200

def pm10_to_aqi(pm10):
    """Convert PM10 (µg/m³) to AQI"""
    if pm10 <= 54:
        return (pm10 / 54) * 50
    elif pm10 <= 154:
        return 50 + ((pm10 - 54) / 100) * 50
    elif pm10 <= 254:
        return 100 + ((pm10 - 154) / 100) * 50
    elif pm10 <= 354:
        return 150 + ((pm10 - 254) / 100) * 50
    elif pm10 <= 424:
        return 200 + ((pm10 - 354) / 70) * 50
    else:
        return 300 + ((pm10 - 424) / 80) * 200

def o3_to_aqi(o3_ppb):
    """Convert O3 (ppb) to AQI"""
    if o3_ppb <= 54.4:
        return (o3_ppb / 54.4) * 50
    elif o3_ppb <= 70.4:
        return 50 + ((o3_ppb - 54.4) / 16) * 50
    elif o3_ppb <= 85.4:
        return 100 + ((o3_ppb - 70.4) / 15) * 50
    elif o3_ppb <= 105.4:
        return 150 + ((o3_ppb - 85.4) / 20) * 50
    elif o3_ppb <= 200.4:
        return 200 + ((o3_ppb - 105.4) / 95) * 50
    else:
        return 300 + ((o3_ppb - 200.4) / 400) * 200

def no2_to_aqi(no2_ppb):
    """Convert NO2 (ppb) to AQI"""
    if no2_ppb <= 53:
        return (no2_ppb / 53) * 50
    elif no2_ppb <= 100:
        return 50 + ((no2_ppb - 53) / 47) * 50
    elif no2_ppb <= 360:
        return 100 + ((no2_ppb - 100) / 260) * 50
    elif no2_ppb <= 649:
        return 150 + ((no2_ppb - 360) / 289) * 50
    elif no2_ppb <= 1249:
        return 200 + ((no2_ppb - 649) / 600) * 50
    else:
        return 300 + ((no2_ppb - 1249) / 1251) * 200

def co_to_aqi(co_ppm):
    """Convert CO (ppm) to AQI"""
    if co_ppm <= 4.4:
        return (co_ppm / 4.4) * 50
    elif co_ppm <= 9.4:
        return 50 + ((co_ppm - 4.4) / 5) * 50
    elif co_ppm <= 12.4:
        return 100 + ((co_ppm - 9.4) / 3) * 50
    elif co_ppm <= 15.4:
        return 150 + ((co_ppm - 12.4) / 3) * 50
    elif co_ppm <= 30.4:
        return 200 + ((co_ppm - 15.4) / 15) * 50
    else:
        return 300 + ((co_ppm - 30.4) / 30) * 200

def get_air_quality_data(location=None):
    """
    Get real air quality data using a multi-source approach:
    1) WAQI API — primary source for pollutant data (PM2.5, PM10, O3, NO2, CO)
    2) AccuWeather — supplements weather data (temperature, humidity, wind)
    3) Simulated fallback — last resort
    """
    try:
        # Determine city name
        if location:
            city = location.split(",")[0].strip()
        else:
            loc_data = get_detailed_location()
            if not loc_data:
                return None
            city = loc_data.get("city", "")

        result = None

        # ---- TIER 1: WAQI API (primary for pollutants) ----
        if WAQI_API_KEY:
            try:
                url = f"https://api.waqi.info/feed/{city}/?token={WAQI_API_KEY}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ok" and data.get("data"):
                        aqi_data = data["data"]
                        aqi_value = aqi_data.get("aqi")
                        iaqi = aqi_data.get("iaqi", {})

                        pm25 = iaqi.get("pm25", {}).get("v")
                        pm10 = iaqi.get("pm10", {}).get("v")
                        o3 = iaqi.get("o3", {}).get("v")
                        no2 = iaqi.get("no2", {}).get("v")
                        co = iaqi.get("co", {}).get("v")
                        temperature = iaqi.get("t", {}).get("v")
                        humidity = iaqi.get("h", {}).get("v")
                        wind_speed = iaqi.get("w", {}).get("v")

                        smog_index = calculate_smog_index(pm25, pm10, o3, no2, co)
                        status, status_color = get_aqi_status(aqi_value)

                        result = {
                            "aqi": aqi_value,
                            "smog_index": smog_index,
                            "status": status,
                            "status_color": status_color,
                            "pm25": pm25,
                            "pm10": pm10,
                            "o3": o3,
                            "no2": no2,
                            "co": co,
                            "temperature": temperature,
                            "humidity": humidity,
                            "wind_speed": wind_speed,
                            "location": aqi_data.get("city", {}).get("name", city),
                            "last_updated": datetime.now().strftime("%I:%M %p"),
                            "timestamp": datetime.now().isoformat(),
                            "data_source": "WAQI",
                            "model_enhanced": False,
                        }
            except Exception as e:
                print(f"WAQI API error: {e}")

        # ---- TIER 2: AccuWeather (supplement weather data) ----
        try:
            from Backend.accuweather_service import search_location, get_current_conditions

            accu_loc = search_location(city)
            if accu_loc:
                conditions = get_current_conditions(accu_loc["location_key"])
                if conditions:
                    if result:
                        # Supplement: override weather fields with more accurate AccuWeather data
                        if conditions.get("temperature") is not None:
                            result["temperature"] = conditions["temperature"]
                        if conditions.get("humidity") is not None:
                            result["humidity"] = conditions["humidity"]
                        if conditions.get("wind_speed") is not None:
                            result["wind_speed"] = conditions["wind_speed"]
                        result["weather_text"] = conditions.get("weather_text", "")
                        result["data_source"] = "WAQI + AccuWeather"
                    else:
                        # AccuWeather-only result (no WAQI data)
                        status, status_color = get_aqi_status(0)
                        resolved_location = f"{accu_loc['city']}, {accu_loc['country']}"
                        result = {
                            "aqi": None,
                            "smog_index": 0,
                            "status": "Unknown",
                            "status_color": "Warning",
                            "pm25": None, "pm10": None, "o3": None, "no2": None, "co": None,
                            "temperature": conditions.get("temperature"),
                            "humidity": conditions.get("humidity"),
                            "wind_speed": conditions.get("wind_speed"),
                            "weather_text": conditions.get("weather_text", ""),
                            "location": resolved_location,
                            "last_updated": datetime.now().strftime("%I:%M %p"),
                            "timestamp": datetime.now().isoformat(),
                            "data_source": "AccuWeather",
                            "model_enhanced": False,
                        }
        except Exception as e:
            print(f"AccuWeather supplement error: {e}")

        if result:
            return result

        # ---- TIER 3: Simulated fallback ----
        import random
        aqi_value = random.randint(50, 200)
        pm25 = random.randint(10, 80)
        pm10 = random.randint(20, 120)
        smog_index = calculate_smog_index(pm25, pm10)
        status, status_color = get_aqi_status(aqi_value)

        return {
            "aqi": aqi_value,
            "smog_index": smog_index,
            "status": status,
            "status_color": status_color,
            "pm25": pm25,
            "pm10": pm10,
            "location": location or "Unknown Location",
            "last_updated": datetime.now().strftime("%I:%M %p"),
            "timestamp": datetime.now().isoformat(),
            "data_source": "Simulated",
            "model_enhanced": False,
        }

    except Exception as e:
        print(f"Error getting air quality data: {e}")
        return None

def get_forecast_data(current_aqi_data=None):
    """Get forecast data for pollution levels using ML model for accurate predictions"""
    try:
        from Backend.ml_model_service import get_model_service
        model_service = get_model_service()

        if model_service.is_loaded and current_aqi_data:
            pm25 = current_aqi_data.get("pm25")
            pm10 = current_aqi_data.get("pm10")
            temperature = current_aqi_data.get("temperature")
            wind_speed = current_aqi_data.get("wind_speed")

            # Use ML model for predictions
            forecasts = {}

            # Predict for tomorrow (1 day)
            tomorrow_aqi = model_service.predict(pm25, pm10, temperature, wind_speed)
            if tomorrow_aqi is None:
                raise ValueError("Model prediction returned None")
            forecasts["tomorrow"] = tomorrow_aqi

            # Predict for next week (7 days) - average
            week_predictions = [tomorrow_aqi]
            for i in range(1, 7):
                # Simulate degradation/improvement trend
                trend_factor = 1.0 - (i * 0.02)  # Slight trend
                pred = int(tomorrow_aqi * trend_factor)
                week_predictions.append(pred)
            forecasts["next_week"] = int(sum(week_predictions) / len(week_predictions))

            # Predict for next month (30 days) - average
            month_predictions = week_predictions[:]
            for i in range(7, 30):
                trend_factor = 1.0 - (i * 0.01)
                pred = int(tomorrow_aqi * trend_factor)
                month_predictions.append(pred)
            forecasts["next_month"] = int(sum(month_predictions) / len(month_predictions))

            # Convert to risk percentage
            def aqi_to_risk(aqi):
                return 0 if aqi is None else min(100, int((aqi / 500) * 100))

            return {
                "tomorrow": aqi_to_risk(forecasts["tomorrow"]),
                "next_week": aqi_to_risk(forecasts["next_week"]),
                "next_month": aqi_to_risk(forecasts["next_month"]),
                "model_used": True,
                "model_name": "GRU Neural Network",
                "raw_predictions": forecasts
            }
    except Exception as e:
        print(f"Error using ML model for forecast: {e}")

    # Fallback: Generate realistic forecast based on current AQI
    current_aqi = current_aqi_data.get("aqi", 100) if current_aqi_data else 100
    tomorrow = int(current_aqi * 0.95) if current_aqi_data else 80
    week_avg = int(current_aqi * 0.92) if current_aqi_data else 75
    month_avg = int(current_aqi * 0.88) if current_aqi_data else 70

    def aqi_to_risk(aqi):
        return min(100, int((aqi / 500) * 100))

    return {
        "tomorrow": aqi_to_risk(tomorrow),
        "next_week": aqi_to_risk(week_avg),
        "next_month": aqi_to_risk(month_avg),
        "model_used": False,
        "data_source": "Trend Analysis"
    }

def get_trends_data():
    """Get trends data for pollution"""
    # This would typically come from historical data in database
    # For now, return realistic trends
    import random
    avg_aqi = random.randint(100, 150)
    trend = "rising" if random.random() > 0.5 else "falling"
    
    return {
        "last_7_days": f"Pollution levels {trend}.",
        "last_30_days": f"Average AQI {avg_aqi}.",
        "trend": trend
    }
