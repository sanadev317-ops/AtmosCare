"""
AccuWeather API Service for AtmosCare
Uses AccuWeather's free trial API for location search and air quality data.
"""
import os
import requests

ACCUWEATHER_API_KEY = os.getenv("ACCUWEATHER_API_KEY", "")
BASE_URL = "http://dataservice.accuweather.com"


def search_location(city_name):
    """
    Search for a location by city name.
    Returns dict with location_key, city, country, or None on failure.
    """
    if not ACCUWEATHER_API_KEY:
        return None
    try:
        url = f"{BASE_URL}/locations/v1/cities/search"
        params = {"apikey": ACCUWEATHER_API_KEY, "q": city_name}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            results = response.json()
            if results and len(results) > 0:
                loc = results[0]
                return {
                    "location_key": loc.get("Key"),
                    "city": loc.get("LocalizedName", ""),
                    "region": loc.get("AdministrativeArea", {}).get("LocalizedName", ""),
                    "country": loc.get("Country", {}).get("LocalizedName", ""),
                    "latitude": loc.get("GeoPosition", {}).get("Latitude", 0),
                    "longitude": loc.get("GeoPosition", {}).get("Longitude", 0),
                }
        elif response.status_code == 503:
            print("AccuWeather API: Rate limit exceeded or service unavailable")
        else:
            print(f"AccuWeather location search failed: {response.status_code}")
        return None
    except Exception as e:
        print(f"AccuWeather location search error: {e}")
        return None


def get_current_conditions(location_key):
    """
    Get current weather conditions for a location key.
    Returns dict with temperature, humidity, wind_speed, uv_index, weather_text.
    """
    if not ACCUWEATHER_API_KEY:
        return None
    try:
        url = f"{BASE_URL}/currentconditions/v1/{location_key}"
        params = {"apikey": ACCUWEATHER_API_KEY, "details": "true"}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            results = response.json()
            if results and len(results) > 0:
                data = results[0]
                temp = data.get("Temperature", {}).get("Metric", {}).get("Value")
                humidity = data.get("RelativeHumidity")
                wind = data.get("Wind", {}).get("Speed", {}).get("Metric", {}).get("Value")
                uv = data.get("UVIndex")
                weather_text = data.get("WeatherText", "")

                return {
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind,
                    "uv_index": uv,
                    "weather_text": weather_text,
                }
        elif response.status_code == 503:
            print("AccuWeather API: Rate limit exceeded")
        else:
            print(f"AccuWeather conditions failed: {response.status_code}")
        return None
    except Exception as e:
        print(f"AccuWeather conditions error: {e}")
        return None


def get_air_quality(location_key):
    """
    Get air quality data for a location key (free tier: index + category only).
    Returns dict with aqi_index, category, dominant_pollutant, or None.
    """
    try:
        url = f"{BASE_URL}/indices/v1/daily/1day/{location_key}/32"
        params = {"apikey": ACCUWEATHER_API_KEY}
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            results = response.json()
            if results and len(results) > 0:
                data = results[0]
                return {
                    "aqi_index": data.get("Value"),
                    "category": data.get("Category", ""),
                    "category_value": data.get("CategoryValue"),
                    "text": data.get("Text", ""),
                }
        elif response.status_code == 503:
            print("AccuWeather API: Rate limit exceeded")
        else:
            print(f"AccuWeather AQ failed: {response.status_code}")
        return None
    except Exception as e:
        print(f"AccuWeather AQ error: {e}")
        return None


def get_full_data(city_name):
    """
    Convenience function: search city → get conditions + air quality.
    Returns combined dict or None.
    """
    location = search_location(city_name)
    if not location:
        return None

    loc_key = location["location_key"]
    conditions = get_current_conditions(loc_key) or {}
    aq_data = get_air_quality(loc_key) or {}

    return {
        "location": location,
        "conditions": conditions,
        "air_quality": aq_data,
    }
