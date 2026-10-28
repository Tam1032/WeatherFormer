import requests
import time
import random

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy with rime",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    80: "rain showers",
    81: "heavy rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "violent thunderstorm with hail"
}

# Neutral lead-in options
NEUTRAL_INTROS = [
    "Looking at the current conditions",
    "According to the weather data",
    "Checking the current weather",
    "Based on the latest report"
]

# ==============================
# WEATHER TOOL
# ==============================

def api_call_with_retry(url, params, max_retries=4):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response
        except Exception:
            delay = 0.2 + random.random() * 0.3
            time.sleep(delay)
    return None

def get_city_coordinates(city_name: str, country: str) -> dict | None:
    params = {"name": f"{city_name}, {country}", "count": 1, "language": "en", "format": "json"}
    response = api_call_with_retry(GEOCODING_API_URL, params)
    if not response:
        return None

    data = response.json()
    if "results" in data and data["results"]:
        m = data["results"][0]
        return {
            "latitude": m["latitude"],
            "longitude": m["longitude"],
            "name": m["name"]
        }
    return None

def get_weather(city: str, country: str) -> dict | None:
    coords = get_city_coordinates(city, country)
    if not coords:
        return None

    params = {
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "current_weather": True,
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh"
    }

    response = api_call_with_retry(WEATHER_API_URL, params)
    if not response:
        return None

    data = response.json()
    if "current_weather" not in data:
        return None

    cw = data["current_weather"]
    temp = cw.get("temperature")
    weather_code = cw.get("weathercode")
    condition = WEATHER_CODE_MAP.get(weather_code, "unknown weather")
    wind_speed = cw.get("windspeed", 0.0)

    # return {
    #     "city": city,
    #     "country": country,
    #     "temperature": temp,
    #     "condition": condition,
    #     "wind_speed": wind_speed
    # }
    return f"{temp}°C, {condition}, wind {wind_speed} km/h"
