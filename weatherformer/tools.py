import asyncio
import os
import time

import httpx

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
# Current weather changes slowly; 10 minutes. Set to 0 to disable (fair benchmarks).
CACHE_TTL_S = float(os.getenv("WEATHER_CACHE_TTL_S", "600"))

WEATHER_CODE_MAP = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with rime",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snowfall", 73: "moderate snowfall", 75: "heavy snowfall",
    80: "rain showers", 81: "heavy rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "violent thunderstorm with hail",
}

_cache: dict[tuple[str, str], tuple[float, str]] = {}


async def _get_json(client: httpx.AsyncClient, url: str, params: dict, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            response = await client.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            if attempt < retries - 1:
                await asyncio.sleep(0.3 * (attempt + 1))
    return None


async def get_weather(city: str, country: str, client: httpx.AsyncClient) -> str | None:
    """Same output format as utils.get_weather: '28.1°C, clear sky, wind 3.6 km/h'."""
    key = (city.lower(), country.lower())
    cached = _cache.get(key)
    if cached and time.monotonic() - cached[0] < CACHE_TTL_S:
        return cached[1]

    geo = await _get_json(client, GEOCODING_API_URL,
                          {"name": f"{city}, {country}", "count": 1, "language": "en", "format": "json"})
    if not geo or not geo.get("results"):
        return None
    place = geo["results"][0]

    data = await _get_json(client, WEATHER_API_URL, {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "current_weather": True,
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh",
    })
    if not data or "current_weather" not in data:
        return None

    cw = data["current_weather"]
    condition = WEATHER_CODE_MAP.get(cw.get("weathercode"), "unknown weather")
    result = f"{cw.get('temperature')}°C, {condition}, wind {cw.get('windspeed', 0.0)} km/h"
    _cache[key] = (time.monotonic(), result)
    return result
