"""
MCP server exposing one tool: get_weather_forecast.

Uses Open-Meteo (https://open-meteo.com/) which needs no API key.

Test standalone first:
    python src/mcp_servers/weather_server.py --test

Run as an actual MCP server (stdio transport) so an agent can connect:
    python src/mcp_servers/weather_server.py
"""

import sys
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

# Singapore's coordinates. Hardcoded since this assignment targets one destination.
SINGAPORE_LAT = 1.3521
SINGAPORE_LON = 103.8198


@mcp.tool()
def get_weather_forecast(days: int = 3) -> dict:
    """Get the weather forecast for Singapore for the next N days (max 7).

    Args:
        days: number of forecast days to return, 1-7.
    """
    days = max(1, min(days, 7))
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": SINGAPORE_LAT,
        "longitude": SINGAPORE_LON,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
        "timezone": "Asia/Singapore",
        "forecast_days": days,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()["daily"]
        forecast = []
        for i, date in enumerate(data["time"]):
            forecast.append({
                "date": date,
                "max_temp_c": data["temperature_2m_max"][i],
                "min_temp_c": data["temperature_2m_min"][i],
                "rain_probability_pct": data["precipitation_probability_max"][i],
            })
        return {"location": "Singapore", "forecast": forecast, "source": "Open-Meteo"}
    except requests.RequestException as e:
        return {"error": f"Weather service unavailable: {e}"}


if __name__ == "__main__":
    if "--test" in sys.argv:
        import json
        print(json.dumps(get_weather_forecast(3), indent=2))
    else:
        mcp.run(transport="stdio")
