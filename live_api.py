"""
live_api.py
OpenWeatherMap Air Quality API integration.
Fetches real-time pollutant readings for any Indian city.

Signed up at https://openweathermap.org/api/air-pollution
Free tier: 60 calls/min, historical data included.
"""

import os
import requests
from datetime import datetime, timedelta
import pandas as pd

# City coordinates for all 26 cities in the Kaggle dataset
#OpenWeatherMap API doesn't accept city names — it only accepts GPS coordinates(latitude ,longitude)
CITY_COORDS = {
    "Ahmedabad":   (23.0225, 72.5714),
    "Aizawl":      (23.7271, 92.7176),
    "Amaravati":   (20.9374, 77.7796),
    "Amritsar":    (31.6340, 74.8723),
    "Bengaluru":   (12.9716, 77.5946),
    "Bhopal":      (23.2599, 77.4126),
    "Brajrajnagar":(21.8218, 83.9205),
    "Chandigarh":  (30.7333, 76.7794),
    "Chennai":     (13.0827, 80.2707),
    "Coimbatore":  (11.0168, 76.9558),
    "Delhi":       (28.6139, 77.2090),
    "Ernakulam":   (9.9816,  76.2999),
    "Gurugram":    (28.4595, 77.0266),
    "Guwahati":    (26.1445, 91.7362),
    "Hyderabad":   (17.3850, 78.4867),
    "Jaipur":      (26.9124, 75.7873),
    "Jorapokhar":  (23.6693, 86.4145),
    "Kochi":       (9.9312,  76.2673),
    "Kolkata":     (22.5726, 88.3639),
    "Lucknow":     (26.8467, 80.9462),
    "Mumbai":      (19.0760, 72.8777),
    "Patna":       (25.5941, 85.1376),
    "Shillong":    (25.5788, 91.8933),
    "Talcher":     (20.9500, 85.2333),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Visakhapatnam":(17.6868, 83.2185),
}

# Map OpenWeatherMap component names to our feature names
# rename the API response fields to match what the model is trained on
OWM_TO_FEATURE = {
    "pm2_5":  "PM2.5",
    "pm10":   "PM10",
    "no":     "NO",
    "no2":    "NO2",
    "co":     "CO",
    "so2":    "SO2",
    "o3":     "O3",
    "nh3":    "NH3",
}


def get_api_key() -> str:
    """Load API key from .env file."""
    key = os.getenv("OWM_API_KEY", "") # gets value from environment variables
    # If key == "" OR None --another method
    if not key:
        try:
            from dotenv import load_dotenv # loads .env file into environment
            load_dotenv()
            key = os.getenv("OWM_API_KEY", "")
        except ImportError:
            pass
    return key


def calculate_aqi_from_pollutants(row) -> float:
    """
    India AQI calculation using CPCB formula.
    All units in µg/m³ to match OpenWeatherMap output.
    CO converted from µg/m³ to mg/m³ before applying breakpoints.
    """
    def sub_index(value, breakpoints):
        for clo, chi, ilo, ihi in breakpoints:
            if clo <= value <= chi:
                return ((ihi - ilo) / (chi - clo)) * (value - clo) + ilo
        # value exceeds highest breakpoint — cap at 500
        return 500

    # PM2.5 breakpoints (µg/m³)
    bp_pm25 = [
        (0, 30, 0, 50), (30, 60, 51, 100),
        (60, 90, 101, 200), (90, 120, 201, 300),
        (120, 250, 301, 400), (250, 500, 401, 500)
    ]

    # PM10 breakpoints (µg/m³)
    bp_pm10 = [
        (0, 50, 0, 50), (50, 100, 51, 100),
        (100, 250, 101, 200), (250, 350, 201, 300),
        (350, 430, 301, 400), (430, 600, 401, 500)
    ]

    # NO2 breakpoints (µg/m³)
    bp_no2 = [
        (0, 40, 0, 50), (40, 80, 51, 100),
        (80, 180, 101, 200), (180, 280, 201, 300),
        (280, 400, 301, 400), (400, 800, 401, 500)
    ]

    # SO2 breakpoints (µg/m³)
    bp_so2 = [
        (0, 40, 0, 50), (40, 80, 51, 100),
        (80, 380, 101, 200), (380, 800, 201, 300),
        (800, 1600, 301, 400), (1600, 2100, 401, 500)
    ]

    # O3 breakpoints (µg/m³)
    bp_o3 = [
        (0, 50, 0, 50), (50, 100, 51, 100),
        (100, 168, 101, 200), (168, 208, 201, 300),
        (208, 748, 301, 400), (748, 1000, 401, 500)
    ]

    # CO breakpoints (mg/m³) — OWM gives µg/m³ so divide by 1000
    bp_co = [
        (0, 1, 0, 50), (1, 2, 51, 100),
        (2, 10, 101, 200), (10, 17, 201, 300),
        (17, 34, 301, 400), (34, 50, 401, 500)
    ]

    sub_indices = []

    pm25 = row.get("PM2.5", 0) or 0
    pm10 = row.get("PM10",  0) or 0
    no2  = row.get("NO2",   0) or 0
    so2  = row.get("SO2",   0) or 0
    o3   = row.get("O3",    0) or 0
    co   = (row.get("CO",   0) or 0) / 1000  # µg/m³ → mg/m³

    if pm25 > 0: sub_indices.append(sub_index(pm25, bp_pm25))
    if pm10 > 0: sub_indices.append(sub_index(pm10, bp_pm10))
    if no2  > 0: sub_indices.append(sub_index(no2,  bp_no2))
    if so2  > 0: sub_indices.append(sub_index(so2,  bp_so2))
    if o3   > 0: sub_indices.append(sub_index(o3,   bp_o3))
    if co   > 0: sub_indices.append(sub_index(co,   bp_co))

    if not sub_indices:
        return 0

    return round(max(sub_indices))


# actual HTTP request happens
def fetch_current_aqi(city: str, api_key: str = None) -> dict:
    """
    Fetch current air quality data for a city.

    Returns dict with pollutant values + raw AQI index from OWM (1-5 scale).
    Returns None if city not found or API call fails.
    """
    if api_key is None:
        api_key = get_api_key()

    if city not in CITY_COORDS:
        return None

    lat, lon = CITY_COORDS[city]
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={api_key}"
    )

    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        components = data["list"][0]["components"]
        result = {
            "city":      city,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "source":    "OpenWeatherMap (live)",
        }

        for owm_key, feature_name in OWM_TO_FEATURE.items():
            result[feature_name] = components.get(owm_key, 0.0)

        # NOx and Benzene/Toluene not in OWM — estimate from NO + NO2
        result["NOx"]     = result["NO"] + result["NO2"]
        result["Benzene"] = 0.0 #  API doesn't provide them
        result["Toluene"] = 0.0

        return result

    except requests.exceptions.RequestException as e:
        print(f"[live_api] API call failed: {e}")
        return None

# calls the history endpoint
def fetch_historical_aqi(city: str, days: int = 7, api_key: str = None) -> pd.DataFrame:
    """
    Fetch past `days` days of hourly air quality data and return daily averages.
    Uses OWM Air Pollution History endpoint (free tier: up to 1 year back).
    """
    if api_key is None:
        api_key = get_api_key()

    if city not in CITY_COORDS:
        return pd.DataFrame()

    lat, lon = CITY_COORDS[city]
    end_ts   = int(datetime.utcnow().timestamp())
    start_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={api_key}"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        rows = []
        for entry in data.get("list", []):
            row = {"datetime": datetime.utcfromtimestamp(entry["dt"])}
            for owm_key, fname in OWM_TO_FEATURE.items():
                row[fname] = entry["components"].get(owm_key, 0.0)
            row["NOx"]     = row["NO"] + row["NO2"]
            row["Benzene"] = 0.0
            row["Toluene"] = 0.0
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        
        df["date"] = df["datetime"].dt.date
        daily = df.groupby("date").mean(numeric_only=True).reset_index()
        daily["City"] = city
        daily["Date"] = pd.to_datetime(daily["date"])

        daily["AQI"] = daily.apply(
            lambda row: calculate_aqi_from_pollutants(row.to_dict()), axis=1
        )
        return daily

    except requests.exceptions.RequestException as e:
        print(f"[live_api] Historical API call failed: {e}")
        return pd.DataFrame()


def get_city_list() -> list:
    """Return sorted list of all supported cities."""
    return sorted(CITY_COORDS.keys())

# new

def fetch_recent_aqi_values(city: str, days: int = 30, 
                             api_key: str = None) -> list:
    """
    Fetch the last `days` days of daily average AQI values for a city.
    Returns a plain Python list of floats, oldest first.
    Returns empty list if API call fails.
    
    Used specifically for computing lag features in build_feature_vector().
    """
    if api_key is None:
        api_key = get_api_key()

    if not api_key:
        print("[live_api] No API key — cannot fetch live lag features")
        return []

    df = fetch_historical_aqi(city, days=days, api_key=api_key)

    if df is None or len(df) == 0:
        print(f"[live_api] No historical data returned for {city}")
        return []

    # Make sure AQI column exists
    if "AQI" not in df.columns:
        df["AQI"] = df.apply(
            lambda row: calculate_aqi_from_pollutants(row.to_dict()), 
            axis=1
        )

    # Sort oldest → newest, return as plain list
    df = df.sort_values("Date")
    aqi_list = df["AQI"].tolist()

    print(f"[live_api] Fetched {len(aqi_list)} days of live AQI for {city}")
    print(f"[live_api] AQI range: {min(aqi_list):.0f} – {max(aqi_list):.0f}")

    return aqi_list


def fetch_recent_aqi_values(city: str, days: int = 30,
                              api_key: str = None) -> list:
    """
    Fetch the last `days` days of daily average AQI for a city.
    Returns a plain list of floats sorted oldest → newest.
    Returns empty list if the API call fails or city not found.

    Used by build_feature_vector() in app.py to populate lag features
    with genuinely recent AQI values instead of 2020 Kaggle data.
    """
    if api_key is None:
        api_key = get_api_key()

    if not api_key:
        print("[live_api] No API key — cannot fetch live lag features")
        return []

    if city not in CITY_COORDS:
        print(f"[live_api] City '{city}' not in CITY_COORDS")
        return []

    # Fetch historical daily averages (already handles AQI calculation)
    df = fetch_historical_aqi(city, days=days, api_key=api_key)

    if df is None or len(df) == 0:
        print(f"[live_api] No historical data returned for {city}")
        return []

    # Calculate AQI from pollutants if not already present
    if "AQI" not in df.columns:
        df["AQI"] = df.apply(
            lambda row: calculate_aqi_from_pollutants(row.to_dict()),
            axis=1
        )

    # Sort oldest first, return as plain Python list
    df = df.sort_values("Date")
    aqi_list = [float(v) for v in df["AQI"].tolist()]

    print(f"[live_api] Fetched {len(aqi_list)} days of live AQI for {city}")
    if aqi_list:
        print(f"[live_api] Range: {min(aqi_list):.0f} – {max(aqi_list):.0f} "
              f"| Most recent: {aqi_list[-1]:.0f}")

    return aqi_list