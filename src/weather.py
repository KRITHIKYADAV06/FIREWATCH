# src/weather.py
"""
FIREWATCH - monthly weather context from the NASA POWER API (no key needed).

monthly_weather(lat, lon, start_year, end_year) returns a DataFrame with columns
    year, month, rain_mm_day, temp_c, humidity_pct, wind_ms
for one point (e.g. the centre of the area of interest), or None on any error.

This is CONTEXT only: FIREWATCH does not claim weather caused any fire.
Data: NASA POWER, https://power.larc.nasa.gov (community AG, monthly point).
"""
import requests
import pandas as pd

try:
    import streamlit as st
    _cache = st.cache_data(ttl=86400, show_spinner=False)
except ImportError:                      # tests and scripts run without Streamlit
    def _cache(func):
        return func

POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
TIMEOUT_S = 20
FILL_VALUE = -999                        # POWER's "no data" marker

# POWER parameter -> our column name
PARAMETERS = {
    "PRECTOTCORR": "rain_mm_day",        # precipitation, mm/day
    "T2M": "temp_c",                     # air temperature at 2 m, deg C
    "RH2M": "humidity_pct",              # relative humidity at 2 m, %
    "WS2M": "wind_ms",                   # wind speed at 2 m, m/s
}
OUT_COLS = ["year", "month"] + list(PARAMETERS.values())


def parse_power_json(payload):
    """Turn a POWER monthly JSON reply into our DataFrame.
    Keys look like "201901".."201912" plus "201913" (the annual value), which we drop."""
    params = payload["properties"]["parameter"]
    rows = {}
    for power_name, column in PARAMETERS.items():
        for key, value in params[power_name].items():
            year, month = int(key[:4]), int(key[4:])
            if not 1 <= month <= 12:     # month 13 = annual summary
                continue
            row = rows.setdefault((year, month), {"year": year, "month": month})
            row[column] = None if value is None or value <= FILL_VALUE else float(value)
    df = pd.DataFrame(list(rows.values()), columns=OUT_COLS)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df[["year", "month"]] = df[["year", "month"]].astype(int)
    return df


@_cache
def _fetch(lat, lon, start_year, end_year):
    """Network call. Raises on error so Streamlit does not cache a failure."""
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "latitude": round(float(lat), 4),
        "longitude": round(float(lon), 4),
        "start": int(start_year),
        "end": int(end_year),
        "format": "JSON",
    }
    response = requests.get(POWER_URL, params=params, timeout=TIMEOUT_S)
    response.raise_for_status()
    return parse_power_json(response.json())


def monthly_weather(lat, lon, start_year, end_year):
    """Monthly weather for one point, or None if anything goes wrong."""
    try:
        if int(start_year) > int(end_year):
            return None
        df = _fetch(float(lat), float(lon), int(start_year), int(end_year))
        return df if df is not None and not df.empty else None
    except Exception:                    # network, HTTP, JSON or format problem
        return None
