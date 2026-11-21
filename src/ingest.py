"""Feature pipeline, part 1: data sources and fusion.

Sources (fusion by PRIORITY: gaps in the primary are filled from the next source):
  1. Open-Meteo weather + air-quality  (primary, keyless, full history -> backfill)
  2. OpenWeather current weather        (optional, needs OPENWEATHER_KEY)
  3. AQICN current AQI                  (optional, needs AQICN_TOKEN)
Imputation (issue #4) and the hourly/backfill entry point (issue #6) are added later.
"""
import math
import os
import time
from datetime import timedelta

import pandas as pd
import requests

from . import config as C
from .features import RAW_COLS

OM_WEATHER = "https://api.open-meteo.com/v1/forecast"
OM_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OM_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
W_VARS = {"temperature_2m": "temperature", "relative_humidity_2m": "humidity", "precipitation": "precip_mm",
          "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir"}
AQ_VARS = {"us_aqi": "us_aqi", "pm2_5": "pm2_5", "pm10": "pm10", "ozone": "ozone", "nitrogen_dioxide": "no2",
           "sulphur_dioxide": "so2", "carbon_monoxide": "co", "uv_index": "uv_index"}
PRIORITY = ["open_meteo", "openweather", "aqicn"]


def _get(url, params, tries=3):
    """GET with retries and exponential backoff (1 s, 2 s)."""
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def _hourly(url, params, mapping):
    h = _get(url, params)["hourly"]
    df = pd.DataFrame({"ts": pd.to_datetime(h["time"])})
    for k, v in mapping.items():
        df[v] = pd.to_numeric(pd.Series(h.get(k, [None] * len(df))), errors="coerce")
    return df


def fetch_open_meteo(start, end):
    """Hourly weather + air quality for [start, end] (dates, UTC). Old dates come from the archive
    API (ERA5 reanalysis), the last ~88 days from the forecast API (model analysis)."""
    today = pd.Timestamp.now("UTC").date()
    cut = today - timedelta(days=88)
    base = {"latitude": C.LAT, "longitude": C.LON, "timezone": "UTC"}
    parts = []
    if start <= cut:
        parts.append(_hourly(OM_ARCHIVE, {**base, "hourly": ",".join(W_VARS), "start_date": start.isoformat(),
                                          "end_date": min(end, cut).isoformat()}, W_VARS))
    if end > cut:
        s = max(start, cut + timedelta(days=1))
        parts.append(_hourly(OM_WEATHER, {**base, "hourly": ",".join(W_VARS), "start_date": s.isoformat(),
                                          "end_date": end.isoformat(), "wind_speed_unit": "kmh"}, W_VARS))
    weather = pd.concat(parts).drop_duplicates("ts")
    aq = _hourly(OM_AQ, {**base, "hourly": ",".join(AQ_VARS), "start_date": start.isoformat(),
                         "end_date": end.isoformat()}, AQ_VARS)
    df = weather.merge(aq, on="ts", how="outer")
    df["source"] = "open_meteo"
    return df


def fetch_openweather(now):
    """Current weather from OpenWeather. Returns None if no key is set or the call fails."""
    key = os.getenv("OPENWEATHER_KEY")
    if not key:
        return None
    try:
        j = _get("https://api.openweathermap.org/data/2.5/weather",
                 {"lat": C.LAT, "lon": C.LON, "appid": key, "units": "metric"})
        return {"ts": now, "source": "openweather", "temperature": j["main"]["temp"],
                "humidity": j["main"]["humidity"], "wind_speed": j["wind"]["speed"] * 3.6,
                "wind_dir": j["wind"].get("deg"), "precip_mm": j.get("rain", {}).get("1h", 0.0)}
    except Exception as e:
        print("OpenWeather failed:", type(e).__name__)


MAX_STATION_KM = 50      # reject AQICN stations farther than this from the city
MAX_AGE_H = 3            # reject readings older than this


def _km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (haversine)."""
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def fetch_aqicn(now):
    """Current AQI from AQICN. Uses the station in AQICN_STATION (uid) if set, otherwise the nearest
    station. Rejected (returns None) when the station is farther than MAX_STATION_KM from the city,
    the reading is older than MAX_AGE_H hours, or the AQI is not a number (dead station)."""
    token = os.getenv("AQICN_TOKEN")
    if not token:
        return None
    uid = os.getenv("AQICN_STATION")
    path = f"@{uid}" if uid else f"geo:{C.LAT};{C.LON}"
    try:
        d = _get(f"https://api.waqi.info/feed/{path}/", {"token": token})["data"]
        st_lat, st_lon = d["city"]["geo"]
        dist = _km(C.LAT, C.LON, float(st_lat), float(st_lon))
        if dist > MAX_STATION_KM:
            print(f"AQICN station '{d['city']['name']}' is {dist:.0f} km away, rejected")
            return None
        aqi = pd.to_numeric(d.get("aqi"), errors="coerce")
        if pd.isna(aqi):
            print(f"AQICN station '{d['city']['name']}' has no valid AQI, rejected")
            return None
        obs = pd.Timestamp(d["time"]["iso"]).tz_convert("UTC").tz_localize(None)
        if now - obs > pd.Timedelta(hours=MAX_AGE_H):
            print(f"AQICN reading from {obs} is stale, rejected")
            return None
        iaqi = d.get("iaqi", {})
        g = lambda k: pd.to_numeric(iaqi.get(k, {}).get("v"), errors="coerce")
        return {"ts": now, "source": "aqicn", "us_aqi": aqi, "pm2_5": g("pm25"), "pm10": g("pm10"),
                "ozone": g("o3"), "no2": g("no2"), "so2": g("so2"), "co": g("co")}
    except Exception as e:
        print("AQICN failed:", type(e).__name__)


def fuse(raw):
    """Source-priority fusion: take the best source, fill its gaps from the next one.
    raw: long frame with columns ts, source and RAW_COLS. Returns one row per timestamp."""
    out = None
    for src in PRIORITY:
        part = raw[raw["source"] == src].drop(columns="source").drop_duplicates("ts").set_index("ts").sort_index()
        if part.empty:
            continue
        out = part if out is None else out.combine_first(part)
    return out.reindex(columns=RAW_COLS).astype(float)