"""Feature pipeline: data sources, fusion and imputation.

Sources (fusion by PRIORITY: gaps in the primary are filled from the next source):
  1. Open-Meteo weather + air-quality  (primary, keyless, full history -> backfill)
  2. OpenWeather current weather        (optional, needs OPENWEATHER_KEY)
  3. AQICN current AQI                  (optional, needs AQICN_TOKEN)

Imputation strategy (applied per column, in this order):
  a) reindex to a complete hourly grid (missing timestamps become NaN rows)
  b) gaps of up to 3 h: linear time interpolation (precipitation: 0 mm)
  c) longer gaps: value of the same hour 1 day earlier, then 2 days earlier
  d) still missing: forward fill for up to 6 h
  e) still missing: rolling 7-day median, then the overall median
  Wind direction is imputed through its sin/cos components, so 350 and 10 degrees are neighbours.
  `n_imputed` counts how many raw fields of a row were imputed.
Usage:
  python -m src.ingest                                 hourly run: last 6 days, upsert the last 48 h
  python -m src.ingest --start 2026-05-01              backfill from a date until today
  python -m src.ingest --start ... --end ... --dry-run fetch and compute only, write nothing
"""
import argparse
import math
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

from . import config as C
from .features import RAW_COLS, add_wind_components, build_features
from .store import get_store

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


MAX_INTERP_GAP_H = 3     # gaps up to this length are interpolated
FFILL_LIMIT_H = 6        # forward-fill limit when no earlier day is available


def _gap_lengths(s):
    """For every NaN, the length of the consecutive NaN run it belongs to."""
    na = s.isna()
    run = na.ne(na.shift(fill_value=False)).cumsum()
    return na, na.groupby(run).transform("sum")


def impute(df):
    """df: output of fuse() (index ts). Returns a gap-free hourly frame plus wind_sin, wind_cos
    and n_imputed. Strategy is described in the module docstring."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full)
    df.index.name = "ts"
    df["n_imputed"] = df[RAW_COLS].isna().sum(axis=1)
    df = add_wind_components(df)
    cols = [c for c in RAW_COLS if c != "wind_dir"] + ["wind_sin", "wind_cos"]
    for c in cols:
        s = df[c]
        na, run_len = _gap_lengths(s)
        short = na & (run_len <= MAX_INTERP_GAP_H)
        if c == "precip_mm":
            s = s.mask(short, 0.0)                                   # short dry gap: assume no rain
        else:
            s = s.mask(short, s.interpolate(method="time", limit_area="inside"))
        s = s.fillna(s.shift(24)).fillna(s.shift(48))                # same hour 1 / 2 days earlier
        s = s.ffill(limit=FFILL_LIMIT_H)
        s = s.fillna(s.rolling(168, min_periods=1).median()).fillna(s.median())
        df[c] = s
    df["wind_dir"] = (np.rad2deg(np.arctan2(df["wind_sin"], df["wind_cos"])) + 360) % 360
    return df

    def _chunks(start, end, days=31):
    """Split [start, end] (dates) into consecutive ranges of at most `days` days (gentle on the API)."""
    s = start
    while s <= end:
        e = min(s + timedelta(days=days - 1), end)
        yield s, e
        s = e + timedelta(days=1)


def _store_ready(df):
    """Column types the feature store accepts: nanosecond timestamps, int64, plain object strings."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype("datetime64[ns]")
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype("int64")
        elif pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object:
            out[c] = out[c].astype(object)
    return out


def ingest(start=None, end=None, dry_run=False):
    """One entry point for both modes.
    hourly (no dates): fetch the last 6 days, upsert the last 48 h of features (late API revisions)
    backfill (start/end dates): fetch the whole range in 31-day chunks, upsert all features
    dry_run: do everything except writing to the feature store."""
    now = pd.Timestamp.now("UTC").tz_convert(None).floor("h")
    live = start is None
    if live:
        start, end = (now - pd.Timedelta(days=6)).date(), now.date()
    om = pd.concat([fetch_open_meteo(s, e) for s, e in _chunks(start, end)], ignore_index=True)
    om = om.drop_duplicates("ts")
    om = om[om["ts"] <= now]                       # the API also returns model forecasts for later hours today
    frames = [om]
    if live:                                       # secondary sources only provide the current hour
        for rec in (fetch_openweather(now), fetch_aqicn(now)):
            if rec:
                frames.append(pd.DataFrame([rec]))
    raw = pd.concat(frames, ignore_index=True).reindex(columns=["ts", "source"] + RAW_COLS)
    raw[RAW_COLS] = raw[RAW_COLS].apply(pd.to_numeric, errors="coerce")

    feats = build_features(impute(fuse(raw)))
    if live:
        feats = feats.tail(48)
    feats = feats.reset_index()
    print(f"{'DRY RUN: ' if dry_run else ''}{len(feats)} feature rows, {feats['ts'].min()} to {feats['ts'].max()}, "
          f"{int((feats['n_imputed'] > 0).sum())} rows with imputed values")
    if dry_run:
        return feats
    store = get_store()
    store.write("raw_observations", _store_ready(raw), pk=["ts", "source"], event_time="ts")
    store.write("weather_features", _store_ready(feats), pk=["ts"], event_time="ts")
    return feats


def main():
    ap = argparse.ArgumentParser(description="Hourly ingestion (no arguments) or historical backfill.")
    ap.add_argument("--start", help="backfill start date, YYYY-MM-DD")
    ap.add_argument("--end", help="backfill end date, YYYY-MM-DD (default: today)")
    ap.add_argument("--dry-run", action="store_true", help="fetch and compute, but do not write to the store")
    a = ap.parse_args()
    if a.end and not a.start:
        ap.error("--end needs --start")
    if a.start:
        s = pd.Timestamp(a.start).date()
        e = pd.Timestamp(a.end).date() if a.end else pd.Timestamp.now("UTC").date()
        ingest(s, e, a.dry_run)
    else:
        ingest(dry_run=a.dry_run)


if __name__ == "__main__":
    main()