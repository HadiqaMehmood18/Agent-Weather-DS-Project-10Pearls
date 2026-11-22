"""Feature engineering (issue #5).

Every feature is CAUSAL: the value in row t only uses rows <= t, so nothing from the future leaks
into a forecast made at time t. Groups:
  * time        : hour, day of week, month + cyclic sin/cos of hour and day of year
  * wind        : direction as sin/cos components (an angle cannot be averaged directly)
  * rain flag   : precipitation >= 0.1 mm
  * per key variable: value 24 h ago (lag24), rate of change over 1/3/6 h (d1, d3, d6),
                      rolling mean over 6 and 24 h (ma6, ma24), rolling std over 24 h (std24)
  * precipitation   : rolling sum over 6 and 24 h
The first WARMUP rows are dropped because their lags / rolling windows are incomplete.
"""
import numpy as np
import pandas as pd

RAW_COLS = ["us_aqi", "pm2_5", "pm10", "ozone", "no2", "so2", "co", "uv_index",
            "temperature", "humidity", "precip_mm", "wind_speed", "wind_dir"]
KEY = ["us_aqi", "pm2_5", "temperature", "humidity", "precip_mm", "wind_speed",
       "wind_sin", "wind_cos", "uv_index", "rain_flag"]
RAIN_MM = 0.1                       # hourly precipitation at or above this counts as rain
WARMUP = 30                         # rows dropped: the longest look-back is 24 h
EXCLUDE = {"ts", "wind_dir", "n_imputed"}   # kept in the store, but not used as model inputs


def add_wind_components(df):
    """Wind direction is an angle (359 and 1 degrees are neighbours), so it is also stored as
    sin/cos components that can be interpolated and modelled safely."""
    rad = np.deg2rad(df["wind_dir"])
    df["wind_sin"], df["wind_cos"] = np.sin(rad), np.cos(rad)
    return df


def build_features(df):
    """df: gap-free hourly frame indexed by ts (output of impute()). Returns the feature frame
    without the first WARMUP rows."""
    f = df.copy()
    idx = f.index
    f["rain_flag"] = (f["precip_mm"] >= RAIN_MM).astype(float)
    new = {
        "hour": idx.hour, "dayofweek": idx.dayofweek, "month": idx.month,
        "hour_sin": np.sin(2 * np.pi * idx.hour / 24), "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
        "doy_sin": np.sin(2 * np.pi * idx.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * idx.dayofyear / 365.25),
    }
    for c in KEY:
        s = f[c]
        new[f"{c}_lag24"] = s.shift(24)             # same hour yesterday
        for k in (1, 3, 6):
            new[f"{c}_d{k}"] = s.diff(k)            # rate of change
        new[f"{c}_ma6"] = s.rolling(6).mean()       # rolling averages
        new[f"{c}_ma24"] = s.rolling(24).mean()
        new[f"{c}_std24"] = s.rolling(24).std()
    new["precip_sum6"] = f["precip_mm"].rolling(6).sum()
    new["precip_sum24"] = f["precip_mm"].rolling(24).sum()
    out = pd.concat([f, pd.DataFrame(new, index=idx)], axis=1)
    return out.iloc[WARMUP:]


def feature_columns(df):
    """Columns the models may use as inputs."""
    return [c for c in df.columns if c not in EXCLUDE and not c.startswith("y_")]