"""Feature engineering. The feature functions are added in issue #5."""
import numpy as np

RAW_COLS = ["us_aqi", "pm2_5", "pm10", "ozone", "no2", "so2", "co", "uv_index",
            "temperature", "humidity", "precip_mm", "wind_speed", "wind_dir"]


def add_wind_components(df):
    """Wind direction is an angle (359 and 1 degrees are neighbours), so it is also stored as
    sin/cos components that can be interpolated and modelled safely."""
    rad = np.deg2rad(df["wind_dir"])
    df["wind_sin"], df["wind_cos"] = np.sin(rad), np.cos(rad)
    return df