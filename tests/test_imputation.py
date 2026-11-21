import numpy as np
import pandas as pd

from src.features import RAW_COLS
from src.ingest import impute

N = 120
FILLED = [c for c in RAW_COLS if c != "wind_dir"] + ["wind_sin", "wind_cos", "wind_dir"]


def _raw():
    idx = pd.date_range("2026-11-01", periods=N, freq="h", name="ts")
    df = pd.DataFrame(50.0, index=idx, columns=RAW_COLS)
    df["us_aqi"] = 100.0 + np.arange(N)                              # linear: interpolation is exact
    df["temperature"] = 20 + 5 * np.sin(2 * np.pi * idx.hour / 24)   # daily cycle: same hour yesterday is exact
    df["precip_mm"] = 2.0
    df["wind_dir"] = 90.0
    return df


def _nan(df, col, start, stop):
    df.iloc[start:stop, df.columns.get_loc(col)] = np.nan


def test_short_gap_is_linearly_interpolated():
    df = _raw()
    _nan(df, "us_aqi", 50, 52)
    out = impute(df)
    assert out["us_aqi"].iloc[50] == 150 and out["us_aqi"].iloc[51] == 151


def test_long_gap_uses_same_hour_of_previous_day():
    df = _raw()
    orig = df["temperature"].copy()
    _nan(df, "temperature", 60, 73)                                   # 13 h outage
    out = impute(df)
    assert np.allclose(out["temperature"].iloc[60:73], orig.iloc[60:73])


def test_precipitation_gaps():
    df = _raw()
    _nan(df, "precip_mm", 30, 32)                                     # short -> 0 mm
    _nan(df, "precip_mm", 80, 90)                                     # long -> same hour yesterday
    out = impute(df)
    assert (out["precip_mm"].iloc[30:32] == 0).all()
    assert (out["precip_mm"].iloc[80:90] == 2.0).all()


def test_wind_direction_is_interpolated_on_the_circle():
    df = _raw()
    df.iloc[40, df.columns.get_loc("wind_dir")] = 350.0
    df.iloc[43, df.columns.get_loc("wind_dir")] = 10.0
    _nan(df, "wind_dir", 41, 43)
    out = impute(df)
    w = out["wind_dir"].iloc[41:43]
    assert (np.minimum(w, 360 - w) < 10).all()                        # near north, not near 180 degrees


def test_leading_gap_falls_back_to_median():
    df = _raw()
    _nan(df, "us_aqi", 0, 5)
    out = impute(df)
    assert (out["us_aqi"].iloc[:5] == 100 + np.median(np.arange(5, N))).all()


def test_missing_timestamp_row_is_recreated_and_counted():
    df = _raw().drop(index=_raw().index[70])
    out = impute(df)
    assert len(out) == N and out["n_imputed"].iloc[70] == len(RAW_COLS)
    assert out.loc[out.index[70], FILLED].notna().all()


def test_n_imputed_counts_missing_fields():
    df = _raw()
    _nan(df, "us_aqi", 50, 51)
    _nan(df, "humidity", 50, 51)
    out = impute(df)
    assert out["n_imputed"].iloc[50] == 2 and out["n_imputed"].iloc[49] == 0


def test_no_nan_left_after_random_gaps_and_outage():
    rng = np.random.default_rng(0)
    df = _raw()
    df[rng.random(df.shape) < 0.05] = np.nan
    _nan(df, "us_aqi", 20, 33)
    out = impute(df)
    assert out[FILLED].notna().all().all()
    assert ((out["wind_dir"] >= 0) & (out["wind_dir"] < 360)).all()
    assert out.index.is_monotonic_increasing and (out.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()