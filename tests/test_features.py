import numpy as np
import pandas as pd

from src.features import RAW_COLS, WARMUP, add_wind_components, build_features, feature_columns

N = 200


def _imputed(n=N, seed=0):
    """A gap-free hourly frame like the output of impute()."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-01", periods=n, freq="h", name="ts")
    df = pd.DataFrame({c: rng.uniform(1, 100, n) for c in RAW_COLS}, index=idx)
    df["precip_mm"] = np.where(rng.random(n) < 0.2, rng.gamma(2, 2, n), 0.0)
    df["wind_dir"] = rng.uniform(0, 360, n)
    df = add_wind_components(df)
    df["n_imputed"] = 0
    return df


def test_warmup_rows_dropped_and_no_nan():
    df = _imputed()
    out = build_features(df)
    assert out.index[0] == df.index[WARMUP] and len(out) == N - WARMUP
    assert out.isna().sum().sum() == 0


def test_features_never_depend_on_the_future_truncation():
    df = _imputed()
    full, cut = build_features(df), build_features(df.iloc[:150])
    pd.testing.assert_frame_equal(full.loc[cut.index], cut)


def test_features_never_depend_on_the_future_perturbation():
    df = _imputed()
    changed = df.copy()
    changed.iloc[150:, :] = changed.iloc[150:, :] * 3 + 7            # rewrite everything after row 149
    a, b = build_features(df), build_features(changed)
    past = df.index[:150]
    pd.testing.assert_frame_equal(a.loc[a.index.isin(past)], b.loc[b.index.isin(past)])


def test_lag_rate_of_change_and_rolling_values():
    df = _imputed()
    out = build_features(df)
    t = df.index[100]
    h = pd.Timedelta(hours=1)
    assert out.loc[t, "us_aqi_lag24"] == df.loc[t - 24 * h, "us_aqi"]
    assert np.isclose(out.loc[t, "us_aqi_d1"], df.loc[t, "us_aqi"] - df.loc[t - h, "us_aqi"])
    assert np.isclose(out.loc[t, "temperature_d6"], df.loc[t, "temperature"] - df.loc[t - 6 * h, "temperature"])
    assert np.isclose(out.loc[t, "humidity_ma6"], df["humidity"].iloc[95:101].mean())
    assert np.isclose(out.loc[t, "uv_index_std24"], df["uv_index"].iloc[77:101].std())
    assert np.isclose(out.loc[t, "precip_sum24"], df["precip_mm"].iloc[77:101].sum())


def test_rain_flag_threshold():
    df = _imputed()
    df.iloc[50, df.columns.get_loc("precip_mm")] = 0.09
    df.iloc[51, df.columns.get_loc("precip_mm")] = 0.10
    out = build_features(df)
    assert out.loc[df.index[50], "rain_flag"] == 0 and out.loc[df.index[51], "rain_flag"] == 1


def test_time_features():
    df = _imputed()
    out = build_features(df)
    t = out.index[out.index.hour == 6][0]
    assert np.isclose(out.loc[t, "hour_sin"], 1) and np.isclose(out.loc[t, "hour_cos"], 0, atol=1e-9)
    assert out.loc[t, "month"] == t.month and out.loc[t, "dayofweek"] == t.dayofweek


def test_feature_columns_exclude_helpers_and_targets():
    out = build_features(_imputed())
    out["y_us_aqi"] = 1.0
    cols = feature_columns(out)
    assert not {"ts", "wind_dir", "n_imputed", "y_us_aqi"} & set(cols)
    assert {"us_aqi", "hour_sin", "wind_sin_lag24", "precip_sum6"} <= set(cols)