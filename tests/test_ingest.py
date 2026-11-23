from datetime import timedelta

import numpy as np
import pandas as pd

from src import ingest
from src.features import RAW_COLS
from src.store import get_store, to_naive_utc


def fake_get(url, params, tries=3):
    """Pretend to be Open-Meteo: hourly rows for every requested hour (including future hours of today)."""
    s, e = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23)
    ts = pd.date_range(s, e, freq="h")
    rng = np.random.default_rng(1)
    h = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in ts]}
    for k in params["hourly"].split(","):
        h[k] = list(np.abs(rng.normal(20, 5, len(ts))))
    h[params["hourly"].split(",")[0]][10:13] = [None] * 3                    # an API gap
    return {"hourly": h}


def _today():
    return pd.Timestamp.now("UTC").date()


def test_chunks_cover_the_range_without_gaps_or_overlap():
    start, end = pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-03-15").date()
    parts = list(ingest._chunks(start, end))
    assert parts[0][0] == start and parts[-1][1] == end
    assert all((e - s).days + 1 <= 31 for s, e in parts)
    assert all(parts[i + 1][0] == parts[i][1] + timedelta(days=1) for i in range(len(parts) - 1))


def test_store_ready_converts_types():
    df = pd.DataFrame({"ts": pd.to_datetime(["2026-11-01"]).astype("datetime64[us]"), "source": ["a"],
                       "n": np.array([1], dtype="int32"), "x": [1.5]})
    out = ingest._store_ready(df)
    assert str(out["ts"].dtype) == "datetime64[us]" and out["n"].dtype == "int64"
    assert out["source"].dtype == object and out["x"].dtype == "float64"


def test_backfill_writes_clean_hourly_features_and_never_the_future(monkeypatch):
    monkeypatch.setattr(ingest, "_get", fake_get)
    ingest.ingest(_today() - timedelta(days=70), _today())                     # > 31 days: several chunks
    s = get_store()
    f = s.read("weather_features")
    ts = to_naive_utc(f["ts"])
    now = pd.Timestamp.now("UTC").tz_convert(None)
    assert len(f) > 24 * 60 and ts.is_unique and ts.max() <= now               # no future hours
    assert (ts.sort_values().diff().dropna() == pd.Timedelta(hours=1)).all()     # complete hourly grid
    assert f.isna().sum().sum() == 0
    assert set(s.read("raw_observations")["source"]) == {"open_meteo"}


def test_rerunning_hourly_mode_does_not_create_duplicates(monkeypatch):
    monkeypatch.setattr(ingest, "_get", fake_get)
    ingest.ingest(_today() - timedelta(days=20), _today())
    n_before = len(get_store().read("weather_features"))
    f2 = ingest.ingest()                                                        # hourly mode
    after = get_store().read("weather_features")
    assert len(f2) == 48 and len(after) == n_before and after["ts"].is_unique


def test_dry_run_writes_nothing(monkeypatch):
    monkeypatch.setattr(ingest, "_get", fake_get)
    ingest.ingest(_today() - timedelta(days=10), _today(), dry_run=True)
    assert get_store().read("weather_features").empty and get_store().read("raw_observations").empty


def test_secondary_source_is_stored_with_numeric_columns(monkeypatch):
    monkeypatch.setattr(ingest, "_get", fake_get)
    monkeypatch.setattr(ingest, "fetch_openweather", lambda now: {
        "ts": now, "source": "openweather", "temperature": 30.0, "humidity": 40, "wind_speed": 12.0,
        "wind_dir": None, "precip_mm": 0.0})
    monkeypatch.setattr(ingest, "fetch_aqicn", lambda now: None)
    ingest.ingest()
    raw = get_store().read("raw_observations")
    assert set(raw["source"]) == {"open_meteo", "openweather"}
    assert all(pd.api.types.is_float_dtype(raw[c]) for c in RAW_COLS)