from datetime import timedelta

import numpy as np
import pandas as pd

from src import ingest
from src.features import RAW_COLS


def _frame(source, ts, **vals):
    df = pd.DataFrame({"ts": pd.to_datetime(ts), "source": source})
    for c in RAW_COLS:
        df[c] = vals.get(c, np.nan)
    return df


def test_fusion_uses_priority_and_fills_gaps_from_secondary():
    ts = ["2026-11-21 10:00", "2026-11-21 11:00"]
    primary = _frame("open_meteo", ts, temperature=[30.0, np.nan])
    secondary = _frame("openweather", ts, temperature=[99.0, 28.0])
    out = ingest.fuse(pd.concat([secondary, primary], ignore_index=True))
    assert out["temperature"].tolist() == [30.0, 28.0]      # primary wins, gap filled from secondary
    assert list(out.columns) == RAW_COLS


def test_fusion_keeps_timestamps_only_present_in_secondary():
    primary = _frame("open_meteo", ["2026-11-21 10:00"], temperature=[30.0])
    secondary = _frame("openweather", ["2026-11-21 11:00"], temperature=[28.0])
    out = ingest.fuse(pd.concat([primary, secondary], ignore_index=True))
    assert len(out) == 2 and out["temperature"].tolist() == [30.0, 28.0]


def _fake_get(url, params, tries=3):
    ts = pd.date_range(params["start_date"], periods=48, freq="h")
    h = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in ts]}
    for k in params["hourly"].split(","):
        h[k] = [1.0] * 48
    return {"hourly": h}


def test_open_meteo_parsing_returns_all_raw_columns(monkeypatch):
    monkeypatch.setattr(ingest, "_get", _fake_get)
    today = pd.Timestamp.now("UTC").date()
    df = ingest.fetch_open_meteo(today - timedelta(days=2), today)
    assert set(RAW_COLS) <= set(df.columns)
    assert (df["source"] == "open_meteo").all() and df["ts"].is_unique


def test_failing_or_unconfigured_secondary_sources_return_none(monkeypatch):
    now = pd.Timestamp("2026-11-21 10:00")
    monkeypatch.delenv("OPENWEATHER_KEY", raising=False)
    monkeypatch.delenv("AQICN_TOKEN", raising=False)
    assert ingest.fetch_openweather(now) is None and ingest.fetch_aqicn(now) is None

    def boom(*a, **k):
        raise RuntimeError("down")
    monkeypatch.setenv("OPENWEATHER_KEY", "x")
    monkeypatch.setenv("AQICN_TOKEN", "x")
    monkeypatch.setattr(ingest, "_get", boom)
    assert ingest.fetch_openweather(now) is None and ingest.fetch_aqicn(now) is None