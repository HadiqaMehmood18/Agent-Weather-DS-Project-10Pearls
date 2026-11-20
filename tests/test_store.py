import os
import tempfile

_t = tempfile.mkdtemp()
os.environ["LOCAL_STORE"] = _t + "/s"
os.environ["LOCAL_MODELS"] = _t + "/m"
os.environ.pop("HOPSWORKS_API_KEY", None)

import pandas as pd

from src.store import get_store, to_naive_utc


def test_upsert_has_no_duplicates():
    s = get_store()
    df = pd.DataFrame({"ts": pd.date_range("2026-11-01", periods=3, freq="h"), "v": [1, 2, 3]})
    s.write("t", df, ["ts"])
    s.write("t", df.assign(v=[9, 9, 9]), ["ts"])
    out = s.read("t")
    assert len(out) == 3 and (out["v"] == 9).all()


def test_empty_read_and_model_roundtrip():
    s = get_store()
    assert s.read("nothing").empty
    assert s.get_current("x") is None
    version = s.promote("x", {"a": 1}, {"model": "m"})
    model, meta = s.get_current("x")
    assert model == {"a": 1} and meta["version"] == version == 1


def test_to_naive_utc_removes_timezone():
    s = pd.Series(pd.to_datetime(["2026-11-01 10:00"]).tz_localize("UTC"))
    assert to_naive_utc(s).dt.tz is None