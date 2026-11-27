import pandas as pd
import pytest

from src import store
from src.store import HopsworksBackend, get_store, to_naive_utc


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


# ---------------------------------------------------------------- retry of transient insert failures
class _FlakyFG:
    def __init__(self, fails):
        self.fails, self.calls = fails, 0

    def insert(self, df, write_options=None):
        self.calls += 1
        if self.calls <= self.fails:
            raise RuntimeError("Broker transport failure")


class _FS:
    def __init__(self, fg):
        self.fg = fg

    def get_or_create_feature_group(self, **kw):
        return self.fg


def _backend(fg):
    b = HopsworksBackend.__new__(HopsworksBackend)      # skip the real login
    b.fs = _FS(fg)
    return b


def test_insert_is_retried_after_transient_failures(monkeypatch):
    monkeypatch.setattr(store.time, "sleep", lambda s: None)
    fg = _FlakyFG(fails=2)
    _backend(fg).write("t", pd.DataFrame({"ts": [1]}), ["ts"])
    assert fg.calls == 3


def test_insert_gives_up_after_three_attempts(monkeypatch):
    monkeypatch.setattr(store.time, "sleep", lambda s: None)
    fg = _FlakyFG(fails=99)
    with pytest.raises(RuntimeError):
        _backend(fg).write("t", pd.DataFrame({"ts": [1]}), ["ts"])
    assert fg.calls == 3