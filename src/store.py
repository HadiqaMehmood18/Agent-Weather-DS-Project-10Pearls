"""Feature Store + Model Registry facade.

Uses Hopsworks when HOPSWORKS_API_KEY is set, otherwise a local parquet/joblib store
(handy for development and tests). Both backends expose the same four methods:
    read(name) / write(name, df, pk, event_time, wait) / get_current(target) / promote(target, model, meta)
"""
import json
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd

from . import config as C

INSERT_ATTEMPTS = 3


class LocalBackend:
    def __init__(self):
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def _p(self, name):
        return C.DATA_DIR / f"{name}.parquet"

    def read(self, name):
        p = self._p(name)
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def write(self, name, df, pk, event_time=None, wait=True):
        if df is None or len(df) == 0:
            return
        old = self.read(name)
        out = pd.concat([old, df], ignore_index=True) if len(old) else df.copy()
        out = out.drop_duplicates(subset=pk, keep="last").reset_index(drop=True)
        out.to_parquet(self._p(name), index=False)

    def get_current(self, target):
        d = C.MODEL_DIR / target
        if not (d / "model.joblib").exists():
            return None
        return joblib.load(d / "model.joblib"), json.loads((d / "meta.json").read_text())

    def promote(self, target, model, meta):
        d = C.MODEL_DIR / target
        d.mkdir(parents=True, exist_ok=True)
        old = json.loads((d / "meta.json").read_text())["version"] if (d / "meta.json").exists() else 0
        meta = {**meta, "version": old + 1}
        joblib.dump(model, d / "model.joblib")
        (d / "meta.json").write_text(json.dumps(meta, default=str))
        return meta["version"]


class HopsworksBackend:
    def __init__(self):
        import hopsworks
        kw = {"api_key_value": C.HOPSWORKS_API_KEY}
        if C.HOPSWORKS_PROJECT:
            kw["project"] = C.HOPSWORKS_PROJECT
        self.project = hopsworks.login(**kw)
        self.fs = self.project.get_feature_store()
        self.mr = self.project.get_model_registry()

    def read(self, name):
        try:
            fg = self.fs.get_feature_group(name, version=C.FG_VERSION)
            if fg is None:
                return pd.DataFrame()
            return fg.read()
        except Exception:
            return pd.DataFrame()

    def write(self, name, df, pk, event_time=None, wait=True):
        if df is None or len(df) == 0:
            return
        fg = self.fs.get_or_create_feature_group(
            name=name, version=C.FG_VERSION, primary_key=pk, event_time=event_time,
            description=f"AgentWeather: {name}", online_enabled=False, stream=C.FG_STREAM)
        for attempt in range(1, INSERT_ATTEMPTS + 1):      # transient Kafka/network errors happen
            try:
                fg.insert(df, write_options={"wait_for_job": wait})
                return
            except Exception as e:
                if attempt == INSERT_ATTEMPTS:
                    raise
                print(f"insert into {name} failed ({type(e).__name__}), "
                      f"retry {attempt}/{INSERT_ATTEMPTS - 1} in {30 * attempt} s")
                time.sleep(30 * attempt)                   # safe to repeat: inserts are upserts on the primary key

    def get_current(self, target):
        name = f"aw_{target}"
        try:
            models = self.mr.get_models(name)
        except Exception:
            return None
        if not models:
            return None
        m = max(models, key=lambda x: x.version)  # latest promoted version
        path = Path(m.download())
        return joblib.load(path / "model.joblib"), json.loads((path / "meta.json").read_text())

    def promote(self, target, model, meta):
        tmp = Path(tempfile.mkdtemp())
        joblib.dump(model, tmp / "model.joblib")
        (tmp / "meta.json").write_text(json.dumps(meta, default=str))
        metrics = {k: float(v) for k, v in meta.get("metrics", {}).items() if v == v}
        m = self.mr.python.create_model(
            name=f"aw_{target}", metrics=metrics,
            description=f"{meta.get('model')} for {target} (24h ahead)")
        m.save(str(tmp))
        return m.version


_store = None


def get_store():
    global _store
    if _store is None:
        _store = HopsworksBackend() if C.HOPSWORKS_API_KEY else LocalBackend()
    return _store


def to_naive_utc(s):
    """Hopsworks may return tz-aware timestamps; normalise everything to naive UTC."""
    return pd.to_datetime(s, utc=True).dt.tz_convert(None)