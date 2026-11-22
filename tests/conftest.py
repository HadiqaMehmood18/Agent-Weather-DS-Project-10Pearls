"""Shared test setup: every test gets its own empty local store and model registry in a temp folder,
so tests never touch real data (or Hopsworks) and never depend on each other."""
import pytest

from src import config as C
from src import store


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATA_DIR", tmp_path / "store")
    monkeypatch.setattr(C, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(C, "HOPSWORKS_API_KEY", "")      # always the local backend in tests
    monkeypatch.setattr(store, "_store", None)           # forget any cached backend
    yield