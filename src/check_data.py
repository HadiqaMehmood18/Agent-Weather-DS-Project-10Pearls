"""Sanity check of the feature store.   python -m src.check_data"""
import pandas as pd

from .store import get_store, to_naive_utc


def main():
    store = get_store()
    raw = store.read("raw_observations")
    print(f"raw_observations : {len(raw)} rows", "" if raw.empty else f"| sources: {sorted(raw['source'].unique())}")
    f = store.read("weather_features")
    if f.empty:
        print("weather_features : empty")
        return
    ts = to_naive_utc(f["ts"]).sort_values()
    full = pd.date_range(ts.min(), ts.max(), freq="h")
    print(f"weather_features : {len(f)} rows x {f.shape[1]} columns")
    print(f"  range          : {ts.min()} to {ts.max()} ({(ts.max() - ts.min()).days} days)")
    print(f"  duplicate ts   : {int(ts.duplicated().sum())}")
    print(f"  missing hours  : {len(full.difference(ts))}")
    print(f"  NaN cells      : {int(f.isna().sum().sum())}")
    print(f"  imputed rows   : {int((f['n_imputed'] > 0).sum())}")


if __name__ == "__main__":
    main()