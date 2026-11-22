# Issues and fixes log
Kept during development; feeds the "issues and fixes" section of the final report.

| # | Date | Issue | Cause | Fix |
|---|---|---|---|---|
| 1 | 2026-11-21 | Hopsworks login failed with `UnicodeEncodeError` | Invisible zero-width space copied with the API key | Strip non-alphanumeric characters from the key; verify length (81) |
| 2 | 2026-11-21 | Forecast API returned future hours of the current day | API serves model forecasts for the rest of today | Filter `ts <= current hour` during ingestion (issue #6) |
| 3 | 2026-11-21 | AQICN nearest-station lookup returned Ahmedabad (~600 km away) for Karachi | No live AQICN station near Karachi | Distance guard (50 km) |
| 4 | 2026-11-21 | Only AQICN station in Karachi (US Consulate) reports no AQI, last update March 2025 | Station decommissioned | Reject non-numeric and stale (>3 h) readings; AQI comes from Open-Meteo/CAMS (modelled), listed as a limitation |
| 5 | 2026-11-22 | First imputation draft interpolated the first 3 h of a long outage linearly | pandas `limit=3` fills the first 3 NaNs of every run, even inside a longer gap | Compute the length of each NaN run and only interpolate runs of at most 3 h; longer gaps use the value of the same hour on the previous day |
| 6 | 2026-11-22 | `test_store` passed alone but failed in the full test run | `config` reads env variables at import time, so a test that sets them later is too late; a model saved by an earlier run polluted the default folder | `tests/conftest.py` gives every test an isolated temp store and registry |