# Data dictionary

## Raw observations (13 columns, hourly, UTC)
| Column | Unit | Source | Meaning |
|---|---|---|---|
| us_aqi | index | Open-Meteo (CAMS model) | US Air Quality Index |
| pm2_5, pm10 | ug/m3 | Open-Meteo | Particulate matter |
| ozone, no2, so2 | ug/m3 | Open-Meteo | Gas pollutants |
| co | ug/m3 | Open-Meteo | Carbon monoxide |
| uv_index | index | Open-Meteo | UV index |
| temperature | deg C | Open-Meteo | Air temperature at 2 m |
| humidity | % | Open-Meteo | Relative humidity at 2 m |
| precip_mm | mm/h | Open-Meteo | Hourly precipitation |
| wind_speed | km/h | Open-Meteo | Wind speed at 10 m |
| wind_dir | deg | Open-Meteo | Wind direction at 10 m (0-360) |

## Derived columns
| Column(s) | Meaning |
|---|---|
| n_imputed | Number of raw fields of that hour that had to be imputed |
| wind_sin, wind_cos | Sine and cosine of the wind direction (circular encoding) |
| rain_flag | 1 if precip_mm >= 0.1 else 0 |
| hour, dayofweek, month | Calendar features (UTC) |
| hour_sin/cos, doy_sin/cos | Cyclic encodings of hour of day and day of year |
| `<var>_lag24` | Value 24 h earlier |
| `<var>_d1`, `_d3`, `_d6` | Change over the last 1, 3 and 6 hours |
| `<var>_ma6`, `_ma24` | Rolling mean over 6 and 24 hours |
| `<var>_std24` | Rolling standard deviation over 24 hours |
| precip_sum6, precip_sum24 | Rolling precipitation sum over 6 and 24 hours |

`<var>` is one of: us_aqi, pm2_5, temperature, humidity, precip_mm, wind_speed, wind_sin, wind_cos,
uv_index, rain_flag. In total 96 columns, of which 94 are model inputs (wind_dir and n_imputed are stored but not used).

## Targets (added in the training pipeline)
`y_<metric>` = the observation 24 hours after the row's timestamp. The newest 24 rows have no target and are used for forecasting only.