"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- City (change these three lines, or set the env vars / GitHub variables) ----
CITY = os.getenv("CITY", "Karachi")
LAT = float(os.getenv("LAT", "24.8882"))
LON = float(os.getenv("LON", "67.0452"))

HORIZON = 24  # hours ahead

# ---- Storage ----
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "")
FG_VERSION = int(os.getenv("FG_VERSION", "3"))          # bump when the schema of a feature group changes
FG_STREAM = os.getenv("FG_STREAM", "1") == "1"          # write through Kafka + materialization job (no direct HopsFS access)
DATA_DIR = Path(os.getenv("LOCAL_STORE", ROOT / "data" / "local_store"))
MODEL_DIR = Path(os.getenv("LOCAL_MODELS", ROOT / "data" / "local_models"))

# ---- LLM (Groq, OpenAI-compatible endpoint) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Verify current model names at console.groq.com/docs/models. The chain is tried in order.
LLM_MODELS = [m for m in os.getenv(
    "LLM_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",") if m]
LLM_TIMEOUT = 25
LLM_RETRIES = 2

# ---- Modelling targets (model-level). Wind direction is predicted as sin/cos. ----
TARGETS = ["us_aqi", "temperature", "precip_mm", "rain_flag", "humidity",
           "uv_index", "wind_speed", "wind_sin", "wind_cos"]

# user-facing metric -> (model target, label, unit)
METRICS = {
    "aqi": ("us_aqi", "Air Quality Index", "US AQI"),
    "temperature": ("temperature", "Temperature", "°C"),
    "precip_mm": ("precip_mm", "Precipitation", "mm/h"),
    "rain_prob": ("rain_flag", "Rain probability", "0-1"),
    "humidity": ("humidity", "Humidity", "%"),
    "uv_index": ("uv_index", "UV Index", ""),
    "wind_speed": ("wind_speed", "Wind speed", "km/h"),
    "wind_dir": (None, "Wind direction", "°"),
}

# ---- Drift ----
PSI_THRESHOLD = 0.25          # data drift: PSI above this = significant shift
MODEL_DRIFT_RATIO = 1.5       # model drift: recent RMSE / validation RMSE above this
LIVE_WINDOW_H = 168           # live window for data drift (7 days)
ERROR_WINDOW_H = 72           # window of matured forecasts for model drift
RETRAIN_COOLDOWN_H = 24
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

REPORT_HOUR_UTC = int(os.getenv("REPORT_HOUR_UTC", "3"))
