import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "Walmart.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
LOGS_DIR = ROOT / "logs"
ARTIFACTS_DIR = ROOT / "artifacts"
TARGET = "Weekly_Sales"
REQUIRED_COLUMNS = {
    "Store",
    "Date",
    TARGET,
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
}
FEATURE_COLUMNS = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "Week",
    "Day",
]


def ensure_directories():
    for directory in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, ARTIFACTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name, filename):
    ensure_directories()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(LOGS_DIR / filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def load_raw_data(path=RAW_DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_raw_data(data):
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Dataset contains no records.")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Dataset contains missing values in required columns.")
    if data.duplicated().any():
        raise ValueError("Dataset contains duplicate rows.")
    dates = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    if dates.isna().any():
        raise ValueError("Date contains values that cannot be parsed.")
    for column in REQUIRED_COLUMNS.difference({"Date"}):
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column {column} must contain numeric values.")


def engineer_features(features):
    engineered = features.copy()
    dates = pd.to_datetime(engineered.pop("Date"), dayfirst=True, errors="coerce")
    if dates.isna().any():
        raise ValueError("Date contains values that cannot be parsed.")
    engineered["Year"] = dates.dt.year
    engineered["Month"] = dates.dt.month
    engineered["Week"] = dates.dt.isocalendar().week.astype(int)
    engineered["Day"] = dates.dt.day
    return engineered[FEATURE_COLUMNS]


def split_raw_data(data):
    validate_raw_data(data)
    features = engineer_features(data.drop(columns=[TARGET]))
    target = data[TARGET]
    return train_test_split(features, target, test_size=0.2, random_state=42)


def create_model():
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)


def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(actual, predicted)),
    }


def save_json(path, content):
    ensure_directories()
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)
        file.write("\n")


def load_processed_split(filename):
    data = pd.read_csv(PROCESSED_DIR / filename)
    return data[FEATURE_COLUMNS], data[TARGET]
