"""Prepare raw Queensland PM2.5 observations for the benchmark."""

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "pm2-5-qld-2024.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_COLUMNS = [
    "station", "pm25_current", "pm25_lag_1h", "pm25_lag_24h",
    "pm25_rollmean_24h", "hour", "day_of_week", "month",
]


def parse_timestamp(frame: pd.DataFrame) -> pd.Series:
    date_text = frame["Date"].astype("string").str.strip()
    time_text = frame["Time"].astype("string").str.strip()
    midnight = time_text.str.match(r"^24:00(?::00)?$")
    normalized_time = time_text.where(~midnight, "00:00")
    timestamps = pd.to_datetime(
        date_text + " " + normalized_time,
        errors="coerce",
        dayfirst=True,
        format="mixed",
    )
    return timestamps + pd.to_timedelta(midnight.fillna(False).astype(int), unit="D")


def normalize_station(value: object) -> str:
    name = " ".join(str(value).replace("\n", " ").split())
    for suffix in (" (ug/m3)", " (µg/m3)", " (ug/m^3)", " [ug/m3]"):
        if name.lower().endswith(suffix.lower()):
            name = name[: -len(suffix)]
    return re.sub(r"\s*[\[(][uµμ]g/m(?:\^?3|³)[\])]$", "", name).strip()


def build_dataset(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}. Run fetch_data.py first.")
    raw = pd.read_csv(raw_path, encoding="utf-8-sig")
    raw.columns = raw.columns.str.strip()
    required = {"Date", "Time"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {sorted(missing)}")

    station_columns = [column for column in raw.columns if column not in required]
    if not station_columns:
        raise ValueError("Raw CSV contains no station columns.")
    raw["timestamp"] = parse_timestamp(raw)
    print(f"Malformed timestamps removed: {raw['timestamp'].isna().sum():,}")
    names = [normalize_station(column) for column in station_columns]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("Station names must be nonempty and unique after normalization.")
    long = raw.melt(
        id_vars=["timestamp"],
        value_vars=station_columns,
        var_name="station",
        value_name="pm25_current",
    )
    long["station"] = long["station"].map(normalize_station)
    long["pm25_current"] = pd.to_numeric(long["pm25_current"], errors="coerce")
    long["pm25_current"] = long["pm25_current"].replace([np.inf, -np.inf], np.nan)
    long = long.dropna(subset=["timestamp"]).sort_values(["station", "timestamp"])
    if long.empty:
        raise ValueError("No valid timestamps found.")
    if long.duplicated(["station", "timestamp"]).any():
        raise ValueError("Duplicate station/timestamp observations are ambiguous.")
    if (long["timestamp"] != long["timestamp"].dt.floor("h")).any():
        raise ValueError("Expected observations on exact hourly boundaries.")
    # Restore missing hours before shifting: a row offset must equal an hour.
    hours = pd.date_range(long.timestamp.min(), long.timestamp.max(), freq="h")
    index = pd.MultiIndex.from_product([sorted(names), hours], names=["station", "timestamp"])
    long = long.set_index(["station", "timestamp"]).reindex(index).reset_index()

    grouped = long.groupby("station", sort=False, group_keys=False)
    long["pm25_lag_1h"] = grouped["pm25_current"].shift(1)
    long["pm25_lag_24h"] = grouped["pm25_current"].shift(24)
    long["pm25_rollmean_24h"] = grouped["pm25_current"].transform(
        lambda values: values.shift(1).rolling(window=24, min_periods=1).mean()
    )
    long["hour"] = long["timestamp"].dt.hour
    long["day_of_week"] = long["timestamp"].dt.dayofweek
    long["month"] = long["timestamp"].dt.month
    long["target_pm25_next_hour"] = grouped["pm25_current"].shift(-1)
    return long.dropna(subset=["target_pm25_next_hour"]).sort_values(["timestamp", "station"]).reset_index(drop=True)


def main() -> None:
    dataset = build_dataset()
    split_time = pd.Timestamp("2024-11-01")
    # Purge the final October origin whose label would be a November observation.
    train = dataset.loc[(dataset["timestamp"] >= pd.Timestamp("2024-01-01")) &
                        (dataset["timestamp"] + pd.Timedelta(hours=1) < split_time)].copy()
    test = dataset.loc[dataset["timestamp"] >= split_time].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split requires nonempty training and test sets.")
    train.insert(0, "row_id", range(len(train)))
    test.insert(0, "row_id", range(len(train), len(train) + len(test)))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train[["row_id", "timestamp", *FEATURE_COLUMNS, "target_pm25_next_hour"]].to_csv(
        PROCESSED_DIR / "train.csv", index=False
    )
    test[["row_id", "timestamp", *FEATURE_COLUMNS]].to_csv(
        PROCESSED_DIR / "test_features.csv", index=False
    )
    test[["row_id", "target_pm25_next_hour"]].to_csv(
        PROCESSED_DIR / "test_labels.csv", index=False
    )
    print(f"Stations: {dataset['station'].nunique()}")
    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print("Missing feature values:")
    print(pd.DataFrame({"train": train[FEATURE_COLUMNS].isna().sum(),
                        "test": test[FEATURE_COLUMNS].isna().sum()}).to_string())


if __name__ == "__main__":
    main()
