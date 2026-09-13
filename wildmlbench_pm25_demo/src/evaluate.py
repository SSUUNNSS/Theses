"""Independently evaluate prediction files with RMSE."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_validated(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"row_id": "string"})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    if frame.empty or frame["row_id"].isna().any() or frame["row_id"].str.strip().eq("").any():
        raise ValueError(f"{path} must have nonempty rows and row IDs.")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate row_id values.")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PM2.5 predictions with RMSE.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    predictions = read_validated(args.predictions, {"row_id", "prediction"})
    labels = read_validated(args.labels, {"row_id", "target_pm25_next_hour"})
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="coerce")
    if not np.isfinite(predictions["prediction"]).all():
        raise ValueError("Predictions must be numeric and must not contain missing values.")
    if set(predictions.row_id) != set(labels.row_id):
        raise ValueError("Predictions have missing or extra row_id values.")
    merged = labels.merge(predictions, on="row_id", how="left", validate="one_to_one")
    if merged["prediction"].isna().any():
        raise ValueError("Predictions are missing for one or more label row_id values.")
    target = pd.to_numeric(merged["target_pm25_next_hour"], errors="coerce")
    if not np.isfinite(target).all():
        raise ValueError("Labels contain non-numeric or missing target values.")
    rmse = float(np.sqrt(np.mean((merged["prediction"] - target) ** 2)))
    if not np.isfinite(rmse):
        raise ValueError("RMSE overflowed; prediction magnitudes are too large.")
    print(f"RMSE: {rmse:.6f}")


if __name__ == "__main__":
    main()
