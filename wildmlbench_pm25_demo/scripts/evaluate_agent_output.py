"""Validate agent predictions and evaluate them using hidden host-side labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"row_id": "string"})
    if set(frame.columns) != required:
        raise ValueError(f"{path} must have exactly columns {sorted(required)}")
    if frame.empty or frame["row_id"].isna().any() or frame["row_id"].str.strip().eq("").any():
        raise ValueError(f"{path} must contain nonempty row IDs")
    if frame["row_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate row_id values")
    return frame


def validate_predictions(predictions_path: Path, features_path: Path) -> dict[str, object]:
    predictions = _read(predictions_path, {"row_id", "prediction"})
    features = _read(features_path, set(pd.read_csv(features_path, nrows=0).columns))
    expected = set(features["row_id"])
    actual = set(predictions["row_id"])
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="coerce")
    if not np.isfinite(predictions["prediction"]).all():
        raise ValueError("Predictions must be numeric, finite, and non-null")
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Prediction row IDs do not match features; missing={missing[:5]}, extra={extra[:5]}")
    return {
        "prediction_rows": len(predictions),
        "expected_rows": len(features),
        "missing_row_ids": len(missing),
        "extra_row_ids": len(extra),
        "duplicate_row_ids": int(predictions["row_id"].duplicated().sum()),
        "nan_predictions": int(predictions["prediction"].isna().sum()),
    }


def evaluate(predictions_path: Path, labels_path: Path) -> float:
    predictions = _read(predictions_path, {"row_id", "prediction"})
    labels = _read(labels_path, {"row_id", "target_pm25_next_hour"})
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="coerce")
    labels["target_pm25_next_hour"] = pd.to_numeric(labels["target_pm25_next_hour"], errors="coerce")
    if not np.isfinite(predictions["prediction"]).all() or not np.isfinite(labels["target_pm25_next_hour"]).all():
        raise ValueError("Predictions and labels must contain finite numeric values")
    if set(predictions["row_id"]) != set(labels["row_id"]):
        raise ValueError("Prediction and label row_id sets differ")
    merged = labels.merge(predictions, on="row_id", validate="one_to_one")
    return float(np.sqrt(np.mean((merged["prediction"] - merged["target_pm25_next_hour"]) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_predictions(args.predictions, args.features)
    if args.labels:
        result["test_rmse"] = evaluate(args.predictions, args.labels)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
