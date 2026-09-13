"""Train and run the reproducible Random Forest benchmark baseline."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
CATEGORICAL_FEATURES = ["station"]
NUMERICAL_FEATURES = [
    "pm25_current", "pm25_lag_1h", "pm25_lag_24h", "pm25_rollmean_24h",
    "hour", "day_of_week", "month",
]


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), NUMERICAL_FEATURES),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown", keep_empty_features=True)),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    model.fit(train[features], train["target_pm25_next_hour"])
    predictions = model.predict(test[features])
    pd.DataFrame({"row_id": test["row_id"], "prediction": predictions}).to_csv(
        PROCESSED_DIR / "baseline_predictions.csv", index=False
    )
    print(f"Saved {len(predictions):,} predictions to {PROCESSED_DIR / 'baseline_predictions.csv'}")


if __name__ == "__main__":
    main()
