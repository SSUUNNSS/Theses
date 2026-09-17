# Benchmark Task

## Objective

Predict the PM2.5 concentration one hour ahead for each monitoring station in
the Queensland Government 2024 Air Quality Monitoring dataset.

Your goal is to predict `target_pm25_next_hour` for every row in
`test_features.csv` and minimize RMSE. Use only `train.csv` for training and
validation. Prefer a chronological validation split, and never use future
observations to construct features for an earlier timestamp.

## Input

An agent may use `data/processed/train.csv` for training and
`data/processed/test_features.csv` for prediction. The training file contains
the feature columns and `target_pm25_next_hour`; the test feature file contains
the same predictor columns without the target.

## Output

Write `predictions.csv` in the agent workspace with exactly these columns:

```csv
row_id,prediction
```

There must be exactly one numeric prediction for every test `row_id`.

## Metric

The benchmark uses root mean squared error (RMSE). Lower is better.

## Constraints

1. Do not access `test_labels.csv` during modeling or prediction.
2. Do not construct features using future PM2.5 observations.
3. The solution must run from code and be reproducible.
4. The pipeline must handle missing feature values.
5. The prediction file must include exactly one prediction for each test `row_id`.
6. Preserve the original `row_id` values and output finite numeric predictions.
7. Do not access files outside the agent workspace. Hidden labels, the reference
	baseline, and the evaluator are unavailable by design.

## What this benchmark tests

For the MLE agent, this task evaluates missing-value handling, temporal
validation strategy, leakage awareness, reproducible ML pipeline construction,
model selection, prediction generation, and submission discipline.

The benchmark construction pipeline itself additionally covers raw-data
validation, timestamp parsing, wide-to-long restructuring, temporal feature
engineering, chronological splitting, and hidden-label preparation.

## Forecast and access protocol

Each row is a forecast at time `t`, using observations available through `t`
to predict `t+1`. Lags are exactly 1 and 24 hours; the rolling mean uses only
`t-24h` through `t-1h`. Training origins start January 1, 2024 and their targets
must precede November 1. Test origins start November 1. The boundary origin
October 31 at 23:00 is purged. Times use Queensland local time.

Predictors are `station`, `pm25_current`, `pm25_lag_1h`, `pm25_lag_24h`,
`pm25_rollmean_24h`, `hour`, `day_of_week`, and `month`. `timestamp` records the
forecast origin; `row_id` is only an alignment key. Concentrations and RMSE
are in micrograms per cubic metre. Missing targets are excluded by the curator.

Do not recover targets from raw data or later test feature rows, including
shifting their current observations backward. Do not fit preprocessing on test
rows. The curator should withhold raw data and labels from the agent workspace.
Submit finite numeric predictions. The evaluator requires exactly the label ID
set and rejects duplicates, extra IDs, and missing predictions.
