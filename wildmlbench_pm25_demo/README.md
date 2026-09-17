# WildMLBench-style PM2.5 Demo

This repository is a small application portfolio project inspired by the idea
of WildMLBench. It is not an official WildMLBench implementation and is not
affiliated with DEEM Lab.

## Motivation

Many current MLE-agent benchmarks rely heavily on popular Kaggle-style
datasets. Those datasets can be relatively clean and simplified, have many
public solutions, and may have appeared in LLM training data. They can also
underrepresent the data-engineering work that surrounds a real ML project.

This prototype starts from public-sector air-quality monitoring data and makes
the engineering workflow part of the task. It is designed to reduce
contamination risk by using real-world public-sector data and a newly defined
benchmark task rather than a widely reused Kaggle competition. Contamination
cannot be proven impossible.

## Why this is more realistic than a normal Kaggle notebook

The repository deliberately preserves a multi-step path:

```text
raw data
  -> parsing
  -> reshaping
  -> cleaning
  -> feature engineering
  -> temporal split
  -> ML pipeline
  -> prediction
  -> independent evaluation
```

## Data source

Source: Queensland Government, [Air Quality Monitoring - 2024 (grouped by pollutant)](https://www.data.qld.gov.au/dataset/air-quality-monitoring-2024-grouped-by-pollutant),
[hourly PM2.5 resource](https://www.data.qld.gov.au/dataset/air-quality-monitoring-2024-grouped-by-pollutant/resource/1cf37e6c-2c68-4409-a8b3-6a80c97e0f52).
The official portal lists [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution: Queensland Government; this project reshapes the source and derives forecasting features.
The [direct CSV](https://files.science-data.qld.gov.au/air_quality/pollutant/pm2-5-qld-2024.csv)
contains `Date`, `Time`, and 44 station columns in micrograms per cubic metre.
Dates are day-first; timestamps retain Queensland local time (AEST, UTC+10).

## Quick Start

From the repository root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and run the complete pipeline:

```bash
pip install -r requirements.txt
python src/fetch_data.py
python src/prepare_data.py
python src/baseline.py
python src/evaluate.py --predictions data/processed/baseline_predictions.csv --labels data/processed/test_labels.csv
```

The downloader uses the official direct CSV URL with portal API discovery as a
fallback, validates cached files, rejects HTML/error responses, and explains where to place the file if
automatic downloading fails.

## Repository Structure

```text
wildmlbench_pm25_demo/
├── data/
│   ├── raw/README.md
│   └── processed/README.md
├── src/
│   ├── fetch_data.py
│   ├── prepare_data.py
│   ├── baseline.py
│   └── evaluate.py
├── tests/test_benchmark.py
├── TASK.md
├── README.md
├── requirements.txt
└── .gitignore
```

Downloaded raw CSVs and generated processed CSVs are ignored by Git.

## Benchmark Design Decisions

- **Chronological split:** January 1 through October 31, 2024 is training;
  November 1 onward is test. The October 31 23:00 forecast origin is
  excluded so its November 1 target cannot enter training.
- **Leakage-safe features:** lags and the 24-hour rolling mean use only prior
  observations. Missing hours are inserted before shifts so lags represent
  elapsed hours. Duplicate station/timestamp pairs are rejected. The current and future target are never included in the roll.
- **Missing values:** invalid measurements become missing values and remain in
  the prepared data. The baseline imputes numeric medians and categorical
  a constant categorical placeholder inside a scikit-learn pipeline.
  Imputation and encoding are fitted on training data only; rows without targets
  are removed. Non-finite measurements become missing values.
- **Separate evaluation:** predictions and labels are joined by `row_id` in a
  separate script with alignment and validity checks.
- **Reproducibility:** paths are repository-relative, the baseline uses
  `random_state=42`, 100 trees, maximum depth 16, and minimum leaf size 2.
  Every step runs from a Python script. Minimum dependency versions are declared;
  exact results can vary between library versions.

### Forecast protocol and evaluator isolation

This is rolling one-hour-ahead forecasting: observations through origin time `t`
are available to predict `t+1`. Earlier test-period observations may therefore
appear in later test features. This is not a fixed-origin two-month forecast.
The rolling mean uses `t-24h` through `t-1h`, with at least one observed value.
`row_id` is an alignment key, and `timestamp` is provenance; neither is a baseline
predictor. RMSE pools all labeled station-hours, weighting each row equally.

The curator runs fetching/preparation and retains test labels. For an actual
agent evaluation, provide only training data and test features in a separate
workspace; keep raw data and labels outside it. Later test rows expose earlier
observations, so agents must also be prohibited from shifting later rows to
recover targets. Local file separation alone does not enforce this rule.
The baseline obeys the protocol, and the evaluator rejects missing/extra IDs,
duplicates, missing IDs, nonnumeric values, and infinities.

## Limitations

- The prototype covers one pollutant and one prediction task.
- Station metadata is limited.
- Contamination cannot be proven impossible.
- The baseline is intentionally simple.
- This is a prototype, not a complete benchmark suite.

## Future Work

Potential V2 extensions include combining PM2.5 with weather data, adding
station metadata, creating true multi-source data integration, held-out
station evaluation, missing-data robustness tests, distribution-shift tests,
fairness or subgroup analyses where conceptually appropriate, evaluating AIDE,
MLE-STAR, and other MLE agents, and adding automatic validity checks.

## AIDE Agent Evaluation

This repository includes a small AIDE evaluation harness. AIDE receives only
the following files in a separate read-only mount: `train.csv`,
`test_features.csv`, and `TASK.md`. The hidden `test_labels.csv`,
`src/baseline.py`, `src/evaluate.py`, README, tests, and raw data remain on the
host and are never mounted into the container. The host performs the final
`row_id`-aligned RMSE evaluation after the container exits. The generated
`aide_task/` workspace is produced at runtime by `scripts/prepare_aide_workspace.py`
and is not a committed project asset.

```text
train.csv + test_features.csv + TASK.md
                |
              AIDE
                |
         predictions.csv
                |
     hidden benchmark evaluator
                |
               RMSE
```

The Docker image uses the inspected `aideml==0.2.2` package and its minimal
runtime dependencies. The default run uses three search steps, and the CLI
`--steps` value is passed through to the container so the actual AIDE config
matches the recorded metadata. AIDE needs an OpenAI key for the configured
`o4-mini` coding and `gpt-4.1-mini` feedback models; provide it at runtime only:

```powershell
$env:OPENAI_API_KEY = '...'
python scripts/run_aide.py
```

The script prepares the allowlisted workspace, builds the image, runs AIDE,
validates the resulting prediction file, and evaluates it on the host. If the
key is missing, it exits before starting any paid call. Results are written to
`runs/<run_id>/`, including logs, metadata, predictions, and `evaluation.json`.
The equivalent container-only command is:

```powershell
docker compose build aide
docker compose run --rm aide
```

The container uses a Docker-isolated AIDE agent workspace with host-side
hidden-label evaluation: the agent mount is read-only, test labels remain on the
host, final RMSE evaluation runs on the host, Linux capabilities are dropped,
and `no-new-privileges` is enabled. Network access is still needed for the LLM
API, so this prototype does not claim network isolation. The entrypoint rejects
symlinks and any workspace contents outside the three-file allowlist. Normal
tests mock no LLM call and run with:

```powershell
python -m unittest discover -s tests -v
```

## Validation

Run the focused regression checks with:

```bash
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

Checks cover `24:00`, malformed timestamps, missing hourly rows, station isolation,
historical rolling means, HTML rejection, shuffled prediction alignment, and
invalid prediction submissions. The real source inspected on September 9, 2026
has 8,784 hourly rows and 44 stations, from January 1 00:00 to December 31 23:00.
Its SHA-256 is
`90424398af70134dffad252c41a7b56fdecc112816b890bb860b5ec7a78378c1`.
Raw source revisions can change the derived benchmark; preserve this fingerprint
when comparing runs. The processed split has 300,459 training rows and 61,095
test rows after excluding missing targets and the split boundary.

The full baseline run on September 9, 2026 achieved **RMSE 2.370575 µg/m³**
on all 61,095 test rows. Validation environment: Python 3.14, pandas 3.0.5,
NumPy 2.5.3, and scikit-learn 1.9.0. This is an observed baseline result,
not a tuned performance claim. Training uses all available CPU cores and may
take several minutes on a laptop.
