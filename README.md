# Thesis Application Projects

This repository contains experimental machine-learning benchmark work developed
while exploring the WildMLBench master's thesis topic.

## Main project

### WildMLBench-style PM2.5 Benchmark

A reproducible one-hour-ahead PM2.5 forecasting benchmark based on
Queensland Government air-quality monitoring data.

Key components:

- real-world public-sector data
- chronological train/test split
- leakage-aware temporal features
- hidden-label evaluation
- reproducible Random Forest baseline
- regression tests
- Docker-isolated AIDE evaluation harness

See [`wildmlbench_pm25_demo/`](./wildmlbench_pm25_demo/).
