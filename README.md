# Verification Records for Ultra-Short-Term Wind Power Prediction Intervals

This repository provides processed prediction records and metric-verification code for the manuscript:

**Leakage-Controlled Residual Quantile Learning for Ultra-Short-Term Wind Power Prediction Intervals**

It contains no raw SCADA datasets, private data, or machine-specific result paths. Raw datasets should be obtained from the public sources listed in metadata/data_sources.csv.

Contents:

- `metadata/data_sources.csv`: public source links for the benchmark datasets.
- `metadata/chronological_split_records.csv`: chronological split counts and sample-index ranges.
- `metadata/validation_selected_configuration.csv`: DPR, degree, loss, and reported calibration operating-point records for the proposed method.
- `metadata/calibration_constants_main088.csv`: validation-only split-conformal constants for the reported gamma=0.88 operating point.
- `metadata/prediction_file_manifest.csv`: manifest for per-dataset, per-seed prediction evidence files.
- `predictions_main088/*.csv.gz`: actual values, raw interval predictions, and calibrated main-table interval predictions for every compared method.
- `tables/*.csv`: aggregate tables, calibration sensitivity, runtime components, and moving-block bootstrap evidence.
- `scripts/*.py`: self-contained verification scripts that operate only on the files in this repository.
- `verification_outputs/*.csv`: derived files produced by the verification script.

For a specified target, conformal constants are computed from validation residuals.
Test targets are included for metric verification. The retained records do not
establish whether gamma=0.88 was chosen before inspecting test outcomes. The
reported operating point and full calibration-target sweep are therefore
exploratory; verification of their metrics does not establish pre-test target
selection or independent confirmation on an untouched chronological period.
In `metadata/chronological_split_records.csv`, `target_name` uses English
target-series names for portability across spreadsheet software; for
Longyuan-SCADA, `actual_power_mw` denotes the actual generated-power target
series in MW.


## Reproducing the reported metrics

Run from the repository root with Python 3.9 or later and NumPy:

```bash
python -m pip install numpy
python scripts/verify_main088_predictions.py
```

The script reads the nine compressed prediction files and writes the three CSV files in `verification_outputs`. It recomputes interval and point metrics and paired circular moving-block bootstrap results with 2,000 resamples and block lengths 12, 24, and 48.

Model-training seeds are 42, 43, and 44. The separate value 20260610 is the bootstrap resampling seed, not a model-training seed.
