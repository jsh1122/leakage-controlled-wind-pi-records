#!/usr/bin/env python3
"""Verify calibrated main-table prediction evidence.

This script is self-contained: it reads only the public repository files under
``predictions_main088`` and writes derived verification tables to
``verification_outputs``. It does not depend on local training directories.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "predictions_main088"
OUT = ROOT / "verification_outputs"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = [
    "Proposed model",
    "MLP-QR",
    "LSTM-QR",
    "GRU-QR",
    "CNN-QR",
    "Transformer-QR",
    "SQR",
    "LUBE-NN",
    "RQR-MLP",
    "KDE",
]
INTERVAL_ALPHA = 0.2
BOOT_SEED = 20260610
N_BOOT = 2000
BLOCK_LENGTHS = [12, 24, 48]


def per_sample_winkler(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    width = upper - lower
    return np.where(
        actual < lower,
        width + 2.0 * (lower - actual) / INTERVAL_ALPHA,
        np.where(actual > upper, width + 2.0 * (actual - upper) / INTERVAL_ALPHA, width),
    )


def metrics(actual: np.ndarray, lower: np.ndarray, median: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((actual - median) ** 2))
    target_range = float(np.max(actual) - np.min(actual))
    picp = float(np.mean((actual >= lower) & (actual <= upper)))
    return {
        "PICP": picp,
        "PINAW": float(np.mean(upper - lower) / target_range) if target_range > 0 else np.nan,
        "Winkler": float(np.mean(per_sample_winkler(actual, lower, upper))),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(actual - median))),
        "R2": float(1.0 - mse / np.var(actual)) if np.var(actual) > 0 else np.nan,
    }


def read_prediction_file(path: Path) -> dict[str, dict[str, np.ndarray | str | int]]:
    grouped: dict[str, dict[str, list[float] | str | int]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            item = grouped.setdefault(
                method,
                {
                    "public_dataset": row["public_dataset"],
                    "seed": int(row["seed"]),
                    "actual": [],
                    "lower": [],
                    "median": [],
                    "upper": [],
                },
            )
            item["actual"].append(float(row["actual"]))  # type: ignore[index]
            item["lower"].append(float(row["calibrated_q10"]))  # type: ignore[index]
            item["median"].append(float(row["calibrated_q50"]))  # type: ignore[index]
            item["upper"].append(float(row["calibrated_q90"]))  # type: ignore[index]

    out: dict[str, dict[str, np.ndarray | str | int]] = {}
    for method, item in grouped.items():
        out[method] = {
            "public_dataset": item["public_dataset"],
            "seed": item["seed"],
            "actual": np.asarray(item["actual"], dtype=float),
            "lower": np.asarray(item["lower"], dtype=float),
            "median": np.asarray(item["median"], dtype=float),
            "upper": np.asarray(item["upper"], dtype=float),
        }
    return out


def circular_block_bootstrap(diff: np.ndarray, block_len: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    block_len = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / block_len))
    offsets = np.arange(block_len)
    means = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]) % n
        means[b] = diff[idx.ravel()[:n]].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p_left = float(np.mean(means <= 0.0))
    p_right = float(np.mean(means >= 0.0))
    return float(diff.mean()), float(lo), float(hi), min(1.0, 2.0 * min(p_left, p_right))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    metric_rows: list[dict[str, object]] = []
    winkler_series: dict[tuple[str, int, str], np.ndarray] = {}

    for path in sorted(PRED_DIR.glob("*_prediction_evidence_main088.csv.gz")):
        grouped = read_prediction_file(path)
        for method, item in grouped.items():
            actual = item["actual"]  # type: ignore[assignment]
            lower = item["lower"]  # type: ignore[assignment]
            median = item["median"]  # type: ignore[assignment]
            upper = item["upper"]  # type: ignore[assignment]
            vals = metrics(actual, lower, median, upper)
            public_dataset = str(item["public_dataset"])
            seed = int(item["seed"])
            metric_rows.append(
                {
                    "calibration_target": 0.88,
                    "public_dataset": public_dataset,
                    "seed": seed,
                    "method": method,
                    "n_test": len(actual),
                    **vals,
                }
            )
            winkler_series[(public_dataset, seed, method)] = per_sample_winkler(actual, lower, upper)

    write_csv(
        OUT / "verified_main088_metrics_long.csv",
        metric_rows,
        [
            "calibration_target",
            "public_dataset",
            "seed",
            "method",
            "n_test",
            "PICP",
            "PINAW",
            "Winkler",
            "RMSE",
            "MAE",
            "R2",
        ],
    )

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        by_method[str(row["method"])].append(row)
    agg_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        vals = by_method.get(method, [])
        if not vals:
            continue
        out = {"calibration_target": 0.88, "method": method, "n": len(vals)}
        for key in ["PICP", "PINAW", "Winkler", "RMSE", "MAE", "R2"]:
            arr = np.asarray([float(v[key]) for v in vals], dtype=float)
            out[f"{key}_mean"] = float(np.nanmean(arr))
            out[f"{key}_std"] = float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0
        agg_rows.append(out)
    write_csv(
        OUT / "verified_main088_metrics_aggregate.csv",
        agg_rows,
        [
            "calibration_target",
            "method",
            "n",
            "PICP_mean",
            "PICP_std",
            "PINAW_mean",
            "PINAW_std",
            "Winkler_mean",
            "Winkler_std",
            "RMSE_mean",
            "RMSE_std",
            "MAE_mean",
            "MAE_std",
            "R2_mean",
            "R2_std",
        ],
    )

    rng = np.random.default_rng(BOOT_SEED)
    boot_rows: list[dict[str, object]] = []
    for (public_dataset, seed, method), baseline_w in sorted(winkler_series.items()):
        if method == "Proposed model":
            continue
        ours = winkler_series.get((public_dataset, seed, "Proposed model"))
        if ours is None:
            continue
        for block_len in BLOCK_LENGTHS:
            mean_diff, lo, hi, p_two = circular_block_bootstrap(baseline_w - ours, block_len, rng)
            boot_rows.append(
                {
                    "calibration_target": 0.88,
                    "public_dataset": public_dataset,
                    "seed": seed,
                    "baseline": method,
                    "block_length": block_len,
                    "n_test": len(ours),
                    "mean_winkler_baseline_minus_ours": mean_diff,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "p_two_sided_block_bootstrap": p_two,
                    "ours_significantly_lower_95": bool(lo > 0.0),
                }
            )
    write_csv(
        OUT / "verified_moving_block_bootstrap_main088.csv",
        boot_rows,
        [
            "calibration_target",
            "public_dataset",
            "seed",
            "baseline",
            "block_length",
            "n_test",
            "mean_winkler_baseline_minus_ours",
            "ci95_low",
            "ci95_high",
            "p_two_sided_block_bootstrap",
            "ours_significantly_lower_95",
        ],
    )


if __name__ == "__main__":
    main()
