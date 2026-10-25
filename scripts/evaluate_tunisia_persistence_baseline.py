"""Evaluate the previous-day-fire persistence baseline on the Tunisia test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUBE = PROJECT_ROOT / "data" / "processed" / "tunisia" / "tunisia_fire_weather_cube_era5_025.nc"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "tunisia_persistence_baseline.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate previous-fire persistence on Tunisia 2024-2025 data.")
    parser.add_argument("--cube", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if not args.cube.exists():
        raise FileNotFoundError(f"Data cube not found: {args.cube}")

    dataset = xr.open_dataset(args.cube)
    test = dataset.sel(valid_time=slice("2024-01-01", "2025-12-31"))
    available = test.target_is_available.values.astype(bool)
    targets = test.next_fire_mask.values[available].reshape(-1).astype(np.int8)
    scores = test.prev_fire_mask.values[available].reshape(-1).astype(np.float32)
    predictions = (scores >= 0.5).astype(np.int8)

    precision, recall, thresholds = precision_recall_curve(targets, scores)
    f1_values = 2 * precision * recall / (precision + recall + 1e-12)
    best_index = int(np.nanargmax(f1_values))
    best_threshold = float(thresholds[min(best_index, len(thresholds) - 1)]) if len(thresholds) else 0.5

    report = {
        "model": "tunisia_persistence_baseline",
        "split": "test 2024-01-01 to 2025-12-30",
        "grid": {"latitude_cells": int(test.sizes["latitude"]), "longitude_cells": int(test.sizes["longitude"])},
        "valid_pixel_days": int(targets.size),
        "positive_pixel_days": int(targets.sum()),
        "fire_rate": float(targets.mean()),
        "average_precision": float(average_precision_score(targets, scores)),
        "fixed_threshold": 0.5,
        "f1_at_fixed_threshold": float(f1_score(targets, predictions, zero_division=0)),
        "precision_at_fixed_threshold": float(precision_score(targets, predictions, zero_division=0)),
        "recall_at_fixed_threshold": float(recall_score(targets, predictions, zero_division=0)),
        "best_f1": float(f1_values[best_index]),
        "best_threshold": best_threshold,
        "definition": "Predict next-day fire mask using the current-day FIRMS fire mask.",
        "caveat": "Evaluation uses the full buffered ERA5 rectangle; a Tunisia land mask will be added in a later refinement.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Tunisia persistence baseline")
    print(f"Test pixel-days: {report['valid_pixel_days']:,}")
    print(f"Fire rate: {report['fire_rate']:.4%}")
    print(f"AP: {report['average_precision']:.4f}")
    print(f"F1 at 0.50: {report['f1_at_fixed_threshold']:.4f}")
    print(f"Precision at 0.50: {report['precision_at_fixed_threshold']:.4f}")
    print(f"Recall at 0.50: {report['recall_at_fixed_threshold']:.4f}")
    print(f"Best F1: {report['best_f1']:.4f} at threshold {report['best_threshold']:.2f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
