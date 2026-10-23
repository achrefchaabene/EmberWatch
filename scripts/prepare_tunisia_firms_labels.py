"""Validate and prepare NASA FIRMS VIIRS detections for the Tunisia pipeline.

The script uses only the Python standard library so the first real-data stage
can begin as soon as the user downloads a FIRMS CSV from NASA.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.tunisia import TUNISIA_BBOX, TEST_END, TRAIN_START


def in_tunisia(latitude: float, longitude: float) -> bool:
    return (
        TUNISIA_BBOX["south"] <= latitude <= TUNISIA_BBOX["north"]
        and TUNISIA_BBOX["west"] <= longitude <= TUNISIA_BBOX["east"]
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FIRMS fire labels for Tunisia.")
    parser.add_argument("--input", type=Path, required=True, help="NASA FIRMS VIIRS CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "tunisia",
        help="Directory for cleaned label outputs",
    )
    parser.add_argument("--min-confidence", default="nominal", choices=["low", "nominal", "high"])
    return parser.parse_args()


def confidence_is_accepted(value: str, minimum: str) -> bool:
    order = {"low": 0, "l": 0, "nominal": 1, "n": 1, "high": 2, "h": 2}
    return order.get(value.strip().lower(), -1) >= order[minimum]


def main() -> None:
    args = parse_arguments()
    if not args.input.exists():
        raise FileNotFoundError(f"FIRMS input not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = args.output_dir / "firms_viirs_tunisia.csv"
    geojson_path = args.output_dir / "firms_viirs_tunisia.geojson"
    daily_counts: Counter[str] = Counter()
    features: list[dict] = []
    kept_rows: list[dict[str, str]] = []
    rejected = Counter()

    with args.input.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"latitude", "longitude", "acq_date", "confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"FIRMS CSV is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            try:
                latitude = float(row["latitude"])
                longitude = float(row["longitude"])
                acquisition_date = date.fromisoformat(row["acq_date"])
            except (TypeError, ValueError):
                rejected["invalid_location_or_date"] += 1
                continue
            if not in_tunisia(latitude, longitude):
                rejected["outside_tunisia"] += 1
                continue
            if not confidence_is_accepted(row["confidence"], args.min_confidence):
                rejected["below_confidence"] += 1
                continue
            if not (date.fromisoformat(TRAIN_START) <= acquisition_date <= date.fromisoformat(TEST_END)):
                rejected["outside_project_period"] += 1
                continue

            row["latitude"] = f"{latitude:.6f}"
            row["longitude"] = f"{longitude:.6f}"
            kept_rows.append(row)
            daily_counts[row["acq_date"]] += 1
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "acq_date": row["acq_date"],
                        "acq_time": row.get("acq_time", ""),
                        "confidence": row["confidence"],
                        "frp": row.get("frp", ""),
                    },
                    "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                }
            )

    fieldnames = list(kept_rows[0]) if kept_rows else ["latitude", "longitude", "acq_date", "confidence"]
    with cleaned_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    with geojson_path.open("w", encoding="utf-8") as output:
        json.dump({"type": "FeatureCollection", "features": features}, output, indent=2)

    print(f"Accepted detections: {len(kept_rows):,}")
    print(f"Days with detections: {len(daily_counts):,}")
    print(f"Date range: {min(daily_counts, default='n/a')} to {max(daily_counts, default='n/a')}")
    print(f"Rejected: {dict(rejected)}")
    print(f"Saved: {cleaned_path}")
    print(f"Saved: {geojson_path}")


if __name__ == "__main__":
    main()
