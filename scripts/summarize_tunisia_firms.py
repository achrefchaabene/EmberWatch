"""Create a temporal quality-control report for the cleaned Tunisia FIRMS data."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.tunisia import TEST_END, TEST_START, TRAIN_END, TRAIN_START


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tunisia" / "firms_viirs_tunisia.csv"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "tunisia_firms_temporal_qc.png"


def period_name(day: date) -> str:
    if date.fromisoformat(TRAIN_START) <= day <= date.fromisoformat(TRAIN_END):
        return "Train (2019–2023)"
    if date.fromisoformat(TEST_START) <= day <= date.fromisoformat(TEST_END):
        return "Test (2024–2025)"
    return "Outside split"


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Cleaned FIRMS file not found: {INPUT_PATH}")

    detections_by_year = Counter()
    detections_by_month = Counter()
    detections_by_period = Counter()
    active_days_by_period: dict[str, set[str]] = {"Train (2019–2023)": set(), "Test (2024–2025)": set()}

    with INPUT_PATH.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            day = date.fromisoformat(row["acq_date"])
            detections_by_year[day.year] += 1
            detections_by_month[day.month] += 1
            period = period_name(day)
            detections_by_period[period] += 1
            if period in active_days_by_period:
                active_days_by_period[period].add(day.isoformat())

    print("Detections by split:")
    for period in ("Train (2019–2023)", "Test (2024–2025)"):
        print(f"  {period}: {detections_by_period[period]:,} detections on {len(active_days_by_period[period]):,} active days")
    print("Detections by year:")
    for year in sorted(detections_by_year):
        print(f"  {year}: {detections_by_year[year]:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    years = sorted(detections_by_year)
    axes[0].bar([str(year) for year in years], [detections_by_year[year] for year in years], color="#f28e2b")
    axes[0].set_title("VIIRS detections by year")
    axes[0].set_ylabel("Detected hotspots")
    axes[0].grid(axis="y", alpha=0.25)

    months = range(1, 13)
    axes[1].bar(months, [detections_by_month[month] for month in months], color="#e15759")
    axes[1].set_title("Seasonality across 2019–2025")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Detected hotspots")
    axes[1].set_xticks(months)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Tunisia NASA FIRMS VIIRS — label quality check", fontweight="bold")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
