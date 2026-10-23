"""Create a presentation-ready chart from the locked test results."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "test_metrics_comparison.png"


def main() -> None:
    models = ("Persistence\nbaseline", "U-Net v2")
    average_precision = (0.0864, 0.2738)
    f1 = (0.2677, 0.3649)

    x = np.arange(len(models))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 5))

    ap_bars = axis.bar(x - width / 2, average_precision, width, label="Average Precision", color="#1565C0")
    f1_bars = axis.bar(x + width / 2, f1, width, label="F1 score", color="#EF6C00")

    axis.set_title("Next-Day Wildfire Spread: Test Performance")
    axis.set_ylabel("Score")
    axis.set_xticks(x, models)
    axis.set_ylim(0, 0.45)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    for bars in (ap_bars, f1_bars):
        axis.bar_label(bars, fmt="%.3f", padding=3)

    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
