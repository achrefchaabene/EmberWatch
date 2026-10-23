"""Evaluate U-Net probability calibration on the eval split."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import brier_score_loss

from src.data.tfrecord_loader import load_preprocessed_dataset


DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "unet_m6_v2.keras"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "reliability_diagram_eval.png"


def collect_scores(model: tf.keras.Model, batches: int) -> tuple[np.ndarray, np.ndarray]:
    labels, scores = [], []
    dataset = load_preprocessed_dataset("eval", batch_size=8, shuffle=False).take(batches)

    for inputs, target, valid_mask in dataset:
        probability = model(inputs, training=False).numpy()
        valid = valid_mask.numpy().astype(bool)
        labels.append(target.numpy()[valid])
        scores.append(probability[valid])

    return np.concatenate(labels).astype(int), np.concatenate(scores)


def calibration_bins(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_scores, observed_rates, counts = [], [], []

    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (scores >= left) & ((scores < right) if right < 1 else (scores <= right))
        count = int(in_bin.sum())
        counts.append(count)
        if count:
            mean_scores.append(float(scores[in_bin].mean()))
            observed_rates.append(float(labels[in_bin].mean()))
        else:
            mean_scores.append(np.nan)
            observed_rates.append(np.nan)

    return np.array(mean_scores), np.array(observed_rates), np.array(counts), edges


def main(model_path: Path, output_path: Path, batches: int) -> None:
    model = tf.keras.models.load_model(model_path, compile=False)
    labels, scores = collect_scores(model, batches)
    means, rates, counts, edges = calibration_bins(labels, scores)

    populated = ~np.isnan(means)
    ece = np.sum(np.abs(means[populated] - rates[populated]) * counts[populated]) / counts.sum()
    brier = brier_score_loss(labels, scores)

    figure, (reliability_axis, histogram_axis) = plt.subplots(1, 2, figsize=(12, 4.8))
    reliability_axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    reliability_axis.plot(means[populated], rates[populated], "o-", color="#1565C0", label="U-Net v2")
    reliability_axis.set(title="Reliability diagram (eval)", xlabel="Mean predicted probability", ylabel="Observed fire rate", xlim=(0, 1), ylim=(0, 1))
    reliability_axis.grid(alpha=0.25)
    reliability_axis.legend()

    histogram_axis.hist(scores, bins=edges, color="#EF6C00", edgecolor="white")
    histogram_axis.set(title="Predicted probability distribution", xlabel="Fire probability", ylabel="Valid pixels")
    histogram_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    figure.suptitle(f"Brier score: {brier:.4f}   |   ECE: {ece:.4f}", y=1.02)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Eval valid pixels: {len(labels):,}")
    print(f"Brier score: {brier:.4f}")
    print(f"Expected calibration error: {ece:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batches", type=int, default=50, help="Eight samples per batch.")
    arguments = parser.parse_args()
    main(arguments.model, arguments.output, arguments.batches)
