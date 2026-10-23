"""Evaluate U-Net v2 by wind regime and next-day fire size on the eval split."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, f1_score

from src.data.preprocessing import CHANNELS, STATS
from src.data.tfrecord_loader import load_preprocessed_dataset


DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "unet_m6_v2.keras"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "error_analysis_eval.png"
THRESHOLD = 0.57


def metric_row(target: np.ndarray, scores: np.ndarray, mask: np.ndarray, selected: np.ndarray) -> tuple[float, float, int, int]:
    """Calculate metrics from all valid pixels within selected patches."""
    valid = mask[selected]
    labels = target[selected][valid].astype(int)
    probabilities = scores[selected][valid]
    prediction = (probabilities >= THRESHOLD).astype(int)
    return (
        float(average_precision_score(labels, probabilities)),
        float(f1_score(labels, prediction, zero_division=0)),
        int(selected.sum()),
        int(labels.sum()),
    )


def main(model_path: Path, output_path: Path, batches: int) -> None:
    model = tf.keras.models.load_model(model_path, compile=False)
    inputs_list, targets_list, masks_list = [], [], []
    dataset = load_preprocessed_dataset("eval", batch_size=8, shuffle=False).take(batches)

    for inputs, target, mask in dataset:
        inputs_list.append(inputs.numpy())
        targets_list.append(target.numpy())
        masks_list.append(mask.numpy().astype(bool))

    inputs = np.concatenate(inputs_list)
    targets = np.concatenate(targets_list)
    masks = np.concatenate(masks_list)
    scores = model.predict(inputs, batch_size=8, verbose=0)

    wind_channel = CHANNELS.index("vs")
    wind_speed = inputs[..., wind_channel].mean(axis=(1, 2)) * STATS["vs"]["std"] + STATS["vs"]["mean"]
    fire_size = (targets * masks).sum(axis=(1, 2, 3))
    positive = fire_size > 0

    wind_q1, wind_q2 = np.quantile(wind_speed[positive], [1 / 3, 2 / 3])
    size_q1, size_q2 = np.quantile(fire_size[positive], [1 / 3, 2 / 3])

    groups = {
        f"Low wind\n<= {wind_q1:.1f}": positive & (wind_speed <= wind_q1),
        f"Medium wind\n{wind_q1:.1f}-{wind_q2:.1f}": positive & (wind_speed > wind_q1) & (wind_speed <= wind_q2),
        f"High wind\n> {wind_q2:.1f}": positive & (wind_speed > wind_q2),
        f"Small fire\n<= {size_q1:.0f}px": positive & (fire_size <= size_q1),
        f"Medium fire\n{size_q1:.0f}-{size_q2:.0f}px": positive & (fire_size > size_q1) & (fire_size <= size_q2),
        f"Large fire\n> {size_q2:.0f}px": positive & (fire_size > size_q2),
    }

    results = {}
    for label, selected in groups.items():
        results[label] = metric_row(targets, scores, masks, selected)
        ap, f1, patches, pixels = results[label]
        print(f"{label.replace(chr(10), ' '):24} patches={patches:3} fire_pixels={pixels:6} AP={ap:.4f} F1={f1:.4f}")

    wind_labels = list(groups)[:3]
    size_labels = list(groups)[3:]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    for axis, labels, title in zip(axes, (wind_labels, size_labels), ("By wind regime", "By next-day fire size")):
        x = np.arange(len(labels))
        ap = [results[label][0] for label in labels]
        f1 = [results[label][1] for label in labels]
        axis.bar(x - 0.18, ap, 0.36, label="AP", color="#1565C0")
        axis.bar(x + 0.18, f1, 0.36, label="F1", color="#EF6C00")
        axis.set(title=title, ylabel="Score", xticks=x, xticklabels=labels, ylim=(0, 0.6))
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    figure.suptitle("U-Net v2 error analysis on positive-fire eval patches", y=1.02)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batches", type=int, default=50, help="Eight eval samples per batch.")
    arguments = parser.parse_args()
    main(arguments.model, arguments.output, arguments.batches)
