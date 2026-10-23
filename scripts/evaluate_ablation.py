"""Measure grouped permutation importance for the M6 U-Net on the eval split."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

from src.data.preprocessing import CHANNELS
from src.data.tfrecord_loader import load_preprocessed_dataset


DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "unet_m6_v2.keras"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "channel_ablation_eval.png"

FEATURE_GROUPS = {
    "Previous fire": ("PrevFireMask",),
    "Elevation": ("elevation",),
    "Wind direction": ("th_sin", "th_cos"),
    "Wind speed": ("vs",),
    "Temperature": ("tmmn", "tmmx"),
    "Humidity": ("sph",),
    "Precipitation": ("pr",),
    "Drought index": ("pdsi",),
    "Fire danger": ("erc",),
    "NDVI": ("NDVI",),
    "Population": ("population",),
}


def collect_dataset(batches: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a bounded eval set into memory for identical ablation comparisons."""
    inputs, targets, masks = [], [], []
    dataset = load_preprocessed_dataset("eval", batch_size=8, shuffle=False).take(batches)
    for x_batch, y_batch, mask_batch in dataset:
        inputs.append(x_batch.numpy())
        targets.append(y_batch.numpy())
        masks.append(mask_batch.numpy().astype(bool))
    return np.concatenate(inputs), np.concatenate(targets), np.concatenate(masks)


def score_model(model: tf.keras.Model, inputs: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> float:
    scores = model.predict(inputs, batch_size=8, verbose=0)
    return float(average_precision_score(targets[masks].astype(int), scores[masks]))


def permute_group(inputs: np.ndarray, channel_names: tuple[str, ...], rng: np.random.Generator) -> np.ndarray:
    """Swap whole spatial maps across patches, preserving within-map geometry."""
    result = inputs.copy()
    permutation = rng.permutation(len(inputs))
    for name in channel_names:
        channel = CHANNELS.index(name)
        result[..., channel] = inputs[permutation, ..., channel]
    return result


def main(model_path: Path, output_path: Path, batches: int, seed: int) -> None:
    model = tf.keras.models.load_model(model_path, compile=False)
    inputs, targets, masks = collect_dataset(batches)
    baseline_ap = score_model(model, inputs, targets, masks)
    rng = np.random.default_rng(seed)

    results = []
    for label, channels in FEATURE_GROUPS.items():
        ablated_ap = score_model(model, permute_group(inputs, channels, rng), targets, masks)
        results.append((label, ablated_ap, baseline_ap - ablated_ap))
        print(f"{label:16} AP={ablated_ap:.4f}  AP drop={baseline_ap - ablated_ap:.4f}")

    results.sort(key=lambda item: item[2], reverse=True)
    labels = [item[0] for item in results]
    drops = [item[2] for item in results]

    figure, axis = plt.subplots(figsize=(9, 6))
    bars = axis.barh(labels[::-1], drops[::-1], color="#1565C0")
    axis.set(title=f"Channel permutation importance (eval AP baseline = {baseline_ap:.4f})", xlabel="Drop in Average Precision")
    axis.grid(axis="x", alpha=0.25)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Samples: {len(inputs):,}")
    print(f"Baseline AP: {baseline_ap:.4f}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batches", type=int, default=50, help="Eight eval samples per batch.")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    main(arguments.model, arguments.output, arguments.batches, arguments.seed)
