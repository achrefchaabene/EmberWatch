"""Export a next-day wildfire prediction figure for a TFRecord sample.

Example:
    .venv\\Scripts\\python.exe scripts\\predict_sample.py --split test --index 0
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.tfrecord_loader import load_preprocessed_dataset
from src.models.unet import WeightedBinaryCrossentropy


DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "unet_m6_v2.keras"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "prediction_sample.png"


def get_sample(split: str, index: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Load the zero-based record index from a dataset split."""
    dataset = load_preprocessed_dataset(split, batch_size=1, shuffle=False).skip(index).take(1)
    try:
        return next(iter(dataset))
    except StopIteration as error:
        raise IndexError(f"No record {index} in the {split!r} split.") from error


def find_positive_sample(split: str, minimum_fire_pixels: int) -> tuple[int, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return the first sample whose known target has enough fire pixels."""
    dataset = load_preprocessed_dataset(split, batch_size=1, shuffle=False)
    for index, (inputs, target, valid_mask) in enumerate(dataset):
        fire_pixels = int(tf.reduce_sum(target * valid_mask).numpy())
        if fire_pixels >= minimum_fire_pixels:
            return index, inputs, target, valid_mask
    raise ValueError(
        f"No {split!r} sample contains at least {minimum_fire_pixels} fire pixels."
    )


def export_prediction(
    model_path: Path,
    output_path: Path,
    split: str,
    index: int,
    threshold: float,
    minimum_fire_pixels: int | None = None,
) -> None:
    """Write a four-panel prediction figure and report basic sample metrics."""
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedBinaryCrossentropy": WeightedBinaryCrossentropy},
    )
    if minimum_fire_pixels is None:
        inputs, target, valid_mask = get_sample(split, index)
    else:
        index, inputs, target, valid_mask = find_positive_sample(split, minimum_fire_pixels)

    probability = model(inputs, training=False).numpy()[0, ..., 0]
    previous_fire = inputs.numpy()[0, ..., 0]
    true_fire = target.numpy()[0, ..., 0]
    valid = valid_mask.numpy()[0, ..., 0].astype(bool)
    prediction = np.where(valid, (probability >= threshold).astype(float), np.nan)

    figure, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = (
        (previous_fire, "Previous fire"),
        (np.where(valid, true_fire, np.nan), "True next-day fire"),
        (np.where(valid, probability, np.nan), "U-Net fire probability"),
        (prediction, f"Prediction (threshold = {threshold:.2f})"),
    )
    for axis, (image, title) in zip(axes, panels):
        plot = axis.imshow(image, cmap="hot", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(plot, ax=axis, fraction=0.046)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    true_positive = int(np.logical_and(prediction == 1, true_fire == 1).sum())
    predicted_fire = int(np.nansum(prediction))
    actual_fire = int((true_fire[valid] == 1).sum())
    print(f"Saved: {output_path}")
    print(f"Split/index: {split}/{index}")
    print(f"Actual fire pixels: {actual_fire}")
    print(f"Predicted fire pixels: {predicted_fire}")
    print(f"True-positive pixels: {true_positive}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "eval", "test"), default="test")
    parser.add_argument("--index", type=int, default=0, help="Zero-based TFRecord sample index.")
    parser.add_argument("--threshold", type=float, default=0.57)
    parser.add_argument(
        "--min-fire-pixels",
        type=int,
        help="Ignore --index and export the first sample with at least this many true fire pixels.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export_prediction(
        model_path=arguments.model,
        output_path=arguments.output,
        split=arguments.split,
        index=arguments.index,
        threshold=arguments.threshold,
        minimum_fire_pixels=arguments.min_fire_pixels,
    )
