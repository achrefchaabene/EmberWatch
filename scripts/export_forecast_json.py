"""Export an M6 next-day forecast as a dashboard-friendly JSON grid."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.data.tfrecord_loader import load_preprocessed_dataset


DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "unet_m6_v2.keras"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "forecast_sample.json"


def select_sample(split: str, index: int, minimum_fire_pixels: int | None):
    dataset = load_preprocessed_dataset(split, batch_size=1, shuffle=False)
    for current_index, sample in enumerate(dataset):
        if minimum_fire_pixels is None and current_index == index:
            return current_index, sample
        if minimum_fire_pixels is not None:
            _, target, valid_mask = sample
            fire_pixels = int(tf.reduce_sum(target * valid_mask).numpy())
            if fire_pixels >= minimum_fire_pixels:
                return current_index, sample
    raise IndexError("No matching sample was found in the requested split.")


def main(
    model_path: Path,
    output_path: Path,
    split: str,
    index: int,
    threshold: float,
    minimum_fire_pixels: int | None,
) -> None:
    model = tf.keras.models.load_model(model_path, compile=False)
    selected_index, (inputs, _, valid_mask) = select_sample(split, index, minimum_fire_pixels)
    probability = model(inputs, training=False).numpy()[0, ..., 0]
    valid = valid_mask.numpy()[0, ..., 0].astype(bool)
    forecast_mask = np.where(valid, (probability >= threshold).astype(np.int8), -1)

    payload = {
        "schema_version": "1.0",
        "model": "unet_m6_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"split": split, "sample_index": selected_index},
        "threshold": threshold,
        "grid": {
            "height": 64,
            "width": 64,
            "coordinate_system": "local_patch_pixels",
            "note": "No geographic coordinates are available in this TFRecord dataset.",
        },
        "summary": {
            "valid_pixels": int(valid.sum()),
            "predicted_fire_pixels": int((forecast_mask == 1).sum()),
            "mean_probability": round(float(probability[valid].mean()), 6),
            "max_probability": round(float(probability[valid].max()), 6),
        },
        "probability_grid": np.round(probability, 4).tolist(),
        "forecast_mask": forecast_mask.tolist(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")
    print(f"Split/index: {split}/{selected_index}")
    print(f"Predicted fire pixels: {payload['summary']['predicted_fire_pixels']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "eval", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.57)
    parser.add_argument("--min-fire-pixels", type=int)
    arguments = parser.parse_args()
    main(
        model_path=arguments.model,
        output_path=arguments.output,
        split=arguments.split,
        index=arguments.index,
        threshold=arguments.threshold,
        minimum_fire_pixels=arguments.min_fire_pixels,
    )
