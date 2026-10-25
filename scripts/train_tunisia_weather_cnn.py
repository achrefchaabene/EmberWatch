"""Train and evaluate the first local Tunisia fire-weather CNN.

Training: 2019-2022. Validation: 2023, used only to select an alert threshold.
Test: 2024-2025, held out until the final report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep the first local Tunisia experiment stable on machines with limited RAM.
# These must be set before NumPy, scikit-learn, or TensorFlow are imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Disable oneDNN for this small CPU-only experiment.  Some Windows CPU builds
# fail to create a convolution-gradient primitive for the 32x18 grid.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
import xarray as xr
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tunisia_weather_cnn import build_tunisia_weather_cnn
from src.models.unet import WeightedBinaryCrossentropy


CUBE_PATH = PROJECT_ROOT / "data" / "processed" / "tunisia" / "tunisia_fire_weather_cube_era5_025.nc"
WEATHER_FEATURES = (
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_min_c",
    "relative_humidity_mean_pct",
    "wind_u_mean_mps",
    "wind_v_mean_mps",
    "wind_speed_mean_mps",
    "wind_speed_max_mps",
    "precipitation_sum_mm",
)
STATIC_FEATURES = (
    "elevation_m",
    "coarse_slope_degrees",
    "tree_cover_fraction",
    "shrubland_fraction",
    "grassland_fraction",
    "cropland_fraction",
    "built_up_fraction",
    "bare_sparse_fraction",
    "water_fraction",
    "wetland_fraction",
)
MODEL_PATH = PROJECT_ROOT / "artifacts" / "tunisia_weather_terrain_landcover_boundary_cnn_v4.keras"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "tunisia_weather_terrain_landcover_boundary_cnn_v4_test_metrics.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Tunisia weather CNN on the ERA5 grid.")
    parser.add_argument("--cube", type=Path, default=CUBE_PATH)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-weight", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cube_to_arrays(cube: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    previous_fire = cube["prev_fire_mask"].values.astype(np.float32)[..., None]
    weather = np.stack([cube[name].values for name in WEATHER_FEATURES], axis=-1).astype(np.float32)
    static = np.stack([cube[name].values for name in STATIC_FEATURES], axis=-1).astype(np.float32)
    # xarray stores terrain as latitude x longitude. Repeat the static values
    # across dates so every daily sample has the same terrain channels.
    static = np.broadcast_to(static[None, ...], (weather.shape[0], *static.shape)).copy()
    covariates = np.concatenate([weather, static], axis=-1)
    features = np.concatenate([previous_fire, covariates], axis=-1)
    targets = cube["next_fire_mask"].values.astype(np.float32)[..., None]
    return features, targets


def tunisia_weights(cube: xr.Dataset) -> np.ndarray:
    """Return a daily loss/metric mask for cells at least half inside Tunisia."""
    cell_mask = cube["is_tunisia_cell"].values.astype(bool)
    return np.broadcast_to(cell_mask[None, ...], (cube.sizes["valid_time"], *cell_mask.shape)).copy()


def normalise_weather(train_x: np.ndarray, train_weights: np.ndarray, *other_x: np.ndarray) -> tuple[np.ndarray, ...]:
    """Standardise all continuous covariates; previous fire remains binary."""
    training_values = train_x[..., 1:][train_weights]
    mean = training_values.mean(axis=0, keepdims=True).reshape(1, 1, 1, -1)
    std = training_values.std(axis=0, keepdims=True).reshape(1, 1, 1, -1)
    std = np.maximum(std, 1e-6)

    normalised = []
    for values in (train_x, *other_x):
        result = values.copy()
        result[..., 1:] = (result[..., 1:] - mean) / std
        normalised.append(result)
    return tuple(normalised)


def select_threshold(targets: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.nanargmax(f1))
    return float(thresholds[min(best, len(thresholds) - 1)]) if len(thresholds) else 0.5


def metrics(targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (scores >= threshold).astype(np.int8)
    precision, recall, _ = precision_recall_curve(targets, scores)
    f1_values = 2 * precision * recall / (precision + recall + 1e-12)
    best_index = int(np.nanargmax(f1_values))
    return {
        "average_precision": float(average_precision_score(targets, scores)),
        "f1": float(f1_score(targets, prediction, zero_division=0)),
        "precision": float(precision_score(targets, prediction, zero_division=0)),
        "recall": float(recall_score(targets, prediction, zero_division=0)),
        "best_f1_oracle": float(f1_values[best_index]),
        "positive_rate": float(targets.mean()),
    }


def main() -> None:
    args = parse_arguments()
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if not args.cube.exists():
        raise FileNotFoundError(f"Tunisia cube not found: {args.cube}")

    cube = xr.open_dataset(args.cube)
    train_cube = cube.sel(valid_time=slice("2019-01-01", "2022-12-31"))
    validation_cube = cube.sel(valid_time=slice("2023-01-01", "2023-12-31"))
    test_cube = cube.sel(valid_time=slice("2024-01-01", "2025-12-31"))

    train_x, train_y = cube_to_arrays(train_cube)
    validation_x, validation_y = cube_to_arrays(validation_cube)
    test_x, test_y = cube_to_arrays(test_cube)
    train_weights = tunisia_weights(train_cube)
    validation_weights = tunisia_weights(validation_cube)
    test_weights = tunisia_weights(test_cube)
    train_x, validation_x, test_x = normalise_weather(train_x, train_weights, validation_x, test_x)

    # Remove only the final available target from test. Earlier periods retain
    # their next-day labels because their next day is in the following split.
    test_available = test_cube["target_is_available"].values.astype(bool)
    test_x, test_y, test_weights = test_x[test_available], test_y[test_available], test_weights[test_available]

    model = build_tunisia_weather_cnn(tuple(train_x.shape[1:]))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=WeightedBinaryCrossentropy(positive_weight=args.positive_weight),
        metrics=[tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]
    model.fit(
        train_x,
        train_y,
        sample_weight=train_weights[..., None].astype(np.float32),
        validation_data=(validation_x, validation_y, validation_weights[..., None].astype(np.float32)),
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
    )

    validation_scores = model.predict(validation_x, batch_size=args.batch_size, verbose=0).reshape(-1)
    validation_targets = validation_y[..., 0][validation_weights].astype(np.int8)
    validation_scores = validation_scores.reshape(validation_y.shape[:-1])[validation_weights]
    threshold = select_threshold(validation_targets, validation_scores)
    test_scores = model.predict(test_x, batch_size=args.batch_size, verbose=0).reshape(-1)
    test_targets = test_y[..., 0][test_weights].astype(np.int8)
    test_scores = test_scores.reshape(test_y.shape[:-1])[test_weights]
    test_report = metrics(test_targets, test_scores, threshold)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    report = {
        "model": model.name,
        "features": ["prev_fire_mask", *WEATHER_FEATURES, *STATIC_FEATURES],
        "grid": {"height": int(train_x.shape[1]), "width": int(train_x.shape[2]), "resolution": "0.25 degrees"},
        "training_period": "2019-01-01 to 2022-12-31",
        "validation_period": "2023-01-01 to 2023-12-31",
        "test_period": "2024-01-01 to 2025-12-30",
        "threshold_selected_on_validation": threshold,
        "positive_weight": args.positive_weight,
        "spatial_training_mask": "is_tunisia_cell == 1 (at least 50% of the ERA5 cell is inside Tunisia)",
        "test": test_report,
        "caveat": "Coarse ERA5-grid baseline. FIRMS active-fire detections are not burned-area perimeters.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Tunisia weather CNN test results")
    print(f"Selected validation threshold: {threshold:.4f}")
    print(f"Test AP: {test_report['average_precision']:.4f}")
    print(f"Test F1: {test_report['f1']:.4f}")
    print(f"Test precision: {test_report['precision']:.4f}")
    print(f"Test recall: {test_report['recall']:.4f}")
    print(f"Test oracle best F1: {test_report['best_f1_oracle']:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {REPORT_PATH}")


if __name__ == "__main__":
    main()
