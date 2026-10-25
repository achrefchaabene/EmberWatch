"""Export a geographically referenced historical Tunisia forecast replay.

The script applies exactly the feature order and training-period scaling used
by the Tunisia v3 model.  Its output is intended for the Role B dashboard,
not as an operational public alert service.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Keep inference stable on the same low-memory Windows machines used to train.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.tunisia import ERA5_GRID_RESOLUTION_DEGREES


CUBE_PATH = PROJECT_ROOT / "data" / "processed" / "tunisia" / "tunisia_fire_weather_cube_era5_025.nc"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "tunisia_weather_terrain_landcover_boundary_cnn_v4.keras"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "tunisia_weather_terrain_landcover_boundary_cnn_v4_test_metrics.json"
JSON_PATH = PROJECT_ROOT / "artifacts" / "tunisia_forecast_latest.json"
GEOJSON_PATH = PROJECT_ROOT / "artifacts" / "tunisia_forecast_alerts.geojson"

WEATHER_FEATURES = (
    "temperature_mean_c", "temperature_max_c", "temperature_min_c",
    "relative_humidity_mean_pct", "wind_u_mean_mps", "wind_v_mean_mps",
    "wind_speed_mean_mps", "wind_speed_max_mps", "precipitation_sum_mm",
)
STATIC_FEATURES = (
    "elevation_m", "coarse_slope_degrees", "tree_cover_fraction",
    "shrubland_fraction", "grassland_fraction", "cropland_fraction",
    "built_up_fraction", "bare_sparse_fraction", "water_fraction",
    "wetland_fraction",
)
FEATURES = ("prev_fire_mask", *WEATHER_FEATURES, *STATIC_FEATURES)


def cube_to_features(cube: xr.Dataset) -> np.ndarray:
    """Build channels in the exact order used by train_tunisia_weather_cnn."""
    previous_fire = cube["prev_fire_mask"].values.astype(np.float32)[..., None]
    weather = np.stack([cube[name].values for name in WEATHER_FEATURES], axis=-1).astype(np.float32)
    static = np.stack([cube[name].values for name in STATIC_FEATURES], axis=-1).astype(np.float32)
    static = np.broadcast_to(static[None, ...], (weather.shape[0], *static.shape))
    return np.concatenate([previous_fire, weather, static], axis=-1)


def training_scaling(cube: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    train_features = cube_to_features(cube.sel(valid_time=slice("2019-01-01", "2022-12-31")))
    train_mask = cube.sel(valid_time=slice("2019-01-01", "2022-12-31"))["is_tunisia_cell"].values.astype(bool)
    train_mask = np.broadcast_to(train_mask[None, ...], train_features.shape[:-1])
    values = train_features[..., 1:][train_mask]
    mean = values.mean(axis=0, keepdims=True).reshape(1, 1, 1, -1)
    std = np.maximum(values.std(axis=0, keepdims=True).reshape(1, 1, 1, -1), 1e-6)
    return mean, std


def cell_polygon(latitude: float, longitude: float) -> list[list[list[float]]]:
    half = ERA5_GRID_RESOLUTION_DEGREES / 2
    west, east = longitude - half, longitude + half
    south, north = latitude - half, latitude + half
    return [[[west, south], [east, south], [east, north], [west, north], [west, south]]]


def main(args: argparse.Namespace) -> None:
    for path in (args.cube, args.model, args.metrics):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    cube = xr.open_dataset(args.cube)
    requested_date = np.datetime64(args.date)
    if requested_date not in cube.valid_time.values:
        raise ValueError(f"Date is not in the Tunisia cube: {args.date}")

    mean, std = training_scaling(cube)
    source_cube = cube.sel(valid_time=[requested_date])
    features = cube_to_features(source_cube).copy()
    features[..., 1:] = (features[..., 1:] - mean) / std

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    threshold = float(metrics["threshold_selected_on_validation"])
    model = tf.keras.models.load_model(args.model, compile=False)
    probability = model.predict(features, batch_size=1, verbose=0)[0, ..., 0]

    water = source_cube["water_fraction"].values
    tunisia_fraction = source_cube["tunisia_fraction"].values
    alert_eligible = tunisia_fraction >= args.min_tunisia_fraction
    alert_mask = (probability >= threshold) & alert_eligible
    source_day = datetime.strptime(args.date, "%Y-%m-%d").date()
    forecast_day = source_day + timedelta(days=1)
    latitudes = cube.latitude.values.astype(float)
    longitudes = cube.longitude.values.astype(float)

    common_metadata = {
        "model": metrics["model"],
        "model_artifact": args.model.name,
        "feature_order": list(FEATURES),
        "source_date": source_day.isoformat(),
        "forecast_for": forecast_day.isoformat(),
        "forecast_type": "historical_replay",
        "threshold_selected_on_2023_validation": threshold,
        "grid_resolution_degrees": ERA5_GRID_RESOLUTION_DEGREES,
        "crs": "EPSG:4326",
        "alert_filter": {
            "minimum_tunisia_fraction": args.min_tunisia_fraction,
            "reason": "Keep cells at least 50% inside Tunisia according to the Natural Earth country boundary.",
        },
        "caveat": "Coarse 0.25-degree ERA5 grid. FIRMS active-fire detections are not burned-area perimeters. This is not an operational warning product.",
    }
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **common_metadata,
        "grid": {"height": int(probability.shape[0]), "width": int(probability.shape[1]), "latitude": latitudes.tolist(), "longitude": longitudes.tolist()},
        "summary": {
            "grid_cells": int(probability.size),
            "eligible_land_cells": int(alert_eligible.sum()),
            "predicted_alert_cells": int(alert_mask.sum()),
            "max_probability": round(float(probability.max()), 6),
            "mean_probability": round(float(probability.mean()), 6),
        },
        "probability_grid": np.round(probability, 6).tolist(),
        "alert_mask": alert_mask.astype(np.uint8).tolist(),
    }

    geojson = {
        "type": "FeatureCollection",
        "metadata": common_metadata,
        "features": [],
    }
    for row, latitude in enumerate(latitudes):
        for column, longitude in enumerate(longitudes):
            if alert_mask[row, column]:
                geojson["features"].append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": cell_polygon(latitude, longitude)},
                    "properties": {
                        "latitude": round(float(latitude), 5),
                        "longitude": round(float(longitude), 5),
                        "probability": round(float(probability[row, column]), 6),
                        "threshold": threshold,
                        "source_date": source_day.isoformat(),
                        "forecast_for": forecast_day.isoformat(),
                        "water_fraction": round(float(water[row, column]), 4),
                        "tunisia_fraction": round(float(tunisia_fraction[row, column]), 4),
                    },
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.geojson.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    print(f"Saved dashboard JSON: {args.output}")
    print(f"Saved alert GeoJSON: {args.geojson}")
    print(f"Forecast: {source_day.isoformat()} -> {forecast_day.isoformat()}")
    print(f"Alert cells: {int(alert_mask.sum())} / {int(alert_eligible.sum())} eligible cells")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2025-12-30", help="Source day in YYYY-MM-DD; target forecast is the following day.")
    parser.add_argument("--cube", type=Path, default=CUBE_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--output", type=Path, default=JSON_PATH)
    parser.add_argument("--geojson", type=Path, default=GEOJSON_PATH)
    parser.add_argument("--min-tunisia-fraction", type=float, default=0.5)
    main(parser.parse_args())
