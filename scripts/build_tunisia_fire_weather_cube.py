"""Build a first Tunisia fire-weather data cube on the native ERA5 grid.

The cube aligns daily NASA FIRMS VIIRS detections with daily ERA5 weather.
Each cell is an ERA5 0.25-degree grid cell.  This is intentionally a first,
coarse local baseline: weather is not upsampled and presented as 1-km data.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.tunisia import TEST_END, TRAIN_START


FIRMS_PATH = PROJECT_ROOT / "data" / "processed" / "tunisia" / "firms_viirs_tunisia.csv"
ERA5_DAILY_DIR = PROJECT_ROOT / "data" / "processed" / "tunisia" / "era5_daily"
TERRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "tunisia" / "terrain_era5_025.nc"
WORLDCOVER_PATH = PROJECT_ROOT / "data" / "processed" / "tunisia" / "worldcover_2021_era5_025.nc"
BOUNDARY_PATH = PROJECT_ROOT / "data" / "processed" / "tunisia" / "tunisia_boundary_fraction_era5_025.nc"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tunisia" / "tunisia_fire_weather_cube_era5_025.nc"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align FIRMS daily fire masks with daily ERA5 weather on the native ERA5 grid."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def open_daily_era5() -> xr.Dataset:
    paths = [ERA5_DAILY_DIR / f"era5_tunisia_daily_{year}.nc" for year in range(2019, 2026)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing daily ERA5 files: " + ", ".join(str(path) for path in missing))
    datasets = [xr.open_dataset(path) for path in paths]
    daily = xr.concat(datasets, dim="valid_time").sortby("valid_time")
    expected_days = np.arange(np.datetime64(TRAIN_START), np.datetime64("2026-01-01"), dtype="datetime64[D]")
    actual_days = daily.valid_time.values.astype("datetime64[D]")
    if not np.array_equal(actual_days, expected_days):
        raise ValueError("ERA5 daily time axis is not complete from 2019-01-01 through 2025-12-31.")
    return daily


def closest_index(values: np.ndarray, value: float) -> int:
    return int(np.abs(values - value).argmin())


def rasterise_firms(days: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[np.ndarray, int]:
    if not FIRMS_PATH.exists():
        raise FileNotFoundError(f"Cleaned FIRMS labels not found: {FIRMS_PATH}")

    day_to_index = {str(day): index for index, day in enumerate(days.astype("datetime64[D]").astype(str))}
    counts = np.zeros((len(days), len(latitudes), len(longitudes)), dtype=np.int16)
    accepted = 0

    with FIRMS_PATH.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            day_index = day_to_index.get(row["acq_date"])
            if day_index is None:
                continue
            lat_index = closest_index(latitudes, float(row["latitude"]))
            lon_index = closest_index(longitudes, float(row["longitude"]))
            if counts[day_index, lat_index, lon_index] < np.iinfo(np.int16).max:
                counts[day_index, lat_index, lon_index] += 1
            accepted += 1
    return counts, accepted


def main() -> None:
    args = parse_arguments()
    weather = open_daily_era5().load()
    if not TERRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Terrain features not found: {TERRAIN_PATH}. Run build_tunisia_terrain_features.py first."
        )
    terrain = xr.open_dataset(TERRAIN_PATH).load()
    if not WORLDCOVER_PATH.exists():
        raise FileNotFoundError(
            f"WorldCover features not found: {WORLDCOVER_PATH}. Run build_tunisia_worldcover_features.py first."
        )
    worldcover = xr.open_dataset(WORLDCOVER_PATH).load()
    if not BOUNDARY_PATH.exists():
        raise FileNotFoundError(
            f"Tunisia boundary mask not found: {BOUNDARY_PATH}. Run build_tunisia_boundary_mask.py first."
        )
    boundary = xr.open_dataset(BOUNDARY_PATH).load()
    _, terrain = xr.align(weather.isel(valid_time=0, drop=True), terrain, join="exact")
    _, worldcover = xr.align(weather.isel(valid_time=0, drop=True), worldcover, join="exact")
    _, boundary = xr.align(weather.isel(valid_time=0, drop=True), boundary, join="exact")
    days = weather.valid_time.values.astype("datetime64[D]")
    latitudes = weather.latitude.values
    longitudes = weather.longitude.values
    detection_count, accepted_detections = rasterise_firms(days, latitudes, longitudes)

    fire_mask = (detection_count > 0).astype(np.int8)
    next_fire_mask = np.zeros_like(fire_mask)
    next_fire_mask[:-1] = fire_mask[1:]
    target_is_available = np.ones(len(days), dtype=np.int8)
    target_is_available[-1] = 0

    cube = weather.assign(
        firms_detection_count=(("valid_time", "latitude", "longitude"), detection_count),
        prev_fire_mask=(("valid_time", "latitude", "longitude"), fire_mask),
        next_fire_mask=(("valid_time", "latitude", "longitude"), next_fire_mask),
        target_is_available=(("valid_time",), target_is_available),
    )
    cube = xr.merge([cube, terrain, worldcover, boundary])
    cube["firms_detection_count"].attrs.update(description="Number of accepted FIRMS VIIRS detections in the ERA5 cell on this date")
    cube["prev_fire_mask"].attrs.update(description="1 when an accepted FIRMS detection occurs in this cell on day t")
    cube["next_fire_mask"].attrs.update(description="1 when an accepted FIRMS detection occurs in this cell on day t plus 1")
    cube["target_is_available"].attrs.update(description="0 only for the final day because no next-day target exists")
    cube.attrs.update(
        title="Tunisia daily fire and weather cube on native ERA5 grid",
        crs="EPSG:4326",
        spatial_resolution="0.25 degrees native ERA5 grid",
        temporal_coverage=f"{TRAIN_START} through {TEST_END}",
        fire_source="NASA FIRMS VIIRS S-NPP accepted nominal/high confidence detections",
        terrain_source="AWS Terrain Tiles SRTM HGT sampled at ERA5 cell centres",
        landcover_source="ESA WorldCover 2021 v200 class fractions aggregated to the ERA5 grid",
        boundary_source="Natural Earth Admin 0 country boundary; training/evaluation use cells at least 50 percent inside Tunisia.",
        caveat="FIRMS active-fire detections are not burned-area perimeters; ERA5 weather and terrain features are coarse at about 25 km.",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: {"zlib": True, "complevel": 4} for name in cube.data_vars}
    cube.to_netcdf(args.output, encoding=encoding)

    print(f"Days: {len(days):,}")
    print(f"Grid: {len(latitudes)} lat x {len(longitudes)} lon")
    print(f"Accepted FIRMS detections rasterised: {accepted_detections:,}")
    print(f"Active fire cell-days: {int(fire_mask.sum()):,}")
    print(f"Target positive cell-days: {int(next_fire_mask[:-1].sum()):,}")
    print("Static features added: terrain and ESA WorldCover class fractions")
    print(f"Train dates: {TRAIN_START} to 2023-12-31")
    print(f"Test dates: 2024-01-01 to {TEST_END}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
