"""Create coarse elevation and slope features aligned with the Tunisia ERA5 grid.

SRTM 1-arc-second HGT tiles are sampled at ERA5 cell centres.  The resulting
slope is deliberately a *coarse 0.25-degree terrain gradient*, not a 30-m
slope product.  It is appropriate for the first local ERA5-grid baseline.
"""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path

import numpy as np
import requests
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ERA5 = PROJECT_ROOT / "data" / "processed" / "tunisia" / "era5_daily" / "era5_tunisia_daily_2019.nc"
RAW_TERRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "tunisia" / "terrain" / "srtm_hgt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tunisia" / "terrain_era5_025.nc"
SRTM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{hemisphere}{latitude:02d}/{name}.hgt.gz"
HGT_SIZE = 3601


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Tunisia ERA5-grid elevation and slope features from SRTM.")
    parser.add_argument("--reference-era5", type=Path, default=REFERENCE_ERA5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--keep-raw-tiles",
        action="store_true",
        help="Keep downloaded .hgt.gz tiles after they are sampled (uses more disk space).",
    )
    return parser.parse_args()


def tile_name(latitude: int, longitude: int) -> str:
    return f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}"


def download_tile(latitude: int, longitude: int) -> Path:
    name = tile_name(latitude, longitude)
    path = RAW_TERRAIN_DIR / f"{name}.hgt.gz"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = SRTM_URL.format(
        hemisphere="N" if latitude >= 0 else "S",
        latitude=abs(latitude),
        name=name,
    )
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"Downloaded: {name}")
    return path


def read_hgt(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as source:
        raw = source.read()
    expected_bytes = HGT_SIZE * HGT_SIZE * 2
    if len(raw) != expected_bytes:
        raise ValueError(f"Unexpected SRTM tile size in {path}: {len(raw):,} bytes")
    return np.frombuffer(raw, dtype=">i2").reshape(HGT_SIZE, HGT_SIZE)


def sample_tile(tile: np.ndarray, latitude: float, longitude: float, tile_latitude: int, tile_longitude: int) -> float:
    # HGT rows run north to south; columns run west to east.
    row = int(round((tile_latitude + 1 - latitude) * 3600))
    col = int(round((longitude - tile_longitude) * 3600))
    row = int(np.clip(row, 0, HGT_SIZE - 1))
    col = int(np.clip(col, 0, HGT_SIZE - 1))
    value = int(tile[row, col])
    return np.nan if value <= -32768 else float(value)


def main() -> None:
    args = parse_arguments()
    if not args.reference_era5.exists():
        raise FileNotFoundError(f"Reference ERA5 file not found: {args.reference_era5}")
    reference = xr.open_dataset(args.reference_era5)
    latitudes = reference.latitude.values.astype(float)
    longitudes = reference.longitude.values.astype(float)

    samples_by_tile: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for lat_index, latitude in enumerate(latitudes):
        for lon_index, longitude in enumerate(longitudes):
            key = (math.floor(latitude), math.floor(longitude))
            samples_by_tile.setdefault(key, []).append((lat_index, lon_index))

    elevation = np.full((len(latitudes), len(longitudes)), np.nan, dtype=np.float32)
    for (tile_latitude, tile_longitude), sample_indices in samples_by_tile.items():
        path = download_tile(tile_latitude, tile_longitude)
        tile = read_hgt(path)
        for lat_index, lon_index in sample_indices:
            latitude, longitude = latitudes[lat_index], longitudes[lon_index]
            elevation[lat_index, lon_index] = sample_tile(
                tile, latitude, longitude, tile_latitude, tile_longitude
            )
        del tile
        if not args.keep_raw_tiles:
            path.unlink(missing_ok=True)

    # Sea or missing samples are encoded as 0 m.  A Tunisia country/land mask
    # will be applied in a later stage before operational mapping.
    elevation = np.nan_to_num(elevation, nan=0.0)
    lat_step_m = abs(float(np.diff(latitudes).mean())) * 111_320
    dz_dlat = np.gradient(elevation, lat_step_m, axis=0)
    longitude_step_m = abs(float(np.diff(longitudes).mean())) * 111_320 * np.cos(np.deg2rad(latitudes))
    dz_dlon = np.empty_like(elevation)
    for row, spacing in enumerate(longitude_step_m):
        dz_dlon[row] = np.gradient(elevation[row], max(float(spacing), 1.0))
    slope_degrees = np.degrees(np.arctan(np.hypot(dz_dlat, dz_dlon))).astype(np.float32)

    terrain = xr.Dataset(
        {
            "elevation_m": (("latitude", "longitude"), elevation),
            "coarse_slope_degrees": (("latitude", "longitude"), slope_degrees),
        },
        coords={"latitude": latitudes, "longitude": longitudes},
        attrs={
            "title": "Tunisia static terrain features aligned to native ERA5 grid",
            "crs": "EPSG:4326",
            "spatial_resolution": "0.25 degrees",
            "elevation_source": "AWS Terrain Tiles SRTM HGT, sampled at ERA5 cell centres",
            "slope_definition": "Terrain gradient computed from the 0.25-degree sampled elevation grid",
            "caveat": "This product includes the full buffered ERA5 rectangle and is not yet clipped to a Tunisia land boundary.",
        },
    )
    terrain["elevation_m"].attrs["units"] = "m"
    terrain["coarse_slope_degrees"].attrs["units"] = "degrees"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    terrain.to_netcdf(args.output, encoding={name: {"zlib": True, "complevel": 4} for name in terrain.data_vars})

    print(f"Grid: {len(latitudes)} lat x {len(longitudes)} lon")
    print(f"SRTM tiles used: {len(samples_by_tile)}")
    print(f"Elevation min/max: {elevation.min():.1f} / {elevation.max():.1f} m")
    print(f"Slope min/max: {slope_degrees.min():.2f} / {slope_degrees.max():.2f} degrees")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
